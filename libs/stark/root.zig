//! STARK backend (F2): constraint composition + quotient + FRI low-degree
//! test + query-time verification.
//!
//! Closes the gap that made the pre-F2 prover useless (TODO "STACK GAP"):
//! libs/prove flattened the trace and asked FRI whether the result was
//! low-degree. A real witness (a running GEMM sum, say) is high-degree in
//! that flattening, so honest proofs were rejected. Here FRI is applied to
//! the QUOTIENT of the composed constraint polynomial, which is low-degree
//! precisely when the trace satisfies the AIR.
//!
//! Protocol (monolithic, per layer — docs/BLUE_PRINT.md §5.2 v1):
//!
//!   H   = H_k      trace domain, n = 2^k rows
//!   LDE = H_{k+b}  subgroup LDE, b chosen from the AIR degree
//!
//!   1. Each column is interpolated through its n values on H (IFFT),
//!      zero-padded to 2^(k+b) and transformed back (FFT) — the LDE.
//!   2. Commit: one Merkle leaf per LDE row over all columns at that row
//!      and its two cyclic neighbours (libs/stark/commit).
//!   3. Fiat-Shamir: absorb the column root, squeeze alpha_0..alpha_{m-1}.
//!   4. Compose: P(x) = sum_i alpha^i * C_i(x), evaluated on the LDE.
//!   5. Quotient Q = P / Z_H with Z_H(x) = x^n - 1: interpolate P (IFFT),
//!      long-divide, transform back. A non-zero remainder means the trace
//!      violates the AIR, and the prover returns ConstraintViolation rather
//!      than producing a proof of nothing.
//!   6. FRI on Q's LDE evaluations (libs/fri) — proves deg Q < bound.
//!   7. Queries: for each FRI query index i the prover opens the column
//!      leaf (authenticating rows i-1, i, i+1) and the verifier recomputes
//!      P(x_i), then checks  P(x_i) == Q(x_i) * (x_i^n - 1).
//!
//! Step 7 is where the AIR is enforced. At a point of H the identity
//! degenerates to P(x) == 0 (the raw constraint); elsewhere it ties the
//! quotient to the authenticated columns.
//!
//! The LDE is a SUBGROUP rather than a coset so the existing norm-1 FRI
//! applies unchanged; the price is the two extra transforms in step 5 that
//! a coset LDE would avoid.
//!
//! Now also here: the LogUp core (logup.zig) and a bit-exact fp16 multiply
//! AIR (float_air.zig, 108 composed constraints), both built on the
//! composition machinery in this file.
//!
//! Not here yet: DIVISION in the IR, pinning a fixed table to a public
//! input (so a lookup proves w is a permutation of u but not that u is
//! *the* table), and multi-layer composition (F3).

const std = @import("std");
const fp2 = @import("../fri/fp2.zig");
const fri = @import("../fri/root.zig");
const domain_lib = @import("../fri/domain.zig");
const transcript_lib = @import("../transcript.zig");
const fft = @import("../fri/fft.zig");
const expr = @import("expr.zig");
const commit_lib = @import("commit.zig");

pub const Fp2 = fp2.Fp2;
pub const Goldilocks = fp2.Goldilocks;
pub const Domain = domain_lib.Domain;
pub const Transcript = transcript_lib.Transcript;
pub const MerkleProof = commit_lib.MerkleProof;
pub const Commitment = commit_lib.Commitment;

pub const System = expr.System;
pub const Constraint = expr.Constraint;
pub const Window = expr.Window;

pub const Error = error{
    OutOfMemory,
    InvalidTrace,
    InvalidConfig,
    /// The prover's own trace does not satisfy the AIR: the quotient
    /// division left a non-zero remainder.
    ConstraintViolation,
    InvalidProof,
};

/// Column-major trace: `columns[c][row]`, each of length `rows`.
pub const Trace = struct {
    rows: usize,
    columns: []const []const Fp2,

    pub fn validate(self: Trace, system: System) Error!void {
        if (self.columns.len == 0) return Error.InvalidTrace;
        if (self.rows == 0 or !std.math.isPowerOfTwo(self.rows)) return Error.InvalidTrace;
        if (system.trace_rows) |expected| if (self.rows != expected) return Error.InvalidTrace;
        for (self.columns) |c| {
            if (c.len != self.rows) return Error.InvalidTrace;
        }
        if (system.hasWideOffsets()) return Error.InvalidTrace;
        if (system.maxColumn()) |m| {
            if (@as(usize, m) >= self.columns.len) return Error.InvalidTrace;
        }
    }
};

pub const Config = struct {
    /// log2 of the trace domain (rows = 2^log_trace).
    log_trace: u6,
    /// Extra LDE bits; must satisfy 2^log_blowup >= max constraint degree.
    log_blowup: u6,
    /// FRI config over the quotient's LDE (log_domain = log_trace+log_blowup).
    fri: fri.Config,

    pub fn logLde(self: Config) u6 {
        return self.log_trace + self.log_blowup;
    }

    pub fn validate(self: Config, system: System) Error!void {
        if (self.log_blowup == 0) return Error.InvalidConfig;
        if (self.fri.log_domain != self.logLde()) return Error.InvalidConfig;
        const d = system.maxDegree();
        // A degree-d constraint composes to degree < d * rows, so the LDE
        // must be at least that large for P to be recoverable by IFFT.
        if (self.log_blowup < ceilLog2(d)) return Error.InvalidConfig;
        // Q = P / Z_H loses exactly the trace domain's degree, so
        // deg Q < (d - 1) * rows. The FRI residual bound must cover it,
        // or an HONEST proof gets rejected for being "too high degree".
        if (self.fri.log_residual_degree < self.log_trace + ceilLog2(d - 1)) {
            return Error.InvalidConfig;
        }
        if (system.trace_rows) |expected| {
            if ((@as(usize, 1) << self.log_trace) != expected) return Error.InvalidConfig;
        }
        // Each exempt row adds one degree of headroom to the quotient (P·E
        // is one degree taller than P), so the FRI residual bound has to
        // cover it or an HONEST proof is rejected as "too high degree".
        // deg Q <= d*(rows-1) + k - rows, which for degree 2 and one
        // exemption is rows-1: inside the existing bound, which is why the
        // GEMM configs do not have to move.
        if (system.transition_exemptions > 0) {
            const n_rows = @as(usize, 1) << self.log_trace;
            const deg_p = d * (n_rows -| 1);
            const deg_q = if (deg_p + system.transition_exemptions <= n_rows)
                0
            else
                deg_p + system.transition_exemptions - n_rows;
            const bound = @as(usize, 1) << self.fri.log_residual_degree;
            if (deg_q >= bound) return Error.InvalidConfig;
        }
        _ = self.fri.validate() catch return Error.InvalidConfig;
    }
};

fn ceilLog2(n: usize) u6 {
    if (n <= 1) return 0;
    var log: u6 = 0;
    var v = n - 1;
    while (v > 0) : (v >>= 1) log += 1;
    return log;
}

/// One authenticated window of the trace: every column at LDE rows
/// i-1, i, i+1, plus the Merkle path for that leaf.
pub const Opening = struct {
    index: usize,
    prev: []Fp2,
    current: []Fp2,
    next: []Fp2,
    path: MerkleProof,

    fn deinit(self: *Opening, allocator: std.mem.Allocator) void {
        self.path.deinit(allocator);
        allocator.free(self.prev);
        allocator.free(self.current);
        allocator.free(self.next);
        self.* = undefined;
    }
};

pub const Proof = struct {
    commitment: Commitment,
    fri_proof: fri.Proof,
    /// Parallel to `fri_proof.queries`: the indices come from the
    /// transcript inside FRI, so reusing them adds no new randomness.
    openings: []Opening,
    /// The first and last trace row, always opened when the system has
    /// boundary constraints. Their indices are FIXED (not sampled): the
    /// constraints only apply at those rows, and the values are
    /// authenticated by the column commitment.
    boundary_openings: []Opening,

    pub fn deinit(self: *Proof, allocator: std.mem.Allocator) void {
        for (self.openings) |*o| o.deinit(allocator);
        allocator.free(self.openings);
        for (self.boundary_openings) |*o| o.deinit(allocator);
        allocator.free(self.boundary_openings);
        self.fri_proof.deinit(allocator);
        self.* = undefined;
    }
};

/// Fill one opening (prev/current/next window + Merkle path) for LDE index i.
fn buildOpening(
    allocator: std.mem.Allocator,
    columns: []const []const Fp2,
    tree: *const commit_lib.MerkleTree,
    i: usize,
    stride: usize,
) Error!Opening {
    const lde_n = columns[0].len;
    const prev_i = (i + lde_n - stride) % lde_n;
    const next_i = (i + stride) % lde_n;
    const ncols = columns.len;

    const prev_vals = try allocator.alloc(Fp2, ncols);
    errdefer allocator.free(prev_vals);
    const cur_vals = try allocator.alloc(Fp2, ncols);
    errdefer allocator.free(cur_vals);
    const next_vals = try allocator.alloc(Fp2, ncols);
    errdefer allocator.free(next_vals);
    for (columns, 0..) |col, k| {
        prev_vals[k] = col[prev_i];
        cur_vals[k] = col[i];
        next_vals[k] = col[next_i];
    }
    const path = tree.prove(i, allocator) catch return Error.OutOfMemory;
    return .{ .index = i, .prev = prev_vals, .current = cur_vals, .next = next_vals, .path = path };
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Interpolate each column through the trace domain and extend to the LDE.
fn buildLde(allocator: std.mem.Allocator, trace: Trace, config: Config) Error![][]Fp2 {
    const trace_dom = Domain.init(config.log_trace);
    const lde_dom = Domain.init(config.logLde());
    const lde_n = lde_dom.size();

    const out = try allocator.alloc([]Fp2, trace.columns.len);
    errdefer allocator.free(out);
    var made: usize = 0;
    errdefer for (out[0..made]) |c| allocator.free(c);

    for (trace.columns, 0..) |col, ci| {
        const buf = try allocator.alloc(Fp2, lde_n);
        @memcpy(buf[0..trace.rows], col);
        @memset(buf[trace.rows..], Fp2.zero);
        // The FFT can only fail on a length mismatch, which the trace
        // validation above rules out; map it to a config error anyway.
        fft.toCoefficients(buf[0..trace.rows], trace_dom) catch return Error.InvalidConfig;
        fft.toEvaluations(buf, lde_dom) catch return Error.InvalidConfig;
        out[ci] = buf;
        made += 1;
    }
    return out;
}

/// P(x) on the LDE: random linear combination of the constraints.
fn compose(
    allocator: std.mem.Allocator,
    columns: []const []const Fp2,
    system: System,
    alphas: []const Fp2,
    stride: usize,
) Error![]Fp2 {
    const n = columns[0].len;
    const acc = try allocator.alloc(Fp2, n);
    errdefer allocator.free(acc);
    @memset(acc, Fp2.zero);

    const prev = try allocator.alloc(Fp2, columns.len);
    defer allocator.free(prev);
    const current = try allocator.alloc(Fp2, columns.len);
    defer allocator.free(current);
    const next = try allocator.alloc(Fp2, columns.len);
    defer allocator.free(next);

    var power = Fp2.one;
    var ci: usize = 0;
    for (system.constraints) |c| {
        if (c.scope != .composed) continue;
        for (0..n) |i| {
            for (columns, 0..) |col, k| {
                prev[k] = col[(i + n - stride) % n];
                current[k] = col[i];
                next[k] = col[(i + stride) % n];
            }
            const v = c.eval(.{ .prev = prev, .current = current, .next = next });
            acc[i] = acc[i].add(power.mul(v));
        }
        power = power.mul(alphas[ci]);
        ci += 1;
    }
    return acc;
}

/// Divide P by Z_H(X) = X^n - 1 in coefficient form. Returns the quotient
/// coefficients, or ConstraintViolation when the remainder is non-zero
/// (an honest trace always divides exactly).
pub fn divideByTraceVanishing(allocator: std.mem.Allocator, coeffs: []const Fp2, n: usize) Error![]Fp2 {
    if (coeffs.len <= n) return Error.ConstraintViolation;
    const work = try allocator.dupe(Fp2, coeffs);
    defer allocator.free(work);

    // Long division by (X^n - 1): at degree i, subtract c*X^(i-n)*(X^n - 1)
    // = c*X^i - c*X^(i-n), which zeroes position i and ADDS c at i-n.
    // The eliminated coefficient IS the quotient's coefficient at i-n, so
    // it must be recorded before position i is cleared.
    const q = try allocator.alloc(Fp2, work.len - n);
    errdefer allocator.free(q);
    @memset(q, Fp2.zero);

    var i = work.len;
    while (i > n) {
        i -= 1;
        const c = work[i];
        if (c.isZero()) continue;
        q[i - n] = c;
        work[i] = Fp2.zero;
        work[i - n] = work[i - n].add(c);
    }
    for (work[0..n]) |r| {
        // The errdefer above releases q on this path.
        if (!r.isZero()) return Error.ConstraintViolation;
    }
    return q;
}

/// Multiply P by the transition-exemption factor `E(X) = prod (X - g^(n-1-i))`
/// in coefficient form, in place over a buffer of length `len`.
///
/// The quotient is taken against `Z_H(X) / E(X)`, so `P·E = Q·Z_H` holds and
/// `P` is only required to vanish where `E` does not, which is exactly the
/// set of exempt rows. `c` must be the trace-domain element of row `n-1-i`.
fn multiplyByExemptionFactor(coeffs: []Fp2, c: Fp2) void {
    // out[i] = orig[i-1] - c*orig[i], with orig[-1] = 0. `prev` must hold
    // the ORIGINAL coefficient: reading coeffs[i-1] here would read the
    // value this loop already overwrote.
    var prev: Fp2 = Fp2.zero;
    for (coeffs) |*slot| {
        const orig = slot.*;
        slot.* = prev.sub(c.mul(orig));
        prev = orig;
    }
}

/// The same factor evaluated at a point: `prod (x - g^(n-1-i))`.
fn exemptionFactorAt(
    system: System,
    log_trace: u6,
    n: usize,
    x: Fp2,
) Fp2 {
    var acc = Fp2.one;
    if (system.transition_exemptions == 0) return acc;
    const trace_dom = Domain.init(log_trace);
    for (0..system.transition_exemptions) |i| {
        const row = n - 1 - i;
        acc = acc.mul(x.sub(trace_dom.at(@intCast(row))));
    }
    return acc;
}

pub fn prove(
    allocator: std.mem.Allocator,
    transcript: *Transcript,
    trace: Trace,
    system: System,
    config: Config,
) Error!Proof {
    try trace.validate(system);
    try config.validate(system);

    const lde_dom = Domain.init(config.logLde());
    const lde_n = lde_dom.size();
    const n = trace.rows;
    // One trace row = `stride` LDE positions. H_k sits inside H_{k+b} at
    // the indices divisible by 2^b (g_{k+b}^{2^b * j} = g_k^j), so a
    // row-shifted constraint reads that far away — NOT one position, and
    // NOT n positions.
    const stride = @as(usize, 1) << config.log_blowup;

    const columns = try buildLde(allocator, trace, config);
    defer {
        for (columns) |c| allocator.free(c);
        allocator.free(columns);
    }

    const commitment = try commit_lib.commit(allocator, columns, stride);
    transcript.absorbBytes(&commitment.root);

    const n_composed = system.composedCount();
    const alphas = try allocator.alloc(Fp2, n_composed);
    defer allocator.free(alphas);
    for (alphas) |*a| a.* = transcript.challengeField(Fp2);

    const p_evals = try compose(allocator, columns, system, alphas, stride);
    defer allocator.free(p_evals);

    const p_coeffs = try allocator.dupe(Fp2, p_evals);
    defer allocator.free(p_coeffs);
    fft.toCoefficients(p_coeffs, lde_dom) catch return Error.InvalidConfig;

    // Transition exemptions: multiply by E(X) = prod (X - g^(n-1-i)) BEFORE
    // the division, so the quotient is P·E / Z_H. The buffer already has
    // lde_n coefficients with the top ones zero, and the shift-by-one only
    // moves data up, so no row of the LDE is disturbed and the quotient stays
    // inside the committed LDE size.
    if (system.transition_exemptions > 0) {
        const trace_dom = Domain.init(config.log_trace);
        for (0..system.transition_exemptions) |i| {
            const row = n - 1 - i;
            multiplyByExemptionFactor(p_coeffs, trace_dom.at(@intCast(row)));
        }
    }

    const q_coeffs = try divideByTraceVanishing(allocator, p_coeffs, n);
    defer allocator.free(q_coeffs);

    const q_evals = try allocator.alloc(Fp2, lde_n);
    defer allocator.free(q_evals);
    @memcpy(q_evals[0..q_coeffs.len], q_coeffs);
    @memset(q_evals[q_coeffs.len..], Fp2.zero);
    fft.toEvaluations(q_evals, lde_dom) catch return Error.InvalidConfig;

    var fri_proof = fri.prove(allocator, transcript, q_evals, config.fri) catch |e| switch (e) {
        error.OutOfMemory => return Error.OutOfMemory,
        // The FRI config was validated above; anything left is a shape bug.
        else => return Error.InvalidConfig,
    };
    errdefer fri_proof.deinit(allocator);

    const leaves = try commit_lib.buildLeaves(allocator, columns, stride);
    defer allocator.free(leaves);
    var tree = commit_lib.MerkleTree.initFromHashes(allocator, leaves) catch return Error.OutOfMemory;
    defer tree.deinit();

    const openings = try allocator.alloc(Opening, fri_proof.queries.len);
    errdefer allocator.free(openings);
    var made: usize = 0;
    errdefer for (openings[0..made]) |*o| o.deinit(allocator);

    for (fri_proof.queries, 0..) |q, qi| {
        openings[qi] = try buildOpening(allocator, columns, &tree, q.pair_index, stride);
        made += 1;
    }

    // Boundary rows: trace row 0 sits at LDE index 0, trace row n-1 at
    // (n-1)*stride.
    const want_boundary = system.hasBoundary();
    const n_boundary: usize = if (want_boundary) 2 else 0;
    const boundary_openings = try allocator.alloc(Opening, n_boundary);
    errdefer allocator.free(boundary_openings);
    var bmade: usize = 0;
    errdefer for (boundary_openings[0..bmade]) |*o| o.deinit(allocator);
    if (want_boundary) {
        boundary_openings[0] = try buildOpening(allocator, columns, &tree, 0, stride);
        bmade += 1;
        if (n > 1) {
            boundary_openings[1] = try buildOpening(allocator, columns, &tree, (n - 1) * stride, stride);
            bmade += 1;
        }
    }

    // The prover checks its own boundary constraints before emitting a
    // proof: with transition exemptions the closing row is no longer
    // pinned by any composed constraint, so a wrong claim now shows up ONLY
    // here. Emitting a proof the prover already knows is unsatisfiable just
    // moves the failure to the verifier.
    for (boundary_openings[0..bmade], 0..) |opening, qi| {
        const w: Window = .{
            .prev = opening.prev,
            .current = opening.current,
            .next = opening.next,
        };
        const want: expr.Scope = if (qi == 0) .boundary_first else .boundary_last;
        for (system.constraints) |c| {
            if (c.scope != want) continue;
            if (!c.eval(w).isZero()) return Error.ConstraintViolation;
        }
    }

    return .{
        .commitment = commitment,
        .fri_proof = fri_proof,
        .openings = openings,
        .boundary_openings = boundary_openings,
    };
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Recompute P(x) from an opening's window and the Fiat-Shamir alphas.
fn recompose(opening: Opening, system: System, alphas: []const Fp2) Fp2 {
    var acc = Fp2.zero;
    var power = Fp2.one;
    var ci: usize = 0;
    const w = Window{ .prev = opening.prev, .current = opening.current, .next = opening.next };
    for (system.constraints) |c| {
        if (c.scope != .composed) continue;
        acc = acc.add(power.mul(c.eval(w)));
        power = power.mul(alphas[ci]);
        ci += 1;
    }
    return acc;
}

/// Upper bound on composed constraints per AIR, so `verify` can keep the
/// Fiat-Shamir alphas in a fixed stack array instead of allocating. It
/// bounds the verifier's own AIR size, never a prover-supplied value, so
/// raising it is a resource decision and never a soundness one.
///
/// It was 1024 while the float multiply cost 108 per row, so eight rows
/// fitted exactly. The classify-and-select multiply costs 163 per row and
/// the subnormal result will cost more, so the guard moves to 4096: the
/// largest legitimate system here is 8 rows of the float AIR.
pub const max_composed_constraints: usize = 4096;

pub fn verify(
    transcript: *Transcript,
    proof: *const Proof,
    system: System,
    config: Config,
) Error!bool {
    if (system.hasWideOffsets()) return Error.InvalidProof;
    if (proof.commitment.log_size != config.logLde()) return Error.InvalidProof;
    if (proof.openings.len != proof.fri_proof.queries.len) return Error.InvalidProof;
    if (proof.fri_proof.queries.len == 0) return Error.InvalidProof;

    const max_col = system.maxColumn() orelse 0;
    if (proof.commitment.num_columns <= max_col) return Error.InvalidProof;
    const ncols: usize = proof.commitment.num_columns;

    const lde_dom = Domain.init(config.logLde());
    const lde_n = lde_dom.size();
    const n = @as(usize, 1) << config.log_trace;
    if (system.trace_rows) |expected| if (n != expected) return Error.InvalidProof;

    transcript.absorbBytes(&proof.commitment.root);
    const n_composed = system.composedCount();
    // Resource guard, not a soundness parameter: `system` is the
    // verifier's own AIR, so a prover cannot inflate this count. The bound
    // exists because the alphas live in a fixed stack array below. 64 was
    // enough for the one-MAC-per-row AIR but not for the chunked layout,
    // which needs 2·slots range checks (1 + 4 constraints each) plus 2·slots
    // dequantization equations — 193 composed constraints at 16 slots.
    if (n_composed > max_composed_constraints) return Error.InvalidProof;
    var alphas: [max_composed_constraints]Fp2 = undefined;
    for (0..n_composed) |i| alphas[i] = transcript.challengeField(Fp2);

    const fri_ok = fri.verify(transcript, &proof.fri_proof, config.fri) catch return false;
    if (!fri_ok) return false;

    for (proof.fri_proof.queries, 0..) |q, qi| {
        const opening = proof.openings[qi];
        const i = q.pair_index;
        if (i >= lde_n) return Error.InvalidProof;
        if (opening.index != i) return Error.InvalidProof;
        if (opening.prev.len != ncols or opening.current.len != ncols or
            opening.next.len != ncols) return Error.InvalidProof;

        // Authenticate the window against the column commitment.
        const leaf = commit_lib.hashWindow(ncols, opening.prev, opening.current, opening.next);
        commit_lib.verifyLeaf(proof.commitment, i, leaf, opening.path) catch return false;

        // P(x) == Q(x) * Z_H(x), with Z_H(x) = x^n - 1. With transition
        // exemptions the quotient is taken against Z_H/E, so the identity
        // the prover built is P(x)*E(x) == Q(x)*Z_H(x).
        const x = lde_dom.at(i);
        const z_h = x.pow(@intCast(n)).sub(Fp2.one);
        // FRI commits antipodal PAIRS: layer 0's leaf j covers positions
        // (j, j + lde_n/2), so the value at `i` is in slot 0 only when
        // i < lde_n/2.
        const half_lde = lde_n / 2;
        const q_value = q.values[0][if (i < half_lde) 0 else 1];
        const p_value = recompose(opening, system, alphas[0..n_composed]);
        const e_value = exemptionFactorAt(system, config.log_trace, n, x);
        if (!p_value.mul(e_value).eql(q_value.mul(z_h))) return false;
    }

    // Boundary constraints, at the two fixed rows the prover must open.
    //
    // The INDEX is part of what has to be checked, not just the leaf: the
    // Merkle proof authenticates the window AT `opening.index`, so without
    // pinning that index a prover could satisfy the boundary constraints on
    // some other row of its own choosing and have them say nothing about the
    // ends. The stride is the same one `prove` uses — one trace row is
    // 2^log_blowup LDE positions — so the last row is (n−1)·stride.
    if (system.hasBoundary()) {
        const expected: usize = if (n > 1) 2 else 1;
        if (proof.boundary_openings.len != expected) return Error.InvalidProof;
        const stride = @as(usize, 1) << config.log_blowup;
        for (proof.boundary_openings, 0..) |opening, qi| {
            if (opening.prev.len != ncols or opening.current.len != ncols or
                opening.next.len != ncols) return Error.InvalidProof;
            const want_index: usize = if (qi == 0) 0 else (n - 1) * stride;
            if (opening.index != want_index) return Error.InvalidProof;
            const leaf = commit_lib.hashWindow(ncols, opening.prev, opening.current, opening.next);
            commit_lib.verifyLeaf(proof.commitment, opening.index, leaf, opening.path) catch return false;
            const w = Window{
                .prev = opening.prev,
                .current = opening.current,
                .next = opening.next,
            };
            // Opening 0 is the first trace row, opening 1 the last.
            const want: expr.Scope = if (qi == 0) .boundary_first else .boundary_last;
            for (system.constraints) |c| {
                if (c.scope != want) continue;
                if (!c.eval(w).isZero()) return false;
            }
        }
    } else if (proof.boundary_openings.len != 0) {
        return Error.InvalidProof;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

fn g(v: u64) Fp2 {
    return Fp2.re(Goldilocks.fromU64(v));
}

/// s' = s + a*b, the running-sum AIR the GEMM gadget compiles to.
///
/// Container-scope storage: a System holds pointers to its terms, so
/// returning a struct built from function locals would dangle as soon as
/// the function returned.
const kNegOne = Fp2.neg(Fp2.one);

const kRunningSumTermSNext = [_]expr.Factor{.{ .column = .{ .index = 2, .offset = 1 } }};
const kRunningSumTermS = [_]expr.Factor{.{ .column = .{ .index = 2 } }};
const kRunningSumTermAB = [_]expr.Factor{
    .{ .column = .{ .index = 0 } },
    .{ .column = .{ .index = 1 } },
};
const kRunningSumTerms = [_]expr.Term{
    .{ .factors = &kRunningSumTermSNext },
    .{ .factors = &kRunningSumTermS, .coefficient = kNegOne },
    .{ .factors = &kRunningSumTermAB, .coefficient = kNegOne },
};
const kRunningSumConstraints = [_]Constraint{
    .{ .name = "s' - s - a*b", .terms = &kRunningSumTerms },
};

fn runningSumSystem() System {
    return .{ .constraints = &kRunningSumConstraints };
}

/// Honest cyclic trace for s' = s + a*b. The domain is a cycle, so the
/// products must sum to zero for the last row to fold back onto s[0];
/// the final row's a is chosen to make that true.
fn honestTrace(a: std.mem.Allocator, rows: usize) !Trace {
    const av = try a.alloc(Fp2, rows);
    const bv = try a.alloc(Fp2, rows);
    const sv = try a.alloc(Fp2, rows);
    for (0..rows - 1) |i| {
        av[i] = g(@intCast(i + 1));
        bv[i] = g(2);
    }
    var partial = Fp2.zero;
    for (0..rows - 1) |i| {
        partial = partial.add(av[i].mul(bv[i]));
    }
    bv[rows - 1] = Fp2.one;
    av[rows - 1] = Fp2.neg(partial);
    var acc = Fp2.zero;
    for (0..rows) |i| {
        sv[i] = acc;
        acc = acc.add(av[i].mul(bv[i]));
    }
    std.debug.assert(acc.isZero());
    const columns = try a.alloc([]Fp2, 3);
    columns[0] = av;
    columns[1] = bv;
    columns[2] = sv;
    return .{ .rows = rows, .columns = columns };
}

/// log_trace rows, blowup 2, and a FRI whose residual bound matches the
/// quotient's degree (< rows for a quadratic AIR) at rate 1/2.
fn testConfig(log_trace: u6) Config {
    const log_lde = log_trace + 2;
    return .{
        .log_trace = log_trace,
        .log_blowup = 2,
        .fri = .{
            .log_domain = log_lde,
            .log_final = log_trace + 1,
            .log_residual_degree = log_trace,
            .num_queries = 6,
        },
    };
}

test "stark: honest running-sum trace proves and verifies" {
    const a = testing.allocator;
    const system = runningSumSystem();
    const config = testConfig(4);
    const trace = try honestTrace(a, 16);
    defer {
        for (trace.columns) |col| a.free(col);
        a.free(trace.columns);
    }

    var pt = Transcript.init("zkml.stark.v1");
    var proof = try prove(a, &pt, trace, system, config);
    defer proof.deinit(a);

    var vt = Transcript.init("zkml.stark.v1");
    try testing.expect(try verify(&vt, &proof, system, config));
}

test "stark: tampered trace is refused by the prover (non-zero remainder)" {
    const a = testing.allocator;
    const system = runningSumSystem();
    const config = testConfig(4);
    const trace = try honestTrace(a, 16);
    defer {
        for (trace.columns) |col| a.free(col);
        a.free(trace.columns);
    }

    // Corrupt the running sum: the AIR no longer holds. Trace.columns is
    // read-only by contract (a proof attests a trace it does not edit),
    // so the tampered column is a separate buffer.
    const s_bad = try a.dupe(Fp2, trace.columns[2]);
    defer a.free(s_bad);
    s_bad[5] = g(999);
    const cols = try a.alloc([]const Fp2, 3);
    defer a.free(cols);
    cols[0] = trace.columns[0];
    cols[1] = trace.columns[1];
    cols[2] = s_bad;

    var pt = Transcript.init("zkml.stark.v1");
    try testing.expectError(
        Error.ConstraintViolation,
        prove(a, &pt, .{ .rows = trace.rows, .columns = cols }, system, config),
    );
}

test "stark: tampered opening is rejected by the verifier" {
    const a = testing.allocator;
    const system = runningSumSystem();
    const config = testConfig(4);
    const trace = try honestTrace(a, 16);
    defer {
        for (trace.columns) |col| a.free(col);
        a.free(trace.columns);
    }

    var pt = Transcript.init("zkml.stark.v1");
    var proof = try prove(a, &pt, trace, system, config);
    defer proof.deinit(a);

    // Flip a column value inside an opening: the leaf hash no longer
    // matches the commitment.
    proof.openings[0].current[0] = proof.openings[0].current[0].add(Fp2.one);
    var vt = Transcript.init("zkml.stark.v1");
    try testing.expect(!(try verify(&vt, &proof, system, config)));
}

test "stark: quotient identity is what ties the columns together" {
    const a = testing.allocator;
    const system = runningSumSystem();
    const config = testConfig(4);
    const trace = try honestTrace(a, 16);
    defer {
        for (trace.columns) |col| a.free(col);
        a.free(trace.columns);
    }

    var pt = Transcript.init("zkml.stark.v1");
    var proof = try prove(a, &pt, trace, system, config);
    defer proof.deinit(a);

    // Perturb the FRI layer-0 value: the opening stays authenticated, but
    // P(x) no longer equals Q(x)*Z_H(x).
    proof.fri_proof.queries[0].values[0][0] =
        proof.fri_proof.queries[0].values[0][0].add(Fp2.one);
    var vt = Transcript.init("zkml.stark.v1");
    try testing.expect(!(try verify(&vt, &proof, system, config)));
}

test "stark: config rejects a blowup smaller than the AIR degree" {
    const system = runningSumSystem();
    var config = testConfig(4);
    config.log_blowup = 0;
    config.fri.log_domain = config.logLde();
    try testing.expectError(Error.InvalidConfig, config.validate(system));
}

test "stark: a transition exemption waives exactly the last row" {
    const a = testing.allocator;
    const system: System = .{
        .constraints = &kRunningSumConstraints,
        .transition_exemptions = 1,
    };
    const config = testConfig(4);
    const trace = try honestTrace(a, 16);
    defer {
        for (trace.columns) |col| a.free(col);
        a.free(trace.columns);
    }

    // Break the wrap: the last row no longer cancels the running sum, so
    // the composed constraint at that row is false. The exemption is what
    // makes this legal — the quotient is taken against Z_H/E, and
    // P·E vanishes on the whole domain.
    const bv = try a.dupe(Fp2, trace.columns[1]);
    defer a.free(bv);
    bv[15] = g(7);
    const cols = try a.alloc([]const Fp2, 3);
    defer a.free(cols);
    cols[0] = trace.columns[0];
    cols[1] = bv;
    cols[2] = trace.columns[2];
    const bad_last: Trace = .{ .rows = trace.rows, .columns = cols };

    var pt = Transcript.init("zkml.stark.v1");
    var proof = try prove(a, &pt, bad_last, system, config);
    defer proof.deinit(a);
    var vt = Transcript.init("zkml.stark.v1");
    try testing.expect(try verify(&vt, &proof, system, config));

    // One row earlier is NOT exempt: same edit, prover refuses. The
    // exemption count is a prefix of the trace's tail, not "wherever the
    // prover finds it convenient".
    bv[14] = g(7);
    var pt2 = Transcript.init("zkml.stark.v1");
    try testing.expectError(
        Error.ConstraintViolation,
        prove(a, &pt2, bad_last, system, config),
    );

    // And without the exemption the last row is enforced like any other.
    bv[14] = trace.columns[1][14];
    var pt3 = Transcript.init("zkml.stark.v1");
    try testing.expectError(
        Error.ConstraintViolation,
        prove(a, &pt3, bad_last, runningSumSystem(), config),
    );
}

test "stark: an exemption the FRI residual bound cannot cover is refused" {
    // One exempt row and degree 2: deg Q <= rows-1, so the standard
    // log_residual_degree = log_trace still covers it and the GEMM configs
    // do not have to move.
    const one: System = .{
        .constraints = &kRunningSumConstraints,
        .transition_exemptions = 1,
    };
    try testConfig(4).validate(one);

    // Two exempt rows: deg Q <= rows, which no longer fits under
    // log_residual_degree = log_trace. The bound is a real requirement, not
    // belt-and-braces: an honest proof here would be rejected by the FRI
    // residual check as "too high degree".
    const two: System = .{
        .constraints = &kRunningSumConstraints,
        .transition_exemptions = 2,
    };
    try testing.expectError(Error.InvalidConfig, testConfig(4).validate(two));
    // One bit of slack fixes it (log_final has to move with it, or the
    // residual would sit at rate 1, and the LDE has to hold log_final).
    var roomy = testConfig(4);
    roomy.log_blowup = 3;
    roomy.fri.log_domain = roomy.logLde();
    roomy.fri.log_residual_degree = 5;
    roomy.fri.log_final = 6;
    try roomy.validate(two);
}

test "stark: config requires the FRI domain to match the LDE" {
    const system = runningSumSystem();
    var config = testConfig(4);
    config.fri.log_domain = config.fri.log_domain + 1;
    try testing.expectError(Error.InvalidConfig, config.validate(system));
}

test "stark: a random trace is refused (the machinery is not vacuous)" {
    const a = testing.allocator;
    const system = runningSumSystem();
    const config = testConfig(4);

    var prng = std.Random.DefaultPrng.init(0xA11CE);
    var cols: [3][]Fp2 = undefined;
    for (&cols) |*col| {
        col.* = try a.alloc(Fp2, 16);
        for (col.*) |*v| v.* = Fp2.random(prng.random());
    }
    defer for (cols) |col| a.free(col);
    const columns = try a.alloc([]const Fp2, 3);
    defer a.free(columns);
    for (cols, 0..) |col, i| columns[i] = col;

    var pt = Transcript.init("zkml.stark.v1");
    try testing.expectError(
        Error.ConstraintViolation,
        prove(a, &pt, .{ .rows = 16, .columns = columns }, system, config),
    );
}

test "stark: a tampered commitment root is rejected" {
    const a = testing.allocator;
    const system = runningSumSystem();
    const config = testConfig(4);
    const trace = try honestTrace(a, 16);
    defer {
        for (trace.columns) |col| a.free(col);
        a.free(trace.columns);
    }

    var pt = Transcript.init("zkml.stark.v1");
    var proof = try prove(a, &pt, trace, system, config);
    defer proof.deinit(a);

    proof.commitment.root[0] ^= 0xFF;
    var vt = Transcript.init("zkml.stark.v1");
    try testing.expect(!(try verify(&vt, &proof, system, config)));
}

test "stark: a proof does not transfer to a different AIR" {
    const a = testing.allocator;
    const system = runningSumSystem();
    const config = testConfig(4);
    const trace = try honestTrace(a, 16);
    defer {
        for (trace.columns) |col| a.free(col);
        a.free(trace.columns);
    }

    var pt = Transcript.init("zkml.stark.v1");
    var proof = try prove(a, &pt, trace, system, config);
    defer proof.deinit(a);

    // Stricter AIR: the running sum must be zero at every row.
    const strict_factors = [_]expr.Factor{.{ .column = .{ .index = 2 } }};
    const strict_terms = [_]expr.Term{
        .{ .factors = &strict_factors },
    };
    const strict_constraints = [_]Constraint{.{ .name = "s == 0", .terms = &strict_terms }};
    const strict = System{ .constraints = &strict_constraints };

    var vt = Transcript.init("zkml.stark.v1");
    try testing.expect(!(try verify(&vt, &proof, strict, config)));
}

test "stark: KNOWN GAP — the running-sum AIR alone admits the zero witness" {
    // Documented weakness of the example AIR, not of the backend: with no
    // boundary constraint tying s to the GEMM output, the all-zero trace
    // satisfies s' - s - a*b = 0. A production AIR MUST add boundary
    // constraints (s[0] = 0, s[n-1] = C[m,n], and the a/b decomposition
    // columns bound to the actual quantized weights). This test pins the
    // behaviour so the gap cannot be forgotten.
    const a = testing.allocator;
    const system = runningSumSystem();
    const config = testConfig(4);

    const zero = try a.alloc(Fp2, 16);
    defer a.free(zero);
    @memset(zero, Fp2.zero);
    const cols = try a.alloc([]const Fp2, 3);
    defer a.free(cols);
    cols[0] = zero;
    cols[1] = zero;
    cols[2] = zero;

    var pt = Transcript.init("zkml.stark.v1");
    var proof = try prove(a, &pt, .{ .rows = 16, .columns = cols }, system, config);
    defer proof.deinit(a);

    var vt = Transcript.init("zkml.stark.v1");
    try testing.expect(try verify(&vt, &proof, system, config));
}
