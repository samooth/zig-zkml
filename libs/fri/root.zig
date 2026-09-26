//! Canonical FRI over Goldilocks F_{p^2} — the zk-zkml L1 low-degree test.
//!
//! Post-spike decision (docs/BLUE_PRINT.md §12): zig-algebra's index-pairing FRI
//! accepts arbitrary data (no RS structure, no degree semantics) — we
//! build a proper one instead, following the Plonky3/Stone design:
//!
//!   - The witness polynomial's evaluations lie on the order-2^k subgroup
//!     H_k of the norm-1 torus of F_{p^2} (p + 1 = 2^61).
//!   - Antipodal pairs x / -x sit at adjacent positions (i, i^1) in the
//!     domain layout, so the fold pairs consecutive memory slots.
//!   - Fold with challenge alpha:
//!         f(x) = f_even(x^2) + x·f_odd(x^2),  w = x^2
//!         g(w) = f_even(w) + alpha·f_odd(w)
//!     computed from an antipodal pair WITHOUT division:
//!         g = (f(x) + f(-x))/2 + alpha·(f(x) - f(-x))/(2x)
//!     (halving is exact: 2 is invertible; 1/x on the torus = conj(x),
//!     but for the fold we only need the pair values — see below).
//!   - Degree halves each round; the prover finishes by sending the
//!     residual polynomial IN COEFFICIENTS (degree < final_length).
//!   - The verifier checks each query's Merkle paths and fold
//!     consistency, and — the degree anchor — that the last committed
//!     layer matches the residual's evaluations on the final domain.
//!
//! Soundness: identical to textbook FRI (queries x roots of committed
//! layers, one challenge per layer, final explicit degree). Error
//! ≈ (d/|H|)^-queries shaped by the codeword distance.

const std = @import("std");
const fp2 = @import("fp2.zig");
const domain = @import("domain.zig");
const merkle_pkg = @import("zig-merkle");
const fft = @import("fft.zig");

/// Blake3 adapter for zig-merkle's `H.hashBytes` node-hash interface.
const NodeHash = struct {
    pub fn hashBytes(input: []const u8) [HASH_LEN]u8 {
        var out: [HASH_LEN]u8 = undefined;
        std.crypto.hash.Blake3.hash(input, &out, .{});
        return out;
    }
};
const MerkleTree = merkle_pkg.MerkleTree(NodeHash);
const MerkleProof = merkle_pkg.MerkleProof;

pub const Fp2 = fp2.Fp2;
pub const Goldilocks = fp2.Goldilocks;
pub const Domain = domain.Domain;

pub const HASH_LEN = 32;

/// Two-to-one leaf hashing for the commitment layer: leaf_j commits to
/// the antipodal pair (f(x), f(-x)) at positions (2j, 2j+1).
fn hashPair(x: Fp2, neg_x: Fp2) [HASH_LEN]u8 {
    var h = std.crypto.hash.Blake3.init(.{});
    h.update("zkml.fri.leaf");
    h.update(&x.toBytes());
    h.update(&neg_x.toBytes());
    var out: [HASH_LEN]u8 = undefined;
    h.final(&out);
    return out;
}

pub const Config = struct {
    /// log2 of the initial domain size (order of H_k). The committed
    /// evaluations are the polynomial on the full domain.
    log_domain: u6,
    /// log2 of the LAST FRI layer's domain size. The residual polynomial
    /// is evaluated on this domain; its degree bound is
    /// 2^log_final / 2^log_blowup — expressed via `log_residual_degree`.
    log_final: u6,
    /// log2 of the residual degree bound (d). The final layer's domain
    /// has size 2^log_final = blowup * d: rate < 1 gives the distance
    /// that makes queries catch cheaters. The residual is sent as
    /// exactly 2^log_residual_degree coefficients.
    log_residual_degree: u6,
    /// Number of random positions checked.
    num_queries: usize,

    /// Sanity: the residual domain must be larger than the degree bound
    /// (rate < 1), and the degree bound must not exceed the original
    /// domain's implicit capacity.
    pub fn validate(self: Config) Error!usize {
        if (self.log_domain > domain.torus_log_order) return Error.InvalidParameters;
        if (self.log_final >= self.log_domain) return Error.InvalidParameters;
        if (self.log_residual_degree > self.log_final) return Error.InvalidParameters;
        const rounds = self.log_domain - self.log_final;
        // Folding halves the degree; the residual degree bound must be
        // consistent: d <= 2^log_final and (for soundness) d < 2^log_final.
        if (self.log_residual_degree == self.log_final) {
            // rate 1: no distance — only sound with num_queries = 0.
            return Error.InvalidParameters;
        }
        return @as(usize, rounds);
    }
};

pub const Error = error{
    OutOfMemory,
    InvalidParameters,
    InvalidProof,
};

/// Resource bound on the final FRI domain, i.e. on the scratch buffer the
/// verifier allocates to evaluate the residual. 2^20 points is 16 MiB of
/// F_{p^2}; beyond that a proof is refused rather than allocated. The
/// config comes from the verifier's own code, so this is a guard against
/// absurd local configurations, not a prover-controlled input.
pub const max_final_domain: usize = 1 << 20;

/// One committed layer (verifier view).
pub const LayerCommit = struct {
    root: [HASH_LEN]u8,
    log_size: u6,
};

/// Per-query opening: values at one antipodal pair per layer + Merkle
/// paths for those pair-leaves.
pub const QueryOpening = struct {
    /// Initial pair index (leaf index in layer 0): the pair covers
    /// positions (2j, 2j+1).
    pair_index: usize,
    /// Per layer r: the antipodal pair (f(x), f(-x)) at the queried spot.
    values: [][2]Fp2,
    /// Per layer r: Merkle proof for the pair's leaf.
    paths: []MerkleProof,
};

pub const Proof = struct {
    /// Commitments to layers 0..R-1 (layer R is the residual).
    layers: []LayerCommit,
    /// The residual polynomial, coefficients, degree < final_size.
    residual: []Fp2,
    /// Final domain size as log2 (residual evaluated on H_{log_final}).
    log_final: u6,
    queries: []QueryOpening,
    /// Initial domain size as log2.
    log_domain: u6,
    /// log2 of the residual degree bound (d). Residual length = 2^d.
    log_residual_degree: u6,

    pub fn deinit(self: *Proof, allocator: std.mem.Allocator) void {
        for (self.queries) |*q| {
            for (q.paths) |*mp| mp.deinit(allocator);
            allocator.free(q.paths);
            allocator.free(q.values);
        }
        allocator.free(self.queries);
        allocator.free(self.layers);
        allocator.free(self.residual);
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Prove that `evals` (length 2^log_domain, the polynomial evaluated on
/// H_{log_domain} in the domain layout) has degree < 2^log_final.
pub fn prove(
    allocator: std.mem.Allocator,
    transcript: anytype,
    evals: []const Fp2,
    config: Config,
) Error!Proof {
    const log_n = config.log_domain;
    const log_final = config.log_final;
    const rounds = try config.validate();
    const n: usize = @as(usize, 1) << log_n;
    if (evals.len != n) return Error.InvalidParameters;
    const residual_len: usize = @as(usize, 1) << config.log_residual_degree;

    // ---------- commit / fold loop ----------
    // layers[r] = current layer's values (length n >> r).
    var layers = std.ArrayList([]Fp2).empty;
    defer {
        for (layers.items) |l| allocator.free(l);
        layers.deinit(allocator);
    }
    var roots = std.ArrayList([HASH_LEN]u8).empty;
    defer roots.deinit(allocator);

    // Ownership: each layer's buffer is APPENDED to `layers` (which frees
    // them at scope exit) and ownership transfers there; `cur` always
    // points to the current buffer, freed only when it is NOT in layers
    // (the final residual). Track with a flag instead of a defer.
    var cur = try allocator.dupe(Fp2, evals);
    var cur_owned = true; // freed at scope exit if still owned
    defer if (cur_owned) allocator.free(cur);
    var log_cur = log_n;

    for (0..rounds) |_| {
        // Commit: one leaf per antipodal pair (positions j, j + half).
        const half = cur.len / 2;
        const leaves = try allocator.alloc([HASH_LEN]u8, half);
        defer allocator.free(leaves);
        for (0..half) |j| leaves[j] = hashPair(cur[j], cur[j + half]);
        var tree = MerkleTree.initFromHashes(allocator, leaves) catch
            return Error.OutOfMemory;
        defer tree.deinit();
        try roots.append(allocator, tree.root());

        // Absorb the root and squeeze the fold challenge.
        transcript.absorbBytes(&tree.root());
        const alpha = transcript.challengeField(Fp2);

        // Transfer ownership of this layer to `layers` (for query
        // extraction); do NOT free it here.
        try layers.append(allocator, cur);
        cur_owned = false;

        // Fold with the antipodal pair (positions j and j + half in the
        // NATURAL layout): even = (f(x)+f(-x))/2, odd = (f(x)-f(-x))/(2x)
        // with 1/x = conj(x) on the norm-1 torus, and the child lands at
        // position j of the half-size natural domain (x^2 = g_{k-1}^j).
        const dom_cur = Domain.init(@intCast(log_cur));
        const next = try allocator.alloc(Fp2, half);
        for (0..half) |j| {
            const x = dom_cur.at(j); // position j holds x; j + half holds -x
            const fx = cur[j];
            const fnegx = cur[j + half];
            const even = fx.add(fnegx).mulReal(halves_inv_g);
            const diff = fx.sub(fnegx);
            const x_inv = x.conj(); // norm 1 => conj(x) = 1/x
            const odd = diff.mul(x_inv).mulReal(halves_inv_g);
            next[j] = even.add(alpha.mul(odd));
        }

        cur_owned = true; // `next` is now the owned current buffer
        cur = next;
        log_cur -= 1;
    }

    // ---------- residual ----------
    // The final layer (length 2^log_final, natural order) interpolates
    // to a polynomial; the PROTOCOL claims degree < 2^log_residual_degree.
    // For an honest prover the high coefficients are exactly zero (each
    // fold halves the degree of a true polynomial), so TRUNCATING the
    // coefficient vector is lossless. For a cheater the truncation drops
    // real energy: the re-evaluated residual then disagrees with the
    // last committed layer almost everywhere — queries catch it.
    const full_coeffs = try interpolateToCoeffs(allocator, cur, log_final);
    defer allocator.free(full_coeffs);
    const residual = try allocator.alloc(Fp2, residual_len);
    errdefer allocator.free(residual);
    for (0..residual_len) |i| residual[i] = full_coeffs[i];
    for (residual) |c| transcript.absorbField(Fp2, c);

    // ---------- queries ----------
    const queries = try buildQueries(
        allocator,
        transcript,
        layers.items,
        roots.items,
        residual,
        config,
    );
    errdefer {
        for (queries) |*q| {
            for (q.paths) |*mp| mp.deinit(allocator);
            allocator.free(q.paths);
            allocator.free(q.values);
        }
        allocator.free(queries);
    }

    const layer_commits = try allocator.alloc(LayerCommit, rounds);
    errdefer allocator.free(layer_commits);
    for (0..rounds) |r| {
        layer_commits[r] = .{ .root = roots.items[r], .log_size = @intCast(log_n - r) };
    }

    return .{
        .layers = layer_commits,
        .residual = residual,
        .log_final = log_final,
        .queries = queries,
        .log_domain = log_n,
        .log_residual_degree = config.log_residual_degree,
    };
}

/// 1/2 in F_p, for the even/odd extraction of the fold.
pub const halves_inv_g = blk: {
    // 1/2 in F_p: (p+1)/2 mod p.
    const g = fp2.Goldilocks;
    break :blk g.fromU64((g.p + 1) / 2);
};
const halves_inv = (fp2.Goldilocks.p + 1) / 2;

/// Interpolate the final layer's natural-order evaluations into
/// coefficients.
///
/// This used to solve the Vandermonde system V·c = v by Gaussian
/// elimination: O(m^3) field operations with m = 2^log_final, which for a
/// 512-row trace meant m = 1024 and ~15 seconds of proving for a single
/// output element. The radix-2 inverse FFT over the same norm-1 torus does
/// it in O(m log m) (libs/fri/fft.zig) and is the same transform the STARK
/// prover already uses for its LDE, so the two can no longer disagree
/// about layout conventions.
fn interpolateToCoeffs(allocator: std.mem.Allocator, values: []const Fp2, log_final: u6) Error![]Fp2 {
    const dom = Domain.init(log_final);
    const coeffs = try allocator.dupe(Fp2, values);
    errdefer allocator.free(coeffs);
    fft.toCoefficients(coeffs, dom) catch return Error.InvalidParameters;
    return coeffs;
}

fn buildQueries(
    allocator: std.mem.Allocator,
    transcript: anytype,
    layers: []const []Fp2,
    roots: []const [HASH_LEN]u8,
    residual: []const Fp2,
    config: Config,
) Error![]QueryOpening {
    _ = roots;
    _ = residual;
    const rounds = layers.len;

    var trees = std.ArrayList(MerkleTree).empty;
    defer {
        for (trees.items) |*t| t.deinit();
        trees.deinit(allocator);
    }
    for (layers) |l| {
        const half = l.len / 2;
        const leaves = try allocator.alloc([HASH_LEN]u8, half);
        defer allocator.free(leaves);
        // SAME pairing as the commit phase: antipodal pair (j, j + half).
        for (0..half) |j| leaves[j] = hashPair(l[j], l[j + half]);
        const t = MerkleTree.initFromHashes(allocator, leaves) catch
            return Error.OutOfMemory;
        try trees.append(allocator, t);
    }

    const queries = try allocator.alloc(QueryOpening, config.num_queries);
    errdefer allocator.free(queries);
    for (queries) |*q| {
        // The query picks a LEAF index in layer 0 (= a pair position,
        // i.e. an element of the half-size domain of the folded view).
        // The same position index walks down all layers: under the
        // natural layout, layer r's position `p` folds into layer
        // r+1's position `p` (when p < half_{r+1}) — but the queried
        // pair at layer r is (p, p + half_r) whose leaf is p, so the
        // leaf chain keeps index p across layers only while p stays
        // below the layer's half. Since halves shrink, p eventually
        // indexes the top half: the leaf there is p - half (the pair
        // (p-half, p) is hashed under leaf p-half). Track both the
        // element index and the leaf index separately.
        const e0 = transcript.challengeU64() % @as(u64, layers[0].len);
        q.pair_index = @intCast(e0);
        q.values = try allocator.alloc([2]Fp2, rounds);
        q.paths = try allocator.alloc(MerkleProof, rounds);

        // e = element position within layer r; leaf = antipodal-pair
        // slot: min(e, e - half) — the pair (j, j+half) hashes at j.
        var e: usize = q.pair_index;
        for (0..rounds) |r| {
            const half = layers[r].len / 2;
            const j = e % half; // pair slot: position j (mod half)
            q.values[r] = .{ layers[r][j], layers[r][j + half] };
            q.paths[r] = trees.items[r].prove(j, allocator) catch
                return Error.OutOfMemory;
            // Child element position under natural fold: j (x^2 lands
            // at child position j). Antipodal collapses to j.
            e = j;
        }
    }
    return queries;
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

pub fn verify(
    transcript: anytype,
    proof: *const Proof,
    config: Config,
) Error!bool {
    if (proof.log_domain != config.log_domain) return Error.InvalidProof;
    if (proof.log_final != config.log_final) return Error.InvalidProof;
    if (proof.log_residual_degree != config.log_residual_degree) return Error.InvalidProof;
    const rounds = try config.validate();
    if (proof.layers.len != rounds) return Error.InvalidProof;
    if (proof.residual.len != @as(usize, 1) << config.log_residual_degree) {
        return Error.InvalidProof;
    }
    if (proof.queries.len != config.num_queries) return Error.InvalidProof;

    // ---------- replay challenges ----------
    var alphas: [64]Fp2 = undefined;
    if (rounds > 64) return Error.InvalidProof;
    for (0..rounds) |r| {
        if (proof.layers[r].log_size != config.log_domain - @as(u6, @intCast(r))) {
            return Error.InvalidProof;
        }
        transcript.absorbBytes(&proof.layers[r].root);
        alphas[r] = transcript.challengeField(Fp2);
    }
    for (proof.residual) |c| transcript.absorbField(Fp2, c);

    // ---------- residual evaluations on the final domain ----------
    // The residual (2^log_residual_degree coefficients) is evaluated on
    // the FULL final domain (2^log_final points) — the domain is larger
    // than the degree bound (rate < 1 gives the soundness distance).
    //
    // Evaluated with the radix-2 FFT, not by Horner: Horner over
    // (2^log_final points) x (2^log_residual_degree coefficients) is
    // O(m·d) = O(m^2) and was already the verifier's dominant cost at
    // 2^12 points (176 ms for a 1024-MAC proof). The buffer is heap
    // allocated because it is config-sized; the cap below is a RESOURCE
    // bound (the old fixed [4096]Fp2 stack buffer rejected any honest
    // proof with log_final > 12, which is what made k=4096 fail), and
    // the config is the verifier's own, never prover-supplied.
    const final_domain_size: usize = @as(usize, 1) << proof.log_final;
    if (final_domain_size > max_final_domain) return Error.InvalidProof;
    const dom_final = Domain.init(proof.log_final);
    const scratch = std.heap.page_allocator.alloc(Fp2, final_domain_size) catch
        return Error.OutOfMemory;
    defer std.heap.page_allocator.free(scratch);
    @memset(scratch, Fp2.zero);
    @memcpy(scratch[0..proof.residual.len], proof.residual);
    fft.toEvaluations(scratch, dom_final) catch return Error.InvalidProof;
    const final_evals = scratch;

    // ---------- queries ----------
    for (proof.queries) |*q| {
        const idx = transcript.challengeU64() % @as(u64, @as(u64, 1) << @intCast(config.log_domain));
        if (q.pair_index != idx) return Error.InvalidProof;

        var e: usize = q.pair_index;
        for (0..rounds) |r| {
            const half_r = (@as(usize, 1) << (config.log_domain - @as(u6, @intCast(r)))) / 2;
            const j = e % half_r; // pair slot in layer r
            const x = q.values[r][0]; // f(x)
            const negx = q.values[r][1]; // f(-x)
            const leaf = hashPair(x, negx);

            // Merkle inclusion of this pair-leaf in layer r (the pair
            // (j, j+half) commits under leaf j — same walk as the prover;
            // leaves are pre-hashed, so verifyHashed, not verify).
            if (!MerkleTree.verifyHashed(
                proof.layers[r].root,
                j,
                leaf,
                q.paths[r],
            )) return false;

            if (r + 1 < rounds) {
                // Fold consistency: THIS pair folds into layer r+1's
                // position j (natural layout). The next round's pair
                // (j', j'+half') contains that child at slot
                // 0 if j < half' else 1.
                const dom_r = Domain.init(@intCast(config.log_domain - @as(u6, @intCast(r))));
                const xv = dom_r.at(j);
                const even = x.add(negx).mulReal(halves_inv_g);
                const odd = x.sub(negx).mul(xv.conj()).mulReal(halves_inv_g);
                const expected = even.add(alphas[r].mul(odd));
                const half_next = half_r / 2;
                const child_slot: usize = if (j < half_next) 0 else 1;
                const child = q.values[r + 1][child_slot];
                if (!expected.eql(child)) return false;
            } else {
                // Last round: the pair folds into the final domain
                // position `j` (x^2 = g_f^j under natural layout).
                const dom_r = Domain.init(@intCast(config.log_domain - @as(u6, @intCast(r))));
                const xv = dom_r.at(j);
                const even = x.add(negx).mulReal(halves_inv_g);
                const odd = x.sub(negx).mul(xv.conj()).mulReal(halves_inv_g);
                const expected = even.add(alphas[r].mul(odd));
                if (!expected.eql(final_evals[j])) return false;
            }
            e = j; // next layer's element position
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Tests — including the mutation suite the zig-algebra FRI failed
// ---------------------------------------------------------------------------

const testing = std.testing;
const TestTranscript = @import("../transcript.zig").Transcript;

fn testConfig(log_n: u6, log_f: u6, queries: usize) Config {
    // Default blowup at the residual: final domain 2^log_f, residual
    // degree 2^(log_f - 3) (rate 1/8 — soundness via distance).
    return .{ .log_domain = log_n, .log_final = log_f, .log_residual_degree = log_f - 3, .num_queries = queries };
}

test "fri: degree-2 poly on the domain verifies" {
    const a = testing.allocator;
    const log_n: u6 = 8; // 256
    const dom = Domain.init(log_n);
    const n = dom.size();
    var evals = try a.alloc(Fp2, n);
    defer a.free(evals);
    const c1 = Fp2.re(fp2.Goldilocks.fromU64(3));
    const c2 = Fp2.re(fp2.Goldilocks.fromU64(7));
    for (0..n) |i| {
        const x = dom.at(i);
        evals[i] = x.sqr().add(x.mul(c1)).add(c2);
    }
    var pt = TestTranscript.init("fri-test");
    var proof = try prove(a, &pt, evals, testConfig(log_n, 6, 8));
    defer proof.deinit(a);
    var vt = TestTranscript.init("fri-test");
    try testing.expect(try verify(&vt, &proof, testConfig(log_n, 6, 8)));
}

test "fri: RANDOM data must be rejected (the zig-algebra failure mode)" {
    const a = testing.allocator;
    const log_n: u6 = 8;
    const dom = Domain.init(log_n);
    const n = dom.size();
    var evals = try a.alloc(Fp2, n);
    defer a.free(evals);
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rng = prng.random();
    for (0..n) |i| evals[i] = Fp2.random(rng);

    var accepted: usize = 0;
    const trials = 16;
    for (0..trials) |t| {
        for (0..n) |i| evals[i] = Fp2.random(rng).add(Fp2.re(fp2.Goldilocks.fromU64(t)));
        var pt = TestTranscript.init("fri-test");
        var proof = try prove(a, &pt, evals, testConfig(log_n, 6, 8));
        defer proof.deinit(a);
        var vt = TestTranscript.init("fri-test");
        if (try verify(&vt, &proof, testConfig(log_n, 6, 8))) accepted += 1;
    }
    // Random data is maximally far from degree < 8: every trial must
    // reject. (zig-algebra's FRI accepted 16/16 of these.)
    try testing.expectEqual(@as(usize, 0), accepted);
}

test "fri: degree-128 poly (>> final 8) must be rejected" {
    const a = testing.allocator;
    const log_n: u6 = 8;
    const dom = Domain.init(log_n);
    const n = dom.size();
    var evals = try a.alloc(Fp2, n);
    defer a.free(evals);

    var accepted: usize = 0;
    const trials = 8;
    for (0..trials) |t| {
        // degree-128 polynomial (final_length = 8): Horner over 129
        // random-ish coefficients.
        for (0..n) |i| {
            const x = dom.at(i);
            var acc = Fp2.zero;
            var xp = Fp2.one;
            for (0..129) |k| {
                const c = Fp2.re(fp2.Goldilocks.fromU64(k *% 2654435761 +% t));
                acc = acc.add(c.mul(xp));
                xp = xp.mul(x);
            }
            evals[i] = acc;
        }
        var pt = TestTranscript.init("fri-test");
        var proof = try prove(a, &pt, evals, testConfig(log_n, 6, 8));
        defer proof.deinit(a);
        var vt = TestTranscript.init("fri-test");
        if (try verify(&vt, &proof, testConfig(log_n, 6, 8))) accepted += 1;
    }
    try testing.expectEqual(@as(usize, 0), accepted);
}

test "fri: interpolateToCoeffs recovers a degree-1 polynomial" {
    const a = testing.allocator;
    const log_f: u6 = 6;
    const dom = Domain.init(log_f);
    const m = dom.size();
    var values = try a.alloc(Fp2, m);
    defer a.free(values);
    // v(x) = 5 + 3x over H_6 natural.
    const c0 = Fp2.re(fp2.Goldilocks.fromU64(5));
    const c1 = Fp2.re(fp2.Goldilocks.fromU64(3));
    for (0..m) |i| {
        const x = dom.at(i);
        values[i] = c0.add(c1.mul(x));
    }
    const coeffs = try interpolateToCoeffs(a, values, log_f);
    defer a.free(coeffs);
    try testing.expectEqual(@as(usize, 64), coeffs.len);
    try testing.expect(coeffs[0].eql(c0));
    try testing.expect(coeffs[1].eql(c1));
    for (coeffs[2..]) |c| try testing.expect(c.isZero());
}
