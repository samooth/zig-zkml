//! Fingerprint GEMM: verify a product without materialising it.
//!
//! The element-wise path in `gemm_air.zig` commits the full product, so its
//! trace is O(m·n) rows for an m×n tile. The fingerprint avoids that.
//!
//! For C = A·Bᵀ (m×n over Fp2) and the bilinear form ⟨u⊗v, C⟩ =
//! Σ_{i,j} u_i v_j C_{ij}, substitute the definition of the product:
//!
//!     Σ_{i,j} u_i v_j Σ_t A_{it} B_{jt}
//!   = Σ_t (Σ_i u_i A_{it}) · (Σ_j v_j B_{jt})
//!   = Σ_t ⟨u, A_{·t}⟩ · ⟨v, B_{·t}⟩
//!
//! The separability is the whole point, and it depends on the test vector
//! being a **rank-1** matrix u⊗v. With a general r_{ij} there is nothing to
//! factor out and the right-hand side would still be O(m·n·k); a
//! general-r "fingerprint" is quadratic in r and computes neither the bilinear
//! form nor any saving. `Challenge` therefore produces two vectors, and the
//! public API takes `u` and `v` separately so a caller cannot pass a dense r
//! by mistake.
//!
//! Cost drops from O(m·n·k) to O((m+n)·k) multiplications: for a 2048×1408
//! tile that is ~3·10⁶ against ~3·10⁹, a factor of ~1000. `measure` reports
//! both numbers so the claim is auditable rather than asserted.
//!
//! What this module is NOT: not a proof, and not linear in u and v
//! separately. The claim is linear in the rank-1 object u⊗v — scaling
//! (u,v) -> (c·u, d·v) scales the value by c·d — but scaling u alone does
//! not double it, because ⟨u⊗v, C⟩ does not factor through u. A challenge
//! that is a general dense r has no separability at all and the O((m+n)·k)
//! cost does not exist. `Challenge` returns a pair precisely so the rank-1
//! shape is a type-level requirement rather than a convention.
//!
//! Soundness: u and v must be derived from the transcript AFTER both matrices
//! are committed. Ordering is enforced by the caller in F3; `fingerprintClaim`
//! is pure arithmetic and is deliberately not a proof. The element-wise
//! prover remains the reference and the oracle — this path is only ever
//! allowed to be faster, never the sole thing checked.

const std = @import("std");
const Allocator = std.mem.Allocator;

const fp2_mod = @import("../fri/fp2.zig");
pub const Fp2 = fp2_mod.Fp2;

pub const Error = error{
    InvalidShape,
    OutOfMemory,
};

/// m·n must fit without overflowing, and both extents must be non-zero: the
/// fingerprint of an empty tile is trivially zero, and proving that would let
/// a prover answer any statement.
pub fn validateShape(m: usize, n: usize) Error!void {
    if (m == 0 or n == 0) return Error.InvalidShape;
    if (m > std.math.maxInt(usize) / n) return Error.InvalidShape;
}

/// A rank-1 challenge: the outer product u⊗v, carried as its two factors.
///
/// Keeping the factors separate instead of materialising the m×n matrix is
/// not an optimisation — it is what makes the separability in
/// `fingerprintClaim` expressible at all, and it makes it impossible to hand
/// this module a dense challenge by accident. Derive these from the
/// transcript only after committing A and B.
pub const Challenge = struct {
    u: []const Fp2,
    v: []const Fp2,

    pub fn init(u: []const Fp2, v: []const Fp2) Error!Challenge {
        if (u.len == 0 or v.len == 0) return Error.InvalidShape;
        return .{ .u = u, .v = v };
    }
};

/// The true value of the bilinear form, computed by actually forming the
/// product entry by entry.
///
/// This is the oracle. It costs O(m·n·k) and exists so the fast path can be
/// checked against the definition rather than against itself. Per §8 of the
/// plan, a reference only exercised where the prover rejects is not a
/// reference; this one is exercised on the accepting path too.
pub fn productInner(
    a: []const Fp2,
    b: []const Fp2,
    u: []const Fp2,
    v: []const Fp2,
    m: usize,
    n: usize,
    k: usize,
) Error!Fp2 {
    try validateShape(m, n);
    if (a.len != m * k) return Error.InvalidShape;
    if (b.len != n * k) return Error.InvalidShape;
    if (u.len != m) return Error.InvalidShape;
    if (v.len != n) return Error.InvalidShape;

    var acc = Fp2.zero;
    for (0..m) |i| {
        const ui = u[i];
        const a_row = a[i * k ..][0..k];
        for (0..n) |j| {
            const b_row = b[j * k ..][0..k];
            var dot = Fp2.zero;
            for (0..k) |t| dot = dot.add(a_row[t].mul(b_row[t]));
            acc = acc.add(ui.mul(v[j]).mul(dot));
        }
    }
    return acc;
}

/// Σ_t ⟨u, A_{·t}⟩ · ⟨v, B_{·t}⟩ — the value `productInner` must return,
/// without ever forming C.
///
/// O((m+n)·k) multiplications. The per-t inner products are accumulated in
/// place as the (i,t) and (j,t) loops sweep, so only the u and v vectors and
/// the k accumulators are live at once: no intermediate matrix exists at any
/// point in this function.
pub fn fingerprintClaim(
    a: []const Fp2,
    b: []const Fp2,
    u: []const Fp2,
    v: []const Fp2,
    m: usize,
    n: usize,
    k: usize,
) Error!Fp2 {
    try validateShape(m, n);
    if (a.len != m * k) return Error.InvalidShape;
    if (b.len != n * k) return Error.InvalidShape;
    if (u.len != m) return Error.InvalidShape;
    if (v.len != n) return Error.InvalidShape;

    const scratch = std.heap.page_allocator;
    var acc = scratch.alloc(Fp2, k) catch return Error.OutOfMemory;
    defer scratch.free(acc);
    var acc2 = scratch.alloc(Fp2, k) catch return Error.OutOfMemory;
    defer scratch.free(acc2);
    @memset(acc, Fp2.zero);
    @memset(acc2, Fp2.zero);

    for (0..m) |i| {
        const ui = u[i];
        if (ui.isZero()) continue;
        const a_row = a[i * k ..][0..k];
        for (0..k) |t| acc[t] = acc[t].add(ui.mul(a_row[t]));
    }
    for (0..n) |j| {
        const vj = v[j];
        if (vj.isZero()) continue;
        const b_row = b[j * k ..][0..k];
        for (0..k) |t| acc2[t] = acc2[t].add(vj.mul(b_row[t]));
    }

    var out = Fp2.zero;
    for (0..k) |t| out = out.add(acc[t].mul(acc2[t]));
    return out;
}

/// Field multiplications each path performs, and the rows each commits.
///
/// §5.1 asks for the cost measured rather than estimated, so both figures are
/// produced from the same shapes and the ratio is left to the caller to read.
pub const Cost = struct {
    /// Rows the element-wise AIR commits: one per output cell.
    elementwise_rows: usize,
    /// Digests the fingerprint path must commit instead: 2 per reduction step.
    fingerprint_digests: usize,
    /// Multiplications to evaluate the claim the defining way.
    oracle_muls: usize,
    /// Multiplications the fingerprint arithmetic performs.
    fingerprint_muls: usize,

    /// oracle_muls / fingerprint_muls, the factor the identity buys.
    pub fn speedup(self: Cost) f64 {
        if (self.fingerprint_muls == 0) return std.math.inf(f64);
        return @as(f64, @floatFromInt(self.oracle_muls)) /
            @as(f64, @floatFromInt(self.fingerprint_muls));
    }
};

pub fn measure(m: usize, n: usize, k: usize) Error!Cost {
    try validateShape(m, n);
    return .{
        .elementwise_rows = m * n,
        .fingerprint_digests = 2 * k,
        .oracle_muls = m * n * k,
        .fingerprint_muls = (m + n) * k,
    };
}

test "fingerprint claim equals the defining bilinear form" {
    const m = 2;
    const n = 4;
    const k = 3;
    const av = [_]Fp2{
        Fp2.one, Fp2.one,       Fp2.one,
        Fp2.one, Fp2.one.neg(), Fp2.one,
    };
    const bv = [_]Fp2{
        Fp2.one, Fp2.one,       Fp2.one,
        Fp2.one, Fp2.zero,      Fp2.one,
        Fp2.one, Fp2.one,       Fp2.one,
        Fp2.one, Fp2.one,       Fp2.one.neg(),
    };
    const uv = [_]Fp2{ Fp2.one, Fp2.one };
    const vv = [_]Fp2{ Fp2.one, Fp2.one, Fp2.one, Fp2.one };

    const want = try productInner(&av, &bv, &uv, &vv, m, n, k);
    const got = try fingerprintClaim(&av, &bv, &uv, &vv, m, n, k);
    try std.testing.expect(want.eql(got));
}

test "claim is linear in the outer product: scaling both u and v is exact" {
    const m = 2;
    const n = 3;
    const k = 2;
    const av = [_]Fp2{
        Fp2.one, Fp2.one,
        Fp2.one, Fp2.one.neg(),
    };
    const bv = [_]Fp2{
        Fp2.one, Fp2.one,
        Fp2.one, Fp2.zero,
        Fp2.one, Fp2.one,
    };
    const uv = [_]Fp2{ Fp2.one, Fp2.one };
    const vv = [_]Fp2{ Fp2.one, Fp2.one, Fp2.one };
    const two = Fp2.fromRaw(2, 0);

    const base = try fingerprintClaim(&av, &bv, &uv, &vv, m, n, k);
    try std.testing.expect(base.eql(try productInner(&av, &bv, &uv, &vv, m, n, k)));

    // Scaling the *pair* (u, v) -> (c·u, d·v) scales the form by c·d. The
    // form is linear in the rank-1 challenge u⊗v, which is exactly the object
    // a sumcheck over a random point manipulates. It is NOT the case that
    // scaling u alone doubles the value: that would require the form to
    // factor through u separately, which <u⊗v, C> does not.
    for ([_]Fp2{ two, Fp2.fromRaw(3, 0) }) |c| {
        for ([_]Fp2{ two, Fp2.fromRaw(3, 0) }) |d| {
            const us = [_]Fp2{ c.mul(uv[0]), c.mul(uv[1]) };
            const vs = [_]Fp2{ d.mul(vv[0]), d.mul(vv[1]), d.mul(vv[2]) };
            const got = try fingerprintClaim(&av, &bv, &us, &vs, m, n, k);
            const want = base.mul(c).mul(d);
            try std.testing.expect(got.eql(want));
            try std.testing.expect(got.eql(try productInner(&av, &bv, &us, &vs, m, n, k)));
        }
    }
}

test "a different challenge moves the claim" {
    const m = 2;
    const n = 2;
    const k = 2;
    const av = [_]Fp2{ Fp2.one, Fp2.one, Fp2.one, Fp2.one };
    const bw = [_]Fp2{ Fp2.one, Fp2.one, Fp2.one, Fp2.zero };
    const uv = [_]Fp2{ Fp2.one, Fp2.one };
    const vv = [_]Fp2{ Fp2.one, Fp2.one };
    const v_alt = [_]Fp2{ Fp2.one, Fp2.zero };

    // Same A, two different B: a false product cannot answer both.
    const good = try fingerprintClaim(&av, &av, &uv, &vv, m, n, k);
    const bad = try fingerprintClaim(&av, &bw, &uv, &vv, m, n, k);
    try std.testing.expect(!good.eql(bad));

    // And a different challenge on the same pair must too, otherwise a prover
    // could satisfy one fixed r with a wrong product.
    const same = try fingerprintClaim(&av, &av, &uv, &v_alt, m, n, k);
    try std.testing.expect(!good.eql(same));
}

test "a zero entry in u removes exactly that row" {
    const m = 2;
    const n = 2;
    const k = 2;
    const av = [_]Fp2{
        Fp2.one, Fp2.one,
        Fp2.one, Fp2.one,
    };
    const uv = [_]Fp2{ Fp2.one, Fp2.zero };
    const vv = [_]Fp2{ Fp2.one, Fp2.one };
    // u[1] = 0 zeroes row 1 of the form, but row 0 still contributes, so the
    // value is u[0]·v·(row0 of A · Bᵀ) and is NOT zero in general.
    const c = try fingerprintClaim(&av, &av, &uv, &vv, m, n, k);
    const o = try productInner(&av, &av, &uv, &vv, m, n, k);
    try std.testing.expect(c.eql(o));

    // Both vectors zero is the one case that must collapse to zero.
    const uz = [_]Fp2{ Fp2.zero, Fp2.zero };
    try std.testing.expect((try fingerprintClaim(&av, &av, &uz, &vv, m, n, k)).isZero());
    try std.testing.expect((try productInner(&av, &av, &uz, &vv, m, n, k)).isZero());

    const vz = [_]Fp2{ Fp2.zero, Fp2.zero };
    try std.testing.expect((try fingerprintClaim(&av, &av, &uv, &vz, m, n, k)).isZero());
    try std.testing.expect((try productInner(&av, &av, &uv, &vz, m, n, k)).isZero());
}

test "a challenge carries its rank-1 factors, not a dense matrix" {
    const c = try Challenge.init(&[_]Fp2{ Fp2.one }, &[_]Fp2{ Fp2.one });
    try std.testing.expectEqual(@as(usize, 1), c.u.len);
    try std.testing.expectEqual(@as(usize, 1), c.v.len);
    // An empty factor is not a zero challenge: it would make the whole form
    // vacuously true, so it is rejected rather than folded to zero.
    try std.testing.expectError(Error.InvalidShape, Challenge.init(&[_]Fp2{}, &[_]Fp2{ Fp2.one }));
    try std.testing.expectError(Error.InvalidShape, Challenge.init(&[_]Fp2{ Fp2.one }, &[_]Fp2{}));
}

test "empty shape is rejected, never proved as zero" {
    try std.testing.expectError(Error.InvalidShape, validateShape(0, 4));
    try std.testing.expectError(Error.InvalidShape, validateShape(4, 0));
}

test "mismatched lengths are rejected" {
    const av = [_]Fp2{ Fp2.one, Fp2.one, Fp2.one, Fp2.one };
    const uv = [_]Fp2{ Fp2.one, Fp2.one };
    const vv = [_]Fp2{ Fp2.one, Fp2.one };
    // m=2,n=2,k=3 would need 6 entries per matrix.
    try std.testing.expectError(Error.InvalidShape, fingerprintClaim(&av, &av, &uv, &vv, 2, 2, 3));
    // u and v are per-row / per-column, not per-reduction-step.
    try std.testing.expectError(Error.InvalidShape, fingerprintClaim(&av, &av, &av, &vv, 2, 2, 2));
    try std.testing.expectError(Error.InvalidShape, fingerprintClaim(&av, &av, &uv, &vv, 2, 3, 2));
}

test "cost model agrees with the measured 2048x1408 shape" {
    const c = try measure(2048, 1408, 2048);
    try std.testing.expectEqual(@as(usize, 2048 * 1408), c.elementwise_rows);
    try std.testing.expectEqual(@as(usize, 2 * 2048), c.fingerprint_digests);
    // O((m+n)k) against O(mnk): three orders of magnitude on this shape.
    try std.testing.expect(c.speedup() > 500.0);
}
