//! 2-adic multiplicative subgroups of the norm-1 torus of F_{p^2}.
//!
//! The torus T = {x ∈ F_{p^2} : N(x) = 1} has order p + 1 = 2^61 —
//! a pure power of two. A generator g of T has order 2^61, and
//! g^(2^60) = -1 (the unique order-2 element). The subgroup
//! H_k = <g^(2^(61-k))> has order 2^k; squaring maps H_k -> H_{k-1}
//! two-to-one (x and -x collide), which is exactly the FRI fold.
//!
//! Domains are laid out in "bit-reversed natural order" for LDE
//! friendliness: domain[i] = g^(2^(61-log_n) * rev_bits(i)) — the
//! standard STARK layout where adjacent-in-memory elements are far
//! apart in the group, so truncating the array keeps a coset-like
//! spread. Element pairing x / -x for the fold is by index arithmetic
//! (rev structure: positions i and i ^ (n/2) hold x and -x).

const std = @import("std");
const fp2 = @import("fp2.zig");

pub const Fp2 = fp2.Fp2;
pub const Goldilocks = fp2.Goldilocks;

/// The 2-adicity of the torus: p + 1 = 2^61.
pub const torus_log_order: u6 = 61;

/// Generator of the order-2^61 torus, found at comptime by trial:
/// x^(p-1) for small x until order exactly 2^61 holds (g^(2^60) = -1).
pub const generator: Fp2 = findGenerator();

fn pow2(e: u32) u64 {
    return @as(u64, 1) << @intCast(e);
}

fn findGenerator() Fp2 {
    // The comptime trial search runs thousands of field operations.
    @setEvalBranchQuota(1_000_000);
    // Candidate base values as Fp2 elements (real axis + a few with i).
    const candidates = [_]Fp2{
        Fp2.fromRaw(3, 0),  Fp2.fromRaw(5, 0),  Fp2.fromRaw(7, 0),
        Fp2.fromRaw(11, 0), Fp2.fromRaw(13, 0), Fp2.fromRaw(17, 0),
        Fp2.fromRaw(19, 0), Fp2.fromRaw(23, 0), Fp2.fromRaw(29, 0),
        Fp2.fromRaw(2, 3),  Fp2.fromRaw(3, 5),  Fp2.fromRaw(7, 11),
        Fp2.fromRaw(1, 2),  Fp2.fromRaw(3, 1),  Fp2.fromRaw(5, 2),
    };
    const neg_one = Fp2.re(Goldilocks.zero.sub(Goldilocks.one));

    for (candidates) |c| {
        // Torus element: c^(p-1).
        const g = c.pow(Goldilocks.p - 1);
        if (g.isZero()) continue;
        // Order exactly 2^61 iff g^(2^60) = -1 (then g^(2^61) = 1 follows).
        const h = g.pow(pow2(60));
        if (h.eql(neg_one)) return g;
    }
    @compileError("no 2^61-order generator found — check torus order math");
}

comptime {
    // The generator must have norm 1 and order 2^61 (its 2^60 power is -1).
    std.debug.assert(generator.norm().eql(Goldilocks.one));
    const neg_one = Fp2.re(Goldilocks.zero.sub(Goldilocks.one));
    std.debug.assert(generator.pow(pow2(60)).eql(neg_one));
}

/// The order-2^k subgroup H_k, as powers of g_k = generator^(2^(61-k)).
///
/// Layout: NATURAL exponent order — domain[i] = g_k^i. The antipodal
/// pairs x / -x (exponents differing by 2^(k-1)) sit at positions
/// (i, i + n/2). Squaring maps position i to position i (of the
/// half-size subgroup: g_k^(2i) = g_{k-1}^i), so the FRI fold
/// j <- (cur[j], cur[j + n/2]) preserves the natural layout at every
/// layer — the residual lands in natural order too, which keeps
/// interpolation and verification index arithmetic trivial.
pub const Domain = struct {
    log_n: u6,
    /// g_k: order-2^log_n generator (H_k = <g_k>).
    step_gen: Fp2,

    pub fn init(log_n: u6) Domain {
        // log_n = 0 is the trivial subgroup {1} (valid fold target).
        std.debug.assert(log_n <= torus_log_order);
        return .{
            .log_n = log_n,
            .step_gen = generator.pow(pow2(torus_log_order - @as(u32, log_n))),
        };
    }

    pub fn size(self: Domain) usize {
        return @as(usize, 1) << self.log_n;
    }

    /// The i-th domain element (natural order): g_k^i.
    pub fn at(self: Domain, i: usize) Fp2 {
        return self.step_gen.pow(@intCast(i));
    }

    /// Fill `buf` (len == size) with the domain elements.
    pub fn fill(self: Domain, buf: []Fp2) void {
        std.debug.assert(buf.len == self.size());
        for (buf, 0..) |*x, i| x.* = self.at(i);
    }

    /// The 2^log_n-th primitive root (step_gen), for NTT butterflies.
    pub fn root(self: Domain) Fp2 {
        return self.step_gen;
    }
};

test "domain: generator has order 2^61, norm 1" {
    const t = std.testing;
    // comptime assertions already check this; re-verify at runtime for
    // belt-and-suspenders on the comptime math.
    try t.expect(generator.norm().eql(Goldilocks.one));
    const neg_one = Fp2.re(Goldilocks.zero.sub(Goldilocks.one));
    try t.expect(generator.pow(pow2(60)).eql(neg_one));
    try t.expect(generator.pow(pow2(61)).eql(Fp2.one));
}

test "domain: H_k sizes and negation structure" {
    const t = std.testing;

    for ([_]u6{ 1, 2, 3, 5, 8 }) |k| {
        const d = Domain.init(k);
        const n = d.size();
        try t.expect(n == pow2(k));

        var buf: [256]Fp2 = undefined;
        std.debug.assert(n <= 256);
        d.fill(buf[0..n]);

        // all elements distinct and of norm 1
        for (buf[0..n], 0..) |x, i| {
            try t.expect(x.norm().eql(Goldilocks.one));
            for (buf[0..i]) |y| try t.expect(!x.eql(y));
        }

        // x and x + n/2 are negatives (exponent differs by 2^(k-1)).
        for (0..n / 2) |i| {
            const j = i + n / 2;
            try t.expect(buf[i].add(buf[j]).isZero());
        }

        // closed under squaring into the half-order domain
        const d2 = Domain.init(k - 1);
        const n2 = d2.size();
        var buf2: [128]Fp2 = undefined;
        d2.fill(buf2[0..n2]);
        for (0..n) |i| {
            const sq = buf[i].sqr();
            var found = false;
            for (buf2) |y| {
                if (sq.eql(y)) {
                    found = true;
                    break;
                }
            }
            try t.expect(found);
        }
    }
}

test "domain: fold pairing halves exactly 2-to-1" {
    const t = std.testing;
    const k: u6 = 6;
    const d = Domain.init(k);
    const n = d.size();
    var buf: [64]Fp2 = undefined;
    d.fill(&buf);

    // Squaring collapses {x, -x} pairs: exactly n/2 distinct squares.
    var seen: [64]bool = undefined;
    @memset(&seen, false);
    var distinct: usize = 0;
    for (0..n) |i| {
        const sq = buf[i].sqr();
        var already = false;
        for (0..i) |j| {
            if (buf[j].sqr().eql(sq)) {
                already = true;
                break;
            }
        }
        seen[i] = !already;
        if (!already) distinct += 1;
    }
    try t.expectEqual(n / 2, distinct);

    // And each square is hit by exactly one {x, -x} pair.
    for (0..n) |i| {
        if (!seen[i]) continue;
        const sq = buf[i].sqr();
        var hits: usize = 0;
        for (buf) |y| {
            if (y.sqr().eql(sq)) hits += 1;
        }
        try t.expectEqual(@as(usize, 2), hits);
    }
}
