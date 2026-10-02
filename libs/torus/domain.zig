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

/// Largest domain this library will build, as log2 of the element count.
///
/// This is a **resource limit, not a soundness parameter**: nothing about the
/// protocol requires a larger domain, and a caller cannot widen it. It is
/// deliberately not `torus_log_order`. Those are two different questions and
/// conflating them is what made `size()` a live bug — `init` accepted any
/// `log_n` up to 61 because that is where the generator stays valid, and then
/// `size()` computed `1 << 61`, which fits a 64-bit `usize` and asks for 2.6e18
/// bytes. It compiled, passed every check, and detonated on first use.
///
/// 2^30 domain points is above any single AIR trace; beyond that the trace
/// needs blocking regardless of the torus. It is also below
/// `@bitSizeOf(usize) - 1` on every supported target, so the shift in `size()`
/// is representable on 32-bit `usize` as well as 64-bit.
pub const max_log_domain: u6 = 30;

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

    /// `error.DomainTooLarge` when `log_n` exceeds `max_log_domain`.
    ///
    /// This was a pre-condition assertion, and replacing it was not a
    /// formality: `log_n` is a `u6` that reaches this function from outside —
    /// `libs/fri/root.zig` reads it out of the prover config and out of the
    /// proof — so a value above 61 makes `torus_log_order - log_n` underflow,
    /// and in ReleaseFast the subtraction wraps and `pow2` returns a generator
    /// of the wrong order with no diagnostic. `zig_algebra` v0.5.1 hit the same
    /// defect in its own `Domain.init` and fixed it the same way.
    ///
    /// The bound is `max_log_domain` and not `torus_log_order`, because the
    /// underflow and the allocation are separate limits and a guard that only
    /// checks the first leaves `size()` returning 2^61. See `max_log_domain`.
    pub fn init(log_n: u6) error{DomainTooLarge}!Domain {
        // log_n = 0 is the trivial subgroup {1} (valid fold target).
        if (log_n > max_log_domain) return error.DomainTooLarge;
        return .{
            .log_n = log_n,
            .step_gen = generator.pow(pow2(torus_log_order - @as(u32, log_n))),
        };
    }

    /// `1 << log_n`, for the `log_n` that `init` accepted.
    ///
    /// The shift amount is `@intCast` so it infers `u5` on a 32-bit `usize`
    /// and `u6` on a 64-bit one, rather than the destination entering the
    /// library. That is only sound because `init` caps `log_n` at
    /// `max_log_domain`, which is below `@bitSizeOf(usize) - 1` everywhere —
    /// an unchecked shift here is what `tools/wasm_expected.txt` was recording
    /// as a compile error on wasm32.
    pub fn size(self: Domain) usize {
        return @as(usize, 1) << @as(std.math.Log2Int(usize), @intCast(self.log_n));
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
        const d = try Domain.init(k);
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
        const d2 = try Domain.init(k - 1);
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
    const d = try Domain.init(k);
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

test "init accepts exactly [0, max_log_domain] and size() is safe across it" {
    // The guard has to agree with the thing it guards, and that agreement is
    // the test. `init` used to accept up to `torus_log_order` (61) while
    // `size()` returned 1 << log_n, so log_n = 61 passed every check and then
    // asked for 2.6e18 bytes. The bug was invisible because nothing asserted
    // the pair — only each half was believed.
    //
    // Pinning the exact named error matters: a test that merely asserted
    // "accept or reject" would have passed on a guard that rejected for the
    // wrong reason, or on one that accepted and then failed later.
    var log_n: u6 = 0;
    while (log_n <= max_log_domain) : (log_n += 1) {
        const d = try Domain.init(log_n);
        try std.testing.expectEqual(log_n, d.log_n);

        // size() must not overflow, and must not exceed what `usize` can hold.
        const n = d.size();
        // The same `@intCast` as `size()`, and for the same reason: a bare
        // u6 shift is a COMPILE error on wasm32, where the shift amount must be
        // u5. Asserting the arithmetic here must not reintroduce the defect the
        // arithmetic is there to prove absent.
        try std.testing.expectEqual(@as(usize, 1) << @as(std.math.Log2Int(usize), @intCast(log_n)), n);
        try std.testing.expect(n <= std.math.maxInt(usize));

        // The real invariant: representable is not the same as allocatable.
        // 2^61 is a fine usize and an impossible buffer.
        try std.testing.expect(n <= (@as(usize, 1) << max_log_domain));
    }

    // One past the bound is rejected, by name.
    try std.testing.expectError(error.DomainTooLarge, Domain.init(max_log_domain + 1));

    // And the value the external dev passed, which is the one that detonated.
    // It is inside the torus order and outside the resource limit, which is
    // exactly the distinction the old guard was missing.
    try std.testing.expect(torus_log_order > max_log_domain);
    try std.testing.expectError(error.DomainTooLarge, Domain.init(torus_log_order));
}

test "the bound is below what any supported usize can shift" {
    // `size()` casts the shift amount to Log2Int(usize). If `max_log_domain`
    // ever grew to or past the bit width, that cast becomes a panic in Debug
    // and undefined behaviour in ReleaseFast — the same defect one level down,
    // and the reason the wasm sweep recorded this file as a compile error.
    try std.testing.expect(max_log_domain < @bitSizeOf(usize) - 1);
}

test "max_log_domain is above what the library actually asks for" {
    // The other half of a resource limit, and the half that is easy to get
    // wrong in the opposite direction. A limit that rejects legitimate work is
    // the same defect as a limit that accepts impossible work: both are a
    // guard that disagrees with the thing it guards. The largest `log_n` any
    // code in this repository currently requests is 8 (256 points, the fixed
    // prover path at `libs/prove/root.zig` and the FRI tests); the prover's
    // variable path is `ceilLog2(raw_len)`, so the bound is a judgement about
    // future trace sizes rather than a number this suite can derive.
    //
    // What is checkable, and checked here: the bound must clear every
    // in-tree request, so tightening or widening it cannot silently reject the
    // library's own use of it. The *value* of the bound is a resource
    // decision, documented at its declaration, not a fact a test can discover.
    try std.testing.expect(max_log_domain >= 8);

    // La asercion que ESTUVO aqui —`expect(max_log_domain <= 30)`— era una
    // tautologia: max_log_domain ES 30, asi que aseguraba una constante contra
    // si misma y solo podia fallar si alguien editaba el numero, lo cual no es
    // una medicion. Parecia una prueba y no lo era.
    //
    // Lo que hace las veces de puerta es `tools/domain_cost.sh`, que mide el
    // coste real del prover y el verifier en una escalera de tamanos y falla si
    // el limite baja por debajo de la frontera medida mas un margen. Ese es el
    // sitio donde la relacion entre el limite y lo pagable queda verificada, y
    // por eso aqui solo queda lo que se puede derivar de los tipos.
}
