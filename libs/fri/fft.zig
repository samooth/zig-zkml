//! Radix-2 FFT over the norm-1 torus subgroups of F_{p^2} (p = 2^61 - 1).
//!
//! The FRI backend (libs/fri) works on evaluations over H_k = {g^i} in
//! NATURAL order, where g is the order-2^k generator of the norm-1 torus.
//! Because every domain element has norm 1, the inverse of a root of unity
//! is simply its conjugate: w^-1 = conj(w) = (a, -b) for w = (a, b).
//! That is what makes an in-place radix-2 transform possible with no
//! separate twiddle-inverse table.
//!
//! Layout: natural order in, natural order out (bit-reversal is applied
//! and undone internally), matching libs/fri/domain.zig `Domain.at(i) =
//! step_gen^i` — so an evaluation vector can be handed straight to FRI.
//!
//! It lives here, not under libs/stark, because FRI needs it too: the
//! final-layer interpolation used to be a naive O(m^3) Vandermonde solve,
//! which made proving a 512-row trace take 15 seconds. Both sides now
//! share this transform.
//!
//! NOTE: this replaces the `libs/ntt.zig` spike, whose inverse applied the
//! conjugate twice (a no-op) and whose test never exercised the inverse
//! path. It was not compiled by any test target.

const std = @import("std");
const fp2 = @import("fp2.zig");
const domain = @import("domain.zig");

pub const Fp2 = fp2.Fp2;
pub const Goldilocks = fp2.Goldilocks;
pub const Domain = domain.Domain;

/// 1/n as an Fp2 element (n is a power of two, hence a real field element).
fn invPow2(n: usize) error{NotPowerOfTwo}!Fp2 {
    if (n == 0 or !std.math.isPowerOfTwo(n)) return error.NotPowerOfTwo;
    const real = Goldilocks.fromU64(@intCast(n));
    return Fp2.re(real).inv() catch unreachable; // a real element is never 0
}

pub const Error = error{ NotPowerOfTwo, LengthMismatch };

/// Reverse the low `bits` bits of `x` (the DIT input permutation).
/// std.math has no bitReverse in 0.16, and this is O(log n) per index —
/// same order as the transform itself.
fn reverseLowBits(x: usize, bits: u6) usize {
    var r: usize = 0;
    var v = x;
    for (0..bits) |_| {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    return r;
}

/// In-place radix-2 decimation-in-time transform.
///
/// `forward == true`:  coefficients -> evaluations, v[i] = sum_j c[j] w^(i*j)
/// `forward == false`: evaluations -> coefficients, scaled by 1/n
///
/// `values.len` must equal `dom.size()`.
pub fn transform(values: []Fp2, dom: Domain, forward: bool) Error!void {
    const n = values.len;
    if (n != dom.size()) return Error.LengthMismatch;
    if (n == 1) return;

    // Bit-reversal permutation (DIT requires bit-reversed input order).
    for (0..n) |i| {
        const rev = reverseLowBits(i, dom.log_n);
        if (i < rev) std.mem.swap(Fp2, &values[i], &values[rev]);
    }

    // twiddle = w or w^-1. Norm 1 => w^-1 = conj(w).
    const root = if (forward) dom.step_gen else dom.step_gen.conj();

    var len: usize = 2;
    while (len <= n) : (len <<= 1) {
        const half = len >> 1;
        // w_len = root^(n/len) has order `len`.
        const w_len = root.pow(@intCast(n / len));
        var block: usize = 0;
        while (block < n / len) : (block += 1) {
            const base = block * len;
            var w = Fp2.one;
            for (0..half) |j| {
                const u = values[base + j];
                const v = values[base + j + half].mul(w);
                values[base + j] = u.add(v);
                values[base + j + half] = u.sub(v);
                w = w.mul(w_len);
            }
        }
    }

    if (!forward) {
        const inv_n = try invPow2(n);
        for (values) |*v| v.* = v.mul(inv_n);
    }
}

/// coefficients -> evaluations over `dom` (in place).
pub fn toEvaluations(values: []Fp2, dom: Domain) !void {
    return transform(values, dom, true);
}

/// evaluations over `dom` -> coefficients (in place, scaled by 1/n).
pub fn toCoefficients(values: []Fp2, dom: Domain) !void {
    return transform(values, dom, false);
}

/// Evaluate a polynomial (ascending coefficients) at a single point.
pub fn evalAt(coeffs: []const Fp2, x: Fp2) Fp2 {
    var acc = Fp2.zero;
    var xp = Fp2.one;
    for (coeffs) |c| {
        acc = acc.add(c.mul(xp));
        xp = xp.mul(x);
    }
    return acc;
}

/// Evaluate on a coset of `dom`: point i is `offset * g^i`. Needed to reach
/// points outside H (where the trace domain's vanishing polynomial would be
/// zero). The coset offset must not lie in H.
pub fn evalOnCoset(allocator: std.mem.Allocator, coeffs: []const Fp2, dom: Domain, offset: Fp2) ![]Fp2 {
    const n = dom.size();
    const out = try allocator.alloc(Fp2, n);
    errdefer allocator.free(out);
    var g = offset;
    for (0..n) |i| {
        out[i] = evalAt(coeffs, g);
        g = g.mul(dom.step_gen);
    }
    return out;
}

const testing = std.testing;

test "fft: roundtrip coefficients -> evaluations -> coefficients" {
    const a = testing.allocator;
    for ([_]u6{ 1, 2, 3, 4, 6 }) |log_n| {
        const dom = Domain.init(log_n);
        const n = dom.size();
        var prng = std.Random.DefaultPrng.init(@as(u64, 0xF17) +% @as(u64, log_n));

        const coeffs = try a.alloc(Fp2, n);
        defer a.free(coeffs);
        for (coeffs) |*c| c.* = Fp2.random(prng.random());

        // `work` is mutated through the slice, not rebound.
        const work = try a.dupe(Fp2, coeffs);
        defer a.free(work);

        try toEvaluations(work, dom);
        try toCoefficients(work, dom);
        for (coeffs, work) |c, r| try testing.expect(c.eql(r));
    }
}

test "fft: evaluations match direct evaluation on the domain" {
    const a = testing.allocator;
    const log_n: u6 = 5;
    const dom = Domain.init(log_n);
    const n = dom.size();

    var prng = std.Random.DefaultPrng.init(0xD1FF);
    const coeffs = try a.alloc(Fp2, n);
    defer a.free(coeffs);
    for (coeffs, 0..) |*c, i| c.* = Fp2.random(prng.random()).add(Fp2.re(Goldilocks.fromU64(@intCast(i))));

    const evals = try a.dupe(Fp2, coeffs);
    defer a.free(evals);
    try toEvaluations(evals, dom);

    for (0..n) |i| {
        const x = dom.at(i);
        try testing.expect(evalAt(coeffs, x).eql(evals[i]));
    }
}

test "fft: conjugate root is the inverse root" {
    const dom = Domain.init(8);
    const w = dom.step_gen;
    try testing.expect(w.mul(w.conj()).eql(Fp2.one));
    // The transform's inverse scaling is exact: forward then inverse is
    // the identity on a random vector (covered above); check 1/n directly.
    const inv4 = try invPow2(4);
    try testing.expect(Fp2.re(Goldilocks.fromU64(4)).mul(inv4).eql(Fp2.one));
}
