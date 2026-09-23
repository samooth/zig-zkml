// NTT over Fp2 torus subgroups — natural layout, radix-2 iterative DIT

const std = @import("std");
const fp2 = @import("fri/fp2.zig");
const domain = @import("fri/domain.zig");

const testing = std.testing;

pub const Fp2 = fp2.Fp2;
pub const Goldilocks = fp2.Goldilocks;

/// In-place radix-2 NTT (natural input/output order).
/// forward: evals[i] = Σ_j coeffs[j] * ω^(i*j) where ω = domain.step_gen (primitive 2^log_n-th root of unity)
/// inverse: coeffs = (1/n) * Σ_i evals[i] * ω^(-i*j)
pub fn transform(values: []Fp2, dom: domain.Domain, invert: bool) void {
    const n = values.len;
    std.debug.assert(n == dom.size());

    // Bit-reversal permutation (in-place)
    for (values, 0..) |*v, i| {
        _ = v;
        const rev = std.math.bitReverse(i);
        if (i < rev) {
            std.mem.swap(Fp2, &values[i], &values[rev]);
        }
    }

    var len: usize = 2;
    while (len <= n) : (len *= 2) {
        const half = len / 2;
        // Root of unity of order `len`: ω_len = domain.step_gen^(size / len)
        // domain.step_gen is primitive n-th root; (step_gen)^(size/len) has order `len`
        const wlen = dom.step_gen.pow(dom.size() / len);

        for (0..n) |k| {
            if (k % len != 0) continue;
            var w: Fp2 = Fp2.one;
            for (0..half) |j| {
                const u = values[k + j];
                const v = values[k + j + half].mul(w);
                values[k + j] = u.add(v);
                values[k + j + half] = u.sub(v);
                w = w.mul(wlen);
            }
        }
    }

    if (invert) {
        const inv_n = Fp2.one.div(Fp2.fromU64(n));
        // conjugate roots: ω^{-1} = conj(ω) since norm(ω) = 1
        // Instead of recomputing roots, we can re-use the forward loop with conjugated twiddles
        // Easier: re-run with conjugated roots and then scale by inv_n
        // Re-use the same loop but with conjugated twiddles and then scale
        var rev = 0;
        for (values, 0..) |*v, i| {
            _ = v;
            rev = std.math.bitReverse(i);
            if (i < rev) {
                std.mem.swap(Fp2, &values[i], &values[rev]);
            }
        }

        len = 2;
        while (len <= n) : (len *= 2) {
            const half = len / 2;
            const wlen = dom.step_gen.pow(dom.size() / len).conj(); // ω^{-1}
            for (0..n) |k| {
                if (k % len != 0) continue;
                var w: Fp2 = Fp2.one;
                for (0..half) |j| {
                    const u = values[k + j];
                    const v = values[k + j + half].mul(w);
                    values[k + j] = u.add(v);
                    values[k + j + half] = u.sub(v);
                    w = w.mul(wlen.conj()); // w *= (wlen)^{-1} = conj(wlen) since norm=1
                }
            }
        }

        // Scale by n^{-1}
        for (values) |*v| {
            v.* = v.mul(inv_n);
        }
    }
}

/// Forward NTT: coeffs (length n) -> evals (length n)
/// Both arrays must have length = dom.size()
pub fn forward(values: []Fp2, dom: domain.Domain) void {
    transform(values, dom, false);
}

/// Inverse NTT: evals (length n) -> coeffs (length n)
/// Both arrays must have length = dom.size()
pub fn inverse(values: []Fp2, dom: domain.Domain) void {
    transform(values, dom, true);
}

/// Evaluate a polynomial given in coefficient form at a single point `x`.
/// `coeffs` must be in ascending degree order (c[0] + c[1]*x + ...)
/// `x` is an Fp2 element.
pub fn eval(coeffs: []const Fp2, x: Fp2) Fp2 {
    var acc: Fp2 = Fp2.zero;
    var xp: Fp2 = Fp2.one;
    for (coeffs) |c| {
        acc = acc.add(c.mul(xp));
        xp = xp.mul(x);
    }
    return acc;
}

/// Interpolate a polynomial given its values on the domain's points (natural order).
/// Returns coefficients in ascending degree order.
/// Uses O(n^2) Lagrange interpolation (fine for small n; NTT-based for large).
pub fn interpolate(allocator: std.mem.Allocator, dom: domain.Domain, values: []const Fp2) error{OutOfMemory}![]Fp2 {
    const n = values.len;
    std.debug.assert(n == dom.size());
    if (n == 1) return allocator.dup(values);

    // Solve Vandermonde system V * c = v where V[i][j] = x_i^j, x_i = dom.at(i)
    var mat = try allocator.alloc(Fp2, n * (n + 1));
    errdefer allocator.free(mat);

    for (0..n) |i| {
        const xi = dom.at(i);
        var xp = Fp2.one;
        for (0..n) |j| {
            mat[i * (n + 1) + j] = xp;
            xp = xp.mul(xi);
        }
        mat[i * (n + 1) + n] = values[i];
    }

    // Gaussian elimination
    var coeffs = try allocator.alloc(Fp2, n);
    errdefer allocator.free(coeffs);

    for (0..n) |col| {
        // Find pivot
        var pivot: ?usize = null;
        for (col..n) |r| {
            if (!mat[r * (n + 1) + col].isZero()) {
                pivot = r;
                break;
            }
        }
        const p = pivot orelse return error.OutOfMemory;

        if (p != col) {
            for (0..(n + 1)) |c| {
                const tmp = mat[col * (n + 1) + c];
                mat[col * (n + 1) + c] = mat[p * (n + 1) + c];
                mat[p * (n + 1) + c] = tmp;
            }
        }

        const inv = mat[col * (n + 1) + col].inv() catch return error.OutOfMemory;
        for (0..(n + 1)) |c| {
            mat[col * (n + 1) + c] = mat[col * (n + 1) + c].mul(inv);
        }

        // Eliminate other rows
        for (0..n) |r| {
            if (r == col) continue;
            const f = mat[r * (n + 1) + col];
            if (f.isZero()) continue;
            for (0..(n + 1)) |c| {
                mat[r * (n + 1) + c] = mat[r * (n + 1) + c].sub(mat[col * (n + 1) + c].mul(f));
            }
        }
    }

    for (0..n) |i| coeffs[i] = mat[i * (n + 1) + n];
    return coeffs;
}

/// Multiply two polynomials via NTT (pad to size 2n, pointwise mul, inverse).
/// `a` and `b` have degree < n. Result has degree < 2n.
/// Returns allocated array of length 2n-1 (coefficient representation).
pub fn multiply(allocator: std.mem.Allocator, dom: domain.Domain, a: []const Fp2, b: []const Fp2) error{OutOfMemory}![]Fp2 {
    const n = a.len;
    std.debug.assert(b.len == n);
    std.debug.assert(n.isPowerOfTwo());

    // Pad to 2n for convolution
    var a_pad = try allocator.alloc(Fp2, 2 * n);
    errdefer allocator.free(a_pad);
    @memcpy(a_pad[0..n], a);
    @memset(a_pad[n..], 0);

    var b_pad = try allocator.alloc(Fp2, 2 * n);
    errdefer allocator.free(b_pad);
    @memcpy(b_pad[0..n], b);
    @memset(b_pad[n..], 0);

    const dom2 = domain.Domain.init(dom.log_n + 1);
    forward(a_pad, dom2);
    forward(b_pad, dom2);

    for (a_pad, b_pad) |*av, bv| {
        av.* = av.mul(bv);
    }

    inverse(a_pad, dom2);

    const result = try allocator.alloc(Fp2, 2 * n - 1);
    errdefer allocator.free(result);
    @memcpy(result, a_pad[0 .. 2 * n - 1]);
    return result;
}

/// Add two polynomials (in-place on `a`, which must have length >= len(b)).
pub fn add(a: []Fp2, b: []const Fp2) void {
    std.debug.assert(a.len >= b.len);
    for (a, b) |*av, bv| av.* = av.add(bv);
}

/// Subtract b from a (in-place on `a`).
pub fn sub(a: []Fp2, b: []const Fp2) void {
    std.debug.assert(a.len >= b.len);
    for (a, b) |*av, bv| av.* = av.sub(bv);
}

/// Scale polynomial by scalar.
pub fn scale(poly: []Fp2, s: Fp2) void {
    for (poly) |*v| v.* = v.mul(s);
}

/// Divide polynomial C by Z(X) = X^N - 1 (exact division, C must vanish on domain).
/// Returns quotient Q of degree < deg(C) - N + 1.
/// C and Q share the same coefficient array (in-place fold).
/// `N` = domain.size().
/// Returns slice of Q coefficients (length = C.len - N).
pub fn divideByVanishing(C: []Fp2, N: usize) []Fp2 {
    // C has length L < 2N (honest: deg < 2N)
    // Q[k] = C[k+N] for k < N (since C = Z * Q, C[k+N] = Q[k])
    // In-place: fold C[0..N-1] += C[N..2N-1]
    const L = C.len;
    std.debug.assert(L <= 2 * N);
    for (0..N) |i| {
        if (i + N < L) C[i] = C[i].add(C[i + N]);
    }
    return C[0..N];
}

/// Verify that C vanishes on the domain (C divisible by X^N - 1).
/// Checks C[j] + C[j+N] == 0 for all j < N (C must have length >= 2N, or pad with zeros).
pub fn checkVanishing(dom: domain.Domain, C: []const Fp2) error{VanishingError}!void {
    const N = dom.size();
    if (C.len < 2 * N) return error.VanishingError;
    for (0..N) |i| {
        if (!C[i].add(C[i + N]).isZero()) return error.VanishingError;
    }
}

pub const VanishingError = error{VanishingError};

test "ntt forward/inverse roundtrip" {
    const a = testing.allocator;
    const log_n: u6 = 4;
    const dom = domain.Domain.init(log_n);
    const n = dom.size();

    const coeffs = try a.alloc(Fp2, n);
    defer a.free(coeffs);
    for (coeffs, 0..) |*c, i| c.* = Fp2.random(std.Random.DefaultPrng.init(0xBEEF + i)).add(Fp2.fromU64(@intCast(i)));

    const evals = try a.dup(coeffs);
    forward(evals, domain.Domain.init(log_n));

    const coeffs2 = try a.alloc(Fp2, n);
    defer a.free(coeffs2);
    @memcpy(coeffs2, coeffs);
    forward(coeffs2, domain.Domain.init(log_n));

    const coeffs3 = try a.alloc(Fp2, n);
    defer a.free(coeffs3);
    @memcpy(coeffs3, coeffs);
    forward(coeffs3, domain.Domain.init(log_n));
    inverse(coeffs3, domain.Domain.init(log_n));

    const t = testing;
    for (coeffs, coeffs3) |c1, c3| try t.expectEqual(c1, c3);
}

test "ntt multiply roundtrip" {
    const a = testing.allocator;
    const log_n: u6 = 3;
    const dom = domain.Domain.init(log_n);
    const n = dom.size();

    const a1 = try a.alloc(Fp2, n);
    defer a.free(a1);
    const a2 = try a.alloc(Fp2, n);
    defer a.free(a2);

    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    for (a1) |*c| c.* = Fp2.random(prng.random());
    for (a2) |*c| c.* = Fp2.random(prng.random());

    const prod = try multiply(a, dom, a1, a2);
    defer a.free(prod);

    // Verify by naive evaluation at a random point
    const test_pt = Fp2.random(prng.random());
    const val_a = eval(a1, test_pt);
    const val_b = eval(a2, test_pt);
    const val_prod = eval(prod[0 .. 2 * dom.size() - 1], test_pt);
    const expected = val_a.mul(val_b);
    try testing.expectEqual(expected, val_prod);
}

test "interpolate/ntt consistency" {
    const a = testing.allocator;
    const log_n: u6 = 3;
    const dom = domain.Domain.init(log_n);
    const n = dom.size();

    const coeffs = try a.alloc(Fp2, n);
    defer a.free(coeffs);
    var prng = std.Random.DefaultPrng.init(0xDEAD);
    for (coeffs) |*c| c.* = Fp2.random(prng.random()).add(Fp2.fromU64(1));

    const evals = try a.alloc(Fp2, n);
    defer a.free(evals);
    forward(evals, dom);

    const recovered = try interpolate(a, dom, evals);
    defer a.free(recovered);

    const t = testing;
    for (coeffs, recovered) |c1, c2| try t.expectEqual(c1, c2);
}

test "divideByVanishing" {
    const a = testing.allocator;
    const N = 8;
    var C = try a.alloc(Fp2, 2 * N);
    defer a.free(C);

    // C(x) = (X^N - 1) * (1 + X + X^2) = (X^8 - 1) * (1 + X + X^2)
    // = X^10 + X^9 + X^8 - X^2 - X - 1
    // Coeffs: [-1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1]
    const coeffs = [_]Fp2{
        Fp2.fromU64(0xFFFFFFFFFFFFFFFF), // -1
        Fp2.fromU64(0xFFFFFFFFFFFFFFFF), // -1
        Fp2.fromU64(0xFFFFFFFFFFFFFFFF), // -1
        Fp2.zero,
        Fp2.zero,
        Fp2.zero,
        Fp2.zero,
        Fp2.zero,
        Fp2.one,
        Fp2.one,
        Fp2.one,
    };
    @memcpy(C[0..11], &coeffs);

    const Q = divideByVanishing(C, N);
    try testing.expectEqual(Q.len, 3);
    try testing.expectEqual(Q[0], Fp2.one);
    try testing.expectEqual(Q[1], Fp2.one);
    try testing.expectEqual(Q[2], Fp2.one);
}

test "checkVanishing" {
    const a = testing.allocator;
    const dom = domain.Domain.init(3);

    var C = try a.alloc(Fp2, 16);
    defer a.free(C);
    // (X^8 - 1) * (1 + X + X^2)
    const coeffs = [_]Fp2{
        Fp2.fromU64(0xFFFFFFFFFFFFFFFF), // -1
        Fp2.fromU64(0xFFFFFFFFFFFFFFFF), // -1
        Fp2.fromU64(0xFFFFFFFFFFFFFFFF), // -1
        Fp2.zero,
        Fp2.zero,
        Fp2.zero,
        Fp2.zero,
        Fp2.zero,
        Fp2.one,
        Fp2.one,
        Fp2.one,
    };
    @memcpy(C[0..11], &coeffs);
    @memset(C[11..], 0);

    try checkVanishing(dom, C);

    // Corrupt one coefficient
    C[0] = Fp2.one;
    try testing.expectError(error.VanishingError, checkVanishing(dom, C));
}
