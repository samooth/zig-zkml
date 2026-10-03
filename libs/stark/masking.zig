//! Zero-knowledge masking for the STARK backend — **a negative result, pinned**.
//!
//! ## What this file is for
//!
//! The obvious way to hide a committed column is to mask it by a multiple of
//! the vanishing polynomial of the trace domain:
//!
//! ```text
//!     g = f + Z_H · h        h uniformly random, kept secret
//! ```
//!
//! `g` agrees with `f` on `H`, so every AIR constraint still reads the true
//! value. That is the whole appeal of the trick, and **it does not hide
//! anything.** The two tests below compute `g mod Z_H` and find `f`.
//!
//! ## Why, in one line
//!
//! `Z_H` is the vanishing polynomial of `H`, so `(Z_H · h)(w) = 0` for every
//! `w ∈ H` and `g(w) = f(w)` on all of `H`. Since `deg f < |H|` and `H` has
//! `|H|` points, interpolation on `H` determines `f` uniquely — so **the
//! verifier recovers `f` by taking the remainder of `g` modulo `Z_H`.**
//!
//! This is not a bug in the code; it is a property of the scheme. Any mask that
//! vanishes on all of `H` is removable by anyone, so it cannot be a privacy
//! mechanism.
//!
//! ## What does hide
//!
//! The mask has to vanish on the points that are **opened** and nowhere else.
//! Those are the shifted query positions, which the trace domain does not
//! contain — which is precisely the problem **DEEP-FRI** solves, and precisely
//! the DEEP factor that ties the auxiliary polynomial back to the trace.
//! A mask that vanishes only on the opened points leaves the committed codeword
//! free everywhere else, and `g mod Z_H` is then no longer `f`.
//!
//! So hiding here is not a primitive you can add to the existing commitment.
//! **It requires the DEEP composition, and the backend has a quotient instead**
//! (`libs/stark/root.zig`). That is the real prerequisite, and it is a bigger
//! piece of work than this file.
//!
//! ## What is still useful here
//!
//! The degree arithmetic, because it is the same arithmetic either way: the mask
//! term has degree `n_rows + deg h`, and whatever bound admits it is the bound
//! the FRI residual test has to use. Those are functions, not remarks, and the
//! mutation gate `tools/masking_mutation.sh` pins them.

const std = @import("std");

const fp2 = @import("../torus/fp2.zig");

/// The base field. The masking question is a question about polynomial
/// arithmetic over a field, so it is asked over Goldilocks and not over the
/// torus element, whose subnormal semantics belong to the float AIR.
pub const Goldilocks = fp2.Goldilocks;

/// The degree arithmetic of a masked column, kept because it is needed by
/// whichever masking scheme is eventually chosen.
pub const MaskPlan = struct {
    /// Size of the trace domain, `2^log_trace`. This is also `deg Z_H`.
    n_rows: usize,
    /// `deg h`.
    mask_degree: usize,

    pub fn init(n_rows: usize, mask_degree: usize) MaskPlan {
        return .{ .n_rows = n_rows, .mask_degree = mask_degree };
    }

    /// `deg f` for a column evaluated on the trace domain.
    pub fn columnDegree(self: MaskPlan) usize {
        return self.n_rows - 1;
    }

    /// `deg (Z_H · h)`.
    pub fn maskDegree(self: MaskPlan) usize {
        return self.n_rows + self.mask_degree;
    }

    /// `deg g`, what a low-degree test would be handed.
    pub fn committedDegree(self: MaskPlan) usize {
        return @max(self.columnDegree(), self.maskDegree());
    }

    /// Whether `g` is inside a low-degree bound of `2^k`.
    pub fn fits(self: MaskPlan, residual_bound: usize) bool {
        return self.committedDegree() < residual_bound;
    }
};

/// The mask degree this repository would pick for a trace of `n_rows`, and the
/// bound it would need. Which mask degree hides enough is a decision, not a
/// derivation — see `docs/decisions/ADR-0004-pesos-ocultos.md`.
pub fn defaultMaskDegree(n_rows: usize) usize {
    return n_rows / 8;
}

/// Smallest power of two above `committedDegree`. The bound is strict.
pub fn requiredResidualBound(n_rows: usize, mask_degree: usize) usize {
    const plan = MaskPlan.init(n_rows, mask_degree);
    var bound: usize = 1;
    while (bound <= plan.committedDegree()) bound *= 2;
    return bound;
}

// -- the falsification, executed -------------------------------------------

/// Dense polynomial as `[]Goldilocks`, little-endian by degree.
const Poly = []Goldilocks;

/// Remainder of `poly` modulo `X^n - 1`, returned as `n` coefficients.
///
/// No division: reducing modulo `X^n - 1` is exactly folding every coefficient
/// onto its exponent class, because `X^j = X^(j mod n)` there. Doing it this
/// way keeps the demonstration free of any long-division bug, which matters
/// because the point of the test is *what* the remainder is, not that some
/// arithmetic happened to produce it.
fn reduceModXnMinus1(poly: []const Goldilocks, n: usize) !Poly {
    const a = std.testing.allocator;
    const out = try a.alloc(Goldilocks, n);
    errdefer a.free(out);
    @memset(out, Goldilocks.zero);
    for (poly, 0..) |c, i| {
        out[i % n] = out[i % n].add(c);
    }
    return out;
}

test "masking by a multiple of Z_H does not hide: the remainder is the column" {
    const n: usize = 8;
    const a = std.testing.allocator;
    var f: [n]Goldilocks = undefined;
    var h: [2]Goldilocks = undefined;
    var g: [n + 2]Goldilocks = undefined;

    for (&f, 0..) |*c, i| c.* = Goldilocks.fromU64(i * 7 + 3);
    h[0] = Goldilocks.fromU64(0x5eed);
    h[1] = Goldilocks.fromU64(0xbeef);

    // g = f + Z_H * h, with Z_H = X^n - 1, written out coefficient by
    // coefficient: (X^n - 1) * (h0 + h1*X) = h0*X^n + h1*X^(n+1) - h0 - h1*X.
    @memset(&g, Goldilocks.zero);
    @memcpy(g[0..n], &f);
    g[n] = g[n].add(h[0]);
    g[0] = g[0].sub(h[0]);
    g[n + 1] = g[n + 1].add(h[1]);
    g[1] = g[1].sub(h[1]);

    // The mask is not zero, or none of this would mean anything.
    try std.testing.expect(!h[0].eql(Goldilocks.zero));

    const rem = try reduceModXnMinus1(&g, n);
    defer a.free(rem);

    // And the verifier gets the column back, exactly.
    try std.testing.expectEqualSlices(Goldilocks, &f, rem);
}

test "the remainder is f for any h, which is what makes it a scheme property" {
    // Same statement across four different masks. One counterexample would only
    // be an accident; all four agreeing is the property.
    const n: usize = 8;
    const a = std.testing.allocator;
    for ([_]u64{ 1, 2, 3, 99 }) |seed| {
        var f: [n]Goldilocks = undefined;
        var g: [n + 2]Goldilocks = undefined;
        for (&f, 0..) |*c, i| c.* = Goldilocks.fromU64(seed * 31 + i + 1);
        @memset(&g, Goldilocks.zero);
        @memcpy(g[0..n], &f);
        const t0 = Goldilocks.fromU64(seed * 17 + 5);
        const t1 = Goldilocks.fromU64(seed * 23 + 9);
        g[n] = g[n].add(t0);
        g[0] = g[0].sub(t0);
        g[n + 1] = g[n + 1].add(t1);
        g[1] = g[1].sub(t1);

        const rem = try reduceModXnMinus1(&g, n);
        defer a.free(rem);
        try std.testing.expectEqualSlices(Goldilocks, &f, rem);
    }
}

// -- the arithmetic, kept ---------------------------------------------------

test "the mask degree is what a low-degree test is handed" {
    const plan = MaskPlan.init(1024, 128);
    try std.testing.expectEqual(1023, plan.columnDegree());
    try std.testing.expectEqual(1152, plan.maskDegree());
    try std.testing.expectEqual(1152, plan.committedDegree());
}

test "a bound one power of two short is refused" {
    const plan = MaskPlan.init(1024, 128);
    try std.testing.expect(!plan.fits(1024));
    try std.testing.expect(plan.fits(2048));
}

test "the required bound is the next power of two above the committed degree" {
    try std.testing.expectEqual(2048, requiredResidualBound(1024, 128));
    try std.testing.expectEqual(1152, MaskPlan.init(1024, 128).committedDegree());
    // A power-of-two committed degree needs the next one: the bound is strict.
    try std.testing.expectEqual(4096, requiredResidualBound(1024, 1024));
}

test "at the shipped geometry the bound does not move" {
    // Unmasked a column needs n_rows; masked it needs n_rows + n_rows/8, and the
    // bound that was already required is 2 * n_rows. So the degree bookkeeping
    // is not what blocks masking — the DEEP composition is.
    const n_rows: usize = 2048;
    const mask_degree = defaultMaskDegree(n_rows);
    try std.testing.expectEqual(2 * n_rows, requiredResidualBound(n_rows, mask_degree));
    try std.testing.expectEqual(
        n_rows + mask_degree,
        MaskPlan.init(n_rows, mask_degree).committedDegree(),
    );
}
