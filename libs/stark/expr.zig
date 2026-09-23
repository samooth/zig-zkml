//! Constraint IR for the STARK backend (F2).
//!
//! libs/air carries the *shape* of a trace (columns, schemes, bounds) but
//! its `Constraint` only records a degree — there is no expression to
//! evaluate, which is why the pre-F2 prover could only flatten and FRI the
//! trace. This module adds the missing half: an evaluable form.
//!
//! A constraint is a sum of terms; a term is a coefficient times a product
//! of factors; a factor is a column value (at the current, next or previous
//! LDE row) or a constant. Flat by design: composition is a loop with no
//! tree allocation, and the degree of a term is just its factor count.
//!
//! Covers the shape the F2 milestone needs:
//!   running sum   s' = s + a*b        -> [s@+1] - [s@0] - [a][b]
//!   linear bind   out = W·x + b      -> [out] - [W][x] - b
//!   range/boolean x*(x-1) = 0         -> [x][x] - [x]
//! Division, inverses and lookups are NOT in this IR (they need the LogUp
//! argument, BLUE_PRINT §7.4) — a constraint that needs them is not
//! expressible yet, by design, rather than silently mis-proven.

const std = @import("std");
const fp2 = @import("../fri/fp2.zig");

pub const Fp2 = fp2.Fp2;
pub const Goldilocks = fp2.Goldilocks;

/// A column value, optionally shifted by whole rows. Offset +1 reads the
/// next LDE row (the running-sum shift); -1 reads the previous one.
pub const ColumnRef = struct {
    index: u16,
    offset: i8 = 0,
};

pub const Factor = union(enum) {
    column: ColumnRef,
    constant: Fp2,
};

pub const Term = struct {
    factors: []const Factor,
    coefficient: Fp2 = Fp2.one,
};

pub const Constraint = struct {
    name: []const u8,
    terms: []const Term,

    /// Degree as a polynomial in the row variable: a product of `k` column
    /// factors has degree < k·(degree of one column) and constants add
    /// nothing, so the term degree bound is its non-constant factor count.
    pub fn degree(self: Constraint) usize {
        var max: usize = 0;
        for (self.terms) |t| {
            var k: usize = 0;
            for (t.factors) |f| {
                switch (f) {
                    .column => k += 1,
                    .constant => {},
                }
            }
            if (k > max) max = k;
        }
        return max;
    }

    pub fn eval(self: Constraint, w: Window) Fp2 {
        var acc = Fp2.zero;
        for (self.terms) |t| {
            var prod = t.coefficient;
            for (t.factors) |f| {
                switch (f) {
                    .column => |c| prod = prod.mul(w.get(c)),
                    .constant => |k| prod = prod.mul(k),
                }
            }
            acc = acc.add(prod);
        }
        return acc;
    }
};

/// Column values around one point: the LDE rows i-1, i, i+1. Offsets
/// outside {-1,0,+1} are rejected by the caller (the IR does not support
/// them yet).
pub const Window = struct {
    prev: []const Fp2,
    current: []const Fp2,
    next: []const Fp2,

    pub fn get(self: Window, ref: ColumnRef) Fp2 {
        const row = switch (ref.offset) {
            -1 => self.prev,
            0 => self.current,
            else => self.next,
        };
        return row[ref.index];
    }
};

/// A named set of constraints evaluated together (one AIR).
pub const System = struct {
    constraints: []const Constraint,

    pub fn maxDegree(self: System) usize {
        var max: usize = 0;
        for (self.constraints) |c| {
            const d = c.degree();
            if (d > max) max = d;
        }
        return max;
    }

    /// Highest column index referenced, or null if the system is constant.
    pub fn maxColumn(self: System) ?u16 {
        var max: ?u16 = null;
        for (self.constraints) |c| {
            for (c.terms) |t| {
                for (t.factors) |f| {
                    switch (f) {
                        .column => |r| {
                            if (max == null or r.index > max.?) max = r.index;
                        },
                        .constant => {},
                    }
                }
            }
        }
        return max;
    }

    /// True if any constraint needs a row outside {-1,0,+1}.
    pub fn hasWideOffsets(self: System) bool {
        for (self.constraints) |c| {
            for (c.terms) |t| {
                for (t.factors) |f| {
                    switch (f) {
                        .column => |r| if (r.offset < -1 or r.offset > 1) return true,
                        .constant => {},
                    }
                }
            }
        }
        return false;
    }
};

const testing = std.testing;

fn g(v: u64) Fp2 {
    return Fp2.re(Goldilocks.fromU64(v));
}

test "expr: degree counts column factors only" {
    const linear = Constraint{
        .name = "linear",
        .terms = &.{
            .{ .factors = &.{.{ .column = .{ .index = 0 } }} },
            .{ .factors = &.{.{ .column = .{ .index = 1 } }}, .coefficient = g(5) },
        },
    };
    try testing.expectEqual(@as(usize, 1), linear.degree());

    const quad = Constraint{
        .name = "quad",
        .terms = &.{
            .{ .factors = &.{
                .{ .column = .{ .index = 0 } },
                .{ .column = .{ .index = 1 } },
                .{ .constant = g(3) },
            } },
        },
    };
    try testing.expectEqual(@as(usize, 2), quad.degree());
}

test "expr: running-sum constraint evaluates to zero on an honest trace" {
    // s[i+1] - s[i] - a[i]*b[i] = 0
    const a_idx: u16 = 0;
    const b_idx: u16 = 1;
    const s_idx: u16 = 2;

    const neg = Fp2.neg(Fp2.one);
    const running_sum = Constraint{
        .name = "s' - s - a*b",
        .terms = &.{
            .{ .factors = &.{.{ .column = .{ .index = s_idx, .offset = 1 } }} },
            .{ .factors = &.{.{ .column = .{ .index = s_idx } }}, .coefficient = neg },
            .{ .factors = &.{
                .{ .column = .{ .index = a_idx } },
                .{ .column = .{ .index = b_idx } },
            }, .coefficient = neg },
        },
    };
    try testing.expectEqual(@as(usize, 2), running_sum.degree());

    // trace rows: a = [2, 3], b = [5, 7], s = [0, 10] (s wraps: s[0] = 0)
    const a = [_]Fp2{ g(2), g(3) };
    const b = [_]Fp2{ g(5), g(7) };
    const s = [_]Fp2{ g(0), g(10) };

    // Evaluate on row 0 (with the cyclic wrap row 2 == row 0).
    const w0 = Window{
        .prev = &s,
        .current = &s,
        .next = &.{ s[1], s[0] },
    };
    var w = w0;
    w.current = &.{ a[0], b[0], s[0] };
    w.next = &.{ a[1], b[1], s[1] };
    w.prev = &.{ a[1], b[1], s[1] };
    try testing.expect(running_sum.eval(w).isZero());
}

test "expr: tampered running sum is non-zero" {
    const running_sum = Constraint{
        .name = "s' - s - a*b",
        .terms = &.{
            .{ .factors = &.{.{ .column = .{ .index = 2, .offset = 1 } }} },
            .{ .factors = &.{.{ .column = .{ .index = 2 } }}, .coefficient = Fp2.neg(Fp2.one) },
            .{ .factors = &.{
                .{ .column = .{ .index = 0 } },
                .{ .column = .{ .index = 1 } },
            }, .coefficient = Fp2.neg(Fp2.one) },
        },
    };
    // s[1] tampered from 10 to 11.
    const a = [_]Fp2{ g(2), g(3) };
    const b = [_]Fp2{ g(5), g(7) };
    const s = [_]Fp2{ g(0), g(11) };
    const w = Window{
        .prev = &.{ a[1], b[1], s[1] },
        .current = &.{ a[0], b[0], s[0] },
        .next = &.{ a[1], b[1], s[1] },
    };
    try testing.expect(!running_sum.eval(w).isZero());
}

test "expr: system reports max degree, max column, wide offsets" {
    const sys = System{
        .constraints = &.{
            .{
                .name = "c0",
                .terms = &.{.{ .factors = &.{.{ .column = .{ .index = 3 } }} }},
            },
            .{
                .name = "c1",
                .terms = &.{.{ .factors = &.{
                    .{ .column = .{ .index = 1 } },
                    .{ .column = .{ .index = 7 } },
                    .{ .column = .{ .index = 2, .offset = 1 } },
                } }},
            },
        },
    };
    try testing.expectEqual(@as(usize, 3), sys.maxDegree());
    try testing.expectEqual(@as(?u16, 7), sys.maxColumn());
    try testing.expect(!sys.hasWideOffsets());
}

test "expr: wide offsets are detected" {
    const sys = System{
        .constraints = &.{.{
            .name = "wide",
            .terms = &.{.{ .factors = &.{.{ .column = .{ .index = 0, .offset = 5 } }} }},
        }},
    };
    try testing.expect(sys.hasWideOffsets());
}
