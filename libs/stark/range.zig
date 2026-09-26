//! Range and dequantization binding for trace operands (F2, LogUp-lite).
//!
//! The GEMM AIR proves `C = A·B` over whatever `a`/`b` columns the prover
//! supplies. On its own that says nothing about WHERE those values came
//! from: a prover could pick convenient operands and still produce a valid
//! proof. Binding them to the actual quantized weights needs each operand to
//! be shown to be a *dequantizable* value:
//!
//!   nibble in [0, 15]        Q4_K nibble, GGML convention: raw - 8
//!   value  == (nibble - 8) * scale
//!
//! This is the range-check half of the LogUp argument (docs/BLUE_PRINT.md §7.4):
//! a value is decomposed into bits and each bit is pinned by
//! `b * (b - 1) = 0`, which is quadratic and therefore expressible in the
//! existing IR. What is NOT here is the *lookup* half (membership in a
//! fixed table: the SiLU LUT, GGML's sub-scale table) nor fp16 parsing;
//! those still need the log-derivative argument. Tracked in TODO under F2.
//!
//! Checks are built into owned storage (`init`/`deinit`) rather than
//! comptime constants: the IR is pointer-heavy, and hand-rolled static
//! tables for a 26-bit scale would be unreadable.

const std = @import("std");
const expr = @import("expr.zig");
const field = @import("../field.zig");

const Goldilocks = field.Goldilocks;

pub const Fp2 = expr.Fp2;

pub const Constraint = expr.Constraint;
pub const Term = expr.Term;
pub const Factor = expr.Factor;
pub const System = expr.System;

pub const Error = error{ OutOfMemory, BadWidth };

/// Bit-decomposition range check: `target = sum 2^i * bit_i` with every
/// `bit_i` boolean, proving `0 <= target < 2^width`.
///
/// Storage is three flat allocations carved up, so the Factor/Term slices
/// inside each Constraint stay valid for the check's lifetime.
pub const RangeCheck = struct {
    allocator: std.mem.Allocator,
    target: u16,
    bit_base: u16,
    width: u8,
    constraints: []Constraint,
    terms: []Term,
    factors: []Factor,

    pub fn init(
        allocator: std.mem.Allocator,
        target: u16,
        bit_base: u16,
        width: u8,
    ) Error!RangeCheck {
        if (width == 0 or width > 32) return Error.BadWidth;
        const w: usize = width;
        // 1 reconstruction term (width+1 terms) + 2 per booleanity term.
        const n_terms = (w + 1) + 2 * w;
        // reconstruction: target + w bits, then the same bit twice per
        // booleanity (b*b and -b)
        const n_factors = (w + 1) + 2 * w;
        const n_constraints = 1 + w;

        const constraints = try allocator.alloc(Constraint, n_constraints);
        errdefer allocator.free(constraints);
        const terms = try allocator.alloc(Term, n_terms);
        errdefer allocator.free(terms);
        const factors = try allocator.alloc(Factor, n_factors);
        errdefer allocator.free(factors);

        // Layout: reconstruction factors occupy [0, w+1) — the target at 0,
        // then bit i at 1+i. Booleanity factors come after.
        //
        // (1) target - sum 2^i bit_i = 0 — a SUM, so one factor per term
        // (a single term holding every factor would be their product).
        factors[0] = .{ .column = .{ .index = target } };
        terms[0] = .{ .factors = factors[0..1] };
        for (0..w) |i| {
            factors[1 + i] = .{ .column = .{ .index = bit_base + @as(u16, @intCast(i)) } };
            // -2^i as a field element
            const two_i: u64 = @as(u64, 1) << @intCast(i);
            terms[1 + i] = .{
                .factors = factors[1 + i .. 2 + i],
                .coefficient = Fp2.re(negField(two_i)),
            };
        }
        constraints[0] = .{
            .name = "range.reconstruct",
            .scope = .composed,
            .terms = terms[0 .. w + 1],
        };

        // (2..) bit_i * (bit_i - 1) = 0, i.e. b*b - b. The second term
        // must repeat the COLUMN (with coefficient -1); a constant there
        // would state b*b = 1 and reject every zero bit.
        var t = w + 1;
        var f = w + 1;
        for (0..w) |i| {
            const col = bit_base + @as(u16, @intCast(i));
            factors[f] = .{ .column = .{ .index = col } };
            factors[f + 1] = .{ .column = .{ .index = col } };
            f += 2;
            terms[t] = .{ .factors = factors[f - 2 .. f] };
            terms[t + 1] = .{
                .factors = factors[f - 1 .. f],
                .coefficient = Fp2.neg(Fp2.one),
            };
            t += 2;
            constraints[i + 1] = .{
                .name = "range.bit_boolean",
                .scope = .composed,
                .terms = terms[t - 2 .. t],
            };
        }

        return .{
            .allocator = allocator,
            .target = target,
            .bit_base = bit_base,
            .width = width,
            .constraints = constraints,
            .terms = terms,
            .factors = factors,
        };
    }

    pub fn deinit(self: *RangeCheck) void {
        self.allocator.free(self.constraints);
        self.allocator.free(self.terms);
        self.allocator.free(self.factors);
        self.* = undefined;
    }

    pub fn bitCol(self: RangeCheck, i: u8) u16 {
        return self.bit_base + i;
    }
};

/// -v as a Goldilocks element, for the reconstruction's -2^i coefficients.
fn negField(v: u64) Goldilocks {
    return Goldilocks.zero.sub(Goldilocks.fromU64(v));
}

/// A system assembled from a base AIR plus range checks on named columns.
/// Owns the merged constraint list; the base system's constraints are
/// copied, so the base may be static.
pub const BuiltSystem = struct {
    allocator: std.mem.Allocator,
    constraints: []Constraint,
    checks: []RangeCheck,
    system: System,

    pub fn init(
        allocator: std.mem.Allocator,
        base: System,
        specs: []const Spec,
    ) Error!BuiltSystem {
        var n_base = base.constraints.len;
        for (specs) |spec| n_base += 1 + spec.width;

        const merged = try allocator.alloc(Constraint, n_base);
        errdefer allocator.free(merged);
        const checks = try allocator.alloc(RangeCheck, specs.len);
        errdefer allocator.free(checks);

        @memcpy(merged[0..base.constraints.len], base.constraints);
        var at = base.constraints.len;
        for (specs, 0..) |spec, i| {
            const check = RangeCheck.init(allocator, spec.column, spec.bit_base, spec.width) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.BadWidth => return error.BadWidth,
            };
            checks[i] = check;
            for (check.constraints) |c| {
                merged[at] = c;
                at += 1;
            }
        }
        return .{
            .allocator = allocator,
            .constraints = merged,
            .checks = checks,
            .system = base.replaceConstraints(merged),
        };
    }

    pub fn deinit(self: *BuiltSystem) void {
        for (self.checks) |*c| c.deinit();
        self.allocator.free(self.checks);
        self.allocator.free(self.constraints);
        self.* = undefined;
    }
};

pub const Spec = struct {
    /// Column whose value is range-checked.
    column: u16,
    /// First column index of that check's bit decomposition.
    bit_base: u16,
    width: u8,
};

const testing = std.testing;

test "range: a 4-bit check emits one reconstruction and four booleanity constraints" {
    const a = testing.allocator;
    var c = try RangeCheck.init(a, 0, 8, 4);
    defer c.deinit();
    try testing.expectEqual(@as(usize, 5), c.constraints.len);
    try testing.expectEqual(@as(u16, 8), c.bitCol(0));
    try testing.expectEqual(@as(u16, 11), c.bitCol(3));
}

test "range: the system builder merges base and check constraints" {
    const a = testing.allocator;
    const base_terms = [_]Term{.{ .factors = &[_]Factor{.{ .column = .{ .index = 0 } }} }};
    const base_constraints = [_]Constraint{.{ .name = "base", .terms = &base_terms }};
    const base = System{ .constraints = &base_constraints };

    const specs = [_]Spec{
        .{ .column = 0, .bit_base = 1, .width = 4 },
        .{ .column = 2, .bit_base = 5, .width = 8 },
    };
    var built = try BuiltSystem.init(a, base, &specs);
    defer built.deinit();

    // 1 base + (1+4) + (1+8)
    try testing.expectEqual(@as(usize, 15), built.system.constraints.len);
    try testing.expectEqual(@as(usize, 2), built.checks.len);
    try testing.expectEqual(@as(?u16, 12), built.system.maxColumn());
    try testing.expectEqual(@as(usize, 2), built.system.maxDegree());
}

test "range: a zero-width check is rejected" {
    const a = testing.allocator;
    try testing.expectError(Error.BadWidth, RangeCheck.init(a, 0, 4, 0));
}
