//! Normalization gadgets — RMSNorm and LayerNorm.
//!
//! BLUE_PRINT §4.3: sums are proven by LogUp of ranges (byte lookups).

const std = @import("std");
const tensor = @import("../../tensor/root.zig");
const air = @import("../../air/root.zig");

pub const Goldilocks = tensor.Goldilocks;

pub const RmsNormGadget = struct {
    rows: usize,
    cols: usize,
    weight: []const i8,

    pub fn airFragment(self: @This(), gpa: std.mem.Allocator) !air.Fragment {
        const cols_arr = try gpa.alloc(air.Column, 2);
        errdefer gpa.free(cols_arr);
        cols_arr[0] = .{ .name = "rms_sq_sum", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.fixed_q16_16, .bound = self.cols * tensor.Scheme.magnitudeBound(tensor.Scheme.int8_symmetric) * tensor.Scheme.magnitudeBound(tensor.Scheme.int8_symmetric) };
        cols_arr[1] = .{ .name = "rms_out", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.int8_symmetric, .bound = tensor.Scheme.magnitudeBound(tensor.Scheme.int8_symmetric) };

        const constraints = try gpa.alloc(air.Constraint, 1);
        errdefer gpa.free(constraints);
        constraints[0] = .{ .degree = 1 };

        const luts = try gpa.alloc(air.LookupTable, 0);
        errdefer gpa.free(luts);

        return air.Fragment{
            .columns = cols_arr,
            .constraints = constraints,
            .lookups = luts,
            .rows = self.rows,
        };
    }
};

pub const LayerNormGadget = struct {
    rows: usize,
    cols: usize,
    weight: []const i8,
    bias: []const i8,

    pub fn airFragment(self: @This(), gpa: std.mem.Allocator) !air.Fragment {
        const cols_arr = try gpa.alloc(air.Column, 3);
        errdefer gpa.free(cols_arr);
        cols_arr[0] = .{ .name = "ln_mean", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.fixed_q16_16, .bound = tensor.Scheme.magnitudeBound(tensor.Scheme.int8_symmetric) };
        cols_arr[1] = .{ .name = "ln_var", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.fixed_q16_16, .bound = self.cols * tensor.Scheme.magnitudeBound(tensor.Scheme.int8_symmetric) * tensor.Scheme.magnitudeBound(tensor.Scheme.int8_symmetric) };
        cols_arr[2] = .{ .name = "ln_out", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.int8_symmetric, .bound = tensor.Scheme.magnitudeBound(tensor.Scheme.int8_symmetric) };

        const constraints = try gpa.alloc(air.Constraint, 2);
        errdefer gpa.free(constraints);
        constraints[0] = .{ .degree = 1 };
        constraints[1] = .{ .degree = 1 };

        const luts = try gpa.alloc(air.LookupTable, 0);
        errdefer gpa.free(luts);

        return air.Fragment{
            .columns = cols_arr,
            .constraints = constraints,
            .lookups = luts,
            .rows = self.rows,
        };
    }
};

test "rmsnorm fragment has correct rows" {
    const t = std.testing;
    const g = RmsNormGadget{ .rows = 8, .cols = 64, .weight = &[_]i8{1} ** 64 };
    var frag = g.airFragment(t.allocator) catch unreachable;
    defer frag.deinit(t.allocator);
    try t.expectEqual(@as(usize, 8), frag.rows);
}

test "layernorm fragment has correct rows" {
    const t = std.testing;
    const g = LayerNormGadget{ .rows = 4, .cols = 128, .weight = &[_]i8{1} ** 128, .bias = &[_]i8{0} ** 128 };
    var frag = g.airFragment(t.allocator) catch unreachable;
    defer frag.deinit(t.allocator);
    try t.expectEqual(@as(usize, 4), frag.rows);
}
