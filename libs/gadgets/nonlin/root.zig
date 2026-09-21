//! Non-linear activation lookups — SiLU, GELU, softmax-step.
//!
//! BLUE_PRINT §4.3: non-linearities are always LogUp lookups over precomputed
//! tables. SiLU output (i16 q8.8) is proven with 2 byte lookups (high + low),
//! never a single 2^16 table.

const std = @import("std");
const tensor = @import("../../tensor/root.zig");
const air = @import("../../air/root.zig");

pub const Goldilocks = tensor.Goldilocks;

pub const SiLULookup = struct {
    pub const silu_q8_8: [256]i16 = blk: {
        var t: [256]i16 = undefined;
        for (0..256) |i| {
            const x: f32 = @floatFromInt(@as(i8, @bitCast(@as(u8, @intCast(i)))));
            const s = x / (1.0 + @exp(-x));
            const scaled = @round(s * 256.0);
            t[i] = @intFromFloat(scaled);
        }
        break :blk t;
    };

    pub fn airFragment(gpa: std.mem.Allocator) !air.Fragment {
        const lut_low = try gpa.alloc(i16, 256);
        errdefer gpa.free(lut_low);
        const lut_high = try gpa.alloc(i16, 256);
        errdefer gpa.free(lut_high);
        for (0..256) |i| {
            const v = silu_q8_8[i];
            lut_low[i] = v & 0xff;
            lut_high[i] = (v >> 8) & 0xff;
        }
        const luts = try gpa.alloc(air.LookupTable, 2);
        errdefer gpa.free(luts);
        luts[0] = .{ .table = lut_low, .width = 8 };
        luts[1] = .{ .table = lut_high, .width = 8 };

        const cols = try gpa.alloc(air.Column, 1);
        errdefer gpa.free(cols);
        cols[0] = .{ .name = "silu_out", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.fixed_q8_8, .bound = 256 };

        const constraints = try gpa.alloc(air.Constraint, 1);
        errdefer gpa.free(constraints);
        constraints[0] = .{ .degree = 1 };

        return air.Fragment{
            .columns = cols,
            .constraints = constraints,
            .lookups = luts,
            .rows = 1,
        };
    }
};

pub const GELULookup = struct {
    pub const gelu_q8_8: [256]i16 = blk: {
        var t: [256]i16 = undefined;
        for (0..256) |i| {
            const x: f32 = @floatFromInt(@as(i8, @bitCast(@as(u8, @intCast(i)))));
            const sigmoid = 1.0 / (1.0 + @exp(-1.702 * x));
            const gelu = x * sigmoid;
            const scaled = @round(gelu * 256.0);
            t[i] = @intFromFloat(scaled);
        }
        break :blk t;
    };

    pub fn airFragment(gpa: std.mem.Allocator) !air.Fragment {
        const lut_low = try gpa.alloc(i16, 256);
        errdefer gpa.free(lut_low);
        const lut_high = try gpa.alloc(i16, 256);
        errdefer gpa.free(lut_high);
        for (0..256) |i| {
            const v = gelu_q8_8[i];
            lut_low[i] = v & 0xff;
            lut_high[i] = (v >> 8) & 0xff;
        }
        const luts = try gpa.alloc(air.LookupTable, 2);
        errdefer gpa.free(luts);
        luts[0] = .{ .table = lut_low, .width = 8 };
        luts[1] = .{ .table = lut_high, .width = 8 };

        const cols = try gpa.alloc(air.Column, 1);
        errdefer gpa.free(cols);
        cols[0] = .{ .name = "gelu_out", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.fixed_q8_8, .bound = 256 };

        const constraints = try gpa.alloc(air.Constraint, 1);
        errdefer gpa.free(constraints);
        constraints[0] = .{ .degree = 1 };

        return air.Fragment{
            .columns = cols,
            .constraints = constraints,
            .lookups = luts,
            .rows = 1,
        };
    }
};

test "silu table is bounded and smooth-ish at 0" {
    const t = std.testing;
    try t.expectEqual(@as(i16, 0), SiLULookup.silu_q8_8[128]);
    try t.expect(SiLULookup.silu_q8_8[0] == 0);
    try t.expect(SiLULookup.silu_q8_8[255] < 0);
    try t.expect(SiLULookup.silu_q8_8[127] > 0);
}

test "gelu table is bounded" {
    const t = std.testing;
    try t.expect(GELULookup.gelu_q8_8[0] >= 0);
    try t.expect(GELULookup.gelu_q8_8[255] <= 256);
}
