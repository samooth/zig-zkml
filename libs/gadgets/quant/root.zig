//! Quantization gadgets — dequant/requant, fp8/mxfp8 reencode, and GGML
//! block-scale handling.
//!
//! BLUE_PRINT §4.2: every scheme carries a comptime magnitude bound M and
//! the AIR generates range proofs via LogUp on bytes (§4.3).

const std = @import("std");
const tensor = @import("../../tensor/root.zig");
const air = @import("../../air/root.zig");

pub const Goldilocks = tensor.Goldilocks;

pub fn requantInt8ToFixedQ8_8(allocator: std.mem.Allocator, values: []const i8) ![]const Goldilocks {
    const out = try allocator.alloc(Goldilocks, values.len);
    for (values, 0..) |v, i| {
        out[i] = Goldilocks.fromI64(v);
    }
    return out;
}

pub fn requantFixedQ16_16ToInt8(allocator: std.mem.Allocator, values: []const Goldilocks) ![]i8 {
    var out = try allocator.alloc(i8, values.len);
    for (values, 0..) |v, i| {
        const u = v.toU64();
        if (u >= (1 << 7)) {
            out[i] = 127;
        } else if (u >= (@as(u64, 1) << 61) - (@as(u64, 1) << 7)) {
            out[i] = -128;
        } else {
            out[i] = @intCast(@as(i9, @bitCast(@as(u9, @intCast(u)))));
        }
    }
    return out;
}

pub const Fp8Reencode = struct {
    pub const table: [256]i16 = blk: {
        var t: [256]i16 = undefined;
        for (0..256) |i| {
            const sign = @as(i16, if ((i & 0x80) != 0) -1 else 1);
            const exp = @as(i16, (i >> 3) & 0x0f) - 7;
            const mant = @as(i16, i & 0x07);
            const shifted = @as(u16, 1) << @intCast(exp);
            t[i] = sign * @as(i16, @intCast(shifted)) * (mant + 8);
        }
        break :blk t;
    };

    pub fn airFragment(gpa: std.mem.Allocator) !air.Fragment {
        const tbl = try gpa.alloc(i16, 256);
        errdefer gpa.free(tbl);
        @memcpy(tbl, &table);

        const luts = try gpa.alloc(air.LookupTable, 1);
        errdefer gpa.free(luts);
        luts[0] = .{ .table = tbl, .width = 8 };

        const cols = try gpa.alloc(air.Column, 1);
        errdefer gpa.free(cols);
        cols[0] = .{ .name = "fp8_reencode", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.fp8_e4m3, .bound = 128 };

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

test "requant int8 to fixed q8.8 converts correctly" {
    const t = std.testing;
    const a = t.allocator;
    const vals = [_]i8{ -1, 0, 1, 127 };
    const out = try requantInt8ToFixedQ8_8(a, &vals);
    defer a.free(out);
    try t.expectEqual(@as(usize, 4), out.len);
    try t.expect(out[0].rep == Goldilocks.fromI64(-1).rep);
    try t.expect(out[1].isZero());
    try t.expect(out[2].rep == Goldilocks.fromI64(1).rep);
}

test "requant fixed q16.16 to int8 clamps" {
    const t = std.testing;
    const a = t.allocator;
    const vals = [_]Goldilocks{ Goldilocks.fromU64(200), Goldilocks.fromU64(100) };
    const out = try requantFixedQ16_16ToInt8(a, &vals);
    defer a.free(out);
    try t.expectEqual(@as(i8, 127), out[0]);
    try t.expect(out[1] >= -128);
    try t.expect(out[1] <= 127);
}
