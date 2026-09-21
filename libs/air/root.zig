//! AIR fragment IR — intermediate representation emitted by L2 gadgets and
//! consumed by the L3 compiler (CircuitGraph → AirGraph) and L1 (zig-zk).
//!
//! A `Fragment` is a self-contained subgraph with dimension holes that L3
//! instantiates and links. Gadgets never produce a full AirGraph; they only
//! declare columns, constraints, and lookup tables.

const std = @import("std");
const tensor = @import("../tensor/root.zig");

pub const Goldilocks = tensor.Goldilocks;

pub const ColumnRole = enum(u8) {
    public = 0,
    advice = 1,
    fixed = 2,
};

pub const Column = struct {
    name: []const u8,
    role: ColumnRole,
    scheme: tensor.Scheme,
    bound: usize,

    pub fn widthBytes(self: Column) u8 {
        return switch (self.scheme) {
            .int8_symmetric, .fp8_e4m3, .mxfp8_e4m3 => 1,
            .int4_gguf_q4_k, .int4_q8_0 => 1,
            .fixed_q16_16 => 4,
            .fixed_q8_8 => 2,
        };
    }
};

pub const LookupTable = struct {
    table: []const i16,
    width: usize,
};

pub const Constraint = struct {
    degree: u8,
};

pub const Fragment = struct {
    columns: []const Column,
    constraints: []const Constraint,
    lookups: []const LookupTable,
    rows: usize,

    pub fn deinit(self: *Fragment, allocator: std.mem.Allocator) void {
        if (self.columns.len > 0) allocator.free(self.columns);
        if (self.constraints.len > 0) allocator.free(self.constraints);
        if (self.lookups.len > 0) {
            for (self.lookups) |lut| {
                if (lut.table.len > 0) allocator.free(lut.table);
            }
            allocator.free(self.lookups);
        }
        self.* = undefined;
    }
};

test "column widthBytes matches scheme rangeBits" {
    const t = std.testing;
    const col = Column{
        .name = "a",
        .role = .advice,
        .scheme = .int8_symmetric,
        .bound = 128,
    };
    try t.expectEqual(@as(u8, 1), col.widthBytes());

    const col32 = Column{
        .name = "b",
        .role = .advice,
        .scheme = .fixed_q16_16,
        .bound = 32768,
    };
    try t.expectEqual(@as(u8, 4), col32.widthBytes());
}

test "fragment deinit frees owned slices" {
    const t = std.testing;
    const a = t.allocator;

    const cols = try a.alloc(Column, 2);
    errdefer a.free(cols);
    cols[0] = .{ .name = "x", .role = .advice, .scheme = .int8_symmetric, .bound = 128 };
    cols[1] = .{ .name = "y", .role = .advice, .scheme = .int8_symmetric, .bound = 128 };

    const lut_table = try a.alloc(i16, 4);
    errdefer a.free(lut_table);
    lut_table[0] = 0;
    lut_table[1] = 1;
    lut_table[2] = 2;
    lut_table[3] = 3;

    const luts = try a.alloc(LookupTable, 1);
    errdefer a.free(luts);
    luts[0] = .{ .table = lut_table, .width = 8 };

    var frag = Fragment{
        .columns = cols,
        .constraints = &.{},
        .lookups = luts,
        .rows = 16,
    };

    frag.deinit(a);
}
