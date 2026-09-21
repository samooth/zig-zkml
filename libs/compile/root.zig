//! L3 compiler — CircuitGraph → AirGraph.
//!
//! BLUE_PRINT §5.1/§7.1: gadgets emit `air.Fragment` instances with dimension
//! holes. The compiler instantiates those holes, merges columns/constraints/
//! lookups across fragments, and produces a single `AirGraph` consumable by
//! the L1 STARK prover (zig-zk).
//!
//! v1 strategy: monolithic AIR per layer (BLUE_PRINT §5.2). No cross-fragment
//! composition — the compiler concatenates gadgets into one trace.

const std = @import("std");
const air = @import("../air/root.zig");
const tensor = @import("../tensor/root.zig");

pub const Goldilocks = tensor.Goldilocks;

pub const ColumnRole = air.ColumnRole;

pub const Column = air.Column;
pub const LookupTable = air.LookupTable;
pub const Constraint = air.Constraint;

pub const PublicInput = struct {
    name: []const u8,
    value: []const Goldilocks,
};

pub const AirGraph = struct {
    columns: []Column,
    constraints: []Constraint,
    lookups: []LookupTable,
    public_inputs: []PublicInput,
    rows: usize,

    pub fn deinit(self: *AirGraph, allocator: std.mem.Allocator) void {
        if (self.columns.len > 0) allocator.free(self.columns);
        if (self.constraints.len > 0) allocator.free(self.constraints);
        if (self.lookups.len > 0) {
            for (self.lookups) |lut| {
                if (lut.table.len > 0) allocator.free(lut.table);
            }
            allocator.free(self.lookups);
        }
        if (self.public_inputs.len > 0) allocator.free(self.public_inputs);
        self.* = undefined;
    }
};

pub const CircuitGraph = struct {
    gpa: std.mem.Allocator,
    fragments: std.ArrayList(air.Fragment),
    columns: std.StringHashMap(Column),
    constraints: std.ArrayList(Constraint),
    lookups: std.ArrayList(LookupTable),
    public_inputs: std.ArrayList(PublicInput),
    rows: usize,

    pub fn init(gpa: std.mem.Allocator) CircuitGraph {
        return .{
            .gpa = gpa,
            .fragments = std.ArrayList(air.Fragment).empty,
            .columns = std.StringHashMap(Column).init(gpa),
            .constraints = std.ArrayList(Constraint).empty,
            .lookups = std.ArrayList(LookupTable).empty,
            .public_inputs = std.ArrayList(PublicInput).empty,
            .rows = 0,
        };
    }

    pub fn deinit(self: *CircuitGraph) void {
        for (self.fragments.items) |*frag| frag.deinit(self.gpa);
        self.fragments.deinit(self.gpa);
        self.columns.deinit();
        self.constraints.deinit(self.gpa);
        self.lookups.deinit(self.gpa);
        self.public_inputs.deinit(self.gpa);
        self.* = undefined;
    }

    /// Add a gadget fragment. The compiler merges its columns/constraints/
    /// lookups into the graph and accumulates rows.
    pub fn addFragment(self: *CircuitGraph, frag: air.Fragment) !void {
        try self.fragments.append(self.gpa, frag);

        // Merge columns (deduplicate by name).
        for (frag.columns) |col| {
            const gop = try self.columns.getOrPut(col.name);
            if (!gop.found_existing) {
                gop.value_ptr.* = col;
            }
        }

        // Merge constraints.
        try self.constraints.appendSlice(self.gpa, frag.constraints);

        // Merge lookups (deduplicate by table pointer identity).
        for (frag.lookups) |lut| {
            var dup = false;
            for (self.lookups.items) |existing| {
                if (existing.table.ptr == lut.table.ptr and existing.width == lut.width) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                try self.lookups.append(self.gpa, lut);
            }
        }

        self.rows += frag.rows;
    }

    /// Add a public input binding.
    pub fn addPublicInput(self: *CircuitGraph, name: []const u8, value: []const Goldilocks) !void {
        try self.public_inputs.append(self.gpa, .{ .name = name, .value = value });
    }

    /// Finalize: build the AirGraph.
    pub fn build(self: *CircuitGraph, allocator: std.mem.Allocator) !AirGraph {
        var columns_out = try allocator.alloc(Column, self.columns.count());
        var col_idx: usize = 0;
        var it = self.columns.iterator();
        while (it.next()) |entry| {
            columns_out[col_idx] = entry.value_ptr.*;
            col_idx += 1;
        }

        const constraints_out = try self.constraints.toOwnedSlice(allocator);
        const lookups_out = try self.lookups.toOwnedSlice(allocator);
        const public_inputs_out = try self.public_inputs.toOwnedSlice(allocator);

        return AirGraph{
            .columns = columns_out,
            .constraints = constraints_out,
            .lookups = lookups_out,
            .public_inputs = public_inputs_out,
            .rows = self.rows,
        };
    }
};

test "circuit graph merges fragments" {
    const t = std.testing;
    const a = t.allocator;

    var graph = CircuitGraph.init(a);
    defer graph.deinit();

    const col_a = Column{ .name = "a", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.int8_symmetric, .bound = 128 };
    const col_b = Column{ .name = "b", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.int8_symmetric, .bound = 128 };

    const frag1_cols = try a.alloc(Column, 1);
    frag1_cols[0] = col_a;

    const frag2_cols = try a.alloc(Column, 1);
    frag2_cols[0] = col_b;

    const frag1_constraints = try a.alloc(Constraint, 1);
    frag1_constraints[0] = .{ .degree = 1 };

    const frag2_constraints = try a.alloc(Constraint, 1);
    frag2_constraints[0] = .{ .degree = 2 };

    const frag1 = air.Fragment{
        .columns = frag1_cols,
        .constraints = frag1_constraints,
        .lookups = &.{},
        .rows = 8,
    };

    const frag2 = air.Fragment{
        .columns = frag2_cols,
        .constraints = frag2_constraints,
        .lookups = &.{},
        .rows = 16,
    };

    try graph.addFragment(frag1);
    try graph.addFragment(frag2);

    var air_graph = try graph.build(a);
    defer air_graph.deinit(a);

    try t.expectEqual(@as(usize, 24), air_graph.rows);
    try t.expectEqual(@as(usize, 2), air_graph.columns.len);
    try t.expectEqual(@as(usize, 2), air_graph.constraints.len);
}
