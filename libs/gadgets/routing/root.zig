//! Routing gadgets — top-k and group-top2 selection for MoE layers.
//!
//! docs/BLUE_PRINT.md §7.1: DeepSeek-V3 uses group-top2 among 256 experts; the
//! selection logic is proven as binary comparisons (LogUp range proofs
//! on the gate scores).

const std = @import("std");
const tensor = @import("../../tensor/root.zig");
const air = @import("../../air/root.zig");

pub const Goldilocks = tensor.Goldilocks;

pub const TopKGadget = struct {
    rows: usize,
    k: usize,
    gate_scores: []const Goldilocks,

    pub fn airFragment(self: @This(), gpa: std.mem.Allocator) !air.Fragment {
        const cols_arr = try gpa.alloc(air.Column, 2);
        errdefer gpa.free(cols_arr);
        cols_arr[0] = .{ .name = "gate_score", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.fixed_q16_16, .bound = tensor.Scheme.magnitudeBound(tensor.Scheme.fixed_q16_16) };
        cols_arr[1] = .{ .name = "selected", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.int8_symmetric, .bound = tensor.Scheme.magnitudeBound(tensor.Scheme.int8_symmetric) };

        const constraints = try gpa.alloc(air.Constraint, 1);
        errdefer gpa.free(constraints);
        constraints[0] = .{ .degree = 1 };

        const luts = try gpa.alloc(air.LookupTable, 0);
        errdefer gpa.free(luts);

        return air.Fragment{
            .columns = cols_arr,
            .constraints = constraints,
            .lookups = luts,
            .rows = self.rows * self.k,
        };
    }
};

pub const GroupTop2Gadget = struct {
    rows: usize,
    num_experts: usize,
    gate_scores: []const Goldilocks,

    pub fn airFragment(self: @This(), gpa: std.mem.Allocator) !air.Fragment {
        const cols_arr = try gpa.alloc(air.Column, 3);
        errdefer gpa.free(cols_arr);
        cols_arr[0] = .{ .name = "group_score", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.fixed_q16_16, .bound = tensor.Scheme.magnitudeBound(tensor.Scheme.fixed_q16_16) };
        cols_arr[1] = .{ .name = "top1_idx", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.int8_symmetric, .bound = self.num_experts };
        cols_arr[2] = .{ .name = "top2_idx", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.int8_symmetric, .bound = self.num_experts };

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
            .rows = self.rows * 2,
        };
    }
};

test "topk fragment rows = rows * k" {
    const t = std.testing;
    const scores = [_]Goldilocks{ Goldilocks.one, Goldilocks.one, Goldilocks.one } ** 8;
    const g = TopKGadget{ .rows = 4, .k = 2, .gate_scores = &scores };
    var frag = g.airFragment(t.allocator) catch unreachable;
    defer frag.deinit(t.allocator);
    try t.expectEqual(@as(usize, 8), frag.rows);
}

test "group top2 fragment rows = rows * 2" {
    const t = std.testing;
    const scores = [_]Goldilocks{Goldilocks.one} ** 256;
    const g = GroupTop2Gadget{ .rows = 8, .num_experts = 256, .gate_scores = &scores };
    var frag = g.airFragment(t.allocator) catch unreachable;
    defer frag.deinit(t.allocator);
    try t.expectEqual(@as(usize, 16), frag.rows);
}
