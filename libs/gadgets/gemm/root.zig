//! GEMM gadget — C[m,n] = dequant(A[m,k]) · dequant(B[k,n]) verified via
//! running-sum AIR with 16 MACs per row (BLUE_PRINT §7.3).
//!
//! v1: monolithic AIR per layer. The witness comes from the exact-path
//! kernel (§3 dual-path), not the fast path.

const std = @import("std");
const tensor = @import("../../tensor/root.zig");
const air = @import("../../air/root.zig");

pub const Goldilocks = tensor.Goldilocks;

pub const GemmGadget = struct {
    m: usize,
    k: usize,
    n: usize,
    a: tensor.QuantTensor(.int4_gguf_q4_k),
    b: tensor.QuantTensor(.int4_gguf_q4_k),
    c: []const Goldilocks,

    pub const macs_per_row: usize = 16;

    pub fn airFragment(self: @This(), gpa: std.mem.Allocator) !air.Fragment {
        const A_scheme = tensor.Scheme.int4_gguf_q4_k;
        const B_scheme = tensor.Scheme.int4_gguf_q4_k;

        const cols = try gpa.alloc(air.Column, 4);
        errdefer gpa.free(cols);
        cols[0] = .{ .name = "a_chunk", .role = air.ColumnRole.advice, .scheme = A_scheme, .bound = tensor.Scheme.magnitudeBound(A_scheme) };
        cols[1] = .{ .name = "b_chunk", .role = air.ColumnRole.advice, .scheme = B_scheme, .bound = tensor.Scheme.magnitudeBound(B_scheme) };
        cols[2] = .{ .name = "s_running", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.fixed_q16_16, .bound = self.k * tensor.Scheme.magnitudeBound(A_scheme) * tensor.Scheme.magnitudeBound(B_scheme) };
        cols[3] = .{ .name = "s_final", .role = air.ColumnRole.advice, .scheme = tensor.Scheme.fixed_q16_16, .bound = tensor.Scheme.magnitudeBound(tensor.Scheme.fixed_q16_16) };

        const constraints = try gpa.alloc(air.Constraint, 1);
        errdefer gpa.free(constraints);
        constraints[0] = .{ .degree = 2 };

        return air.Fragment{
            .columns = cols,
            .constraints = constraints,
            .lookups = &.{},
            .rows = (self.m * self.n * self.k) / macs_per_row,
        };
    }
};

test "gemm air fragment basic structure" {
    const t = std.testing;
    const a = t.allocator;

    const gadget = GemmGadget{
        .m = 2,
        .k = 2,
        .n = 2,
        .a = tensor.QuantTensor(.int4_gguf_q4_k){ .rows = 2, .cols = 2, .data = &[_]u8{0} ** 2, .scales = &[_]u16{0} ** 1 },
        .b = tensor.QuantTensor(.int4_gguf_q4_k){ .rows = 2, .cols = 2, .data = &[_]u8{0} ** 2, .scales = &[_]u16{0} ** 1 },
        .c = &[_]Goldilocks{Goldilocks.zero} ** 4,
    };

    var frag = gadget.airFragment(a) catch |err| {
        std.debug.print("airFragment error: {}\n", .{err});
        return error.TestUnexpectedResult;
    };
    defer frag.deinit(a);

    try t.expect(frag.columns.len == 4);
    try t.expect(frag.constraints.len == 1);
    try t.expect(frag.constraints[0].degree == 2);
    try t.expect(frag.rows == 0); // 2*2*2/16 = 0
}
