//! End-to-end tests for the chunked layout WITH operand binding (F2).
//!
//! gemm_chunk_test.zig proves the chunked AIR on its own; this file
//! checks that the 16× smaller trace is still tied to the quantized
//! weights — that is, that chunking cost no soundness.

const std = @import("std");
const tensor = @import("../tensor/root.zig");
const stark = @import("root.zig");
const chunk = @import("gemm_chunk.zig");
const cb = @import("chunk_binding.zig");

const testing = std.testing;
const Goldilocks = tensor.Goldilocks;
const Fp2 = stark.Fp2;

const k_macs: usize = 256;

const CONFIG: stark.Config = blk: {
    const log_trace: u6 = 5;
    const log_blowup: u6 = 2;
    break :blk .{
        .log_trace = log_trace,
        .log_blowup = log_blowup,
        .fri = .{
            .log_domain = log_trace + log_blowup,
            .log_final = log_trace + 1,
            .log_residual_degree = log_trace,
            .num_queries = 4,
        },
    };
};

/// A Q4_K block whose bytes encode known raw nibbles: byte j is (j%16) in
/// both halves, so element 2j and 2j+1 both have raw nibble j%16.
fn blockWithRamp() [128]u8 {
    var out: [128]u8 = undefined;
    for (0..128) |j| {
        const raw: u8 = @intCast(j % 16);
        out[j] = (raw << 4) | raw;
    }
    return out;
}

fn rawNibble(i: usize) u8 {
    return @intCast((i / 2) % 16);
}

const Case = struct {
    a: []Goldilocks,
    b: []Goldilocks,
    nib_a: []u8,
    nib_b: []u8,
    scale_a: []Goldilocks,
    scale_b: []Goldilocks,
    c_true: Goldilocks,

    fn deinit(self: *Case, allocator: std.mem.Allocator) void {
        allocator.free(self.a);
        allocator.free(self.b);
        allocator.free(self.nib_a);
        allocator.free(self.nib_b);
        allocator.free(self.scale_a);
        allocator.free(self.scale_b);
        self.* = undefined;
    }
};

/// A real Q4_K reduction: one block per operand, so all 256 MACs share
/// one block scale each.
fn realCase(allocator: std.mem.Allocator) !Case {
    const block = blockWithRamp();
    const deq_a = tensor.dequantQ4K(&block, 0x3C00) catch unreachable; // fp16 1.0
    const deq_b = tensor.dequantQ4K(&block, 0x3800) catch unreachable; // fp16 0.5
    const s_a = Goldilocks.fromU64(tensor.fp16ToFixedQ4_22(0x3C00) catch unreachable);
    const s_b = Goldilocks.fromU64(tensor.fp16ToFixedQ4_22(0x3800) catch unreachable);

    const a = try allocator.alloc(Goldilocks, k_macs);
    errdefer allocator.free(a);
    const b = try allocator.alloc(Goldilocks, k_macs);
    errdefer allocator.free(b);
    const nib_a = try allocator.alloc(u8, k_macs);
    errdefer allocator.free(nib_a);
    const nib_b = try allocator.alloc(u8, k_macs);
    errdefer allocator.free(nib_b);
    const scale_a = try allocator.alloc(Goldilocks, k_macs);
    errdefer allocator.free(scale_a);
    const scale_b = try allocator.alloc(Goldilocks, k_macs);
    errdefer allocator.free(scale_b);

    var acc = Goldilocks.zero;
    for (0..k_macs) |i| {
        a[i] = deq_a[i];
        b[i] = deq_b[i];
        nib_a[i] = rawNibble(i);
        nib_b[i] = rawNibble(i);
        scale_a[i] = s_a;
        scale_b[i] = s_b;
        acc = acc.add(a[i].mul(b[i]));
    }
    return .{
        .a = a,
        .b = b,
        .nib_a = nib_a,
        .nib_b = nib_b,
        .scale_a = scale_a,
        .scale_b = scale_b,
        .c_true = acc,
    };
}

fn boundTrace(allocator: std.mem.Allocator, case: *const Case) !struct { gemm: chunk.Trace, bound: cb.Trace } {
    var gemm = try chunk.buildTrace(allocator, case.a, case.b, case.c_true);
    errdefer gemm.deinit(allocator);
    const bound = try cb.bindOperands(
        allocator,
        &gemm,
        case.nib_a,
        case.scale_a,
        case.nib_b,
        case.scale_b,
    );
    return .{ .gemm = gemm, .bound = bound };
}

test "chunk_binding: a real Q4_K reduction proves and verifies in 32 rows" {
    const a = testing.allocator;
    var sys = try cb.buildSystem(a);
    defer sys.deinit();

    var case = try realCase(a);
    defer case.deinit(a);
    try testing.expect(!case.c_true.isZero());

    var bt = try boundTrace(a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);
    try testing.expectEqual(@as(usize, 32), bt.bound.rows);
    try testing.expectEqual(@as(usize, cb.column_count), bt.bound.columns.len);

    var pt = stark.Transcript.init("zkml.chunkbind.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = bt.bound.rows,
        .columns = bt.bound.columns,
    }, sys.system(), CONFIG);
    defer proof.deinit(a);

    var vt = stark.Transcript.init("zkml.chunkbind.v1");
    try testing.expect(try stark.verify(&vt, &proof, sys.system(), CONFIG));
}

test "chunk_binding: a tampered chunk operand is rejected" {
    const a = testing.allocator;
    var sys = try cb.buildSystem(a);
    defer sys.deinit();

    var case = try realCase(a);
    defer case.deinit(a);
    var bt = try boundTrace(a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);

    // Slot 5 of chunk 3, deep inside the trace: the dequantization
    // equation for that slot is what notices.
    const row = 3;
    const slot = 5;
    bt.bound.columns[chunk.colA(slot)][row] = Fp2.re(Goldilocks.fromU64(999));

    var pt = stark.Transcript.init("zkml.chunkbind.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{
            .rows = bt.bound.rows,
            .columns = bt.bound.columns,
        }, sys.system(), CONFIG),
    );
}

test "chunk_binding: a non-boolean bit deep in a chunk is rejected" {
    const a = testing.allocator;
    var sys = try cb.buildSystem(a);
    defer sys.deinit();

    var case = try realCase(a);
    defer case.deinit(a);
    var bt = try boundTrace(a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);

    // Nibble 3 re-decomposed as -1 + 2*2: the sum is still 3, so only the
    // booleanity constraint of that slot can catch it. MAC 38 is the one
    // with raw nibble 3 ((38/2) % 16), which is row 2, slot 6.
    const row = 2;
    const slot = 6;
    const nib_col = cb.colNibA(slot);
    try testing.expectEqual(@as(u8, 3), bt.bound.columns[nib_col][row].a.toU64());
    bt.bound.columns[cb.colBitsA(slot) + 0][row] = Fp2.re(Goldilocks.zero.sub(Goldilocks.one));
    bt.bound.columns[cb.colBitsA(slot) + 1][row] = Fp2.re(Goldilocks.fromU64(2));

    var pt = stark.Transcript.init("zkml.chunkbind.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{
            .rows = bt.bound.rows,
            .columns = bt.bound.columns,
        }, sys.system(), CONFIG),
    );
}

test "chunk_binding: operands inconsistent with their scale are refused up front" {
    const a = testing.allocator;
    var case = try realCase(a);
    defer case.deinit(a);

    var gemm = try chunk.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm.deinit(a);

    const wrong = try a.alloc(Goldilocks, k_macs);
    defer a.free(wrong);
    for (0..k_macs) |i| wrong[i] = case.scale_a[i].add(Goldilocks.one);

    try testing.expectError(
        cb.BindError.InconsistentOperands,
        cb.bindOperands(a, &gemm, case.nib_a, wrong, case.nib_b, case.scale_b),
    );
}

test "chunk_binding: a raw nibble of 16 is refused" {
    const a = testing.allocator;
    var case = try realCase(a);
    defer case.deinit(a);

    var gemm = try chunk.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm.deinit(a);

    const bad = try a.alloc(u8, k_macs);
    defer a.free(bad);
    @memcpy(bad, case.nib_a);
    bad[100] = 16;

    try testing.expectError(
        cb.BindError.NibbleOutOfRange,
        cb.bindOperands(a, &gemm, bad, case.scale_a, case.nib_b, case.scale_b),
    );
}
