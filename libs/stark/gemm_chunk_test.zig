//! End-to-end tests for the chunked GEMM AIR (F2): 16 MACs per trace row.
//!
//! The point of the chunked layout is a 16× smaller trace for the same
//! reduction, so the first test checks both that a real Q4_K chunked
//! reduction verifies and that the row count actually drops. The sharp
//! negative is the closing row: an attacker can park a product in an
//! unused slot and move the attested output *while every composed
//! constraint still holds* — only the 30 boundary pins catch that.

const std = @import("std");
const tensor = @import("../tensor/root.zig");
const stark = @import("root.zig");
const gemm_air = @import("gemm_air.zig");
const chunk = @import("gemm_chunk.zig");
const Constraint = chunk.Constraint;

const testing = std.testing;
const Goldilocks = tensor.Goldilocks;
const Fp2 = stark.Fp2;

/// One Q4_K block per operand: 256 dequantized elements, so k = 256 fills
/// exactly 16 chunks of 16 MACs.
const k_macs: usize = 256;

const CONFIG: stark.Config = blk: {
    // 16 chunks + closing row -> 32 trace rows (log_trace 5), blowup 2.
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

const Case = struct {
    a: []Goldilocks,
    b: []Goldilocks,
    c_true: Goldilocks,

    fn deinit(self: *Case, allocator: std.mem.Allocator) void {
        allocator.free(self.a);
        allocator.free(self.b);
        self.* = undefined;
    }
};

/// A real Q4_K reduction, operands straight out of tensor.dequantQ4K.
fn realCase(allocator: std.mem.Allocator) !Case {
    const block = blockWithRamp();
    const deq_a = tensor.dequantQ4K(&block, 0x3C00) catch unreachable; // fp16 1.0
    const deq_b = tensor.dequantQ4K(&block, 0x3800) catch unreachable; // fp16 0.5

    const a = try allocator.alloc(Goldilocks, k_macs);
    errdefer allocator.free(a);
    const b = try allocator.alloc(Goldilocks, k_macs);
    errdefer allocator.free(b);

    var acc = Goldilocks.zero;
    for (0..k_macs) |i| {
        a[i] = deq_a[i];
        b[i] = deq_b[i];
        acc = acc.add(a[i].mul(b[i]));
    }
    return .{ .a = a, .b = b, .c_true = acc };
}

test "chunk: a real Q4_K reduction proves and verifies in 16x fewer rows" {
    const a = testing.allocator;
    const system = chunk.system();

    // The whole point: same reduction, 16x shorter trace.
    const chunked_rows = chunk.rowsFor(k_macs);
    const per_mac_rows = gemm_air.rowsFor(k_macs);
    try testing.expectEqual(@as(usize, 32), chunked_rows);
    try testing.expectEqual(@as(usize, 512), per_mac_rows);
    try testing.expectEqual(per_mac_rows / chunked_rows, @as(usize, 16));

    var case = try realCase(a);
    defer case.deinit(a);
    try testing.expect(!case.c_true.isZero());

    var trace = try chunk.buildTrace(a, case.a, case.b, case.c_true);
    defer trace.deinit(a);
    try testing.expect(case.c_true.eql(chunk.traceOutput(&trace)));

    var pt = stark.Transcript.init("zkml.chunk.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG);
    defer proof.deinit(a);

    var vt = stark.Transcript.init("zkml.chunk.v1");
    try testing.expect(try stark.verify(&vt, &proof, system, CONFIG));
}

test "chunk: claiming a wrong output is rejected" {
    const a = testing.allocator;
    const system = chunk.system();
    var case = try realCase(a);
    defer case.deinit(a);

    var trace = try chunk.buildTrace(a, case.a, case.b, case.c_true.add(Goldilocks.one));
    defer trace.deinit(a);

    var pt = stark.Transcript.init("zkml.chunk.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{ .rows = trace.rows, .columns = trace.columns }, system, CONFIG),
    );
}

test "chunk: a product parked in an unused closing slot is rejected" {
    const a = testing.allocator;
    const system = chunk.system();
    var case = try realCase(a);
    defer case.deinit(a);

    var trace = try chunk.buildTrace(a, case.a, case.b, case.c_true);
    defer trace.deinit(a);

    // The attack: put an extra product in slot 1 of the closing row and
    // raise the claimed output to match. The recursion forces
    // s[last] = P (the row before it is all padding), and the composed
    // constraint at the last row then only asks
    //   0 = s[last] + Σ_last = P + (b₀ + a₁b₁)
    // so setting c = P + 1 with a₁ = b₁ = 1, b₀ = -c satisfies EVERY
    // composed constraint and both old boundary pins. The only thing left
    // to catch it is `a₁[last] = 0`.
    const last = trace.rows - 1;
    const forged = case.c_true.add(Goldilocks.one);
    trace.columns[chunk.colA(1)][last] = Fp2.one;
    trace.columns[chunk.colB(1)][last] = Fp2.one;
    trace.columns[chunk.col_c][last] = Fp2.re(forged);
    trace.columns[chunk.colB(0)][last] = Fp2.re(Goldilocks.zero.sub(forged));

    // The prover accepts: from the composed constraints alone this trace
    // is indistinguishable from an honest one.
    var pt = stark.Transcript.init("zkml.chunk.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG);
    defer proof.deinit(a);

    // The boundary pins reject it.
    var vt = stark.Transcript.init("zkml.chunk.v1");
    try testing.expect(!try stark.verify(&vt, &proof, system, CONFIG));
}

test "chunk: editing the authenticated output column is rejected" {
    const a = testing.allocator;
    const system = chunk.system();
    var case = try realCase(a);
    defer case.deinit(a);

    var trace = try chunk.buildTrace(a, case.a, case.b, case.c_true);
    defer trace.deinit(a);

    var pt = stark.Transcript.init("zkml.chunk.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG);
    defer proof.deinit(a);

    // One q4.22 ulp on the output, in the authenticated last-row opening.
    proof.boundary_openings[1].current[chunk.col_c] =
        proof.boundary_openings[1].current[chunk.col_c].add(Fp2.one);

    var vt = stark.Transcript.init("zkml.chunk.v1");
    try testing.expect(!try stark.verify(&vt, &proof, system, CONFIG));
}

test "chunk: the unused-slot pins are load-bearing, not decorative" {
    const a = testing.allocator;
    const system = chunk.system();
    var case = try realCase(a);
    defer case.deinit(a);

    // Same attack as above, but judged against a WEAKENED system with the
    // 30 `aᵢ/bᵢ[last] = 0` pins removed. If the forged proof verifies
    // here, those pins are the only thing rejecting it — and a future
    // refactor that drops them reopens a real hole.
    var trace = try chunk.buildTrace(a, case.a, case.b, case.c_true);
    defer trace.deinit(a);
    const last = trace.rows - 1;
    const forged = case.c_true.add(Goldilocks.one);
    trace.columns[chunk.colA(1)][last] = Fp2.one;
    trace.columns[chunk.colB(1)][last] = Fp2.one;
    trace.columns[chunk.col_c][last] = Fp2.re(forged);
    trace.columns[chunk.colB(0)][last] = Fp2.re(Goldilocks.zero.sub(forged));

    var pt = stark.Transcript.init("zkml.chunk.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG);
    defer proof.deinit(a);

    var weak = try a.alloc(Constraint, 0);
    for (system.constraints) |c| {
        if (std.mem.endsWith(u8, c.name, "[last] = 0")) continue;
        weak = try a.realloc(weak, weak.len + 1);
        weak[weak.len - 1] = c;
    }
    defer a.free(weak);
    try testing.expectEqual(system.constraints.len - 30, weak.len);

    var vt = stark.Transcript.init("zkml.chunk.v1");
    try testing.expect(try stark.verify(&vt, &proof, .{ .constraints = weak }, CONFIG));
}

test "chunk: a trace with one fewer MAC still proves (ragged chunk)" {
    const a = testing.allocator;
    const system = chunk.system();
    var case = try realCase(a);
    defer case.deinit(a);

    // k = 250: the last chunk is half empty, so those slots must be zero
    // rather than uninitialised. Slot 10 is real work in chunks 0..14 and
    // absent from chunk 15 onwards (padding and the closing row).
    const k_ragged: usize = 250;
    var c_ragged = Goldilocks.zero;
    for (0..k_ragged) |i| c_ragged = c_ragged.add(case.a[i].mul(case.b[i]));

    var trace = try chunk.buildTrace(a, case.a[0..k_ragged], case.b[0..k_ragged], c_ragged);
    defer trace.deinit(a);
    try testing.expectEqual(@as(usize, 32), trace.rows);
    for (chunk.chunkRowsFor(k_ragged)..trace.rows) |r| {
        try testing.expect(trace.columns[chunk.colA(10)][r].a.isZero());
        try testing.expect(trace.columns[chunk.colB(10)][r].a.isZero());
    }

    var pt = stark.Transcript.init("zkml.chunk.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG);
    defer proof.deinit(a);

    var vt = stark.Transcript.init("zkml.chunk.v1");
    try testing.expect(try stark.verify(&vt, &proof, system, CONFIG));
}
