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

/// One Q4_K block is reused across the reduction; k = 496 fills 31 full
/// chunks of 16 MACs.
const k_macs: usize = 496;

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
        a[i] = deq_a[i % 256];
        b[i] = deq_b[i % 256];
        acc = acc.add(a[i].mul(b[i]));
    }
    return .{ .a = a, .b = b, .c_true = acc };
}

test "chunk: a real Q4_K reduction proves and verifies in 16x fewer rows" {
    const a = testing.allocator;
    const system = try chunk.system(k_macs);
    try testing.expectEqual(@as(?usize, 32), system.trace_rows);

    // The whole point: 31 full chunks plus the closing row in 32 rows;
    // the comparable 1-MAC shape is k=511, whose 512 rows are 16x more.
    const chunked_rows = try chunk.rowsFor(k_macs);
    const per_mac_rows = try gemm_air.rowsFor(511);
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
    const system = try chunk.system(k_macs);
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

test "chunk: a product parked in an unused closing slot cannot move the claim" {
    const a = testing.allocator;
    const system = try chunk.system(k_macs);
    var case = try realCase(a);
    defer case.deinit(a);

    var trace = try chunk.buildTrace(a, case.a, case.b, case.c_true);
    defer trace.deinit(a);

    // The attack this layout once needed 30 boundary pins for: park a
    // product in slot 1 of the closing row and raise the claim. It used to
    // satisfy every composed constraint and both old pins, leaving only
    // `aᵢ[last] = 0` to catch it.
    //
    // The closing row is now EXEMPT, so those slots are not in any equation
    // at all — but neither is s[last], and `s[last] = c[last]` is a
    // boundary, so moving the claim means moving the sum with it, which the
    // telescoped data rows forbid.
    const last = trace.rows - 1;
    const forged = case.c_true.add(Goldilocks.one);
    trace.columns[chunk.colA(1)][last] = Fp2.one;
    trace.columns[chunk.colB(1)][last] = Fp2.one;
    trace.columns[chunk.col_c][last] = Fp2.re(forged);
    // s[last] is left at the true sum: the boundary now rejects.
    var pt = stark.Transcript.init("zkml.chunk.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{
            .rows = trace.rows,
            .columns = trace.columns,
        }, system, CONFIG),
    );

    // Move the sum too, to keep the boundary happy, and the data rows no
    // longer telescope: rejected on a composed row this time.
    trace.columns[chunk.col_s][last] = Fp2.re(forged);
    var pt2 = stark.Transcript.init("zkml.chunk.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt2, .{
            .rows = trace.rows,
            .columns = trace.columns,
        }, system, CONFIG),
    );
}

test "chunk: editing the authenticated output column is rejected" {
    const a = testing.allocator;
    const system = try chunk.system(k_macs);
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

test "chunk: the exemption, not a wall of pins, is what closes the row" {
    const a = testing.allocator;
    const system = try chunk.system(k_macs);
    var case = try realCase(a);
    defer case.deinit(a);

    var trace = try chunk.buildTrace(a, case.a, case.b, case.c_true);
    defer trace.deinit(a);
    const last = trace.rows - 1;
    // The closing row's slots are free witness now. Park a product in one
    // and leave the honest sum: the claim does not move.
    trace.columns[chunk.colA(1)][last] = Fp2.one;
    trace.columns[chunk.colB(1)][last] = Fp2.one;

    var pt = stark.Transcript.init("zkml.chunk.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG);
    defer proof.deinit(a);
    var vt = stark.Transcript.init("zkml.chunk.v1");
    try testing.expect(try stark.verify(&vt, &proof, system, CONFIG));

    // Strip the exemption and the same trace is a violation: the closing
    // row participates in the sum again and `Σ_last` must cancel. This is
    // the property the field is load-bearing for, and the reason it is
    // declared per-system instead of being an ambient default.
    var without = system;
    without.transition_exemptions = 0;
    var pt2 = stark.Transcript.init("zkml.chunk.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt2, .{
            .rows = trace.rows,
            .columns = trace.columns,
        }, without, CONFIG),
    );
}

test "chunk: a ragged reduction is refused instead of creating hidden slots" {
    const a = testing.allocator;
    var case = try realCase(a);
    defer case.deinit(a);

    const k_ragged: usize = 250;
    var c_ragged = Goldilocks.zero;
    for (0..k_ragged) |i| c_ragged = c_ragged.add(case.a[i].mul(case.b[i]));

    try testing.expectError(
        chunk.BuildError.RaggedReduction,
        chunk.buildTrace(a, case.a[0..k_ragged], case.b[0..k_ragged], c_ragged),
    );
}

test "chunk: unsupported reduction lengths are refused" {
    try testing.expectError(chunk.BuildError.EmptyReduction, chunk.system(0));
    try testing.expectError(chunk.BuildError.RaggedReduction, chunk.system(250));
    try testing.expectError(chunk.BuildError.UnsupportedTraceSize, chunk.system(256));
    try testing.expectError(chunk.BuildError.RaggedReduction, chunk.system(240 + 1));
}
