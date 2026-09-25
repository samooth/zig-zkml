//! End-to-end F2 milestone test: a real quantized GEMM output element,
//! proved and verified by the STARK backend (libs/stark).
//!
//! The operands come from the same Q4_K dequantization the engine uses
//! (tensor.dequantQ4K), so the witness is a genuine q4.22-scaled MAC
//! reduction rather than synthetic numbers.

const std = @import("std");
const tensor = @import("../tensor/root.zig");
const stark = @import("root.zig");
const gemm_air = @import("gemm_air.zig");

const testing = std.testing;
const Goldilocks = tensor.Goldilocks;
const Fp2 = stark.Fp2;

fn u(v: u64) Goldilocks {
    return Goldilocks.fromU64(v);
}

/// A Q4_K block: 128 nibble bytes + an fp16 scale, dequantized to 256
/// Goldilocks exactly as the engine's loader would.
fn makeBlock(nibbles: [128]u8, scale_bits: u16) [256]Goldilocks {
    return tensor.dequantQ4K(&nibbles, scale_bits) catch unreachable;
}

/// 256 nibbles encoding 0..15 as raw values (q4Nibble maps them to -8..7).
fn rampNibbles() [128]u8 {
    var out: [128]u8 = undefined;
    for (0..128) |byte_i| {
        out[byte_i] = @intCast(((byte_i % 8) * 2) << 4 | (((byte_i / 8) % 8) * 2));
    }
    return out;
}

const CONFIG_A: stark.Config = blk: {
    // k = 31 MACs -> 31 data rows + 1 closing row = 32 trace rows
    // (log_trace 5), blowup 2, FRI on a 32-point quotient at rate 1/2.
    const log_trace: u6 = 5;
    const log_blowup: u6 = 2;
    const log_lde = log_trace + log_blowup;
    break :blk .{
        .log_trace = log_trace,
        .log_blowup = log_blowup,
        .fri = .{
            .log_domain = log_lde,
            .log_final = log_trace + 1,
            .log_residual_degree = log_trace,
            .num_queries = 4,
        },
    };
};

/// MAC operands for k=31: two dequantized Q4_K blocks worth of elements.
fn operands(allocator: std.mem.Allocator) !struct { a: []Goldilocks, b: []Goldilocks } {
    const k: usize = 31;
    const a = try allocator.alloc(Goldilocks, k);
    errdefer allocator.free(a);
    const b = try allocator.alloc(Goldilocks, k);
    errdefer allocator.free(b);

    const block_a = makeBlock(rampNibbles(), 0x3C00); // fp16 1.0
    const block_b = makeBlock(rampNibbles(), 0x3800); // fp16 0.5
    for (0..k) |i| {
        a[i] = block_a[i];
        b[i] = block_b[i];
    }
    return .{ .a = a, .b = b };
}

fn trueOutput(a: []const Goldilocks, b: []const Goldilocks) Goldilocks {
    var acc = Goldilocks.zero;
    for (a, b) |x, y| acc = acc.add(x.mul(y));
    return acc;
}

fn g(v: u64) Fp2 {
    return Fp2.re(Goldilocks.fromU64(v));
}

test "gemm: real Q4_K reduction proves and verifies" {
    const a = testing.allocator;
    const ops = try operands(a);
    defer a.free(ops.a);
    defer a.free(ops.b);
    const system = try gemm_air.system(ops.a.len);

    const c_true = trueOutput(ops.a, ops.b);
    try testing.expect(!c_true.isZero());

    var trace = try gemm_air.buildTrace(a, ops.a, ops.b, c_true);
    defer trace.deinit(a);
    try testing.expectEqual(@as(usize, try gemm_air.rowsFor(ops.a.len)), trace.rows);
    try testing.expect(c_true.eql(gemm_air.traceOutput(&trace)));

    var pt = stark.Transcript.init("zkml.gemm.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG_A);
    defer proof.deinit(a);

    var vt = stark.Transcript.init("zkml.gemm.v1");
    try testing.expect(try stark.verify(&vt, &proof, system, CONFIG_A));
}

test "gemm: exact trace length is part of the statement" {
    const a = testing.allocator;
    const ops = try operands(a);
    defer a.free(ops.a);
    defer a.free(ops.b);
    const system = try gemm_air.system(ops.a.len);
    const c_true = trueOutput(ops.a, ops.b);
    var trace = try gemm_air.buildTrace(a, ops.a, ops.b, c_true);
    defer trace.deinit(a);

    var pt = stark.Transcript.init("zkml.gemm.shape");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG_A);
    defer proof.deinit(a);

    var wrong = CONFIG_A;
    wrong.log_trace += 1;
    wrong.fri.log_domain = wrong.log_trace + wrong.log_blowup;
    wrong.fri.log_final = wrong.log_trace + 1;
    wrong.fri.log_residual_degree = wrong.log_trace;

    var vt = stark.Transcript.init("zkml.gemm.shape");
    try testing.expectError(stark.Error.InvalidProof, stark.verify(&vt, &proof, system, wrong));

    var pt2 = stark.Transcript.init("zkml.gemm.shape");
    try testing.expectError(
        stark.Error.InvalidConfig,
        stark.prove(a, &pt2, .{ .rows = trace.rows, .columns = trace.columns }, system, wrong),
    );
}

test "gemm: unsupported reduction lengths are refused" {
    try testing.expectError(gemm_air.BuildError.EmptyReduction, gemm_air.system(0));
    try testing.expectEqual(@as(usize, 2), try gemm_air.rowsFor(1));
    for ([_]usize{ 16, 17, 250 }) |k| {
        try testing.expectError(gemm_air.BuildError.UnsupportedTraceSize, gemm_air.system(k));
    }
}

test "gemm: claiming a wrong output is rejected" {
    const a = testing.allocator;
    const ops = try operands(a);
    defer a.free(ops.a);
    defer a.free(ops.b);
    const system = try gemm_air.system(ops.a.len);

    const c_true = trueOutput(ops.a, ops.b);
    const c_lie = c_true.add(Goldilocks.one);

    // A prover that claims a different output has to break the cycle: the
    // closing row is built from the TRUE sum, so the composed constraint no
    // longer vanishes on the domain and the prover refuses to prove.
    var trace = try gemm_air.buildTrace(a, ops.a, ops.b, c_lie);
    defer trace.deinit(a);

    var pt = stark.Transcript.init("zkml.gemm.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{ .rows = trace.rows, .columns = trace.columns }, system, CONFIG_A),
    );
}

test "gemm: a verifier rejects a proof whose output column was edited" {
    const a = testing.allocator;
    const ops = try operands(a);
    defer a.free(ops.a);
    defer a.free(ops.b);
    const system = try gemm_air.system(ops.a.len);

    const c_true = trueOutput(ops.a, ops.b);
    var trace = try gemm_air.buildTrace(a, ops.a, ops.b, c_true);
    defer trace.deinit(a);

    var pt = stark.Transcript.init("zkml.gemm.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG_A);
    defer proof.deinit(a);

    // Edit the authenticated boundary opening: the leaf hash no longer
    // matches the commitment, so the closing constraint is not reached.
    proof.boundary_openings[1].current[gemm_air.col_c] =
        proof.boundary_openings[1].current[gemm_air.col_c].add(Fp2.one);
    var vt = stark.Transcript.init("zkml.gemm.v1");
    try testing.expect(!(try stark.verify(&vt, &proof, system, CONFIG_A)));
}

test "gemm: ±1 ulp in an operand changes the attestsed output" {
    const a = testing.allocator;
    const ops = try operands(a);
    defer a.free(ops.a);
    defer a.free(ops.b);
    const system = try gemm_air.system(ops.a.len);

    const c_true = trueOutput(ops.a, ops.b);
    const before = c_true;

    // Perturb one operand by one q4.22 ulp. The AIR ties the output to the
    // products of the trace's own a/b, so the attestsed value must move —
    // a proof of the old output can no longer be produced.
    const bumped = ops.a[5].add(Goldilocks.one);
    const b2 = try a.dupe(Goldilocks, ops.b);
    defer a.free(b2);
    const a2 = try a.dupe(Goldilocks, ops.a);
    defer a.free(a2);
    a2[5] = bumped;

    const after = trueOutput(a2, b2);
    try testing.expect(!before.eql(after));

    // The old output is no longer provable against the perturbed operands.
    var trace = try gemm_air.buildTrace(a, a2, b2, c_true);
    defer trace.deinit(a);
    var pt = stark.Transcript.init("zkml.gemm.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{ .rows = trace.rows, .columns = trace.columns }, system, CONFIG_A),
    );
}

test "gemm: a boundary opening at the WRONG row is rejected" {
    const a = testing.allocator;
    const n = 31;
    const av = try a.alloc(Goldilocks, n);
    defer a.free(av);
    const bv = try a.alloc(Goldilocks, n);
    defer a.free(bv);
    for (0..n) |i| {
        av[i] = Goldilocks.fromU64(@intCast(i + 1));
        bv[i] = Goldilocks.fromU64(@intCast(2 * i + 1));
    }
    // The claimed output has to be the real sum, or the closing constraint
    // fails and there is no honest proof to tamper with.
    var total = Goldilocks.zero;
    for (0..n) |i| total = total.add(av[i].mul(bv[i]));
    var trace = try gemm_air.buildTrace(a, av, bv, total);
    defer trace.deinit(a);
    const system = try gemm_air.system(n);

    var pt = stark.Transcript.init("zkml.gemm.boundary.index");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG_A);
    defer proof.deinit(a);
    var vt = stark.Transcript.init("zkml.gemm.boundary.index");
    try testing.expect(try stark.verify(&vt, &proof, system, CONFIG_A));

    // The attack the pin closes: claim the boundary window lives at some
    // OTHER row. The Merkle proof still authenticates a real leaf — it is
    // just the wrong leaf — and the boundary constraints are evaluated
    // there instead of at the ends, so before the index was checked this
    // was accepted. The index is now part of the claim.
    // A wrong index is a MALFORMED proof rather than a failed check, so
    // verify reports it as InvalidProof — the same treatment the boundary
    // opening count already had.
    const stride: usize = 1 << CONFIG_A.log_blowup;
    proof.boundary_openings[0].index = 2 * stride;
    var vt2 = stark.Transcript.init("zkml.gemm.boundary.index");
    try testing.expectError(
        stark.Error.InvalidProof,
        stark.verify(&vt2, &proof, system, CONFIG_A),
    );

    // And the last one has to be exactly (n-1)*stride, not one row before.
    proof.boundary_openings[0].index = 0;
    proof.boundary_openings[1].index -= stride;
    var vt3 = stark.Transcript.init("zkml.gemm.boundary.index");
    try testing.expectError(
        stark.Error.InvalidProof,
        stark.verify(&vt3, &proof, system, CONFIG_A),
    );
}

test "gemm: the boundary constraints are what make the AIR non-vacuous" {
    const a = testing.allocator;

    // All-zero operands: the COMPOSED constraint s' = s - a*b is satisfied
    // by the all-zero trace, so it alone attests nothing.
    const zero = try a.alloc(Goldilocks, 31);
    defer a.free(zero);
    @memset(zero, Goldilocks.zero);

    var trace = try gemm_air.buildTrace(a, zero, zero, Goldilocks.zero);
    const system = try gemm_air.system(zero.len);
    defer trace.deinit(a);

    // buildTrace's closing row satisfies the boundaries, so this verifies.
    var pt = stark.Transcript.init("zkml.gemm.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG_A);
    defer proof.deinit(a);
    var vt = stark.Transcript.init("zkml.gemm.v1");
    try testing.expect(try stark.verify(&vt, &proof, system, CONFIG_A));

    // The closing row is EXEMPT, so its a and b are now free witness: any
    // value verifies, which is the point — a synthetic row cannot be a
    // dequantized operand, so nothing should try to constrain it.
    const last = trace.rows - 1;
    trace.columns[gemm_air.col_a][last] = g(1234567);
    trace.columns[gemm_air.col_b][last] = g(7654321);
    var pt2 = stark.Transcript.init("zkml.gemm.v1");
    var loose = try stark.prove(a, &pt2, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG_A);
    defer loose.deinit(a);
    var vt2 = stark.Transcript.init("zkml.gemm.v1");
    try testing.expect(try stark.verify(&vt2, &loose, system, CONFIG_A));

    // What is NOT free is the claim. Moving s[last] away from c[last] is
    // the attack the boundary exists for: the composed constraint does not
    // see it (the row is exempt), the boundary does.
    trace.columns[gemm_air.col_s][last] = g(1);
    var pt3 = stark.Transcript.init("zkml.gemm.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt3, .{
            .rows = trace.rows,
            .columns = trace.columns,
        }, system, CONFIG_A),
    );

    // And a REAL row is not exempt: moving the running sum there breaks the
    // telescoping, and the composed constraint catches it (the sum is read
    // with offset +1, so row `last-1` reads s[last]).
    trace.columns[gemm_air.col_s][last] = Fp2.zero;
    trace.columns[gemm_air.col_s][last - 1] = g(1);
    var pt4 = stark.Transcript.init("zkml.gemm.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt4, .{
            .rows = trace.rows,
            .columns = trace.columns,
        }, system, CONFIG_A),
    );
}
