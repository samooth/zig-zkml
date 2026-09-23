//! End-to-end tests for the quantized-operand binding (F2 LogUp-lite):
//! the GEMM operands are no longer free witness columns but are pinned to
//! a 4-bit raw nibble times one block scale, with the nibble's bits proven
//! boolean.
//!
//! The last test is a KNOWN GAP marker: it proves something that should be
//! unsound (a fabricated scale). Delete or invert it when the scale lookup
//! lands — it exists so the gap cannot be forgotten, not because it is fine.

const std = @import("std");
const tensor = @import("../tensor/root.zig");
const stark = @import("root.zig");
const gemm_air = @import("./gemm_air.zig");
const quant = @import("./quant_binding.zig");

const testing = std.testing;
const Goldilocks = tensor.Goldilocks;
const Fp2 = stark.Fp2;

const k_macs: usize = 16;

/// Same shape as gemm_test: 16 rows + closing row -> 32 trace rows, blowup 2.
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

/// Raw nibble of element i in `blockWithRamp`'s layout.
fn rawNibble(i: usize) u8 {
    return @intCast((i / 2) % 16);
}

const Case = struct {
    a: []Goldilocks,
    b: []Goldilocks,
    nib_a: [k_macs]u8,
    nib_b: [k_macs]u8,
    scale_a: [k_macs]Goldilocks,
    scale_b: [k_macs]Goldilocks,
    c_true: Goldilocks,

    fn deinit(self: *Case, allocator: std.mem.Allocator) void {
        allocator.free(self.a);
        allocator.free(self.b);
        self.* = undefined;
    }
};

/// A real Q4_K reduction: operands straight out of tensor.dequantQ4K.
fn realCase(allocator: std.mem.Allocator) !Case {
    const block = blockWithRamp();
    const deq_a = tensor.dequantQ4K(&block, 0x3C00) catch unreachable; // fp16 1.0
    const deq_b = tensor.dequantQ4K(&block, 0x3800) catch unreachable; // fp16 0.5

    const a = try allocator.alloc(Goldilocks, k_macs);
    errdefer allocator.free(a);
    const b = try allocator.alloc(Goldilocks, k_macs);
    errdefer allocator.free(b);

    var case = Case{
        .a = a,
        .b = b,
        .nib_a = undefined,
        .nib_b = undefined,
        .scale_a = undefined,
        .scale_b = undefined,
        .c_true = Goldilocks.zero,
    };
    for (0..k_macs) |i| {
        a[i] = deq_a[i];
        b[i] = deq_b[i];
        case.nib_a[i] = rawNibble(i);
        case.nib_b[i] = rawNibble(i);
        case.scale_a[i] = Goldilocks.fromU64(tensor.fp16ToFixedQ4_22(0x3C00) catch unreachable);
        case.scale_b[i] = Goldilocks.fromU64(tensor.fp16ToFixedQ4_22(0x3800) catch unreachable);
        case.c_true = case.c_true.add(a[i].mul(b[i]));
    }
    return case;
}

fn boundTrace(allocator: std.mem.Allocator, case: *const Case) !struct { gemm: gemm_air.Trace, bound: quant.Trace } {
    var gemm = try gemm_air.buildTrace(allocator, case.a, case.b, case.c_true);
    errdefer gemm.deinit(allocator);
    const bound = try quant.bindOperands(
        allocator,
        &gemm,
        &case.nib_a,
        &case.scale_a,
        &case.nib_b,
        &case.scale_b,
    );
    return .{ .gemm = gemm, .bound = bound };
}

test "quant: real Q4_K operands prove and verify with the binding" {
    const a = testing.allocator;
    var sys = try quant.buildSystem(a);
    defer sys.deinit();

    // The binding must actually be in the system: two range checks (1 + 4
    // constraints each) and both dequantization equations.
    try testing.expectEqual(@as(usize, 4 + 10 + 2), sys.system().constraints.len);
    try testing.expectEqual(@as(?u16, quant.col_scale_b), sys.system().maxColumn());
    // The nibble range check is quadratic; the running sum already was, so
    // the blowup requirement is unchanged.
    try testing.expectEqual(@as(usize, 2), sys.system().maxDegree());

    var case = try realCase(a);
    defer case.deinit(a);
    try testing.expect(!case.c_true.isZero());

    var bt = try boundTrace(a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);

    var pt = stark.Transcript.init("zkml.quant.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = bt.bound.rows,
        .columns = bt.bound.columns,
    }, sys.system(), CONFIG);
    defer proof.deinit(a);

    var vt = stark.Transcript.init("zkml.quant.v1");
    try testing.expect(try stark.verify(&vt, &proof, sys.system(), CONFIG));
}

test "quant: a bit column that is not boolean is rejected" {
    const a = testing.allocator;
    var sys = try quant.buildSystem(a);
    defer sys.deinit();

    var case = try realCase(a);
    defer case.deinit(a);
    var bt = try boundTrace(a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);

    // Re-decompose nibble 3 (bits 1+2) as -1 + 2*2: the sum still equals 3,
    // so the reconstruction constraint is satisfied and ONLY the
    // booleanity constraints can catch it.
    const row: usize = 6; // raw nibble 3
    try testing.expectEqual(@as(u8, 3), bt.bound.columns[quant.col_nib_a][row].a.toU64());
    bt.bound.columns[quant.col_bits_a + 0][row] = Fp2.re(Goldilocks.zero.sub(Goldilocks.one));
    bt.bound.columns[quant.col_bits_a + 1][row] = Fp2.re(Goldilocks.fromU64(2));

    var pt = stark.Transcript.init("zkml.quant.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{
            .rows = bt.bound.rows,
            .columns = bt.bound.columns,
        }, sys.system(), CONFIG),
    );
}

test "quant: an operand that does not match its scale is rejected" {
    const a = testing.allocator;
    var sys = try quant.buildSystem(a);
    defer sys.deinit();

    var case = try realCase(a);
    defer case.deinit(a);
    var bt = try boundTrace(a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);

    // Nibble and scale untouched, operand moved: the dequantization
    // equation is what rejects this.
    bt.bound.columns[gemm_air.col_a][3] = Fp2.re(Goldilocks.fromU64(12345));

    var pt = stark.Transcript.init("zkml.quant.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{
            .rows = bt.bound.rows,
            .columns = bt.bound.columns,
        }, sys.system(), CONFIG),
    );
}

test "quant: a raw nibble outside [0, 15] is refused before proving" {
    const a = testing.allocator;
    var case = try realCase(a);
    defer case.deinit(a);

    var gemm = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm.deinit(a);

    var bad = case.nib_a;
    bad[2] = 16;
    try testing.expectError(
        quant.BindError.NibbleOutOfRange,
        quant.bindOperands(a, &gemm, &bad, &case.scale_a, &case.nib_b, &case.scale_b),
    );
}

test "quant: operands that are not the dequantization of their nibble are refused" {
    const a = testing.allocator;
    var case = try realCase(a);
    defer case.deinit(a);

    // Correct nibbles, but a block scale that does not produce these
    // operands: buildOperands catches it rather than the AIR.
    var gemm = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm.deinit(a);

    var wrong: [k_macs]Goldilocks = undefined;
    for (0..k_macs) |i| wrong[i] = case.scale_a[i].add(Goldilocks.one);
    try testing.expectError(
        quant.BindError.InconsistentOperands,
        quant.bindOperands(a, &gemm, &case.nib_a, &wrong, &case.nib_b, &case.scale_b),
    );
}

test "quant: KNOWN GAP — a fabricated scale still proves (no scale lookup yet)" {
    const a = testing.allocator;
    var sys = try quant.buildSystem(a);
    defer sys.deinit();

    // A scale no fp16 could have produced. Every constraint still holds:
    // nibble in range, bits boolean, operands equal (nibble-8)*scale. The
    // proof therefore attests an output that no real Q4_K tensor yields.
    // This test exists to mark the hole — when the scale lookup (LogUp)
    // lands, this must stop verifying.
    const fake = Goldilocks.fromU64(1234567);
    const operands_a = try a.alloc(Goldilocks, k_macs);
    defer a.free(operands_a);
    const operands_b = try a.alloc(Goldilocks, k_macs);
    defer a.free(operands_b);

    var case = Case{
        .a = operands_a,
        .b = operands_b,
        .nib_a = undefined,
        .nib_b = undefined,
        .scale_a = undefined,
        .scale_b = undefined,
        .c_true = Goldilocks.zero,
    };
    const eight = Goldilocks.fromU64(8);
    for (0..k_macs) |i| {
        const na = Goldilocks.fromU64(rawNibble(i));
        const nb = na;
        case.nib_a[i] = rawNibble(i);
        case.nib_b[i] = rawNibble(i);
        case.scale_a[i] = fake;
        case.scale_b[i] = fake;
        case.a[i] = na.sub(eight).mul(fake);
        case.b[i] = nb.sub(eight).mul(fake);
        case.c_true = case.c_true.add(case.a[i].mul(case.b[i]));
    }

    var bt = try boundTrace(a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);

    var pt = stark.Transcript.init("zkml.quant.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = bt.bound.rows,
        .columns = bt.bound.columns,
    }, sys.system(), CONFIG);
    defer proof.deinit(a);

    var vt = stark.Transcript.init("zkml.quant.v1");
    try testing.expect(try stark.verify(&vt, &proof, sys.system(), CONFIG));
}

test "quant: a 4096-MAC reduction (8192 rows) proves and verifies" {
    // Regression: fri.verify used to carry a fixed [4096]Fp2 stack buffer
    // for the residual, so any proof whose final FRI domain exceeded 2^12
    // was rejected as InvalidProof — an HONEST proof, silently. k=4096
    // puts the final domain at 2^14. The residual is now evaluated with
    // the FFT, so this both passes and stops dominating verify time.
    const a = testing.allocator;
    const k: usize = 4096;
    var sys = try quant.buildSystem(a);
    defer sys.deinit();

    const av = try a.alloc(Goldilocks, k);
    defer a.free(av);
    const bv = try a.alloc(Goldilocks, k);
    defer a.free(bv);
    const nibs_a = try a.alloc(u8, k);
    defer a.free(nibs_a);
    const nibs_b = try a.alloc(u8, k);
    defer a.free(nibs_b);
    const sc_a = try a.alloc(Goldilocks, k);
    defer a.free(sc_a);
    const sc_b = try a.alloc(Goldilocks, k);
    defer a.free(sc_b);

    // Four Q4_K blocks, all sharing the fp16 scale 1.0, so the reduction
    // spans several blocks and the per-MAC scale array is exercised.
    const block = blockWithRamp();
    const deq = tensor.dequantQ4K(&block, 0x3C00) catch unreachable;
    const scale = Goldilocks.fromU64(tensor.fp16ToFixedQ4_22(0x3C00) catch unreachable);
    var c = Goldilocks.zero;
    for (0..k) |i| {
        av[i] = deq[i % 256];
        bv[i] = deq[i % 256];
        nibs_a[i] = rawNibble(i % 256);
        nibs_b[i] = rawNibble(i % 256);
        sc_a[i] = scale;
        sc_b[i] = scale;
        c = c.add(av[i].mul(bv[i]));
    }

    var gemm = try gemm_air.buildTrace(a, av, bv, c);
    defer gemm.deinit(a);
    var bound = try quant.bindOperands(a, &gemm, nibs_a, sc_a, nibs_b, sc_b);
    defer bound.deinit(a);
    try testing.expectEqual(@as(usize, 8192), bound.rows);

    const log_trace: u6 = 13;
    const cfg: stark.Config = .{
        .log_trace = log_trace,
        .log_blowup = 2,
        .fri = .{
            .log_domain = log_trace + 2,
            .log_final = log_trace + 1,
            .log_residual_degree = log_trace,
            .num_queries = 4,
        },
    };

    var pt = stark.Transcript.init("zkml.quant.big.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = bound.rows,
        .columns = bound.columns,
    }, sys.system(), cfg);
    defer proof.deinit(a);

    var vt = stark.Transcript.init("zkml.quant.big.v1");
    try testing.expect(try stark.verify(&vt, &proof, sys.system(), cfg));
}

test "quant: no malformed fp16 can reach a provable witness" {
    const a = testing.allocator;
    var sys = try quant.buildSystem(a);
    defer sys.deinit();

    // Every one of these must fail at the fp16 seam. The point is not the
    // error code but the ORDER: the conversion happens before the trace
    // exists, so there is no witness to prove and nothing to reinterpret
    // silently. 0x7C00 = +inf, 0xFC00 = -inf, 0x7E00 = NaN, 0x4C00 =
    // 16.0 (>= 2^4), 0x0A00 = 2^-13 (inexact in q4.22), 0x00FF =
    // subnormal, 0x0000 / 0x8000 = +/-0.
    const malformed = [_]u16{ 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x4C00, 0xCC00, 0x0A00, 0x00FF, 0x0000, 0x8000 };
    for (malformed) |bits| {
        if (quant.scaleFromFp16(bits)) |v| {
            _ = v;
            return error.MalformedFp16WasAccepted;
        } else |_| {}
        // The dequantizer that reads real tensor bytes refuses the same
        // patterns, so a GGUF with such a scale cannot be witnessed.
        const block = blockWithRamp();
        if (tensor.dequantQ4K(&block, bits)) |v| {
            _ = v;
            return error.MalformedFp16WasAccepted;
        } else |_| {}
    }

    // And the positive control: a valid scale flows all the way through
    // to a verifying proof, so the rejections above are not vacuous.
    const ok_bits: u16 = 0x3C00;
    const scale = try quant.scaleFromFp16(ok_bits);
    var case = try realCase(a);
    defer case.deinit(a);
    try testing.expect(scale.eql(case.scale_a[0]));

    var trace = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer trace.deinit(a);
    var bound = try quant.bindOperands(a, &trace, &case.nib_a, &case.scale_a, &case.nib_b, &case.scale_b);
    defer bound.deinit(a);

    var pt = stark.Transcript.init("zkml.quant.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = bound.rows,
        .columns = bound.columns,
    }, sys.system(), CONFIG);
    defer proof.deinit(a);

    var vt = stark.Transcript.init("zkml.quant.v1");
    try testing.expect(try stark.verify(&vt, &proof, sys.system(), CONFIG));
}
