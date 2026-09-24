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

const k_macs: usize = 31;

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
    scale_a: [k_macs]u16,
    scale_b: [k_macs]u16,
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
        case.scale_a[i] = 0x3C00; // fp16 1.0
        case.scale_b[i] = 0x3800; // fp16 0.5
        case.c_true = case.c_true.add(a[i].mul(b[i]));
    }
    return case;
}

fn boundTrace(allocator: std.mem.Allocator, case: *const Case) !struct { gemm: gemm_air.Trace, bound: quant.Trace } {
    var gemm = try gemm_air.buildTrace(allocator, case.a, case.b, case.c_true);
    errdefer gemm.deinit(allocator);
    const bound = try quant.Q4_0.bindOperands(
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
    var sys = try quant.Q4_0.buildSystem(a, k_macs);
    defer sys.deinit();

    // The binding must actually be in the system: two range checks (1 + 4
    // constraints each), both dequantization equations, and the scale
    // provenance gadget (21 per side). The GEMM side is 3 (one composed,
    // two boundary) rather than 4: the closing row is exempt, so it needs
    // no operand pin, only `s[last] = c[last]`.
    try testing.expectEqual(@as(usize, 3 + 10 + 2 + 42), sys.system().constraints.len);
    try testing.expectEqual(@as(usize, 1), sys.system().transition_exemptions);
    try testing.expectEqual(@as(?u16, quant.Q4_0.column_count - 1), sys.system().maxColumn());
    // The nibble range check is quadratic; the running sum already was, so
    // the blowup requirement is unchanged.
    try testing.expectEqual(@as(usize, 2), sys.system().maxDegree());
    try testing.expectEqual(@as(?usize, 32), sys.system().trace_rows);

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

test "quant: a padded GEMM trace is refused before binding" {
    const a = testing.allocator;
    var case = try realCase(a);
    defer case.deinit(a);
    var gemm = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm.deinit(a);
    gemm.rows = 64;

    try testing.expectError(
        quant.BindError.PaddedTrace,
        quant.Q4_0.bindOperands(a, &gemm, &case.nib_a, &case.scale_a, &case.nib_b, &case.scale_b),
    );
    try testing.expectError(quant.BuildError.InvalidReductionLength, quant.Q4_0.buildSystem(a, 16));
}

test "quant: a bit column that is not boolean is rejected" {
    const a = testing.allocator;
    var sys = try quant.Q4_0.buildSystem(a, k_macs);
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
    try testing.expectEqual(@as(u8, 3), bt.bound.columns[quant.Q4_0.col_q_a][row].a.toU64());
    bt.bound.columns[quant.Q4_0.col_bits_a + 0][row] = Fp2.re(Goldilocks.zero.sub(Goldilocks.one));
    bt.bound.columns[quant.Q4_0.col_bits_a + 1][row] = Fp2.re(Goldilocks.fromU64(2));

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
    var sys = try quant.Q4_0.buildSystem(a, k_macs);
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
        quant.BindError.QuantOutOfRange,
        quant.Q4_0.bindOperands(a, &gemm, &bad, &case.scale_a, &case.nib_b, &case.scale_b),
    );
}

test "quant: operands that are not the dequantization of their nibble are refused" {
    const a = testing.allocator;
    var case = try realCase(a);
    defer case.deinit(a);

    // Correct nibbles, but a block scale that does not produce these
    // operands: bindOperands catches it rather than the AIR.
    var gemm = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm.deinit(a);

    // A neighbouring fp16 scale: still a real fp16, so the AIR would
    // accept the provenance, but it does not dequantize to these operands.
    var wrong: [k_macs]u16 = undefined;
    for (0..k_macs) |i| wrong[i] = 0x3C01; // fp16 1.0009765625
    try testing.expectError(
        quant.BindError.InconsistentOperands,
        quant.Q4_0.bindOperands(a, &gemm, &case.nib_a, &wrong, &case.nib_b, &case.scale_b),
    );
}

test "quant: a fabricated scale no longer proves" {
    const a = testing.allocator;
    var sys = try quant.Q4_0.buildSystem(a, k_macs);
    defer sys.deinit();

    // The gap this test used to mark: a scale no fp16 could produce, with
    // every other constraint satisfied, produced a valid proof. The scale
    // is now pinned to the image of `fp16ToFixedQ4_22` by scale_air, so
    // the binder cannot even BUILD such a trace: the fp16 pattern has to
    // exist, and the scale column is derived from it.
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
        // A real fp16 (1.0), so the pattern is accepted...
        case.scale_a[i] = 0x3C00;
        case.scale_b[i] = 0x3C00;
        // ...but the operands are built with the fabricated scale.
        case.a[i] = na.sub(eight).mul(fake);
        case.b[i] = nb.sub(eight).mul(fake);
        case.c_true = case.c_true.add(case.a[i].mul(case.b[i]));
    }

    var gemm = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm.deinit(a);
    // bindOperands refuses: the dequantization equation does not hold for
    // the scale the fp16 pattern produces.
    try testing.expectError(
        quant.BindError.InconsistentOperands,
        quant.Q4_0.bindOperands(a, &gemm, &case.nib_a, &case.scale_a, &case.nib_b, &case.scale_b),
    );

    // And an fp16 that is not a usable q4.22 scale is refused outright.
    var gemm2 = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm2.deinit(a);
    var subnormal: [k_macs]u16 = undefined;
    var too_big: [k_macs]u16 = undefined;
    var infinite: [k_macs]u16 = undefined;
    for (0..k_macs) |i| {
        subnormal[i] = 0x0001; // fp16 smallest subnormal
        too_big[i] = 0x7C00; // fp16 +inf
        infinite[i] = 0x7E00; // fp16 NaN
    }
    try testing.expectError(
        quant.BindError.BadScale,
        quant.Q4_0.bindOperands(a, &gemm2, &case.nib_a, &subnormal, &case.nib_b, &case.scale_b),
    );
    var gemm3 = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm3.deinit(a);
    try testing.expectError(
        quant.BindError.BadScale,
        quant.Q4_0.bindOperands(a, &gemm3, &case.nib_a, &too_big, &case.nib_b, &case.scale_b),
    );
    var gemm4 = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm4.deinit(a);
    try testing.expectError(
        quant.BindError.BadScale,
        quant.Q4_0.bindOperands(a, &gemm4, &case.nib_a, &infinite, &case.nib_b, &case.scale_b),
    );
}

test "quant: tampering with a scale's provenance witness is rejected" {
    const a = testing.allocator;
    var sys = try quant.Q4_0.buildSystem(a, k_macs);
    defer sys.deinit();
    var case = try realCase(a);
    defer case.deinit(a);

    var bt = try boundTrace(a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);

    // The honest trace proves and verifies.
    var pt = stark.Transcript.init("zkml.quant.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = bt.bound.rows,
        .columns = bt.bound.columns,
    }, sys.system(), CONFIG);
    defer proof.deinit(a);
    var vt = stark.Transcript.init("zkml.quant.v1");
    try testing.expect(try stark.verify(&vt, &proof, sys.system(), CONFIG));

    // Claim a different shift: the scale column no longer matches the
    // (1024 + m)·2^s the selector and mantissa produce. This is the
    // attack the gadget exists to stop, and it is now caught by the AIR
    // rather than by the binder refusing to build the trace.
    const bad = try a.dupe(Fp2, bt.bound.columns[quant.Q4_0.col_scale_a]);
    defer a.free(bad);
    bad[0] = bad[0].add(Fp2.one);
    const cols = try a.alloc([]const Fp2, quant.Q4_0.column_count);
    defer a.free(cols);
    for (bt.bound.columns, 0..) |c, i| cols[i] = c;
    cols[quant.Q4_0.col_scale_a] = bad;

    var pt2 = stark.Transcript.init("zkml.quant.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt2, .{
            .rows = bt.bound.rows,
            .columns = cols,
        }, sys.system(), CONFIG),
    );
}

test "quant: a 4095-MAC reduction (4096 rows) proves and verifies" {
    // Regression: fri.verify used to carry a fixed [4096]Fp2 stack buffer
    // for the residual, so any proof whose final FRI domain exceeded 2^12
    // was rejected as InvalidProof — an HONEST proof, silently. k=4095
    // puts the final domain at 2^14. The residual is now evaluated with
    // the FFT, so this both passes and stops dominating verify time.
    const a = testing.allocator;
    const k: usize = 4095;
    var sys = try quant.Q4_0.buildSystem(a, k);
    defer sys.deinit();

    const av = try a.alloc(Goldilocks, k);
    defer a.free(av);
    const bv = try a.alloc(Goldilocks, k);
    defer a.free(bv);
    const nibs_a = try a.alloc(u8, k);
    defer a.free(nibs_a);
    const nibs_b = try a.alloc(u8, k);
    defer a.free(nibs_b);
    const sc_a = try a.alloc(u16, k);
    defer a.free(sc_a);
    const sc_b = try a.alloc(u16, k);
    defer a.free(sc_b);

    // Four Q4_K blocks, all sharing the fp16 scale 1.0, so the reduction
    // spans several blocks and the per-MAC scale array is exercised.
    const block = blockWithRamp();
    const deq = tensor.dequantQ4K(&block, 0x3C00) catch unreachable;
    const scale: u16 = 0x3C00; // fp16 1.0
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
    var bound = try quant.Q4_0.bindOperands(a, &gemm, nibs_a, sc_a, nibs_b, sc_b);
    defer bound.deinit(a);
    try testing.expectEqual(@as(usize, 4096), bound.rows);

    const log_trace: u6 = 12;
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
    var sys = try quant.Q4_0.buildSystem(a, k_macs);
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
    try testing.expectEqual(ok_bits, case.scale_a[0]);
    try testing.expect(scale.eql(quant.scaleFromFp16(case.scale_a[0]) catch unreachable));

    var trace = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer trace.deinit(a);
    var bound = try quant.Q4_0.bindOperands(a, &trace, &case.nib_a, &case.scale_a, &case.nib_b, &case.scale_b);
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

test "quant: the width template lays out columns without overlap" {
    // The provenance gadget's columns start after the quant and bit
    // columns, and the two gadgets' blocks do not touch. Getting this
    // wrong is silent — the constraints still evaluate, they just read the
    // wrong columns, which is how a gadget ends up attesting nothing.
    const B = quant.Q4_0;
    try testing.expectEqual(@as(u16, 6), B.col_bits_a);
    try testing.expectEqual(@as(u16, 10), B.col_bits_b);
    try testing.expectEqual(@as(u16, 14), B.col_scale_a);
    try testing.expectEqual(@as(u16, 15), B.col_scale_b);
    // Bits and scale come first, then the gadgets.
    try testing.expect(B.col_bits_b + B.width > B.col_bits_b);
    try testing.expect(B.col_scale_a > B.col_bits_b + B.width - 1);
    try testing.expectEqual(@as(usize, 74), B.column_count);
    // The gadget's own layout, checked against the declared column count.
    try testing.expectEqual(@as(u16, 16), B.gadget_a.mant_base);
    try testing.expectEqual(@as(u16, 26), B.gadget_a.sel_base);
    try testing.expectEqual(@as(u16, 42), B.gadget_a.shift_col);
    try testing.expectEqual(@as(u16, 45), B.gadget_b.mant_base);
    try testing.expectEqual(@as(u16, 55), B.gadget_b.sel_base);
    try testing.expectEqual(@as(u16, 71), B.gadget_b.shift_col);
    try testing.expectEqual(B.column_count - 1, B.gadget_b.sign_col);
    try testing.expect(B.gadget_a.mant_base > B.col_scale_b);
    try testing.expect(B.gadget_b.mant_base > B.gadget_a.sign_col);
}

test "quant: Q8_0 proves the representation but not the scale's origin" {
    // Q8_0's scale is not an fp16 in q4.22, so the provenance gadget would
    // attest a convention Q8_0 does not use. It runs without, and that gap
    // is deliberate: the representation is pinned, the scale's origin is
    // not. This test exists so the gap stays visible.
    const B = quant.Q8_0;
    try testing.expectEqual(@as(u8, 8), B.width);
    // No gadget columns: the layout stops after the two scales.
    try testing.expectEqual(@as(usize, B.col_scale_b + 1), B.column_count);

    var sys = try B.buildSystem(testing.allocator, k_macs);
    defer sys.deinit();
    // 3 GEMM + 2 range checks x (1 + 8) + 2 dequant equations, and no
    // provenance gadget.
    try testing.expectEqual(@as(usize, 3 + 2 * 9 + 2), sys.system().constraints.len);
    try testing.expectEqual(@as(?u16, B.col_scale_b), sys.system().maxColumn());
}
