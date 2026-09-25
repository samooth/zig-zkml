//! End-to-end tests for the Q4_0 / Q8_0 quantized-operand binding:
//! the GEMM operands are no longer free witness columns but are pinned to
//! a raw quant value times one block scale, with its bits proven boolean.

const std = @import("std");
const tensor = @import("../tensor/root.zig");
const stark = @import("root.zig");
const gemm_air = @import("./gemm_air.zig");
const quant = @import("./quant_binding.zig");

const testing = std.testing;
const Goldilocks = tensor.Goldilocks;
const Fp2 = stark.Fp2;
const Q4_0 = quant.Q4_0;
const Q4_1 = quant.Q4_1;
const Q8_0 = quant.Q8_0;

const k_macs: usize = 31;
const k_q4_1_macs: usize = 127;

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

/// A Q4 block whose bytes encode known raw nibbles: byte j is (j%16) in
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
fn rawQ4(i: usize) u8 {
    return @intCast((i / 2) % 16);
}

fn rawQuant(comptime quant_width: u8, i: usize) u8 {
    const modulus: usize = @as(usize, 1) << @intCast(quant_width);
    const value = if (i == 0) @as(usize, 3) else (i * 37 + 11) % modulus;
    return @intCast(value);
}

const Case = struct {
    a: []Goldilocks,
    b: []Goldilocks,
    q_a: [k_macs]u8,
    q_b: [k_macs]u8,
    scale_a: [k_macs]Goldilocks,
    scale_b: [k_macs]Goldilocks,
    /// The fp16 PATTERN each scale came from. The binding takes the
    /// pattern, not the field element, so the AIR can witness the
    /// provenance gadget from the same source the dequant equation uses.
    scale_bits_a: [k_macs]u16,
    scale_bits_b: [k_macs]u16,
    c_true: Goldilocks,

    fn deinit(self: *Case, allocator: std.mem.Allocator) void {
        allocator.free(self.a);
        allocator.free(self.b);
        self.* = undefined;
    }
};

fn symmetricCase(allocator: std.mem.Allocator, comptime quant_width: u8) !Case {
    const a = try allocator.alloc(Goldilocks, k_macs);
    errdefer allocator.free(a);
    const b = try allocator.alloc(Goldilocks, k_macs);
    errdefer allocator.free(b);

    var case = Case{
        .a = a,
        .b = b,
        .q_a = undefined,
        .q_b = undefined,
        .scale_a = undefined,
        .scale_b = undefined,
        .scale_bits_a = undefined,
        .scale_bits_b = undefined,
        .c_true = Goldilocks.zero,
    };
    const eight = Goldilocks.fromU64(8);
    for (0..k_macs) |i| {
        case.q_a[i] = rawQuant(quant_width, i);
        case.q_b[i] = rawQuant(quant_width, i + 1);
        case.scale_bits_a[i] = 0x3C00; // fp16 1.0
        case.scale_bits_b[i] = 0x3800; // fp16 0.5
        case.scale_a[i] = try quant.scaleFromFp16(case.scale_bits_a[i]);
        case.scale_b[i] = try quant.scaleFromFp16(case.scale_bits_b[i]);
        case.a[i] = Goldilocks.fromU64(case.q_a[i]).sub(eight).mul(case.scale_a[i]);
        case.b[i] = Goldilocks.fromU64(case.q_b[i]).sub(eight).mul(case.scale_b[i]);
        case.c_true = case.c_true.add(case.a[i].mul(case.b[i]));
    }
    return case;
}

fn boundTrace(
    comptime Binding: type,
    allocator: std.mem.Allocator,
    case: *const Case,
) !struct { gemm: gemm_air.Trace, bound: quant.Trace } {
    var gemm = try gemm_air.buildTrace(allocator, case.a, case.b, case.c_true);
    errdefer gemm.deinit(allocator);
    const bound = try Binding.bindOperands(
        allocator,
        &gemm,
        &case.q_a,
        &case.scale_bits_a,
        &case.q_b,
        &case.scale_bits_b,
    );
    return .{ .gemm = gemm, .bound = bound };
}

fn expectSymmetricPaddingRejected(comptime Binding: type) !void {
    const allocator = testing.allocator;
    var case = try symmetricCase(allocator, Binding.width);
    defer case.deinit(allocator);
    const k: usize = 31;
    const a = try allocator.dupe(Goldilocks, case.a[0..k]);
    defer allocator.free(a);
    const b = try allocator.dupe(Goldilocks, case.b[0..k]);
    defer allocator.free(b);
    var q_a: [k]u8 = undefined;
    var q_b: [k]u8 = undefined;
    var scale_a: [k]u16 = undefined;
    var scale_b: [k]u16 = undefined;
    @memcpy(&q_a, case.q_a[0..k]);
    @memcpy(&q_b, case.q_b[0..k]);
    @memcpy(&scale_a, case.scale_bits_a[0..k]);
    @memcpy(&scale_b, case.scale_bits_b[0..k]);
    var c = Goldilocks.zero;
    for (0..k) |i| c = c.add(a[i].mul(b[i]));
    var gemm = try gemm_air.buildTrace(allocator, a, b, c);
    defer gemm.deinit(allocator);
    // The honest 32-row trace binds...
    var ok = try Binding.bindOperands(allocator, &gemm, &q_a, &scale_a, &q_b, &scale_b);
    defer ok.deinit(allocator);
    // ...and the same operands on a 64-row domain are refused: that is the
    // shape a prover would inflate to hide a product.
    gemm.rows = 64;
    try testing.expectError(
        quant.BindError.PaddedTrace,
        Binding.bindOperands(allocator, &gemm, &q_a, &scale_a, &q_b, &scale_b),
    );
}

fn expectHonestProof(comptime Binding: type) !void {
    const a = testing.allocator;
    var sys = try Binding.buildSystem(a, k_macs);
    defer sys.deinit();

    const width: usize = Binding.width;
    // 3 GEMM (closing row exempt) + 2 range checks x (1 + width) + 2 dequant,
    // plus 21 per side when the format carries the provenance gadget. The
    // flag is the binding's own `has_provenance`, not a structural probe:
    // Q4_0 and Q8_0 are the same template, so a probe cannot tell them apart.
    const provenance: usize = if (Binding.has_provenance) 2 * 21 else 0;
    try testing.expectEqual(3 + 2 * (1 + width) + 2 + provenance, sys.system().constraints.len);
    const last: ?u16 = if (provenance != 0) @intCast(Binding.column_count - 1) else @as(?u16, Binding.col_scale_b);
    try testing.expectEqual(last, sys.system().maxColumn());
    try testing.expectEqual(@as(usize, 2), sys.system().maxDegree());

    var case = try symmetricCase(a, Binding.width);
    defer case.deinit(a);
    try testing.expect(!case.c_true.isZero());

    var bt = try boundTrace(Binding, a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);

    const label = if (Binding.width == 4) "zkml.quant.q4_0.v1" else "zkml.quant.q8_0.v1";
    var pt = stark.Transcript.init(label);
    var proof = try stark.prove(a, &pt, .{
        .rows = bt.bound.rows,
        .columns = bt.bound.columns,
    }, sys.system(), CONFIG);
    defer proof.deinit(a);

    var vt = stark.Transcript.init(label);
    try testing.expect(try stark.verify(&vt, &proof, sys.system(), CONFIG));
}

fn expectNonBooleanBitsRejected(comptime Binding: type) !void {
    const a = testing.allocator;
    var sys = try Binding.buildSystem(a, k_macs);
    defer sys.deinit();

    var case = try symmetricCase(a, Binding.width);
    defer case.deinit(a);
    var bt = try boundTrace(Binding, a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);

    const row: usize = 0;
    try testing.expectEqual(@as(u8, 3), bt.bound.columns[Binding.col_q_a][row].a.toU64());
    bt.bound.columns[Binding.col_bits_a + 0][row] = Fp2.re(Goldilocks.zero.sub(Goldilocks.one));
    bt.bound.columns[Binding.col_bits_a + 1][row] = Fp2.re(Goldilocks.fromU64(2));

    const label = if (Binding.width == 4) "zkml.quant.q4_0.v1" else "zkml.quant.q8_0.v1";
    var pt = stark.Transcript.init(label);
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{
            .rows = bt.bound.rows,
            .columns = bt.bound.columns,
        }, sys.system(), CONFIG),
    );
}

fn expectTamperedScaleRejected(comptime Binding: type) !void {
    const a = testing.allocator;
    var sys = try Binding.buildSystem(a, k_macs);
    defer sys.deinit();

    var case = try symmetricCase(a, Binding.width);
    defer case.deinit(a);
    var bt = try boundTrace(Binding, a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);

    const row: usize = 3;
    const scale = bt.bound.columns[Binding.col_scale_a][row].a;
    bt.bound.columns[Binding.col_scale_a][row] = Fp2.re(scale.add(Goldilocks.one));

    const label = if (Binding.width == 4) "zkml.quant.q4_0.v1" else "zkml.quant.q8_0.v1";
    var pt = stark.Transcript.init(label);
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{
            .rows = bt.bound.rows,
            .columns = bt.bound.columns,
        }, sys.system(), CONFIG),
    );
}

fn expectTamperedQuantRejected(comptime Binding: type) !void {
    const a = testing.allocator;
    var sys = try Binding.buildSystem(a, k_macs);
    defer sys.deinit();

    var case = try symmetricCase(a, Binding.width);
    defer case.deinit(a);
    var bt = try boundTrace(Binding, a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);

    const row: usize = 1;
    const old_quant = rawQuant(Binding.width, row);
    const new_quant = if (old_quant == Binding.max_quant) old_quant - 1 else old_quant + 1;
    bt.bound.columns[Binding.col_q_a][row] = Fp2.re(Goldilocks.fromU64(new_quant));
    for (0..Binding.width) |bit| {
        const shift: u3 = @intCast(bit);
        const bit_index: u16 = @intCast(bit);
        bt.bound.columns[Binding.col_bits_a + bit_index][row] =
            Fp2.re(Goldilocks.fromU64((new_quant >> shift) & 1));
    }

    const label = if (Binding.width == 4) "zkml.quant.q4_0.v1" else "zkml.quant.q8_0.v1";
    var pt = stark.Transcript.init(label);
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{
            .rows = bt.bound.rows,
            .columns = bt.bound.columns,
        }, sys.system(), CONFIG),
    );
}

const AffineCase = struct {
    a: []Goldilocks,
    b: []Goldilocks,
    q_a: []u8,
    q_b: []u8,
    scale_a: []Goldilocks,
    scale_b: []Goldilocks,
    minimum_a: []Goldilocks,
    minimum_b: []Goldilocks,
    c_true: Goldilocks,

    fn deinit(self: *AffineCase, allocator: std.mem.Allocator) void {
        allocator.free(self.a);
        allocator.free(self.b);
        allocator.free(self.q_a);
        allocator.free(self.q_b);
        allocator.free(self.scale_a);
        allocator.free(self.scale_b);
        allocator.free(self.minimum_a);
        allocator.free(self.minimum_b);
        self.* = undefined;
    }
};

fn affineCase(allocator: std.mem.Allocator, k: usize) !AffineCase {
    const a = try allocator.alloc(Goldilocks, k);
    errdefer allocator.free(a);
    const b = try allocator.alloc(Goldilocks, k);
    errdefer allocator.free(b);
    const q_a = try allocator.alloc(u8, k);
    errdefer allocator.free(q_a);
    const q_b = try allocator.alloc(u8, k);
    errdefer allocator.free(q_b);
    const scale_a = try allocator.alloc(Goldilocks, k);
    errdefer allocator.free(scale_a);
    const scale_b = try allocator.alloc(Goldilocks, k);
    errdefer allocator.free(scale_b);
    const blocks = k / Q4_1.block_size + @intFromBool(k % Q4_1.block_size != 0);
    const minimum_a = try allocator.alloc(Goldilocks, blocks);
    errdefer allocator.free(minimum_a);
    const minimum_b = try allocator.alloc(Goldilocks, blocks);
    errdefer allocator.free(minimum_b);

    var case = AffineCase{
        .a = a,
        .b = b,
        .q_a = q_a,
        .q_b = q_b,
        .scale_a = scale_a,
        .scale_b = scale_b,
        .minimum_a = minimum_a,
        .minimum_b = minimum_b,
        .c_true = Goldilocks.zero,
    };
    for (0..blocks) |block| {
        case.minimum_a[block] = if (block % 2 == 0)
            try quant.scaleFromFp16(0xBC00)
        else
            try quant.scaleFromFp16(0x3800);
        case.minimum_b[block] = if (block % 2 == 0)
            try quant.scaleFromFp16(0x3C00)
        else
            try quant.scaleFromFp16(0xB800);
    }
    for (0..k) |i| {
        const block = i / Q4_1.block_size;
        case.q_a[i] = @intCast((i * 7 + 3) % 16);
        case.q_b[i] = @intCast((i * 11 + 5) % 16);
        case.scale_a[i] = if (block % 2 == 0)
            try quant.scaleFromFp16(0x3C00)
        else
            try quant.scaleFromFp16(0x3800);
        case.scale_b[i] = if (block % 2 == 0)
            try quant.scaleFromFp16(0x3800)
        else
            try quant.scaleFromFp16(0x3C00);
        case.a[i] = Goldilocks.fromU64(case.q_a[i]).mul(case.scale_a[i]).add(case.minimum_a[block]);
        case.b[i] = Goldilocks.fromU64(case.q_b[i]).mul(case.scale_b[i]).add(case.minimum_b[block]);
        case.c_true = case.c_true.add(case.a[i].mul(case.b[i]));
    }
    return case;
}

fn configForRows(rows: usize) stark.Config {
    const log_trace: u6 = @intCast(std.math.log2_int(usize, rows));
    return .{
        .log_trace = log_trace,
        .log_blowup = 2,
        .fri = .{
            .log_domain = log_trace + 2,
            .log_final = log_trace + 1,
            .log_residual_degree = log_trace,
            .num_queries = 4,
        },
    };
}

fn boundAffine(
    allocator: std.mem.Allocator,
    case: *const AffineCase,
) !struct { gemm: gemm_air.Trace, bound: quant.Trace } {
    var gemm = try gemm_air.buildTrace(allocator, case.a, case.b, case.c_true);
    errdefer gemm.deinit(allocator);
    const bound = try Q4_1.bindOperands(
        allocator,
        &gemm,
        case.q_a,
        case.scale_a,
        case.minimum_a,
        case.q_b,
        case.scale_b,
        case.minimum_b,
    );
    return .{ .gemm = gemm, .bound = bound };
}

const AffineMutation = enum {
    none,
    scale,
    minimum,
    magnitude_range,
    quant,
    inside_block,
    block_boundary,
    phase_rotation,
};

fn refreshAffineGemm(allocator: std.mem.Allocator, bound: *quant.Trace, case: *AffineCase) !void {
    var c = Goldilocks.zero;
    for (0..case.a.len) |i| c = c.add(case.a[i].mul(case.b[i]));
    case.c_true = c;
    var replacement = try gemm_air.buildTrace(allocator, case.a, case.b, case.c_true);
    defer replacement.deinit(allocator);
    for (0..gemm_air.column_count) |column| {
        @memcpy(bound.columns[column], replacement.columns[column]);
    }
    const last_block = case.minimum_a.len - 1;
    for (case.a.len..bound.rows) |row| {
        bound.columns[Q4_1.col_scale_a][row] = Fp2.re(
            bound.columns[gemm_air.col_a][row].a.sub(case.minimum_a[last_block]),
        );
        bound.columns[Q4_1.col_scale_b][row] = Fp2.re(
            bound.columns[gemm_air.col_b][row].a.sub(case.minimum_b[last_block]),
        );
    }
}

fn writeZeroMinimum(bound: *quant.Trace, row: usize, is_a: bool) void {
    const minimum_col = if (is_a) Q4_1.col_minimum_a else Q4_1.col_minimum_b;
    const sign_col = if (is_a) Q4_1.col_minimum_sign_a else Q4_1.col_minimum_sign_b;
    const magnitude_col = if (is_a) Q4_1.col_minimum_magnitude_a else Q4_1.col_minimum_magnitude_b;
    const magnitude_bits_col = if (is_a) Q4_1.col_minimum_magnitude_bits_a else Q4_1.col_minimum_magnitude_bits_b;
    bound.columns[minimum_col][row] = Fp2.zero;
    bound.columns[sign_col][row] = Fp2.zero;
    bound.columns[magnitude_col][row] = Fp2.zero;
    for (0..Q4_1.minimum_magnitude_bits) |bit| {
        const bit_index: u16 = @intCast(bit);
        bound.columns[magnitude_bits_col + bit_index][row] = Fp2.zero;
    }
}

fn rotateAffinePhase(bound: *quant.Trace) void {
    for (0..bound.rows) |row| {
        const phase: u8 = @intCast((row + 1) % Q4_1.block_size);
        const plus_one: u8 = phase + 1;
        bound.columns[Q4_1.col_block_phase][row] = Fp2.re(Goldilocks.fromU64(phase));
        bound.columns[Q4_1.col_block_phase_plus_one][row] = Fp2.re(Goldilocks.fromU64(plus_one));
        for (0..5) |bit| {
            const shift: u3 = @intCast(bit);
            const bit_index: u16 = @intCast(bit);
            bound.columns[Q4_1.col_block_phase_bits + bit_index][row] =
                Fp2.re(Goldilocks.fromU64((phase >> shift) & 1));
        }
        for (0..6) |bit| {
            const shift: u3 = @intCast(bit);
            const bit_index: u16 = @intCast(bit);
            bound.columns[Q4_1.col_block_phase_plus_one_bits + bit_index][row] =
                Fp2.re(Goldilocks.fromU64((plus_one >> shift) & 1));
        }
    }
}

fn refreshAffineMinimumDeltas(bound: *quant.Trace) void {
    for (0..bound.rows) |row| {
        const next = if (row + 1 == bound.rows) 0 else row + 1;
        bound.columns[Q4_1.col_minimum_delta_a][row] =
            bound.columns[Q4_1.col_minimum_a][next].sub(bound.columns[Q4_1.col_minimum_a][row]);
        bound.columns[Q4_1.col_minimum_delta_b][row] =
            bound.columns[Q4_1.col_minimum_b][next].sub(bound.columns[Q4_1.col_minimum_b][row]);
    }
}

fn expectAffineCase(k: usize, mutation: AffineMutation) !void {
    const allocator = testing.allocator;
    var case = try affineCase(allocator, k);
    defer case.deinit(allocator);
    var bt = try boundAffine(allocator, &case);
    defer bt.gemm.deinit(allocator);
    defer bt.bound.deinit(allocator);
    var sys = try Q4_1.buildSystem(allocator, k);
    defer sys.deinit();

    const phased = bt.bound.rows >= 2 * Q4_1.block_size;
    // 3 GEMM (closing row exempt) + the affine constraints; one fewer than
    // before, because the closing row no longer needs a dequant equation.
    const expected_constraints: usize = if (phased) 93 else 76;
    const expected_max_column: u16 = if (phased) Q4_1.col_minimum_delta_b else Q4_1.col_block_phase;
    try testing.expectEqual(expected_constraints, sys.system().constraints.len);
    try testing.expectEqual(@as(?u16, expected_max_column), sys.system().maxColumn());
    try testing.expectEqual(@as(usize, 2), sys.system().maxDegree());
    try testing.expectEqual(Q4_1.column_count, bt.bound.columns.len);
    try testing.expect(!case.c_true.isZero());
    try testing.expectEqual(@as(u64, 1), bt.bound.columns[Q4_1.col_minimum_sign_a][0].a.toU64());
    try testing.expectEqual(@as(u64, 1) << 22, bt.bound.columns[Q4_1.col_minimum_magnitude_a][0].a.toU64());
    if (bt.bound.rows >= 2 * Q4_1.block_size) {
        try testing.expectEqual(@as(u64, 0), bt.bound.columns[Q4_1.col_minimum_sign_a][Q4_1.block_size].a.toU64());
        try testing.expectEqual(@as(u64, 1) << 21, bt.bound.columns[Q4_1.col_minimum_magnitude_a][Q4_1.block_size].a.toU64());
        try testing.expectEqual(@as(u64, 1), bt.bound.columns[Q4_1.col_minimum_sign_b][Q4_1.block_size].a.toU64());
    }

    switch (mutation) {
        .none => {},
        .scale => {
            const row: usize = 3;
            const scale = bt.bound.columns[Q4_1.col_scale_a][row].a;
            bt.bound.columns[Q4_1.col_scale_a][row] = Fp2.re(scale.add(Goldilocks.one));
        },
        .minimum => {
            bt.bound.columns[Q4_1.col_minimum_a][0] =
                bt.bound.columns[Q4_1.col_minimum_a][0].add(Fp2.one);
            refreshAffineMinimumDeltas(&bt.bound);
        },
        .magnitude_range => {
            const old_magnitude = @as(u64, 1) << 22;
            const new_magnitude = @as(u64, 1) << Q4_1.minimum_magnitude_bits;
            const new_minimum = Goldilocks.fromU64(new_magnitude).neg();
            case.a[0] = case.a[0].sub(Goldilocks.fromU64(new_magnitude - old_magnitude));
            bt.bound.columns[Q4_1.col_minimum_a][0] = Fp2.re(new_minimum);
            bt.bound.columns[Q4_1.col_minimum_magnitude_a][0] =
                Fp2.re(Goldilocks.fromU64(new_magnitude));
            refreshAffineMinimumDeltas(&bt.bound);
            try refreshAffineGemm(allocator, &bt.bound, &case);
        },
        .quant => {
            const row: usize = 1;
            const new_quant: u8 = if (case.q_a[row] == 15) 0 else case.q_a[row] + 1;
            bt.bound.columns[Q4_1.col_q_a][row] = Fp2.re(Goldilocks.fromU64(new_quant));
            for (0..4) |bit| {
                const shift: u3 = @intCast(bit);
                const bit_index: u16 = @intCast(bit);
                bt.bound.columns[Q4_1.col_q_bits_a + bit_index][row] =
                    Fp2.re(Goldilocks.fromU64((new_quant >> shift) & 1));
            }
        },
        .inside_block => {
            case.a[0] = case.a[0].sub(case.minimum_a[0]);
            writeZeroMinimum(&bt.bound, 0, true);
            refreshAffineMinimumDeltas(&bt.bound);
            try refreshAffineGemm(allocator, &bt.bound, &case);
        },
        .block_boundary => {
            for (0..Q4_1.block_size) |row| {
                case.a[row] = case.a[row].sub(case.minimum_a[0]);
                writeZeroMinimum(&bt.bound, row, true);
            }
            refreshAffineMinimumDeltas(&bt.bound);
            try refreshAffineGemm(allocator, &bt.bound, &case);
        },
        .phase_rotation => {
            for (0..case.a.len) |row| {
                const block = row / Q4_1.block_size;
                case.a[row] = case.a[row].sub(case.minimum_a[block]);
                case.b[row] = case.b[row].sub(case.minimum_b[block]);
            }
            for (case.minimum_a) |*minimum| minimum.* = Goldilocks.zero;
            for (case.minimum_b) |*minimum| minimum.* = Goldilocks.zero;
            for (0..case.a.len) |row| {
                writeZeroMinimum(&bt.bound, row, true);
                writeZeroMinimum(&bt.bound, row, false);
            }
            writeZeroMinimum(&bt.bound, bt.bound.rows - 1, true);
            writeZeroMinimum(&bt.bound, bt.bound.rows - 1, false);
            refreshAffineMinimumDeltas(&bt.bound);
            try refreshAffineGemm(allocator, &bt.bound, &case);
            rotateAffinePhase(&bt.bound);
        },
    }

    const config = configForRows(bt.bound.rows);
    const should_verify = mutation == .none or mutation == .block_boundary;
    var pt = stark.Transcript.init("zkml.quant.q4_1.v1");
    if (should_verify) {
        var proof = try stark.prove(allocator, &pt, .{
            .rows = bt.bound.rows,
            .columns = bt.bound.columns,
        }, sys.system(), config);
        defer proof.deinit(allocator);
        var vt = stark.Transcript.init("zkml.quant.q4_1.v1");
        try testing.expect(try stark.verify(&vt, &proof, sys.system(), config));
    } else if (mutation == .phase_rotation) {
        // The prover now checks its own boundary constraints, so a rotated
        // block phase is caught there rather than at the verifier. The
        // verifier's copy of the check is unchanged; what moved is which
        // side of the protocol refuses first.
        try testing.expectError(
            stark.Error.ConstraintViolation,
            stark.prove(allocator, &pt, .{
                .rows = bt.bound.rows,
                .columns = bt.bound.columns,
            }, sys.system(), config),
        );
    } else {
        try testing.expectError(
            stark.Error.ConstraintViolation,
            stark.prove(allocator, &pt, .{
                .rows = bt.bound.rows,
                .columns = bt.bound.columns,
            }, sys.system(), config),
        );
    }
}

test "quant: Q4_0 operands prove and verify with the binding" {
    try expectHonestProof(Q4_0);
}

test "quant: Q8_0 operands prove and verify with the binding" {
    try expectHonestProof(Q8_0);
}

test "quant: Q4_0 and Q8_0 refuse padded GEMM traces" {
    try expectSymmetricPaddingRejected(Q4_0);
    try expectSymmetricPaddingRejected(Q8_0);
}

test "quant: non-boolean bits are rejected for Q4_0 and Q8_0" {
    try expectNonBooleanBitsRejected(Q4_0);
    try expectNonBooleanBitsRejected(Q8_0);
}

test "quant: a tampered scale is rejected for Q4_0 and Q8_0" {
    try expectTamperedScaleRejected(Q4_0);
    try expectTamperedScaleRejected(Q8_0);
}

test "quant: a tampered quant value is rejected for Q4_0 and Q8_0" {
    try expectTamperedQuantRejected(Q4_0);
    try expectTamperedQuantRejected(Q8_0);
}

test "quant: Q4_1 signed minima prove and verify across two blocks" {
    try expectAffineCase(k_q4_1_macs, .none);
}

test "quant: Q4_1 uses one minimum for a partial block" {
    try expectAffineCase(k_macs, .none);
}

test "quant: Q4_1 permits the minimum to change at a block boundary" {
    try expectAffineCase(k_q4_1_macs, .block_boundary);
}

test "quant: Q4_1 rejects a tampered scale" {
    try expectAffineCase(k_q4_1_macs, .scale);
}

test "quant: Q4_1 rejects a tampered minimum" {
    try expectAffineCase(k_q4_1_macs, .minimum);
}

test "quant: Q4_1 rejects a magnitude outside the 26-bit AIR range" {
    try expectAffineCase(k_q4_1_macs, .magnitude_range);
}

test "quant: Q4_1 rejects a tampered quant value" {
    try expectAffineCase(k_q4_1_macs, .quant);
}

test "quant: Q4_1 rejects a minimum change inside a block" {
    try expectAffineCase(k_q4_1_macs, .inside_block);
}

test "quant: Q4_1 rejects a rotated block phase at the first row" {
    try expectAffineCase(k_q4_1_macs, .phase_rotation);
}

test "quant: Q4_1 refuses a minimum outside the signed q4.22 range" {
    const allocator = testing.allocator;
    var case = try affineCase(allocator, k_q4_1_macs);
    defer case.deinit(allocator);
    var gemm = try gemm_air.buildTrace(allocator, case.a, case.b, case.c_true);
    defer gemm.deinit(allocator);

    var bad_minimum = case.minimum_a;
    bad_minimum[0] = Goldilocks.fromU64(@as(u64, 1) << Q4_1.minimum_magnitude_bits);
    try testing.expectError(
        quant.BindError.MinimumOutOfRange,
        Q4_1.bindOperands(
            allocator,
            &gemm,
            case.q_a,
            case.scale_a,
            bad_minimum,
            case.q_b,
            case.scale_b,
            case.minimum_b,
        ),
    );
}

test "quant: Q4_1 refuses a padded GEMM trace" {
    const allocator = testing.allocator;
    // k=63 gives a 64-row domain; claim 128 for it, which is the shape a
    // prover would inflate to hide a product.
    var case = try affineCase(allocator, 63);
    defer case.deinit(allocator);
    var gemm = try gemm_air.buildTrace(allocator, case.a, case.b, case.c_true);
    defer gemm.deinit(allocator);
    gemm.rows = 128;
    try testing.expectError(
        quant.BindError.PaddedTrace,
        Q4_1.bindOperands(
            allocator,
            &gemm,
            case.q_a,
            case.scale_a,
            case.minimum_a,
            case.q_b,
            case.scale_b,
            case.minimum_b,
        ),
    );
}

test "quant: Q4_1 refuses the wrong number of block minima" {
    const allocator = testing.allocator;
    var case = try affineCase(allocator, k_q4_1_macs);
    defer case.deinit(allocator);
    var gemm = try gemm_air.buildTrace(allocator, case.a, case.b, case.c_true);
    defer gemm.deinit(allocator);

    try testing.expectError(
        quant.BindError.InvalidBlockCount,
        Q4_1.bindOperands(
            allocator,
            &gemm,
            case.q_a,
            case.scale_a,
            case.minimum_a[0..1],
            case.q_b,
            case.scale_b,
            case.minimum_b,
        ),
    );
}

test "quant: an operand that does not match its scale is rejected" {
    const a = testing.allocator;
    var sys = try Q4_0.buildSystem(a, k_macs);
    defer sys.deinit();

    var case = try symmetricCase(a, Q4_0.width);
    defer case.deinit(a);
    var bt = try boundTrace(Q4_0, a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);

    bt.bound.columns[gemm_air.col_a][3] = Fp2.re(Goldilocks.fromU64(12345));

    var pt = stark.Transcript.init("zkml.quant.q4_0.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{
            .rows = bt.bound.rows,
            .columns = bt.bound.columns,
        }, sys.system(), CONFIG),
    );
}

test "quant: a raw Q4_0 value outside [0, 15] is refused before proving" {
    const a = testing.allocator;
    var case = try symmetricCase(a, Q4_0.width);
    defer case.deinit(a);

    var gemm = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm.deinit(a);

    var bad = case.q_a;
    bad[2] = 16;
    try testing.expectError(
        quant.BindError.QuantOutOfRange,
        Q4_0.bindOperands(a, &gemm, &bad, &case.scale_bits_a, &case.q_b, &case.scale_bits_b),
    );
}

test "quant: operands that are not the dequantization of their quant value are refused" {
    const a = testing.allocator;
    var case = try symmetricCase(a, Q4_0.width);
    defer case.deinit(a);

    var gemm = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm.deinit(a);

    var wrong: [k_macs]u16 = undefined;
    for (0..k_macs) |i| wrong[i] = 0x3C01; // fp16 1.0009765625
    try testing.expectError(
        quant.BindError.InconsistentOperands,
        Q4_0.bindOperands(a, &gemm, &case.q_a, &wrong, &case.q_b, &case.scale_bits_b),
    );
}

test "quant: a fabricated scale no longer proves" {
    const a = testing.allocator;
    var sys = try Q4_0.buildSystem(a, k_macs);
    defer sys.deinit();

    // The gap this test used to mark: a scale no fp16 could produce, with
    // every other constraint satisfied, produced a valid proof. The scale is
    // now pinned to the image of fp16ToFixedQ4_22 by scale_air, so the
    // binder cannot even BUILD such a trace: the fp16 pattern has to exist.
    const operands_a = try a.alloc(Goldilocks, k_macs);
    defer a.free(operands_a);
    const operands_b = try a.alloc(Goldilocks, k_macs);
    defer a.free(operands_b);

    var case = Case{
        .a = operands_a,
        .b = operands_b,
        .q_a = undefined,
        .q_b = undefined,
        .scale_a = undefined,
        .scale_b = undefined,
        .scale_bits_a = undefined,
        .scale_bits_b = undefined,
        .c_true = Goldilocks.zero,
    };
    const eight = Goldilocks.fromU64(8);
    const fake = Goldilocks.fromU64(1234567);
    for (0..k_macs) |i| {
        const qa = Goldilocks.fromU64(rawQ4(i));
        case.q_a[i] = rawQ4(i);
        case.q_b[i] = rawQ4(i);
        // A real fp16 pattern, so it is accepted...
        case.scale_bits_a[i] = 0x3C00;
        case.scale_bits_b[i] = 0x3C00;
        // ...but the operands are built with the fabricated scale, so the
        // dequantization equation does not hold for what the pattern gives.
        case.scale_a[i] = try quant.scaleFromFp16(0x3C00);
        case.scale_b[i] = case.scale_a[i];
        case.a[i] = qa.sub(eight).mul(fake);
        case.b[i] = qa.sub(eight).mul(fake);
        case.c_true = case.c_true.add(case.a[i].mul(case.b[i]));
    }

    var gemm = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm.deinit(a);
    // bindOperands refuses: the operands are not the dequantization of the
    // scale the pattern produces.
    try testing.expectError(
        quant.BindError.InconsistentOperands,
        Q4_0.bindOperands(a, &gemm, &case.q_a, &case.scale_bits_a, &case.q_b, &case.scale_bits_b),
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
        Q4_0.bindOperands(a, &gemm2, &case.q_a, &subnormal, &case.q_b, &case.scale_bits_b),
    );
    var gemm3 = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm3.deinit(a);
    try testing.expectError(
        quant.BindError.BadScale,
        Q4_0.bindOperands(a, &gemm3, &case.q_a, &too_big, &case.q_b, &case.scale_bits_b),
    );
    var gemm4 = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer gemm4.deinit(a);
    try testing.expectError(
        quant.BindError.BadScale,
        Q4_0.bindOperands(a, &gemm4, &case.q_a, &infinite, &case.q_b, &case.scale_bits_b),
    );
}

test "quant: a 4095-MAC reduction (4096 rows) proves and verifies" {
    // Regression: fri.verify used to carry a fixed [4096]Fp2 stack buffer
    // for the residual, so any proof whose final FRI domain exceeded 2^12
    // was rejected as InvalidProof — an HONEST proof, silently. This
    // unpadded reduction puts the final domain at 2^13. The residual is now
    // evaluated with the FFT, so this both passes and stops dominating verify time.
    const a = testing.allocator;
    const k: usize = 4095;
    var sys = try Q4_0.buildSystem(a, k);
    defer sys.deinit();

    const av = try a.alloc(Goldilocks, k);
    defer a.free(av);
    const bv = try a.alloc(Goldilocks, k);
    defer a.free(bv);
    const qs_a = try a.alloc(u8, k);
    defer a.free(qs_a);
    const qs_b = try a.alloc(u8, k);
    defer a.free(qs_b);
    const sc_bits_a = try a.alloc(u16, k);
    defer a.free(sc_bits_a);
    const sc_bits_b = try a.alloc(u16, k);
    defer a.free(sc_bits_b);

    // Many symmetric Q4 blocks share the fp16 scale 1.0, so the reduction
    // spans block boundaries and the per-MAC scale array is exercised.
    const block = blockWithRamp();
    const deq = tensor.dequantQ4K(&block, 0x3C00) catch unreachable;
    const scale_bits: u16 = 0x3C00; // fp16 1.0
    var c = Goldilocks.zero;
    for (0..k) |i| {
        av[i] = deq[i % 256];
        bv[i] = deq[i % 256];
        qs_a[i] = rawQ4(i % 256);
        qs_b[i] = rawQ4(i % 256);
        sc_bits_a[i] = scale_bits;
        sc_bits_b[i] = scale_bits;
        c = c.add(av[i].mul(bv[i]));
    }

    var gemm = try gemm_air.buildTrace(a, av, bv, c);
    defer gemm.deinit(a);
    var bound = try Q4_0.bindOperands(a, &gemm, qs_a, sc_bits_a, qs_b, sc_bits_b);
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
    var sys = try Q4_0.buildSystem(a, k_macs);
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
    var case = try symmetricCase(a, Q4_0.width);
    defer case.deinit(a);
    try testing.expectEqual(ok_bits, case.scale_bits_a[0]);
    try testing.expect(scale.eql(case.scale_a[0]));

    var trace = try gemm_air.buildTrace(a, case.a, case.b, case.c_true);
    defer trace.deinit(a);
    var bound = try Q4_0.bindOperands(a, &trace, &case.q_a, &case.scale_bits_a, &case.q_b, &case.scale_bits_b);
    defer bound.deinit(a);

    var pt = stark.Transcript.init("zkml.quant.q4_0.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = bound.rows,
        .columns = bound.columns,
    }, sys.system(), CONFIG);
    defer proof.deinit(a);

    var vt = stark.Transcript.init("zkml.quant.q4_0.v1");
    try testing.expect(try stark.verify(&vt, &proof, sys.system(), CONFIG));
}

test "quant: tampering with a scale's provenance witness is rejected" {
    const a = testing.allocator;
    var sys = try Q4_0.buildSystem(a, k_macs);
    defer sys.deinit();
    var case = try symmetricCase(a, Q4_0.width);
    defer case.deinit(a);

    var bt = try boundTrace(Q4_0, a, &case);
    defer bt.gemm.deinit(a);
    defer bt.bound.deinit(a);

    // The honest trace proves and verifies, so the negatives below are not
    // passing because everything is rejected.
    var pt = stark.Transcript.init("zkml.quant.q4_0.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = bt.bound.rows,
        .columns = bt.bound.columns,
    }, sys.system(), CONFIG);
    defer proof.deinit(a);
    var vt = stark.Transcript.init("zkml.quant.q4_0.v1");
    try testing.expect(try stark.verify(&vt, &proof, sys.system(), CONFIG));

    // Claim a different shift. The scale column no longer matches the
    // (1024 + m)·2^s that the selector and the mantissa produce: this is
    // the attack the gadget exists to stop, and it is caught by the AIR
    // rather than by the binder refusing to build the trace.
    const bad = try a.dupe(Fp2, bt.bound.columns[Q4_0.col_scale_a]);
    defer a.free(bad);
    bad[0] = bad[0].add(Fp2.one);
    const cols = try a.alloc([]const Fp2, Q4_0.column_count);
    defer a.free(cols);
    for (bt.bound.columns, 0..) |c, i| cols[i] = c;
    cols[Q4_0.col_scale_a] = bad;

    var pt2 = stark.Transcript.init("zkml.quant.q4_0.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt2, .{
            .rows = bt.bound.rows,
            .columns = cols,
        }, sys.system(), CONFIG),
    );

    // And a non-boolean selector bit is refused too: with two bits set, M
    // would no longer be a power of two even if the sum constraint were
    // dropped.
    const sel_bad = try a.dupe(Fp2, bt.bound.columns[Q4_0.gadget_a.sel_base]);
    defer a.free(sel_bad);
    sel_bad[0] = Fp2.re(Goldilocks.fromU64(3));
    const cols2 = try a.alloc([]const Fp2, Q4_0.column_count);
    defer a.free(cols2);
    for (bt.bound.columns, 0..) |c, i| cols2[i] = c;
    cols2[Q4_0.gadget_a.sel_base] = sel_bad;

    var pt3 = stark.Transcript.init("zkml.quant.q4_0.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt3, .{
            .rows = bt.bound.rows,
            .columns = cols2,
        }, sys.system(), CONFIG),
    );
}
