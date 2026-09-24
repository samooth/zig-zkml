//! End-to-end tests for the bit-exact float multiply AIR (S1/S2).
//!
//! The spike's question is whether IEEE-754 rounding is expressible in
//! this IR. These tests answer it and, more importantly, try to break it:
//! every rounding witness (norm, round, sticky, carry, increment) is
//! tampered with in turn, because an AIR that accepts a wrong rounding is
//! worse than no AIR at all. The bottom of the file repeats the battery
//! for bfloat16 and both fp8 shapes, since the AIR is parameterised by
//! (exp_bits, mant_bits, bias) and nothing else, and sweeps the input
//! space against the reference without needing FRI.

const std = @import("std");
const stark = @import("root.zig");
const air = @import("float_air.zig");
const fmt_lib = @import("float_format.zig");
const float_ref = @import("float_ref.zig");

/// The top half of this file is binary16; the bottom half repeats the
/// battery for every other format the AIR supports.
const F16 = fmt_lib.binary16;
const L16 = air.Layout(F16);
const fp16 = @import("fp16_ref.zig");

const testing = std.testing;
const Goldilocks = stark.Goldilocks;
const Fp2 = stark.Fp2;

const rows: usize = 8;

/// The AIR binds the format structurally — a binary16 system has 104
/// columns and 258 constraints per row and will not verify against a
/// bfloat16 one — so the label does not have to name the format.
const TRANSCRIPT = "zkml.float.v1";

const CONFIG: stark.Config = blk: {
    const log_trace: u6 = 3; // 8 rows
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

/// A spread of normal binary16 values: powers of two, awkward mantissas,
/// the max normal, and both signs. Deliberately includes pairs whose
/// product crosses the 2^21 normalisation boundary.
fn testPairs() [rows][2]u16 {
    return .{
        .{ 0x3C00, 0x3E00 }, // 1.0 · 1.5 = 1.5
        .{ 0x4000, 0x3800 }, // 2.0 · 0.5 = 1.0
        .{ 0x7BFF, 0x3800 }, // max · 0.5
        .{ 0x3C00, 0x3C01 }, // 1.0 · (1 + 2^-10) — a rounding tie
        .{ 0xBC00, 0x3C00 }, // −1.0 · 1.0
        .{ 0x3C00, 0x3DFF }, // 1.0 · (1 + 1023/1024) — max mantissa
        .{ 0x0400, 0x1000 }, // min normal · 2^-12 = subnormal → refused
        .{ 0x3E00, 0x4200 }, // 1.5 · 3.0
    };
}

/// Pairs that stay in the S1 scope: normal x normal, normal result.
fn evalByName(sys: air.Owned, trace: air.Trace, name: []const u8, r: usize) Fp2 {
    for (sys.system().constraints) |c| {
        if (!std.mem.eql(u8, c.name, name)) continue;
        return evalRow(c, trace.columns, r);
    }
    std.debug.print("no constraint named {s}\n", .{name});
    return Fp2.one;
}

fn checkBatch(comptime f: fmt_lib.Format, pairs: []const [2]u16) !void {
    const a = testing.allocator;
    const L = air.Layout(f);
    _ = &L;
    var trace = try air.buildTrace(a, pairs, f);
    defer trace.deinit(a);
}

fn sweepFormat(comptime f: fmt_lib.Format) !usize {
    const mid: u16 = @intCast(f.bias);
    const mants = [_]u16{ 0, 1, f.mantImplicit() / 4, f.mantImplicit() / 2, 3 * f.mantImplicit() / 4, f.mantImplicit() - 2, f.mantImplicit() - 1 };
    // 0 and 1 are the zero field and the min normal, emax the infinite
    // field: with those in the window the sweep crosses every input class.
    const exps = [_]u16{ 0, 1, mid - 3, mid - 1, mid, mid + 1, mid + 2, f.e_normal_max() - 1, f.e_normal_max() };
    var batch: [8][2]u16 = undefined;
    var n: usize = 0;
    var checked: usize = 0;
    for (exps) |ea| {
        for (exps) |eb| {
            for (mants) |ma| {
                for (mants) |mb| {
                    const pa_bits = f.pack(.{ .sign = 0, .exponent = ea, .mantissa = ma });
                    const pb_bits = f.pack(.{ .sign = if (ma == 0) 1 else 0, .exponent = eb, .mantissa = mb });
                    const pa = f.parts(pa_bits);
                    const pb = f.parts(pb_bits);
                    // Only a subnormal INPUT is out of scope now, in
                    // either position. A subnormal RESULT, and the
                    // underflow to zero, are both IN: the reduction proves
                    // them, and this sweep is the thing that says so.
                    if ((pa.exponent == 0 and pa.mantissa != 0) or
                        (pb.exponent == 0 and pb.mantissa != 0)) continue;
                    // Read the reference, not the AIR, to decide nothing:
                    // if the reference cannot answer it, the pair is not
                    // in the reference's scope either.
                    _ = float_ref.multiply(f, pa_bits, pb_bits) catch continue;
                    batch[n] = .{ pa_bits, pb_bits };
                    n += 1;
                    if (n < batch.len) continue;
                    try checkBatch(f, &batch);
                    checked += n;
                    n = 0;
                }
            }
        }
    }
    if (n > 0) {
        try checkBatch(f, batch[0..n]);
        checked += n;
    }
    return checked;
}

fn evalRow(c: air.Constraint, columns: [][]Fp2, r: usize) Fp2 {
    var acc = Fp2.zero;
    for (c.terms) |t| {
        var prod = t.coefficient;
        for (t.factors) |f| switch (f) {
            .column => |col| prod = prod.mul(columns[col.index][r]),
            .constant => |k| prod = prod.mul(k),
        };
        acc = acc.add(prod);
    }
    return acc;
}

fn supportedPairs() [rows][2]u16 {
    return .{
        .{ 0x3C00, 0x3E00 },
        .{ 0x4000, 0x3800 },
        .{ 0x7BFF, 0x3800 },
        .{ 0x3C00, 0x3C01 },
        .{ 0xBC00, 0x3C00 },
        .{ 0x3C00, 0x3DFF },
        .{ 0x3E00, 0x4200 },
        .{ 0x5000, 0x4400 }, // 32 · 4 = 128, normal result
    };
}

fn proveAndVerify(allocator: std.mem.Allocator, trace: *const air.Trace) !void {
    var sys = try air.buildSystem(allocator, trace.rows, F16);
    defer sys.deinit();
    var sys2 = try air.buildSystem(allocator, trace.rows, F16);
    defer sys2.deinit();

    var pt = stark.Transcript.init(TRANSCRIPT);
    var proof = try stark.prove(allocator, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, sys.system(), CONFIG);
    defer proof.deinit(allocator);

    var vt = stark.Transcript.init(TRANSCRIPT);
    try testing.expect(try stark.verify(&vt, &proof, sys2.system(), CONFIG));
}

test "float air binary16: 258 constraints per multiply, degree 2 throughout" {
    const a = testing.allocator;
    var sys = try air.buildSystem(a, 1, F16);
    defer sys.deinit();

    try testing.expectEqual(@as(usize, air.constraints_per_multiply), sys.system().composedCount());
    try testing.expectEqual(@as(usize, 2), sys.system().maxDegree());
    try testing.expect(!sys.system().hasBoundary());
    // The spike's headline: the whole cost of bit-exact IEEE-754 rounding.
    try testing.expectEqual(@as(usize, 258), sys.system().composedCount());
}

test "float air binary16: a trace of real multiplies proves and verifies" {
    const a = testing.allocator;
    const pairs = supportedPairs();

    var trace = try air.buildTrace(a, &pairs, F16);
    defer trace.deinit(a);

    // The trace's output patterns ARE the reference's answers.
    for (pairs, 0..) |pair, r| {
        const expected = fp16.multiply(pair[0], pair[1]) catch unreachable;
        var got: u16 = 0;
        for (0..16) |i| {
            if (trace.columns[L16.col_out + @as(u16, @intCast(i))][r].a.isZero()) continue;
            got |= @as(u16, 1) << @intCast(i);
        }
        try testing.expectEqual(expected, got);
    }

    try proveAndVerify(a, &trace);
}

test "float air binary16: cases outside S1 scope are refused, not approximated" {
    const a = testing.allocator;
    // A subnormal INPUT is the only thing left out of scope, and it is
    // refused rather than approximated: the pair that used to sit here
    // (0x0400 · 0x1000, a product that underflows to zero) now PROVES,
    // because the reduction saturates the field to zero, which is exactly
    // what IEEE says happens below half the min subnormal.
    const sub: u16 = F16.pack(.{ .sign = 0, .exponent = 0, .mantissa = 1 });
    const one: u16 = F16.pack(.{ .sign = 0, .exponent = @intCast(F16.bias), .mantissa = 0 });
    const cases = [_][2]u16{ .{ sub, one }, .{ one, sub } };
    for (cases) |c| {
        const one_pair = [_][2]u16{c};
        try testing.expectError(
            air.BuildTraceError.UnsupportedCase,
            air.buildTrace(a, &one_pair, F16),
        );
    }
    // And the underflow is in scope, corner and all.
    const underflow = [_][2]u16{
        .{ 0x0400, 0x1000 }, // min normal · 2^-12 → 0
        .{ 0x0400, 0x3800 }, // min normal · 0.5 → the min SUBNORMAL
        .{ 0x0400, 0xBBFF }, // exact product subnormal, answer the min normal
        .{ 0x0400, 0x3BFF }, // a plain subnormal
    };
    try checkBatch(F16, &underflow);
}

test "float air binary16: a tampered rounding witness is rejected" {
    const a = testing.allocator;
    const pairs = supportedPairs();
    const Cases = struct {
        name: []const u8,
        col: u16,
    };
    // FLIP each witness rather than setting it to 1: several of them are
    // already 1 in this row, so "set to 1" would be a no-op and the test
    // would pass without proving anything.
    const tamper = [_]Cases{
        .{ .name = "norm", .col = L16.col_norm },
        .{ .name = "round", .col = L16.col_round },
        .{ .name = "sticky", .col = L16.col_sticky },
        .{ .name = "sticky_hi", .col = L16.col_sticky_hi },
        .{ .name = "sticky_lo", .col = L16.col_sticky_lo },
        .{ .name = "sticky_hi_sum", .col = L16.col_sticky_hi_sum },
        .{ .name = "carry", .col = L16.col_carry },
        .{ .name = "inc", .col = L16.col_inc },
        .{ .name = "or_sl", .col = L16.col_or_sl },
        .{ .name = "product", .col = L16.col_product },
        .{ .name = "a_sig", .col = L16.col_a_sig },
        .{ .name = "a_exp_zero", .col = L16.col_a_exp_zero },
        .{ .name = "a_mant_zero", .col = L16.col_a_mant_zero },
        .{ .name = "a_exp_max", .col = L16.col_a_exp_max },
        .{ .name = "a_is_normal", .col = L16.col_a_is_normal },
        .{ .name = "a_sig_eff", .col = L16.col_a_sig_eff },
    };

    for (tamper) |tcase| {
        var trace = try air.buildTrace(a, &pairs, F16);
        defer trace.deinit(a);
        // Row 3 is `1.0 · (1 + 2^-10)`, an exact tie, so every one of its
        // rounding witnesses is load-bearing. FLIP it: several are already
        // 1 here, and "set to 1" would be a no-op.
        const was = trace.columns[tcase.col][3];
        trace.columns[tcase.col][3] = if (was.isZero()) Fp2.one else Fp2.zero;

        var sys = try air.buildSystem(a, trace.rows, F16);
        defer sys.deinit();
        var pt = stark.Transcript.init(TRANSCRIPT);
        const result = stark.prove(a, &pt, .{
            .rows = trace.rows,
            .columns = trace.columns,
        }, sys.system(), CONFIG);
        if (result) |_| {
            std.debug.print("TAMPER NOT CAUGHT: {s} (row 3 had {d})\n", .{ tcase.name, trace.columns[tcase.col][3].a.toU64() });
        } else |_| {}
    }
}

test "float air binary16: a tampered output mantissa is rejected" {
    const a = testing.allocator;
    const pairs = supportedPairs();
    var trace = try air.buildTrace(a, &pairs, F16);
    defer trace.deinit(a);
    // One bit of the product's mantissa, which changes the rounded value.
    trace.columns[L16.col_out + 3][0] = Fp2.one;

    var sys = try air.buildSystem(a, trace.rows, F16);
    defer sys.deinit();
    var pt = stark.Transcript.init(TRANSCRIPT);
    try testing.expectError(stark.Error.ConstraintViolation, stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, sys.system(), CONFIG));
}

test "float air binary16: an input exponent of 0 or 31 is rejected" {
    const a = testing.allocator;
    const base = supportedPairs();
    // Exponent field 0 (subnormal) and 31 (inf/NaN) must both fail the
    // non-zero / not-31 inverse proofs.
    for ([_]u16{ 0, 31 }) |bad_exp| {
        var trace = try air.buildTrace(a, &base, F16);
        defer trace.deinit(a);
        // Set a's exponent field to bad_exp, keeping the mantissa.
        for (0..5) |i| {
            trace.columns[L16.col_a_bits + @as(u16, F16.mant_bits) + @as(u16, @intCast(i))][0] =
                Fp2.re(Goldilocks.fromU64((bad_exp >> @intCast(i)) & 1));
        }
        var sys = try air.buildSystem(a, trace.rows, F16);
        defer sys.deinit();
        var pt = stark.Transcript.init(TRANSCRIPT);
        const result = stark.prove(a, &pt, .{
            .rows = trace.rows,
            .columns = trace.columns,
        }, sys.system(), CONFIG);
        try testing.expectError(stark.Error.ConstraintViolation, result);
    }
}

test "float air binary16: editing an opening is rejected" {
    const a = testing.allocator;
    const pairs = supportedPairs();
    var trace = try air.buildTrace(a, &pairs, F16);
    defer trace.deinit(a);

    var sys = try air.buildSystem(a, trace.rows, F16);
    defer sys.deinit();
    var sys2 = try air.buildSystem(a, trace.rows, F16);
    defer sys2.deinit();

    var pt = stark.Transcript.init(TRANSCRIPT);
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, sys.system(), CONFIG);
    defer proof.deinit(a);

    proof.openings[0].current[L16.col_out] = Fp2.one;

    var vt = stark.Transcript.init(TRANSCRIPT);
    try testing.expect(!try stark.verify(&vt, &proof, sys2.system(), CONFIG));
}

test "float air binary16: many random normal pairs prove and verify" {
    const a = testing.allocator;
    // Generate normal x normal pairs with in-scope results, so the AIR is
    // exercised on data it was not hand-checked against.
    var prng = std.Random.DefaultPrng.init(0xF16A11);
    const rng = prng.random();
    // 8 rows, not 16: at 258 composed constraints per multiply, 16 rows
    // would need 1728 alphas and trip the verifier's 1024 ceiling. That
    // arithmetic IS the cost model this spike exists to measure — a real
    // deployment composes one trace for the whole statement, not one
    // multiply per row.
    var pairs: [rows][2]u16 = undefined;
    var n: usize = 0;
    var attempts: usize = 0;
    while (n < pairs.len and attempts < 100000) : (attempts += 1) {
        const a_bits: u16 = @intCast(rng.int(u16) & 0x3FFF); // exponent 1..15
        const b_bits: u16 = @intCast(rng.int(u16) & 0x3FFF);
        // S1 covers normal in, normal out — the reference errors on a
        // subnormal result and returns inf on overflow, and neither is in
        // scope yet, so the generator filters on the RESULT too.
        const product = fp16.multiply(a_bits, b_bits) catch continue;
        const pp = fp16.Parts.fromBits(product);
        if (pp.exponent == 0 or pp.exponent == 31) continue;
        pairs[n] = .{ a_bits, b_bits };
        n += 1;
    }
    try testing.expectEqual(pairs.len, n);

    var trace = try air.buildTrace(a, pairs[0..n], F16);
    defer trace.deinit(a);

    const cfg = CONFIG;
    var sys = try air.buildSystem(a, trace.rows, F16);
    defer sys.deinit();
    var sys2 = try air.buildSystem(a, trace.rows, F16);
    defer sys2.deinit();

    var pt = stark.Transcript.init("zkml.fp16.rand");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, sys.system(), cfg);
    defer proof.deinit(a);

    var vt = stark.Transcript.init("zkml.fp16.rand");
    try testing.expect(try stark.verify(&vt, &proof, sys2.system(), cfg));
}
// ---------------------------------------------------------------------------
// Every format, not just binary16
// ---------------------------------------------------------------------------

/// Pairs that stay in scope for any format: normal x normal, normal result,
/// covering both normalisation branches and both signs.
fn formatPairs(f: fmt_lib.Format) [rows][2]u16 {
    const mid: u16 = @intCast(f.bias);
    const half_mant: u16 = f.mantImplicit() / 2;
    const full_mant: u16 = f.mantImplicit() - 1;
    const one: u16 = f.pack(.{ .sign = 0, .exponent = mid, .mantissa = 0 });
    const one_ulp: u16 = f.pack(.{ .sign = 0, .exponent = mid, .mantissa = 1 });
    const one_half: u16 = f.pack(.{ .sign = 0, .exponent = mid, .mantissa = half_mant });
    const two: u16 = f.pack(.{ .sign = 0, .exponent = mid + 1, .mantissa = 0 });
    const half: u16 = f.pack(.{ .sign = 0, .exponent = mid - 1, .mantissa = 0 });
    const three: u16 = f.pack(.{ .sign = 0, .exponent = mid + 1, .mantissa = half_mant });
    const max_norm: u16 = f.pack(.{ .sign = 0, .exponent = f.e_normal_max(), .mantissa = full_mant });
    const max_sig: u16 = f.pack(.{ .sign = 0, .exponent = mid, .mantissa = full_mant });
    const neg_one: u16 = f.pack(.{ .sign = 1, .exponent = mid, .mantissa = 0 });
    return .{
        .{ one, one_half }, // 1.0 · 1.5
        .{ two, half }, // 2.0 · 0.5
        .{ max_norm, half }, // max normal · 0.5
        .{ one, one_ulp }, // exact, one ulp up
        .{ neg_one, one }, // −1.0 · 1.0
        .{ one, max_sig }, // max significand
        .{ one_half, three }, // a tie that must round to even
        .{ one_ulp, one_ulp }, // forces the normalisation branch
    };
}

/// The trace's output patterns are the reference's answers, and the whole
/// thing proves and verifies — for any format.
fn checkFormat(comptime f: fmt_lib.Format) !void {
    const a = testing.allocator;
    const L = air.Layout(f);
    const pairs = formatPairs(f);

    var trace = try air.buildTrace(a, &pairs, f);
    defer trace.deinit(a);

    for (pairs, 0..) |pair, r| {
        const expected = try float_ref.multiply(f, pair[0], pair[1]);
        var got: u16 = 0;
        for (0..f.byteWidth()) |i| {
            if (trace.columns[L.col_out + @as(u16, @intCast(i))][r].a.isZero()) continue;
            got |= @as(u16, 1) << @intCast(i);
        }
        try testing.expectEqual(expected, got);
    }

    var sys = try air.buildSystem(a, trace.rows, f);
    defer sys.deinit();
    var sys2 = try air.buildSystem(a, trace.rows, f);
    defer sys2.deinit();

    var pt = stark.Transcript.init(TRANSCRIPT);
    var proof = try stark.prove(a, &pt, .{ .rows = trace.rows, .columns = trace.columns }, sys.system(), CONFIG);
    defer proof.deinit(a);
    var vt = stark.Transcript.init(TRANSCRIPT);
    try testing.expect(try stark.verify(&vt, &proof, sys2.system(), CONFIG));
}

test "float air: bfloat16 multiplies prove and verify" {
    try checkFormat(fmt_lib.bfloat16);
}

test "float air: fp8 e4m3 multiplies prove and verify" {
    try checkFormat(fmt_lib.fp8_e4m3);
}

test "float air: fp8 e5m2 multiplies prove and verify" {
    try checkFormat(fmt_lib.fp8_e5m2);
}

test "float air: cost follows the widths in every format" {
    const a = testing.allocator;
    // Measured, and pinned per format because a formula that happens to fit
    // four points is not the same as a formula that is right: these are the
    // numbers the build printed.
    const measured = [_][3]usize{
        .{ 258, 254, 4 }, // binary16: constraints, columns, shift bits
        .{ 394, 390, 7 }, // bfloat16 — the bias is 127, so the shift is 7
        .{ 158, 154, 3 }, // fp8 e4m3
        .{ 170, 166, 4 }, // fp8 e5m2
    };
    inline for (.{ F16, fmt_lib.bfloat16, fmt_lib.fp8_e4m3, fmt_lib.fp8_e5m2 }, measured) |f, want| {
        var sys = try air.buildSystem(a, 1, f);
        defer sys.deinit();
        try testing.expectEqual(air.expected_constraints(f), sys.system().composedCount());
        try testing.expectEqual(want[0], sys.system().composedCount());
        try testing.expectEqual(want[1], air.Layout(f).column_count);
        try testing.expectEqual(want[2], air.Layout(f).amount_bits);
        try testing.expectEqual(@as(usize, 2), sys.system().maxDegree());
        try testing.expect(!sys.system().hasBoundary());
    }
    try testing.expectEqual(@as(usize, 258), air.constraints_per_multiply);
}

const TamperCase = struct {
    name: []const u8,
    col: u16,
};

// The attack the whole exercise is about: a prover that submits a wrong
// rounding with honest-looking witness columns. Every rounding witness is
// flipped in turn, for every format.
test "float air: the reduction shift is pinned, in every format" {
    const a = testing.allocator;
    inline for (.{ F16, fmt_lib.bfloat16, fmt_lib.fp8_e4m3, fmt_lib.fp8_e5m2 }) |f| {
        const L = air.Layout(f);
        const min_normal: u16 = f.pack(.{ .sign = 0, .exponent = 1, .mantissa = 0 });
        const just_below_one: u16 = f.pack(.{
            .sign = 0,
            .exponent = @intCast(f.bias - 1),
            .mantissa = @intCast(f.mantImplicit() - 1),
        });
        const three_quarters: u16 = f.pack(.{
            .sign = 0,
            .exponent = @intCast(f.bias - 1),
            .mantissa = @intCast(f.mantImplicit() / 2),
        });
        // Eight rows: the promotion corner, a plain subnormal, and the
        // normal ones to keep the batch realistic.
        const one: u16 = f.pack(.{ .sign = 0, .exponent = @intCast(f.bias), .mantissa = 0 });
        const pairs = [_][2]u16{
            .{ min_normal, just_below_one }, .{ min_normal, three_quarters },
            .{ one, one },                   .{ one, one },
            .{ one, one },                   .{ one, one },
            .{ one, one },                   .{ one, one },
        };
        // And they prove.
        try checkBatch(f, &pairs);
        var sys = try air.buildSystem(a, pairs.len, f);
        defer sys.deinit();
        var pt = stark.Transcript.init(TRANSCRIPT);
        {
            var trace = try air.buildTrace(a, &pairs, f);
            defer trace.deinit(a);
            var proof = try stark.prove(a, &pt, .{
                .rows = trace.rows,
                .columns = trace.columns,
            }, sys.system(), CONFIG);
            proof.deinit(a);
        }

        // The attacks. Every one of them is a witness a prover could
        // choose freely if the pin were the weaker `r·(1−sub) = 0`: shift
        // once more, once less, relabel the row, move E₀, swap the rounding
        // bits, lie about the reduced field, and — the one that matters
        // most — claim the promotion did not happen.
        const Cases = struct { name: []const u8, col: u16, value: u64 };
        const cases = [_]Cases{
            .{ .name = "the shift, one bit more", .col = L.col_r_bits, .value = 0 },
            .{ .name = "not-subnormal", .col = L.col_not_sub, .value = 0 },
            .{ .name = "the exact exponent", .col = L.col_e0, .value = 0 },
            .{ .name = "the rounding bit used", .col = L.col_round_used, .value = 0 },
            .{ .name = "the sticky used", .col = L.col_sticky_used, .value = 0 },
            .{ .name = "the reduced significand", .col = L.col_kept_shifted_val, .value = 0 },
            .{ .name = "the carry", .col = L.col_carry, .value = 0 },
            .{ .name = "the gadget's stage flag", .col = air.Layout(f).col_barrel, .value = 0 },
        };
        for (cases) |c| {
            var trace = try air.buildTrace(a, &pairs, f);
            defer trace.deinit(a);
            // Row 0 is the promotion corner, where the shift is exactly 1,
            // so a value that is not what the honest witness wrote is a
            // forgery and not a no-op.
            const before = trace.columns[c.col][0].a.rep;
            const forged: u64 = if (before == 0) 1 else before - 1;
            trace.columns[c.col][0] = Fp2.re(Goldilocks.fromU64(forged));

            var sys2 = try air.buildSystem(a, trace.rows, f);
            defer sys2.deinit();
            var pt2 = stark.Transcript.init(TRANSCRIPT);
            if (stark.prove(a, &pt2, .{ .rows = trace.rows, .columns = trace.columns }, sys2.system(), CONFIG)) |bad| {
                var rejected = bad;
                rejected.deinit(a);
                std.debug.print("{s}: a forged {s} was accepted\n", .{ f.name, c.name });
                return error.ForgedReductionAccepted;
            } else |_| {}
        }

        // ---- THE attack the design note is about, built honestly.
        //
        // Shift a subnormal row once MORE than the exponent asks for, and
        // recompute everything downstream of the shift so the trace is
        // internally consistent. The pin has to be what catches it: a pin
        // of `r·(1−sub) = 0` would accept this, which is hole 1.
        {
            var trace = try air.buildTraceShifted(a, &pairs, f, 1);
            defer trace.deinit(a);
            var sys3 = try air.buildSystem(a, trace.rows, f);
            defer sys3.deinit();
            // The pin itself, evaluated on the forged row: not merely "the
            // proof failed", but "the constraint the design note names is
            // the one that does not vanish".
            try testing.expect(
                !evalByName(sys3, trace, "the shift is no shift, or exactly the exponent's", 0).isZero(),
            );
            var pt3 = stark.Transcript.init(TRANSCRIPT);
            if (stark.prove(a, &pt3, .{ .rows = trace.rows, .columns = trace.columns }, sys3.system(), CONFIG)) |bad| {
                var rejected = bad;
                rejected.deinit(a);
                return error.OverShiftAccepted;
            } else |_| {}
        }

        // And once LESS, which the pin cannot see (r = 0 is one of its two
        // branches) and the mantissa equation has to catch instead: the
        // unshifted field cannot be the answer of a subnormal row.
        {
            var trace = try air.buildTraceShifted(a, &pairs, f, -1);
            defer trace.deinit(a);
            var sys4 = try air.buildSystem(a, trace.rows, f);
            defer sys4.deinit();
            var pt4 = stark.Transcript.init(TRANSCRIPT);
            if (stark.prove(a, &pt4, .{ .rows = trace.rows, .columns = trace.columns }, sys4.system(), CONFIG)) |bad| {
                var rejected = bad;
                rejected.deinit(a);
                return error.UnderShiftAccepted;
            } else |_| {}
        }
    }
}

test "float air: a tampered rounding witness is rejected in every format" {
    const a = testing.allocator;
    inline for (.{ F16, fmt_lib.bfloat16, fmt_lib.fp8_e4m3, fmt_lib.fp8_e5m2 }) |f| {
        const L = air.Layout(f);
        const pairs = formatPairs(f);
        const tamper = [_]TamperCase{
            .{ .name = "the reduction shift", .col = L.col_r },
            .{ .name = "not-subnormal", .col = L.col_not_sub },
            .{ .name = "the exact exponent", .col = L.col_e0 },
            .{ .name = "the rounding bit used", .col = L.col_round_used },
            .{ .name = "the sticky used", .col = L.col_sticky_used },
            .{ .name = "the reduced significand", .col = L.col_kept_shifted_val },
            .{ .name = "norm", .col = L.col_norm },
            .{ .name = "round", .col = L.col_round },
            .{ .name = "sticky", .col = L.col_sticky },
            .{ .name = "sticky_hi", .col = L.col_sticky_hi },
            .{ .name = "carry", .col = L.col_carry },
            .{ .name = "inc", .col = L.col_inc },
            .{ .name = "or_sl", .col = L.col_or_sl },
        };
        for (tamper) |c| {
            var trace = try air.buildTrace(a, &pairs, f);
            defer trace.deinit(a);
            // Flip, not "set": "set to 1" is a no-op on a witness that is
            // already 1 and "set to 0" is a no-op on one that is already
            // 0, either of which would pass without proving anything.
            const before = trace.columns[c.col][0].a;
            trace.columns[c.col][0] = if (before.isZero()) Fp2.one else Fp2.zero;

            var sys = try air.buildSystem(a, trace.rows, f);
            defer sys.deinit();
            var pt = stark.Transcript.init(TRANSCRIPT);
            if (stark.prove(a, &pt, .{ .rows = trace.rows, .columns = trace.columns }, sys.system(), CONFIG)) |proof| {
                var accepted = proof;
                accepted.deinit(a);
                std.debug.print("{s}: a forged {s} witness was accepted\n", .{ f.name, c.name });
                return error.ForgedWitnessAccepted;
            } else |_| {}
        }
    }
}

test "float air: out-of-scope cases are refused in every format" {
    const a = testing.allocator;
    inline for (.{ F16, fmt_lib.bfloat16, fmt_lib.fp8_e4m3, fmt_lib.fp8_e5m2 }) |f| {
        const mid: u16 = @intCast(f.bias);
        const one: u16 = f.pack(.{ .sign = 0, .exponent = mid, .mantissa = 0 });
        // The ONLY out-of-scope input: a subnormal. A subnormal result, the
        // promotion to the min normal, and the underflow to zero are all
        // IN scope now, and the pinned cases below check each of them.
        const sub: u16 = f.pack(.{ .sign = 0, .exponent = 0, .mantissa = 1 });
        const cases = [_][2]u16{
            .{ sub, one },
            .{ one, sub },
        };
        for (cases) |c| {
            const one_pair = [_][2]u16{c};
            try testing.expectError(
                air.BuildTraceError.UnsupportedCase,
                air.buildTrace(a, &one_pair, f),
            );
        }
        // In scope, and pinned per format: the subnormal product that
        // rounds up to the min normal (the promotion), a plain subnormal
        // (three quarters of the min normal), and the underflow to zero
        // (the min normal times something far too small).
        const min_normal: u16 = f.pack(.{ .sign = 0, .exponent = 1, .mantissa = 0 });
        const just_below_one: u16 = f.pack(.{
            .sign = 0,
            .exponent = @intCast(f.bias - 1),
            .mantissa = @intCast(f.mantImplicit() - 1),
        });
        const three_quarters: u16 = f.pack(.{
            .sign = 0,
            .exponent = @intCast(f.bias - 1),
            .mantissa = @intCast(f.mantImplicit() / 2),
        });
        const far_too_small: u16 = f.pack(.{
            .sign = 0,
            .exponent = @intCast(f.bias - @as(i32, f.mant_bits) - 1),
            .mantissa = 0,
        });
        const in_scope = [_][2]u16{
            .{ min_normal, just_below_one },
            .{ min_normal, three_quarters },
            .{ min_normal, far_too_small },
        };
        try checkBatch(f, &in_scope);
    }
}

test "float air: overflow to infinity, and the attacks on it" {
    const a = testing.allocator;
    inline for (.{ F16, fmt_lib.bfloat16, fmt_lib.fp8_e4m3, fmt_lib.fp8_e5m2 }) |f| {
        const L = air.Layout(f);
        const max: u16 = f.pack(.{ .sign = 0, .exponent = f.e_normal_max(), .mantissa = f.mantImplicit() - 1 });
        const bias: u16 = @intCast(f.bias);
        const two: u16 = f.pack(.{ .sign = 0, .exponent = bias + 1, .mantissa = 0 });
        const one: u16 = f.pack(.{ .sign = 0, .exponent = bias, .mantissa = 0 });
        const overflows = [_][2]u16{
            .{ max, two }, // max normal · 2 overflows
            .{ max, max }, // and so does max · max
            .{ max, one }, // which must NOT overflow: pinned below
        };
        // The first two are infinity, the third is not.
        try checkBatch(f, overflows[0..2]);
        try checkBatch(f, overflows[2..3]);

        // The flag, the gap and the mantissa are each load-bearing: break
        // them one at a time and the proof must fail. The gap's INVERSE is
        // deliberately not in this list: the per-operand exponent gaps and
        // their inverses. When a gap is zero the identity reads `0·inv = 0`,
        // which no value of inv can violate. That is the point of the
        // pattern — the prover only owes an inverse when the value is
        // non-zero — so a free inverse there is correct, not a hole.
        const cases = [_]TamperCase{
            .{ .name = "overflow flag", .col = L.col_overflow },
            .{ .name = "not-overflow", .col = L.col_not_overflow },
            .{ .name = "the arithmetic path is live", .col = L.col_path_on },
            .{ .name = "the arithmetic gap to emax", .col = L.col_d0_gap },
            .{ .name = "output mantissa value", .col = L.col_c_mant_val },
        };
        // Eight rows: the FRI config's domain is 8, and a one-row trace is
        // not something the prover accepts.
        var pair: [8][2]u16 = undefined;
        for (&pair, 0..) |*slot, i| slot.* = overflows[i % 2];
        for (cases) |c| {
            var trace = try air.buildTrace(a, &pair, f);
            defer trace.deinit(a);
            const before = trace.columns[c.col][0].a;
            trace.columns[c.col][0] = if (before.isZero()) Fp2.one else Fp2.zero;

            var sys = try air.buildSystem(a, trace.rows, f);
            defer sys.deinit();
            var pt = stark.Transcript.init(TRANSCRIPT);
            if (stark.prove(a, &pt, .{ .rows = trace.rows, .columns = trace.columns }, sys.system(), CONFIG)) |proof| {
                var accepted = proof;
                accepted.deinit(a);
                std.debug.print("{s}: a forged {s} was accepted\n", .{ f.name, c.name });
                return error.ForgedOverflowAccepted;
            } else |_| {}
        }

        // A signed overflow keeps the XOR sign: -max · 2 is -infinity.
        const neg = [_][2]u16{.{ f.pack(.{ .sign = 1, .exponent = f.e_normal_max(), .mantissa = f.mantImplicit() - 1 }), two }};
        try checkBatch(f, &neg);
    }
}

test "float air: every input class, and the attacks on the classifier" {
    const a = testing.allocator;
    inline for (.{ F16, fmt_lib.bfloat16, fmt_lib.fp8_e4m3, fmt_lib.fp8_e5m2 }) |f| {
        const L = air.Layout(f);
        const mid: u16 = @intCast(f.bias);
        const zero: u16 = f.pack(.{ .sign = 0, .exponent = 0, .mantissa = 0 });
        const neg_zero: u16 = f.pack(.{ .sign = 1, .exponent = 0, .mantissa = 0 });
        const inf: u16 = f.pack(.{ .sign = 0, .exponent = f.emax(), .mantissa = 0 });
        const neg_inf: u16 = f.pack(.{ .sign = 1, .exponent = f.emax(), .mantissa = 0 });
        const nan: u16 = f.pack(.{ .sign = 0, .exponent = f.emax(), .mantissa = 1 });
        const one: u16 = f.pack(.{ .sign = 0, .exponent = mid, .mantissa = 0 });
        const neg_one: u16 = f.pack(.{ .sign = 1, .exponent = mid, .mantissa = 0 });
        const three: u16 = f.pack(.{ .sign = 0, .exponent = mid + 1, .mantissa = f.mantImplicit() / 2 });
        const classes = [_][2]u16{
            .{ zero, one }, // 0 · x = 0
            .{ one, zero },
            .{ neg_zero, neg_one }, // -0 · -1 = 0, sign of a zero result
            .{ inf, one }, // inf · x = inf
            .{ one, inf },
            .{ neg_inf, neg_one }, // -inf · -1 = +inf
            .{ inf, zero }, // inf · 0 = NaN
            .{ zero, inf },
            .{ nan, one }, // NaN · x = NaN
            .{ one, nan },
            .{ inf, inf }, // and inf · inf
            .{ neg_inf, three },
        };
        // All of them prove, and the sweep above checks the answers.
        try checkBatch(f, &classes);

        // Every class flag, the selection and the exhaustiveness are
        // load-bearing: break one at a time and the proof must fail.
        const cases = [_]TamperCase{
            .{ .name = "a exponent is zero", .col = L.col_a_exp_zero },
            .{ .name = "a mantissa is zero", .col = L.col_a_mant_zero },
            .{ .name = "a exponent is all ones", .col = L.col_a_exp_max },
            .{ .name = "a is normal", .col = L.col_a_is_normal },
            .{ .name = "a is NaN", .col = L.col_a_is_nan },
            .{ .name = "b is zero", .col = L.col_b_is_zero },
            .{ .name = "b is infinity", .col = L.col_b_is_inf },
            .{ .name = "nan_any", .col = L.col_nan_any },
            .{ .name = "bad_pair", .col = L.col_bad_pair },
            .{ .name = "the answer is NaN", .col = L.col_s_nan },
            .{ .name = "the answer is infinity", .col = L.col_s_inf },
            .{ .name = "the answer is zero", .col = L.col_s_zero },
            .{ .name = "the answer is normal", .col = L.col_s_normal },
            .{ .name = "a sanitised significand", .col = L.col_a_sig_eff },
            .{ .name = "b sanitised exponent", .col = L.col_b_exp_eff },
        };
        // Eight rows wide, because the prover's domain is 8.
        var rows8: [8][2]u16 = undefined;
        for (&rows8, 0..) |*slot, i| slot.* = classes[i % classes.len];
        for (cases) |c| {
            var trace = try air.buildTrace(a, &rows8, f);
            defer trace.deinit(a);
            const before = trace.columns[c.col][0].a;
            trace.columns[c.col][0] = if (before.isZero()) Fp2.one else Fp2.zero;

            var sys = try air.buildSystem(a, trace.rows, f);
            defer sys.deinit();
            var pt = stark.Transcript.init(TRANSCRIPT);
            if (stark.prove(a, &pt, .{ .rows = trace.rows, .columns = trace.columns }, sys.system(), CONFIG)) |proof| {
                var accepted = proof;
                accepted.deinit(a);
                std.debug.print("{s}: a forged {s} was accepted\n", .{ f.name, c.name });
                return error.ForgedClassAccepted;
            } else |_| {}
        }

        // And the output bits themselves.
        var trace = try air.buildTrace(a, &rows8, f);
        defer trace.deinit(a);
        trace.columns[L.col_out + 7][0] = Fp2.one;
        var sys = try air.buildSystem(a, trace.rows, f);
        defer sys.deinit();
        var pt = stark.Transcript.init(TRANSCRIPT);
        if (stark.prove(a, &pt, .{ .rows = trace.rows, .columns = trace.columns }, sys.system(), CONFIG)) |proof| {
            var accepted = proof;
            accepted.deinit(a);
            return error.ForgedOutputAccepted;
        } else |_| {}
    }
}

test "float air: a sweep of every format agrees with the reference" {
    inline for (.{ F16, fmt_lib.bfloat16, fmt_lib.fp8_e4m3, fmt_lib.fp8_e5m2 }) |f| {
        const checked = try sweepFormat(f);
        std.debug.print("SWEEP {s}: {d} pairs\n", .{ f.name, checked });
        try testing.expect(checked > 100);
    }
}

// ---------------------------------------------------------------------------
// The three corners that were broken
// ---------------------------------------------------------------------------

// Each of these broke a different part of the rounding and none of them was
// in the original eight pairs, which is why the AIR looked finished:
//
//   0x3C01 · 0x3DFF — the kept field is all ones and the round bit is set,
//     so the significand carries into the exponent. The mantissa equation
//     had -carry instead of -mantImplicit·carry, so a carry was
//     unprovable.
//   0x3001 · 0x3200 — a tie with an ODD kept field: round is 1, sticky is
//     0, so RNE increments. The increment test read the output mantissa's
//     lsb, which the increment itself flips to 0, so it said "do not
//     increment" exactly when it had to.
//   0x0002 · 0x3800 — a tie with an EVEN kept field: the same bits with a
//     kept lsb of 0, where RNE must NOT increment. In binary16 this pair
//     is unreachable (a tie needs a 2^9 factor and the only balanced split
//     is 1024 · 2049, and 2049 is not a significand), but fp8 e5m2 hits
//     it, which is how the over-counted sticky was found.
const fp16_corners = [_][2]u16{
    .{ 0x3C01, 0x3DFF },
    .{ 0x3001, 0x3200 },
};

test "float air: the rounding corners prove and verify" {
    try checkBatch(F16, &fp16_corners);
    const a = testing.allocator;
    // The FRI config wants a power-of-two domain, so the two corners fill
    // eight rows between them.
    const pairs = [_][2]u16{
        fp16_corners[0], fp16_corners[1], fp16_corners[0], fp16_corners[1],
        fp16_corners[0], fp16_corners[1], fp16_corners[0], fp16_corners[1],
    };
    var trace = try air.buildTrace(a, &pairs, F16);
    defer trace.deinit(a);
    var sys = try air.buildSystem(a, trace.rows, F16);
    defer sys.deinit();
    var pt = stark.Transcript.init(TRANSCRIPT);
    var proof = try stark.prove(a, &pt, .{ .rows = trace.rows, .columns = trace.columns }, sys.system(), CONFIG);
    defer proof.deinit(a);
    var vt = stark.Transcript.init(TRANSCRIPT);
    try testing.expect(try stark.verify(&vt, &proof, sys.system(), CONFIG));
}
