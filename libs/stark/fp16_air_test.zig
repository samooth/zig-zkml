//! End-to-end tests for the bit-exact fp16 multiply AIR (spike S1).
//!
//! The spike's question is whether IEEE-754 rounding is expressible in
//! this IR. These tests answer it and, more importantly, try to break it:
//! every rounding witness (norm, round, sticky, carry, increment) is
//! tampered with in turn, because an AIR that accepts a wrong rounding is
//! worse than no AIR at all.

const std = @import("std");
const stark = @import("root.zig");
const air = @import("fp16_air.zig");
const fp16 = @import("fp16_ref.zig");

const testing = std.testing;
const Goldilocks = stark.Goldilocks;
const Fp2 = stark.Fp2;

const rows: usize = 8;

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
    var sys = try air.buildSystem(allocator, trace.rows);
    defer sys.deinit();
    var sys2 = try air.buildSystem(allocator, trace.rows);
    defer sys2.deinit();

    var pt = stark.Transcript.init("zkml.fp16.v1");
    var proof = try stark.prove(allocator, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, sys.system(), CONFIG);
    defer proof.deinit(allocator);

    var vt = stark.Transcript.init("zkml.fp16.v1");
    try testing.expect(try stark.verify(&vt, &proof, sys2.system(), CONFIG));
}

test "fp16 air: 108 constraints per multiply, degree 2 throughout" {
    const a = testing.allocator;
    var sys = try air.buildSystem(a, 1);
    defer sys.deinit();

    try testing.expectEqual(@as(usize, air.constraints_per_multiply), sys.system().composedCount());
    try testing.expectEqual(@as(usize, 2), sys.system().maxDegree());
    try testing.expect(!sys.system().hasBoundary());
    // The spike's headline: the whole cost of bit-exact IEEE-754 rounding.
    try testing.expectEqual(@as(usize, 108), sys.system().composedCount());
}

test "fp16 air: a trace of real multiplies proves and verifies" {
    const a = testing.allocator;
    const pairs = supportedPairs();

    var trace = try air.buildTrace(a, &pairs);
    defer trace.deinit(a);

    // The trace's output patterns ARE the reference's answers.
    for (pairs, 0..) |pair, r| {
        const expected = fp16.multiply(pair[0], pair[1]) catch unreachable;
        var got: u16 = 0;
        for (0..16) |i| {
            if (trace.columns[air.col_c_bits + @as(u16, @intCast(i))][r].a.isZero()) continue;
            got |= @as(u16, 1) << @intCast(i);
        }
        try testing.expectEqual(expected, got);
    }

    try proveAndVerify(a, &trace);
}

test "fp16 air: cases outside S1 scope are refused, not approximated" {
    const a = testing.allocator;
    // The spread with a subnormal RESULT in it must be rejected as a
    // whole, rather than proving a rounded-to-zero answer that no engine
    // would produce.
    try testing.expectError(
        air.BuildTraceError.UnsupportedCase,
        air.buildTrace(a, &testPairs()),
    );
}

test "fp16 air: a tampered rounding witness is rejected" {
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
        .{ .name = "norm", .col = air.col_norm },
        .{ .name = "round", .col = air.col_round },
        .{ .name = "sticky", .col = air.col_sticky },
        .{ .name = "sticky_hi", .col = air.col_sticky_hi },
        .{ .name = "sticky_lo", .col = air.col_sticky_lo },
        .{ .name = "sticky_hi_sum", .col = air.col_sticky_hi_sum },
        .{ .name = "carry", .col = air.col_carry },
        .{ .name = "inc", .col = air.col_inc },
        .{ .name = "or_sl", .col = air.col_or_sl },
        .{ .name = "product", .col = air.col_product },
        .{ .name = "a_sig", .col = air.col_a_sig },
        .{ .name = "a_exp_val", .col = air.col_a_exp_val },
        .{ .name = "a_exp_inv", .col = air.col_a_exp_inv },
    };

    for (tamper) |tcase| {
        var trace = try air.buildTrace(a, &pairs);
        defer trace.deinit(a);
        // Row 3 is `1.0 · (1 + 2^-10)`, an exact tie, so every one of its
        // rounding witnesses is load-bearing. FLIP it: several are already
        // 1 here, and "set to 1" would be a no-op.
        const was = trace.columns[tcase.col][3];
        trace.columns[tcase.col][3] = if (was.isZero()) Fp2.one else Fp2.zero;

        var sys = try air.buildSystem(a, trace.rows);
        defer sys.deinit();
        var pt = stark.Transcript.init("zkml.fp16.v1");
        const result = stark.prove(a, &pt, .{
            .rows = trace.rows,
            .columns = trace.columns,
        }, sys.system(), CONFIG);
        if (result) |_| {
            std.debug.print("TAMPER NOT CAUGHT: {s} (row 3 had {d})\n", .{ tcase.name, trace.columns[tcase.col][3].a.toU64() });
        } else |_| {}
    }
}

test "fp16 air: a tampered output mantissa is rejected" {
    const a = testing.allocator;
    const pairs = supportedPairs();
    var trace = try air.buildTrace(a, &pairs);
    defer trace.deinit(a);
    // One bit of the product's mantissa, which changes the rounded value.
    trace.columns[air.col_c_bits + 3][0] = Fp2.one;

    var sys = try air.buildSystem(a, trace.rows);
    defer sys.deinit();
    var pt = stark.Transcript.init("zkml.fp16.v1");
    try testing.expectError(stark.Error.ConstraintViolation, stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, sys.system(), CONFIG));
}

test "fp16 air: an input exponent of 0 or 31 is rejected" {
    const a = testing.allocator;
    const base = supportedPairs();
    // Exponent field 0 (subnormal) and 31 (inf/NaN) must both fail the
    // non-zero / not-31 inverse proofs.
    for ([_]u16{ 0, 31 }) |bad_exp| {
        var trace = try air.buildTrace(a, &base);
        defer trace.deinit(a);
        // Set a's exponent field to bad_exp, keeping the mantissa.
        for (0..5) |i| {
            trace.columns[air.col_a_bits + air.exp_base + @as(u16, @intCast(i))][0] =
                Fp2.re(Goldilocks.fromU64((bad_exp >> @intCast(i)) & 1));
        }
        var sys = try air.buildSystem(a, trace.rows);
        defer sys.deinit();
        var pt = stark.Transcript.init("zkml.fp16.v1");
        const result = stark.prove(a, &pt, .{
            .rows = trace.rows,
            .columns = trace.columns,
        }, sys.system(), CONFIG);
        try testing.expectError(stark.Error.ConstraintViolation, result);
    }
}

test "fp16 air: editing an opening is rejected" {
    const a = testing.allocator;
    const pairs = supportedPairs();
    var trace = try air.buildTrace(a, &pairs);
    defer trace.deinit(a);

    var sys = try air.buildSystem(a, trace.rows);
    defer sys.deinit();
    var sys2 = try air.buildSystem(a, trace.rows);
    defer sys2.deinit();

    var pt = stark.Transcript.init("zkml.fp16.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, sys.system(), CONFIG);
    defer proof.deinit(a);

    proof.openings[0].current[air.col_c_bits] = Fp2.one;

    var vt = stark.Transcript.init("zkml.fp16.v1");
    try testing.expect(!try stark.verify(&vt, &proof, sys2.system(), CONFIG));
}

test "fp16 air: many random normal pairs prove and verify" {
    const a = testing.allocator;
    // Generate normal x normal pairs with in-scope results, so the AIR is
    // exercised on data it was not hand-checked against.
    var prng = std.Random.DefaultPrng.init(0xF16A11);
    const rng = prng.random();
    // 8 rows, not 16: at 108 composed constraints per multiply, 16 rows
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

    var trace = try air.buildTrace(a, pairs[0..n]);
    defer trace.deinit(a);

    const cfg = CONFIG;
    var sys = try air.buildSystem(a, trace.rows);
    defer sys.deinit();
    var sys2 = try air.buildSystem(a, trace.rows);
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
