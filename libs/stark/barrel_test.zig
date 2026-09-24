//! The barrel shifter on its own, before anything uses it.
//!
//! Like the widening, the gadget has no arithmetic to get wrong, so the
//! test sweeps it: every shift amount in range, crossed with vectors that
//! hit each interesting case (all zero, all one, a single bit at each
//! end, a single bit in the middle, and a pattern with bits on both
//! sides of the cut). Every constraint is evaluated on every row, with no
//! prover involved.

const std = @import("std");
const stark = @import("root.zig");
const barrel = @import("barrel.zig");
const bld = @import("air_builder.zig");

const testing = std.testing;
const Fp2 = bld.Fp2;
const Builder = bld.Builder;
const TRANSCRIPT = "zkml.barrel.v1";

const rows: usize = 8;

const CONFIG: stark.Config = blk: {
    const log_trace: u6 = 3;
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

/// The gadget's own column layout, so it can be built and proved alone.
fn Layout(comptime width: u16, comptime amount_bits: u16) type {
    return struct {
        const out_base: u16 = width + amount_bits;
        // column_cost needs a Config to measure, and a Config needs the cost
        // to place its own outputs, so the measuring call gets a dummy one:
        // only amount_bits and width matter for the size.
        const cost: usize = barrel.column_cost(.{
            .in_base = 0,
            .width = width,
            .amount_base = width,
            .amount_bits = amount_bits,
            .out_base = out_base,
            .sticky_col = out_base,
            .round_col = out_base,
            .below_col = out_base,
            .round_bits = true,
        });
        const cfg: barrel.Config = .{
            .in_base = 0,
            .width = width,
            .amount_base = width,
            .amount_bits = amount_bits,
            .out_base = out_base,
            // column_cost ends with the caller's three output columns, in
            // the order they appear in Config.
            .sticky_col = out_base + @as(u16, @intCast(cost)) - 3,
            .round_col = out_base + @as(u16, @intCast(cost)) - 2,
            .below_col = out_base + @as(u16, @intCast(cost)) - 1,
            .round_bits = true,
        };
        pub const col_in: u16 = cfg.in_base;
        pub const col_amount: u16 = cfg.amount_base;
        pub const col_out: u16 = cfg.out_base;
        pub const column_count: usize = out_base + cost;
    };
}

/// The reference: shift the vector right, remember whether anything was
/// lost off the bottom.
pub const Shifted = struct { value: u64, sticky: u64, round: u64, below: u64 };

pub fn shiftRight(value: u64, width: u16, amount: u16) Shifted {
    const mask = if (width >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(width)) - 1;
    const v = value & mask;
    // The round bit is the LAST bit the shift drops; the sticky a rounding
    // step wants is everything STRICTLY below it, which is not the same as
    // the OR of everything lost.
    const round: u64 = if (amount == 0) 0 else (v >> @intCast(amount - 1)) & 1;
    const below_mask: u64 = if (amount <= 1) 0 else (@as(u64, 1) << @intCast(amount - 1)) - 1;
    const below: u64 = if (v & below_mask == 0) 0 else 1;
    if (amount >= width) return .{ .value = 0, .sticky = if (v == 0) 0 else 1, .round = round, .below = below };
    const out = (v >> @intCast(amount)) & mask;
    const lost = v & ((@as(u64, 1) << @intCast(amount)) - 1);
    return .{ .value = out, .sticky = if (lost == 0) 0 else 1, .round = round, .below = below };
}

fn evalRow(c: bld.Constraint, columns: [][]Fp2, r: usize) Fp2 {
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

fn buildSystem(comptime width: u16, comptime amount_bits: u16, allocator: std.mem.Allocator, nrows: usize) !bld.Owned {
    const L = Layout(width, amount_bits);
    var b = Builder.init(allocator);
    for (0..width) |i| try b.boolean("input bit is boolean", L.col_in + @as(u16, @intCast(i)));
    try barrel.build(&b, L.cfg);
    return bld.freeze(allocator, &b, nrows);
}

fn buildTrace(comptime width: u16, comptime amount_bits: u16, allocator: std.mem.Allocator, values: []const u64, amounts: []const u16) !bld.Trace {
    const L = Layout(width, amount_bits);
    var trace = try bld.Trace.alloc(allocator, L.column_count, values.len);
    errdefer trace.deinit(allocator);
    const mask: u64 = (@as(u64, 1) << @intCast(width)) - 1;
    for (values, amounts, 0..) |value, amount, r| {
        trace.writeBits(L.col_in, value & mask, width, r);
        trace.writeBits(L.col_amount, amount, amount_bits, r);
        // Walk the stages the same way the constraints do, so the witness
        // is computed rather than read back out of the answer.
        const bases: barrel.Bases = .of(L.cfg);
        var current: u64 = value & mask;
        var sticky: u64 = 0;
        var round_bit: u64 = 0;
        var below: u64 = 0;
        var k: usize = 0;
        while (k < amount_bits) : (k += 1) {
            const stage: u16 = @intCast(k);
            const bit = (amount >> @intCast(amount_bits - 1 - k)) & 1;
            const distance: u6 = @intCast(@as(u32, 1) << @intCast(amount_bits - 1 - @as(u16, @intCast(k))));

            // The prefix OR of the bits this stage can lose, over the
            // vector as it stands BEFORE the shift.
            var prefix_or: u64 = 0;
            for (0..distance) |j| {
                const bit_value = (current >> @intCast(j)) & 1;
                if (bit_value == 1) prefix_or = 1;
                trace.columns[bases.prefixOf(L.cfg, stage) + @as(u16, @intCast(j))][r] =
                    Fp2.re(bld.Goldilocks.fromU64(prefix_or));
            }
            // The block this stage drops is the low `distance` bits of the
            // vector as it stands NOW, so its top and its rest have to be
            // read before the shift.
            const block_top: u64 = (current >> @intCast(distance - 1)) & 1;
            const rest_mask: u64 = if (distance >= 2) (@as(u64, 1) << @intCast(distance - 1)) - 1 else 0;
            const rest: u64 = if (current & rest_mask == 0) 0 else 1;
            // Everything dropped before this stage, counting the previous
            // stage's round bit: that bit stops being the round bit as soon
            // as this stage moves.
            const prev_all: u64 = if (round_bit == 1 or below == 1) 1 else 0;
            const tail: u64 = if (prev_all == 1 or rest == 1) 1 else 0;
            trace.columns[bases.prevAllOf(stage)][r] = Fp2.re(bld.Goldilocks.fromU64(prev_all));
            trace.columns[bases.tailOf(stage)][r] = Fp2.re(bld.Goldilocks.fromU64(tail));
            if (bit == 1) {
                round_bit = block_top;
                below = tail;
                const lost = current & ((@as(u64, 1) << @intCast(distance)) - 1);
                if (lost != 0) sticky = 1;
                current >>= @intCast(distance);
            }
            trace.columns[bases.roundOf(L.cfg, stage)][r] = Fp2.re(bld.Goldilocks.fromU64(round_bit));
            trace.columns[bases.belowOf(L.cfg, stage)][r] = Fp2.re(bld.Goldilocks.fromU64(below));
            const flag: u64 = if (bit == 1) prefix_or else 0;
            trace.columns[bases.flagOf(stage)][r] = Fp2.re(bld.Goldilocks.fromU64(flag));

            // Every stage writes its vector, shift or no shift: the
            // constraints read all of them, not just the ones that moved.
            trace.writeBits(bases.vector(L.cfg, stage), current, width, r);
            trace.columns[bases.stickyOf(L.cfg, stage)][r] = Fp2.re(bld.Goldilocks.fromU64(sticky));
        }
    }
    return trace;
}

fn checkBatch(comptime width: u16, comptime amount_bits: u16, values: []const u64, amounts: []const u16) !void {
    const a = testing.allocator;
    const L = Layout(width, amount_bits);
    var trace = try buildTrace(width, amount_bits, a, values, amounts);
    defer trace.deinit(a);
    for (values, amounts, 0..) |value, amount, r| {
        const want = shiftRight(value, width, amount);
        var got: u64 = 0;
        for (0..width) |i| {
            if (trace.columns[L.col_out + @as(u16, @intCast(i))][r].a.isZero()) continue;
            got |= @as(u64, 1) << @intCast(i);
        }
        const got_sticky = trace.columns[L.cfg.sticky_col][r].a.toU64();
        const got_round = trace.columns[L.cfg.round_col][r].a.toU64();
        const got_below = trace.columns[L.cfg.below_col][r].a.toU64();
        if (got != want.value or got_sticky != want.sticky or
            got_round != want.round or got_below != want.below)
        {
            std.debug.print("shift {x} by {d}: got {x} sticky {d} round {d} below {d}, want {x} sticky {d} round {d} below {d}\n", .{
                value, amount, got, got_sticky, got_round, got_below, want.value, want.sticky, want.round, want.below,
            });
            return error.ShiftMismatch;
        }
    }
    var sys = try buildSystem(width, amount_bits, a, trace.rows);
    defer sys.deinit();
    for (sys.system().constraints) |c| {
        for (0..trace.rows) |r| {
            if (evalRow(c, trace.columns, r).isZero()) continue;
            std.debug.print("shift {x} by {d}: \"{s}\" does not vanish on row {d}\n", .{ values[r], amounts[r], c.name, r });
            return error.WitnessInconsistent;
        }
    }
}

fn vectors(comptime width: u16) []const u64 {
    const mask = (@as(u64, 1) << @intCast(width)) - 1;
    return &.{
        0,
        mask,
        1,
        mask - 1,
        @as(u64, 1) << @intCast(width - 1),
        (@as(u64, 1) << @intCast(width - 1)) - 1,
        0x5555_5555_5555_5555 & mask,
        0xAAAA_AAAA_AAAA_AAAA & mask,
    };
}

fn sweep(comptime width: u16, comptime amount_bits: u16) !usize {
    var values: [rows]u64 = undefined;
    var amounts: [rows]u16 = undefined;
    var n: usize = 0;
    var checked: usize = 0;
    const limit: u32 = @as(u32, 1) << @intCast(amount_bits);
    var amount: u32 = 0;
    while (amount < limit) : (amount += 1) {
        for (vectors(width)) |value| {
            values[n] = value;
            amounts[n] = @intCast(amount);
            n += 1;
            if (n < rows) continue;
            try checkBatch(width, amount_bits, &values, &amounts);
            checked += n;
            n = 0;
        }
    }
    if (n > 0) {
        try checkBatch(width, amount_bits, values[0..n], amounts[0..n]);
        checked += n;
    }
    return checked;
}

test "barrel: every shift amount of a 25-bit vector, checked against the reference" {
    try testing.expectEqual(@as(usize, 8 * 16), try sweep(25, 4));
}

test "barrel: wider shift amounts and narrower vectors" {
    _ = try sweep(25, 5);
    _ = try sweep(12, 4);
    _ = try sweep(49, 6);
    _ = try sweep(8, 2);
}

test "barrel: the round-bit chains cost 4 constraints per stage" {
    const a = testing.allocator;
    // 25 bits with a 4-bit amount: four stages, and stage 0 emits three
    // chains instead of four because its prev_all is a forced zero, so the
    // chains add 15 to the 152 the shift alone costs. Pinned because the
    // roadmap quotes these numbers as a cost model.
    var sys = try buildSystem(25, 4, a, 1);
    defer sys.deinit();
    try testing.expectEqual(@as(usize, 167), sys.system().composedCount());
    try testing.expectEqual(@as(usize, 2), sys.system().maxDegree());
    try testing.expect(!sys.system().hasBoundary());

    // And without the chains the same shift is 152, which is the number the
    // roadmap carried before the extension existed.
    const L = Layout(25, 4);
    var plain = Builder.init(a);
    for (0..25) |i| try plain.boolean("input bit is boolean", L.col_in + @as(u16, @intCast(i)));
    try barrel.build(&plain, .{
        .in_base = L.col_in,
        .width = 25,
        .amount_base = L.col_amount,
        .amount_bits = 4,
        .out_base = L.col_out,
        .sticky_col = 0,
        .round_col = 0,
        .below_col = 0,
    });
    try testing.expectEqual(@as(usize, 152), plain.count());
    plain.deinit();
}

test "barrel: it proves, verifies, and rejects a forged shift" {
    const a = testing.allocator;
    // A 25-bit vector with a 4-bit amount is 152 constraints per row, and
    // the prover's cap is 1024 composed constraints for the WHOLE system,
    // so eight rows of it would not fit. The big configurations are swept
    // above without a prover, which is the cheaper way to cover them.
    const width: u16 = 12;
    const amount_bits: u16 = 3;
    const L = Layout(width, amount_bits);
    var values: [rows]u64 = undefined;
    var amounts: [rows]u16 = undefined;
    for (0..rows) |r| {
        values[r] = 0x0123_4567_89AB_CDEF;
        amounts[r] = @intCast(r);
    }
    var trace = try buildTrace(width, amount_bits, a, &values, &amounts);
    defer trace.deinit(a);
    var sys = try buildSystem(width, amount_bits, a, trace.rows);
    defer sys.deinit();

    var pt = stark.Transcript.init(TRANSCRIPT);
    var proof = try stark.prove(a, &pt, .{ .rows = trace.rows, .columns = trace.columns }, sys.system(), CONFIG);
    defer proof.deinit(a);
    var vt = stark.Transcript.init(TRANSCRIPT);
    try testing.expect(try stark.verify(&vt, &proof, sys.system(), CONFIG));

    trace.columns[L.col_amount][0] = Fp2.one;
    var pt2 = stark.Transcript.init(TRANSCRIPT);
    if (stark.prove(a, &pt2, .{ .rows = trace.rows, .columns = trace.columns }, sys.system(), CONFIG)) |forged| {
        var accepted = forged;
        accepted.deinit(a);
        std.debug.print("barrel: a forged shift amount was accepted\n", .{});
        return error.ForgedShiftAccepted;
    } else |_| {}
}

// The cost of the shift an fp32 ADDER would need, measured rather than
// argued: aligning the smaller operand means shifting a 24-bit significand
// by up to 2·emax = 254 places, which is an 8-bit amount.
//
// This test exists to keep a decision honest. The multiply's reduction
// needs (11, 4) and costs 86 constraints; an fp32 add's alignment needs
// (24, 8) and costs several times that, per operand, per add. Sixteen adds
// per row — the chunk size — would then need tens of thousands of composed
// constraints, and the verifier keeps its Fiat-Shamir alphas in a fixed
// stack array sized by `max_composed_constraints`. So the proof's
// accumulation stays the EXACT field sum (see BLUE_PRINT §3.1, the
// normative QuantScheme) and the fp32 semantics live in the cross-check
// test, where `float_ref.add` is the reference.
test "the fp32 adder's alignment shift, measured" {
    const a = testing.allocator;
    // The shift an fp32 ADDER would need, measured rather than argued:
    // aligning the smaller operand means shifting a 24-bit significand by
    // up to 2·emax = 254 places, which is an 8-bit amount. (11, 4) is the
    // multiply's reduction, for scale.
    //
    // The numbers include the input bits' own booleanity, which the float
    // AIR already emits for its kept field; subtract `width` to compare
    // with the AIR's own gadget cost.
    //
    // This test exists to keep a decision honest. Per operand, per add,
    // this is the FLOOR: an fp32 add also needs the exponent comparison,
    // the sum, its bit decomposition, a normalisation and a rounding.
    // Sixteen adds per row — the chunk size — would put the composed
    // constraints an order of magnitude past what the verifier's alpha
    // array is sized for, so the proof's accumulation stays the EXACT field
    // sum (BLUE_PRINT §3.1's normative QuantScheme) and the fp32 semantics
    // live in the cross-check, where `float_ref.add` is the reference.
    inline for (.{ .{ 11, 4 }, .{ 24, 8 }, .{ 24, 6 } }) |shape| {
        const width: u16 = comptime shape[0];
        const stages: u16 = comptime shape[1];
        var b = Builder.init(a);
        defer b.factors.deinit(a);
        for (0..width) |i| try b.boolean("input bit is boolean", @as(u16, @intCast(i)));
        const out_base: u16 = width + stages;
        const cost: usize = comptime barrel.column_cost(.{
            .in_base = 0,
            .width = width,
            .amount_base = width,
            .amount_bits = stages,
            .out_base = out_base,
            .sticky_col = out_base,
            .round_col = out_base,
            .below_col = out_base,
            .round_bits = true,
        });
        const cfg: barrel.Config = comptime .{
            .in_base = 0,
            .width = width,
            .amount_base = width,
            .amount_bits = stages,
            .out_base = out_base,
            .sticky_col = out_base + @as(u16, @intCast(cost)) - 3,
            .round_col = out_base + @as(u16, @intCast(cost)) - 2,
            .below_col = out_base + @as(u16, @intCast(cost)) - 1,
            .round_bits = true,
        };
        try barrel.build(&b, cfg);
        var owned = try bld.freeze(a, &b, 1);
        defer owned.deinit();

        // MEASURED, pinned: 97 for the multiply's reduction (11 bits, 4
        // stages) and 526 / 272 for the adder's alignment (24 bits, 8 and
        // 6 stages). The first number minus the 11 input bits is the 86
        // the float AIR actually pays.
        const want: usize = switch (shape[0]) {
            11 => 97,
            24 => if (shape[1] == 8) 526 else 272,
            else => unreachable,
        };
        try testing.expectEqual(want, owned.system().composedCount());
    }
}
