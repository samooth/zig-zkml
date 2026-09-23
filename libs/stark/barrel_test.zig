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
        const cfg: barrel.Config = .{
            .in_base = 0,
            .width = width,
            .amount_base = width,
            .amount_bits = amount_bits,
            .out_base = width + amount_bits,
            .sticky_col = width + amount_bits + barrel.column_cost(.{
                .in_base = 0,
                .width = width,
                .amount_base = width,
                .amount_bits = amount_bits,
                .out_base = width + amount_bits,
                .sticky_col = width + amount_bits,
            }) - 1,
        };
        pub const col_in: u16 = cfg.in_base;
        pub const col_amount: u16 = cfg.amount_base;
        pub const col_out: u16 = cfg.out_base;
        pub const column_count: usize = cfg.sticky_col + 1;
    };
}

/// The reference: shift the vector right, remember whether anything was
/// lost off the bottom.
pub const Shifted = struct { value: u64, sticky: u64 };

pub fn shiftRight(value: u64, width: u16, amount: u16) Shifted {
    const mask = if (width >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(width)) - 1;
    const v = value & mask;
    if (amount >= width) return .{ .value = 0, .sticky = if (v == 0) 0 else 1 };
    const out = (v >> @intCast(amount)) & mask;
    const lost = v & ((@as(u64, 1) << @intCast(amount)) - 1);
    return .{ .value = out, .sticky = if (lost == 0) 0 else 1 };
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
            if (bit == 1) {
                const lost = current & ((@as(u64, 1) << @intCast(distance)) - 1);
                if (lost != 0) sticky = 1;
                current >>= @intCast(distance);
            }
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
        if (got != want.value or got_sticky != want.sticky) {
            std.debug.print("shift {x} by {d}: got {x}/{d}, want {x}/{d}\n", .{ value, amount, got, got_sticky, want.value, want.sticky });
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
