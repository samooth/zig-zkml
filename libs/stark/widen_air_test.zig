//! Tests for the widening AIR.
//!
//! The interesting property here is that a widening has no arithmetic to
//! get wrong, so the tests can be exhaustive: every one of the 65536
//! bfloat16 patterns, and every normal finite pattern of the other three
//! formats, goes through a real trace and every constraint is evaluated
//! on it. No prover, no FRI, no sampling — if a bit of the mapping is
//! wrong, some pattern in the space says so.

const std = @import("std");
const air = @import("widen_air.zig");
const fmt_lib = @import("float_format.zig");

const testing = std.testing;
const Fp2 = air.Fp2;
const TRANSCRIPT = "zkml.widen.v1";

const rows: usize = 8;

const CONFIG: @import("root.zig").Config = blk: {
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

/// One batch of patterns: the trace's target bits are the reference's, and
/// every constraint vanishes.
fn checkBatch(comptime f: fmt_lib.Format, patterns: []const u16) !void {
    const a = testing.allocator;
    const L = air.Layout(f);
    var trace = try air.buildTrace(a, patterns, f);
    defer trace.deinit(a);
    for (patterns, 0..) |src, r| {
        const expected = try air.widen(f, src);
        var got: u32 = 0;
        for (0..air.binary32.width) |i| {
            if (trace.columns[L.col_dst + @as(u16, @intCast(i))][r].a.isZero()) continue;
            got |= @as(u32, 1) << @intCast(i);
        }
        if (got != expected) {
            std.debug.print("{s}: {x} widened to {x}, the reference says {x}\n", .{ f.name, src, got, expected });
            return error.ReferenceMismatch;
        }
    }
    var sys = try air.buildSystem(a, trace.rows, f);
    defer sys.deinit();
    for (sys.system().constraints) |c| {
        for (0..trace.rows) |r| {
            if (evalRow(c, trace.columns, r).isZero()) continue;
            std.debug.print("{s}: {x}: \"{s}\" does not vanish on row {d}\n", .{ f.name, patterns[r], c.name, r });
            return error.WitnessInconsistent;
        }
    }
}

fn sweep(comptime f: fmt_lib.Format) !usize {
    var batch: [rows]u16 = undefined;
    var n: usize = 0;
    var checked: usize = 0;
    const limit: u32 = @as(u32, 1) << @intCast(f.byteWidth());
    var src: u32 = 0;
    while (src < limit) : (src += 1) {
        const pattern: u16 = @intCast(src);
        _ = air.widen(f, pattern) catch continue;
        batch[n] = pattern;
        n += 1;
        if (n < batch.len) continue;
        try checkBatch(f, &batch);
        checked += n;
        n = 0;
    }
    if (n > 0) {
        try checkBatch(f, batch[0..n]);
        checked += n;
    }
    return checked;
}

test "widen air: every bfloat16 pattern widens exactly" {
    // Total: all 65536, including subnormals, infinities and NaNs.
    try testing.expectEqual(@as(usize, 65536), try sweep(fmt_lib.bfloat16));
}

test "widen air: every normal finite pattern of the other formats widens exactly" {
    const expected = [_]usize{ 61440, 224, 240 };
    inline for (.{ fmt_lib.binary16, fmt_lib.fp8_e4m3, fmt_lib.fp8_e5m2 }, expected) |f, want| {
        try testing.expectEqual(want, try sweep(f));
    }
}

test "widen air: out-of-scope sources are refused, not rounded" {
    const a = testing.allocator;
    inline for (.{ fmt_lib.binary16, fmt_lib.fp8_e4m3, fmt_lib.fp8_e5m2 }) |f| {
        const sub: u16 = f.pack(.{ .sign = 0, .exponent = 0, .mantissa = 0 });
        const inf: u16 = f.pack(.{ .sign = 0, .exponent = f.emax(), .mantissa = 0 });
        const nan: u16 = f.pack(.{ .sign = 0, .exponent = f.emax(), .mantissa = 1 });
        for ([_]u16{ sub, inf, nan }) |pattern| {
            try testing.expectError(error.UnsupportedCase, air.widen(f, pattern));
            const one = [_]u16{pattern};
            try testing.expectError(error.UnsupportedCase, air.buildTrace(a, &one, f));
        }
    }
}

test "widen air: the cost is one constraint per target bit, plus the carries" {
    const a = testing.allocator;
    inline for (.{ fmt_lib.bfloat16, fmt_lib.binary16, fmt_lib.fp8_e4m3, fmt_lib.fp8_e5m2 }) |f| {
        var sys = try air.buildSystem(a, 1, f);
        defer sys.deinit();
        try testing.expectEqual(air.expected_constraints(f), sys.system().composedCount());
        try testing.expect(!sys.system().hasBoundary());
    }
}

test "widen air: widening proves and verifies, and a forged target does not" {
    const a = testing.allocator;
    const stark = @import("root.zig");
    inline for (.{ fmt_lib.bfloat16, fmt_lib.binary16, fmt_lib.fp8_e4m3, fmt_lib.fp8_e5m2 }) |f| {
        var patterns: [rows]u16 = undefined;
        var n: usize = 0;
        var src: u32 = 0;
        while (n < rows) : (src += 1) {
            const pattern: u16 = @intCast(src);
            _ = air.widen(f, pattern) catch continue;
            patterns[n] = pattern;
            n += 1;
        }

        var trace = try air.buildTrace(a, &patterns, f);
        defer trace.deinit(a);
        var sys = try air.buildSystem(a, trace.rows, f);
        defer sys.deinit();

        var pt = stark.Transcript.init(TRANSCRIPT);
        var proof = try stark.prove(a, &pt, .{ .rows = trace.rows, .columns = trace.columns }, sys.system(), CONFIG);
        defer proof.deinit(a);
        var vt = stark.Transcript.init(TRANSCRIPT);
        try testing.expect(try stark.verify(&vt, &proof, sys.system(), CONFIG));

        // Flip one target bit. The copy constraints are the whole
        // statement, so this must not verify.
        const L = air.Layout(f);
        trace.columns[L.col_dst + 3][0] = Fp2.one;
        var pt2 = stark.Transcript.init(TRANSCRIPT);
        if (stark.prove(a, &pt2, .{ .rows = trace.rows, .columns = trace.columns }, sys.system(), CONFIG)) |forged| {
            var accepted = forged;
            accepted.deinit(a);
            std.debug.print("{s}: a forged widening was accepted\n", .{f.name});
            return error.ForgedWideningAccepted;
        } else |_| {}
    }
}
