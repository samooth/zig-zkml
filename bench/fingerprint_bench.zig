//! Measured cost of the fingerprint claim against the oracle it replaces.
//!
//! `fingerprint.measure()` counts multiplications, which is the right model
//! for comparing asymptotic behaviour but is not what a decision needs: the
//! question is whether the fast path is worth building an AIR for at all.
//! This bench answers that in wall-clock time, on the shapes that matter.
//!
//! Both paths run on identical inputs and their results are compared *before*
//! any timing is reported. A speedup measured against a path that returns a
//! different answer would be meaningless, so disagreement aborts the run.
//!
//! What this does NOT measure: proof size, verification time, or the FRI/AIR
//! cost of discharging the claim — those need the F3 work. It measures the
//! claim arithmetic, which is exactly the part the identity replaces.

const std = @import("std");
const Allocator = std.mem.Allocator;

const zkml = @import("zkml");
const Fp2 = zkml.fingerprint.Fp2;
const fingerprint = zkml.fingerprint;
const fingerprint_bind = zkml.fingerprint_bind;

const Shape = struct { m: usize, n: usize, k: usize };

const Row = struct {
    oracle_ns: u64,
    fast_ns: u64,
};

/// Deterministic non-trivial values. A ramp keeps the multiprecision honest
/// (values grow with the seed rather than being 1) while staying reproducible.
fn ramp(allocator: Allocator, len: usize, seed: u64) ![]Fp2 {
    const out = try allocator.alloc(Fp2, len);
    var acc = seed | 1;
    for (out) |*e| {
        acc = acc *% 6364136223846793005 +% 1442695040888963407;
        e.* = Fp2.fromRaw(@intCast((acc >> 33) % 1013 + 1), @intCast((acc >> 13) % 7 + 1));
    }
    return out;
}

fn benchShape(allocator: Allocator, io: std.Io, s: Shape) !Row {
    const a = try ramp(allocator, s.m * s.k, 0x1111);
    defer allocator.free(a);
    const b = try ramp(allocator, s.n * s.k, 0x2222);
    defer allocator.free(b);

    var t = fingerprint_bind.Transcript.init("zkml.bench.fingerprint");
    const c = try fingerprint_bind.bindMatrices(
        allocator,
        &t,
        [_]u8{0xA5} ** 32,
        [_]u8{0x5A} ** 32,
        s.m,
        s.n,
    );
    defer allocator.free(c.u);
    defer allocator.free(c.v);

    // Correctness gate before any timing is reported.
    const oracle = try fingerprint.productInner(a, b, c.u, c.v, s.m, s.n, s.k);
    const fast = try fingerprint.fingerprintClaim(a, b, c.u, c.v, s.m, s.n, s.k);
    if (!oracle.eql(fast)) return error.FingerprintDisagrees;

    const t0 = std.Io.Timestamp.now(io, .awake);
    var sink = try fingerprint.productInner(a, b, c.u, c.v, s.m, s.n, s.k);
    const t1 = std.Io.Timestamp.now(io, .awake);
    if (sink.isZero()) return error.ImpossibleOracleResult;

    const t2 = std.Io.Timestamp.now(io, .awake);
    sink = try fingerprint.fingerprintClaim(a, b, c.u, c.v, s.m, s.n, s.k);
    const t3 = std.Io.Timestamp.now(io, .awake);
    if (sink.isZero()) return error.ImpossibleFastResult;

    const oracle_ns: u64 = @intCast(t1.nanoseconds - t0.nanoseconds);
    const fast_ns: u64 = @intCast(t3.nanoseconds - t2.nanoseconds);

    return .{ .oracle_ns = oracle_ns, .fast_ns = fast_ns };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());

    var repeat: usize = 3;
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--repeat") and i + 1 < argv.len) {
            i += 1;
            repeat = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--help")) {
            std.debug.print("usage: fingerprint_bench [--repeat N]\n", .{});
            return;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            return error.BadArgument;
        }
    }
    if (repeat == 0) repeat = 1;

    const shapes = [_]Shape{
        .{ .m = 64, .n = 96, .k = 128 },
        .{ .m = 256, .n = 384, .k = 512 },
        .{ .m = 512, .n = 704, .k = 1024 },
    };

    std.debug.print("\nfingerprint claim — wall clock, best of {d}\n", .{repeat});
    std.debug.print(
        "oracle and fast path are compared before timing; a disagreement aborts.\n\n",
        .{},
    );
    std.debug.print(
        "{s:>16} {s:>11} {s:>10} {s:>8} {s:>11} {s:>10} {s:>10}\n",
        .{ "m x n x k", "muls or", "muls fp", "model", "oracle ms", "fast ms", "measured" },
    );

    for (shapes) |s| {
        var best: Row = undefined;
        var have = false;
        for (0..repeat) |_| {
            const r = try benchShape(allocator, init.io, s);
            if (!have or r.fast_ns < best.fast_ns) {
                best = r;
                have = true;
            }
        }
        const model = try fingerprint.measure(s.m, s.n, s.k);
        const measured = @as(f64, @floatFromInt(best.oracle_ns)) /
            @as(f64, @floatFromInt(best.fast_ns));
        std.debug.print(
            "{d:>5} x{d:>4} x{d:>4} {d:>11} {d:>10} {d:>7.1}x {d:>10.2} {d:>9.2} {d:>9.1}x\n",
            .{
                s.m,
                s.n,
                s.k,
                model.oracle_muls,
                model.fingerprint_muls,
                model.speedup(),
                @as(f64, @floatFromInt(best.oracle_ns)) / 1e6,
                @as(f64, @floatFromInt(best.fast_ns)) / 1e6,
                measured,
            },
        );
    }

    const big = try fingerprint.measure(2048, 1408, 2048);
    std.debug.print(
        "\n2048 x 1408 x 2048 model: {:.0}x fewer multiplications ({} vs {})\n",
        .{ big.speedup(), big.fingerprint_muls, big.oracle_muls },
    );
    std.debug.print(
        "not measured: FRI/AIR cost, proof size, verification time — those need F3.\n\n",
        .{},
    );
}
