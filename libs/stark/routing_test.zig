//! End-to-end tests for the compiled routing AIR (F2).
//!
//! The statement under test: the selected expert set IS the k largest gate
//! scores. The interesting negative is not "someone edits a flag" (the
//! commitment catches that) but "the prover picks the threshold to pick
//! the experts" — which the design answers structurally, and which the
//! threshold-independence test pins from the other side.

const std = @import("std");
const stark = @import("root.zig");
const routing = @import("routing_air.zig");

const testing = std.testing;
const Goldilocks = stark.Goldilocks;
const Fp2 = stark.Fp2;

const experts: usize = 16;
const k: u64 = 2;

const CONFIG: stark.Config = blk: {
    const log_trace: u6 = 4; // 16 rows
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

/// Gate scores in q4.12, deliberately non-monotonic so a prover cannot
/// claim the set is "the first k rows" or "the last k rows".
fn scores() [experts]Goldilocks {
    var out: [experts]Goldilocks = undefined;
    const values = [_]u64{
        100, 3000, 50,  900, 12,  2500, 77,  1100,
        5,   1800, 999, 42,  700, 31,   400, 1200,
    };
    for (0..experts) |i| out[i] = Goldilocks.fromU64(values[i]);
    return out;
}

test "routing: the top-k set proves and verifies" {
    const a = testing.allocator;
    const s = scores();

    var sys = try routing.buildSystem(a, experts, k);
    defer sys.deinit();
    const system = sys.system();
    // 1 accumulator + 16·(1+12) score range + 16·(1+13) difference range
    // + 16 ties + 16 selections.
    try testing.expectEqual(@as(usize, 1 + 16 * 13 + 16 * 14 + 32), system.composedCount());
    try testing.expectEqual(@as(usize, 2), system.maxDegree());
    try testing.expectEqual(@as(?usize, experts), system.trace_rows);
    try testing.expect(system.hasBoundary()); // acc[0] = 0

    var trace = try routing.buildTrace(a, s[0..], k);
    defer trace.deinit(a);

    // The trace really selects experts 1 and 5, the two largest scores.
    const flags = routing.selectedFlags(&trace);
    var selected: usize = 0;
    for (flags, 0..) |f, i| {
        if (!f.a.isZero()) {
            selected += 1;
            try testing.expect(i == 1 or i == 5);
        }
    }
    try testing.expectEqual(@as(usize, @intCast(k)), selected);

    var pt = stark.Transcript.init("zkml.routing.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG);
    defer proof.deinit(a);

    var vt = stark.Transcript.init("zkml.routing.v1");
    try testing.expect(try stark.verify(&vt, &proof, system, CONFIG));
}

test "routing: the threshold is a witness, and the set does not depend on it" {
    const a = testing.allocator;
    const s = scores();

    var sys = try routing.buildSystem(a, experts, k);
    defer sys.deinit();
    const system = sys.system();

    // `selected_e = [score_e >= T]`, so a valid T sits in
    // (score_3, score_2] = (1800, 2500]: the top two are 3000 and 2500.
    // Both thresholds below select exactly experts 1 and 5 and must
    // verify — the soundness argument stated as a test: the prover picks T
    // freely, yet cannot move the set.
    const high = Goldilocks.fromU64(2500); // the (k+1)-th largest exactly
    const low = Goldilocks.fromU64(2000); // strictly inside the gap

    var t1 = try routing.buildTraceAtTau(a, s[0..], k, high);
    defer t1.deinit(a);
    var t2 = try routing.buildTraceAtTau(a, s[0..], k, low);
    defer t2.deinit(a);

    for ([_]*routing.Trace{ &t1, &t2 }) |t| {
        for (0..experts) |i| {
            try testing.expect(routing.selectedFlags(t)[i].eql(routing.selectedFlags(&t1)[i]));
        }
        var pt = stark.Transcript.init("zkml.routing.tau");
        var proof = try stark.prove(a, &pt, .{
            .rows = t.rows,
            .columns = t.columns,
        }, system, CONFIG);
        defer proof.deinit(a);
        var vt = stark.Transcript.init("zkml.routing.tau");
        try testing.expect(try stark.verify(&vt, &proof, system, CONFIG));
    }
}

test "routing: a threshold that selects the wrong count is refused" {
    const a = testing.allocator;
    const s = scores();
    // The valid thresholds for k=2 are (1800, 2500]. Asking for k=3 at
    // T=2500 selects only two, so the builder refuses rather than letting
    // the AIR reject it later.
    try testing.expectError(
        routing.BuildTraceError.ThresholdSelectsWrongCount,
        routing.buildTraceAtTau(a, s[0..], 3, Goldilocks.fromU64(2500)),
    );
    // T equal to the k-th largest (3000) selects only ONE expert.
    try testing.expectError(
        routing.BuildTraceError.ThresholdSelectsWrongCount,
        routing.buildTraceAtTau(a, s[0..], 2, Goldilocks.fromU64(3000)),
    );
    // T above every score selects nobody.
    try testing.expectError(
        routing.BuildTraceError.ThresholdSelectsWrongCount,
        routing.buildTraceAtTau(a, s[0..], 1, Goldilocks.fromU64(4000)),
    );
}

test "routing: claiming a different expert set is rejected" {
    const a = testing.allocator;
    const s = scores();
    var sys = try routing.buildSystem(a, experts, k);
    defer sys.deinit();
    const system = sys.system();

    var trace = try routing.buildTrace(a, s[0..], k);
    defer trace.deinit(a);

    // Swap the selection to experts 3 and 6 (scores 900 and 77) instead of
    // 1 and 5. The `selected = top_bit(dshift)` constraints notice, because
    // the threshold still says otherwise.
    trace.columns[routing.col_selected][3] = Fp2.one;
    trace.columns[routing.col_selected][1] = Fp2.zero;
    trace.columns[routing.col_selected][5] = Fp2.zero;
    trace.columns[routing.col_selected][6] = Fp2.one;

    var pt = stark.Transcript.init("zkml.routing.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{
            .rows = trace.rows,
            .columns = trace.columns,
        }, system, CONFIG),
    );
}

test "routing: a score moved without its bits is rejected" {
    const a = testing.allocator;
    const s = scores();
    var sys = try routing.buildSystem(a, experts, k);
    defer sys.deinit();
    const system = sys.system();

    var trace = try routing.buildTrace(a, s[0..], k);
    defer trace.deinit(a);

    // Expert 6 (score 77) is promoted above the threshold, but its bit
    // decomposition and the difference column still describe 77.
    trace.columns[routing.col_score][6] = Fp2.re(Goldilocks.fromU64(3500));

    var pt = stark.Transcript.init("zkml.routing.v1");
    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{
            .rows = trace.rows,
            .columns = trace.columns,
        }, system, CONFIG),
    );
}

test "routing: a score outside the declared range cannot be witnessed" {
    const a = testing.allocator;
    var s = scores();
    s[4] = Goldilocks.fromU64(routing.score_modulus); // exactly 2^m
    try testing.expectError(
        routing.BuildTraceError.ScoreOutOfRange,
        routing.buildTrace(a, s[0..], k),
    );
    s[4] = Goldilocks.fromU64(routing.score_modulus + 1);
    try testing.expectError(
        routing.BuildTraceError.ScoreOutOfRange,
        routing.buildTrace(a, s[0..], k),
    );
}

test "routing: a tie at the k-th position is refused, not silently broken" {
    const a = testing.allocator;
    var s = scores();
    // Make the 2nd and 3rd largest equal (2500 and 1800 -> both 1800): no
    // threshold selects exactly 2, because any T takes the whole tie.
    s[5] = Goldilocks.fromU64(1800);
    try testing.expectError(
        routing.BuildTraceError.TieAtBoundary,
        routing.buildTrace(a, s[0..], k),
    );
}

test "routing: a non-power-of-two expert count is refused" {
    const a = testing.allocator;
    var s: [15]Goldilocks = undefined;
    for (&s, 0..) |*v, i| v.* = Goldilocks.fromU64(@intCast(i * 10));
    try testing.expectError(
        routing.BuildTraceError.NotPowerOfTwo,
        routing.buildTrace(a, &s, 1),
    );
    try testing.expectError(
        routing.BuildError.NotPowerOfTwo,
        routing.buildSystem(a, 15, 1),
    );
}

test "routing: editing the selected column in an opening is rejected" {
    const a = testing.allocator;
    const s = scores();
    var sys = try routing.buildSystem(a, experts, k);
    defer sys.deinit();
    const system = sys.system();

    var trace = try routing.buildTrace(a, s[0..], k);
    defer trace.deinit(a);

    var pt = stark.Transcript.init("zkml.routing.v1");
    var proof = try stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = trace.columns,
    }, system, CONFIG);
    defer proof.deinit(a);

    proof.openings[0].current[routing.col_selected] = Fp2.one;

    var vt = stark.Transcript.init("zkml.routing.v1");
    try testing.expect(!try stark.verify(&vt, &proof, system, CONFIG));
}
