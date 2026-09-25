//! End-to-end tests for the LogUp lookup argument (F2).
//!
//! The argument under test: the values in column `w` are a permutation of
//! the values in column `u`. What it deliberately does NOT test is
//! provenance — `u` is still witness, so nothing here stops a prover from
//! setting u = w. Pinning u to a fixed table needs a preprocessed trace
//! with a public commitment root (F3 plumbing); the KNOWN GAP tests in
//! quant_test.zig track the consequence.

const std = @import("std");
const stark = @import("root.zig");
const logup = @import("logup.zig");
const gemm_air = @import("gemm_air.zig");

const testing = std.testing;
const Fp2 = stark.Fp2;
const Goldilocks = stark.Goldilocks;

const rows: usize = 8;

/// A tiny AIR with a single trivial constraint plus one lookup, so the
/// test isolates the argument from the GEMM machinery.
const Config: stark.Config = blk: {
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

const col_w: u16 = 0;
const col_u: u16 = 1;

/// The base AIR is EMPTY on purpose: an empty composed set is legal (P = 0
/// divides by Z_H), which isolates the lookup argument from the GEMM
/// machinery. The last test below composes it with a real AIR instead.
const empty_constraints = [_]stark.Constraint{};

fn baseSystem() stark.System {
    return .{ .constraints = &empty_constraints };
}

/// The multiset the prover claims: `u` is `w` rotated by one row.
fn rotatedTable() [rows]Fp2 {
    var w: [rows]Fp2 = undefined;
    for (0..rows) |i| w[i] = Fp2.fromRaw(@intCast(3 * i + 1), @intCast(5 * i));
    return w;
}

const Case = struct {
    w: [rows]Fp2,
    u: [rows]Fp2,
};

fn buildCase(allocator: std.mem.Allocator, ch: logup.Challenges, sys: *const logup.Owned) !struct { cols: [2][]Fp2, ext: logup.Extended } {
    var c = Case{ .w = rotatedTable(), .u = undefined };
    for (0..rows) |i| c.u[i] = c.w[(i + 1) % rows];
    const w = try allocator.alloc(Fp2, rows);
    errdefer allocator.free(w);
    const u = try allocator.alloc(Fp2, rows);
    errdefer allocator.free(u);
    @memcpy(w, &c.w);
    @memcpy(u, &c.u);
    const cols = [_][]Fp2{ w, u };
    const ext = try logup.extendTrace(
        allocator,
        .{ .rows = rows, .columns = &cols },
        sys.system.lookups,
        ch,
    );
    return .{ .cols = cols, .ext = ext };
}

test "logup: a genuine permutation proves and verifies" {
    const a = testing.allocator;
    const requests = [_]logup.Request{.{ .value = col_w, .table = col_u }};

    var pt = stark.Transcript.init("zkml.logup.v1");
    const pch = logup.drawChallenges(&pt);
    var psys = try logup.buildSystem(a, baseSystem(), &requests, pch);
    defer psys.deinit();

    var pc = try buildCase(a, pch, &psys);
    defer a.free(pc.cols[0]);
    defer a.free(pc.cols[1]);
    defer pc.ext.deinit();

    var proof = try stark.prove(a, &pt, .{
        .rows = rows,
        .columns = pc.ext.view().columns,
    }, psys.system, Config);
    defer proof.deinit(a);

    // The verifier replays the same three steps.
    var vt = stark.Transcript.init("zkml.logup.v1");
    const vch = logup.drawChallenges(&vt);
    var vsys = try logup.buildSystem(a, baseSystem(), &requests, vch);
    defer vsys.deinit();

    try testing.expect(vch.beta.eql(pch.beta) and vch.gamma.eql(pch.gamma));
    try testing.expect(try stark.verify(&vt, &proof, vsys.system, Config));
}

test "logup: a value with no counterpart in the table cannot be proved" {
    const a = testing.allocator;
    const requests = [_]logup.Request{.{ .value = col_w, .table = col_u }};

    var pt = stark.Transcript.init("zkml.logup.v1");
    const pch = logup.drawChallenges(&pt);
    var psys = try logup.buildSystem(a, baseSystem(), &requests, pch);
    defer psys.deinit();

    var pc = try buildCase(a, pch, &psys);
    defer a.free(pc.cols[0]);
    defer a.free(pc.cols[1]);
    defer pc.ext.deinit();

    // Swap one entry for a value that appears nowhere in u: the
    // multisets differ, so the sum of reciprocals differs and the cyclic
    // accumulator cannot close.
    pc.cols[0][3] = Fp2.fromRaw(999_999, 7);
    // Refresh the witness so p/q are consistent with the edited column:
    // an honest prover would. The failure is then purely the cycle not
    // closing, i.e. the LogUp identity itself.
    const refreshed = try logup.extendTrace(
        a,
        .{ .rows = rows, .columns = &pc.cols },
        psys.system.lookups,
        pch,
    );
    pc.ext.deinit();
    pc.ext = refreshed;

    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{
            .rows = rows,
            .columns = pc.ext.view().columns,
        }, psys.system, Config),
    );
}

test "logup: a duplicated value (and a missing one) cannot be proved" {
    const a = testing.allocator;
    const requests = [_]logup.Request{.{ .value = col_w, .table = col_u }};

    var pt = stark.Transcript.init("zkml.logup.v1");
    const pch = logup.drawChallenges(&pt);
    var psys = try logup.buildSystem(a, baseSystem(), &requests, pch);
    defer psys.deinit();

    var pc = try buildCase(a, pch, &psys);
    defer a.free(pc.cols[0]);
    defer a.free(pc.cols[1]);
    defer pc.ext.deinit();

    // u repeats one entry and drops another: the same number of rows, a
    // different multiset. A count-based check would miss this; the
    // reciprocal sum does not.
    pc.cols[1][2] = pc.cols[1][3];
    // Rebuild the witness so p/q are consistent with the edited u (an
    // honest prover would); the argument must still fail.
    const refreshed = try logup.extendTrace(
        a,
        .{ .rows = rows, .columns = &pc.cols },
        psys.system.lookups,
        pch,
    );
    pc.ext.deinit();
    pc.ext = refreshed;

    try testing.expectError(
        stark.Error.ConstraintViolation,
        stark.prove(a, &pt, .{
            .rows = rows,
            .columns = pc.ext.view().columns,
        }, psys.system, Config),
    );
}

test "logup: editing a looked-up value in an opening is rejected" {
    const a = testing.allocator;
    const requests = [_]logup.Request{.{ .value = col_w, .table = col_u }};

    var pt = stark.Transcript.init("zkml.logup.v1");
    const pch = logup.drawChallenges(&pt);
    var psys = try logup.buildSystem(a, baseSystem(), &requests, pch);
    defer psys.deinit();

    var pc = try buildCase(a, pch, &psys);
    defer a.free(pc.cols[0]);
    defer a.free(pc.cols[1]);
    defer pc.ext.deinit();

    var proof = try stark.prove(a, &pt, .{
        .rows = rows,
        .columns = pc.ext.view().columns,
    }, psys.system, Config);
    defer proof.deinit(a);

    // A proof of a TRUE permutation, then one unit added to a looked-up
    // value inside an authenticated opening: the statement no longer
    // matches what the lookup argument proved, so it must be rejected.
    proof.openings[0].current[col_w] = proof.openings[0].current[col_w].add(Fp2.one);

    var vt = stark.Transcript.init("zkml.logup.v1");
    const vch = logup.drawChallenges(&vt);
    var vsys = try logup.buildSystem(a, baseSystem(), &requests, vch);
    defer vsys.deinit();

    try testing.expect(!try stark.verify(&vt, &proof, vsys.system, Config));
}

test "logup: a proof does not verify under a different challenge" {
    const a = testing.allocator;
    const requests = [_]logup.Request{.{ .value = col_w, .table = col_u }};

    var pt = stark.Transcript.init("zkml.logup.v1");
    const pch = logup.drawChallenges(&pt);
    var psys = try logup.buildSystem(a, baseSystem(), &requests, pch);
    defer psys.deinit();

    var pc = try buildCase(a, pch, &psys);
    defer a.free(pc.cols[0]);
    defer a.free(pc.cols[1]);
    defer pc.ext.deinit();

    var proof = try stark.prove(a, &pt, .{
        .rows = rows,
        .columns = pc.ext.view().columns,
    }, psys.system, Config);
    defer proof.deinit(a);

    // A verifier that drew its challenges from a different transcript has
    // different betas, so the p/q constraints it checks are not the ones
    // that were proved. This is what pins the protocol ORDER: the
    // challenges must come from the same transcript, before the
    // commitment is absorbed.
    var vt = stark.Transcript.init("zkml.logup.other");
    const vch = logup.drawChallenges(&vt);
    var vsys = try logup.buildSystem(a, baseSystem(), &requests, vch);
    defer vsys.deinit();
    try testing.expect(!vch.beta.eql(pch.beta) or !vch.gamma.eql(pch.gamma));

    try testing.expect(!try stark.verify(&vt, &proof, vsys.system, Config));
}

test "logup: a lookup composes with a real AIR (the GEMM system)" {
    const a = testing.allocator;
    // Prove the same Q4_K reduction as gemm_test, plus a lookup on the
    // running-sum column against a rotated copy of itself.
    const k: usize = 7;
    const av = try a.alloc(Goldilocks, k);
    defer a.free(av);
    const bv = try a.alloc(Goldilocks, k);
    defer a.free(bv);
    var c = Goldilocks.zero;
    for (0..k) |i| {
        av[i] = Goldilocks.fromU64(i + 1);
        bv[i] = Goldilocks.fromU64(2 * i + 3);
        c = c.add(av[i].mul(bv[i]));
    }

    const requests = [_]logup.Request{.{ .value = gemm_air.col_s, .table = gemm_air.col_c }};

    var pt = stark.Transcript.init("zkml.logup-gemm.v1");
    const pch = logup.drawChallenges(&pt);
    var psys = try logup.buildSystem(a, try gemm_air.system(k), &requests, pch);
    defer psys.deinit();
    try testing.expectEqual(@as(?usize, 8), psys.system.trace_rows);

    // s and c are NOT permutations of each other (c is nonzero only on
    // the closing row), so this lookup must NOT verify — it pins that
    // the argument composes with a real AIR and still bites.
    var trace = try gemm_air.buildTrace(a, av, bv, c);
    defer trace.deinit(a);
    const view = logup.TraceView{ .rows = trace.rows, .columns = trace.columns };
    var ext = try logup.extendTrace(a, view, psys.system.lookups, pch);
    defer ext.deinit();

    const cfg: stark.Config = blk: {
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

    // The prover cannot even build it: extendTrace succeeds, but the
    // accumulator does not close, so prove refuses.
    const result = stark.prove(a, &pt, .{
        .rows = trace.rows,
        .columns = ext.view().columns,
    }, psys.system, cfg);
    try testing.expectError(stark.Error.ConstraintViolation, result);
}
