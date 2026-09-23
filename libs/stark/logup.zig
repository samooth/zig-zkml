//! LogUp: the lookup argument (BLUE_PRINT §7.4).
//!
//! Everything else in libs/stark is a sum of products, which cannot state
//! "this value is in that set". LogUp can, and it fits this backend
//! exactly:
//!
//!   p_i = 1 / (beta·w_i + gamma)          p_i·(beta·w_i + gamma) = 1
//!   q_i = 1 / (beta·u_i + gamma)          q_i·(beta·u_i + gamma) = 1
//!   acc' = acc + p - q                    acc[0] = 0
//!
//! The last line is the part this domain makes free. The trace domain is
//! CYCLIC, so telescoping `acc` all the way around returns to acc[0] = 0
//! and leaves
//!
//!   sum_i p_i - q_i = 0
//!
//! with no sumcheck, no extra query and no extra proof item: a plain
//! degree-1 running-sum constraint, the same shape the GEMM AIR already
//! uses to close its cycle. Over a random (beta, gamma) the identity
//! `sum 1/(beta·w + gamma) = sum 1/(beta·u + gamma)` holds only when
//! {w} and {u} are the same multiset.
//!
//! ## Protocol order
//!
//! The inverse witness depends on (beta, gamma), so they must be drawn
//! BEFORE the trace commitment is absorbed — the opposite of the RLC
//! alphas, which come after. That makes the order part of the protocol,
//! and it lives here rather than inside prove/verify: the caller draws
//! the challenges, builds the system and extends the trace with them.
//! Both sides run the same three steps, so the transcripts match.
//!
//! ## What this does NOT do
//!
//! It proves the two columns are permutations of each other. It does not
//! prove either one is THE table: `u` is still a witness column, so a
//! prover who sets u = w has proved nothing about provenance. Pinning `u`
//! needs the table in a preprocessed trace whose commitment root is a
//! public input (the fp16 table is 2^16 entries — far too large to pin
//! row-wise in this trace). That is the remaining step for the KNOWN GAP
//! in quant_test.zig / chunk_binding.zig and it needs the multi-trace
//! public-input plumbing of F3. TODO tracks it.

const std = @import("std");
const expr = @import("expr.zig");
const transcript_lib = @import("../transcript.zig");

pub const Fp2 = expr.Fp2;
pub const Goldilocks = expr.Goldilocks;
pub const System = expr.System;
pub const Constraint = expr.Constraint;
pub const Term = expr.Term;
pub const Factor = expr.Factor;
pub const LookupSpec = expr.LookupSpec;
pub const Transcript = transcript_lib.Transcript;

/// Per-lookup input: the two columns whose multisets must match.
pub const Request = struct {
    value: u16,
    table: u16,
};

/// Fiat-Shamir challenges. Drawn before the commitment is absorbed.
pub const Challenges = struct {
    beta: Fp2,
    gamma: Fp2,

    pub fn betaFor(self: Challenges) Fp2 {
        return self.beta;
    }
};

/// Draw beta and gamma. One pair serves every lookup in the system: the
/// lookups differ in their columns, and the arguments are separated by
/// the RLC alphas that mix the composed constraints afterwards.
pub fn drawChallenges(transcript: *Transcript) Challenges {
    return .{
        .beta = transcript.challengeField(Fp2),
        .gamma = transcript.challengeField(Fp2),
    };
}

pub const BuildError = error{OutOfMemory};

/// The base AIR plus the LogUp constraints and the filled-in specs.
pub const Owned = struct {
    allocator: std.mem.Allocator,
    constraints: []Constraint,
    lookups: []LookupSpec,
    terms: []Term,
    factors: []Factor,
    system: System,

    pub fn deinit(self: *Owned) void {
        self.allocator.free(self.constraints);
        self.allocator.free(self.lookups);
        self.allocator.free(self.terms);
        self.allocator.free(self.factors);
        self.* = undefined;
    }
};

/// Append the LogUp machinery to `base`. New witness columns (p, q, acc)
/// are placed after everything the base references, so the layout is a
/// pure function of the base and both sides derive the same indices.
pub fn buildSystem(
    allocator: std.mem.Allocator,
    base: System,
    requests: []const Request,
    challenges: Challenges,
) BuildError!Owned {
    const n = requests.len;
    // Per lookup: 3 composed + 1 boundary_first; 11 terms; 12 factors
    // (the acc[0] boundary reuses the acc factor).
    const constraints = try allocator.alloc(Constraint, base.constraints.len + 4 * n);
    errdefer allocator.free(constraints);
    const lookups = try allocator.alloc(LookupSpec, n);
    errdefer allocator.free(lookups);
    const terms = try allocator.alloc(Term, 11 * n);
    errdefer allocator.free(terms);
    const factors = try allocator.alloc(Factor, 12 * n);
    errdefer allocator.free(factors);

    @memcpy(constraints[0..base.constraints.len], base.constraints);

    // Start after everything any constraint *or request* touches. The
    // table column is the trap here: no base constraint references it, so
    // base.maxColumn() alone would place p/q/acc on top of it.
    var highest: i32 = if (base.maxColumn()) |m| @intCast(m) else -1;
    for (requests) |req| {
        highest = @max(highest, @as(i32, req.value));
        highest = @max(highest, @as(i32, req.table));
    }
    var next_col: u16 = @intCast(highest + 1);

    for (requests, 0..) |req, i| {
        const p = next_col;
        const q = next_col + 1;
        const acc = next_col + 2;
        next_col += 3;
        lookups[i] = .{ .value = req.value, .table = req.table, .p = p, .q = q, .acc = acc };

        const fb = 12 * i;
        const tb = 11 * i;
        const beta = challenges.betaFor();
        const gamma = challenges.gamma;

        // p·(beta·w + gamma) - 1 = 0
        factors[fb + 0] = .{ .column = .{ .index = p } };
        factors[fb + 1] = .{ .column = .{ .index = req.value } };
        factors[fb + 2] = .{ .column = .{ .index = p } };
        factors[fb + 3] = .{ .constant = Fp2.one };
        terms[tb + 0] = .{ .factors = factors[fb + 0 .. fb + 2], .coefficient = beta };
        terms[tb + 1] = .{ .factors = factors[fb + 2 .. fb + 3], .coefficient = gamma };
        terms[tb + 2] = .{ .factors = factors[fb + 3 .. fb + 4], .coefficient = Fp2.neg(Fp2.one) };

        // q·(beta·u + gamma) - 1 = 0
        factors[fb + 4] = .{ .column = .{ .index = q } };
        factors[fb + 5] = .{ .column = .{ .index = req.table } };
        factors[fb + 6] = .{ .column = .{ .index = q } };
        factors[fb + 7] = .{ .constant = Fp2.one };
        terms[tb + 3] = .{ .factors = factors[fb + 4 .. fb + 6], .coefficient = beta };
        terms[tb + 4] = .{ .factors = factors[fb + 6 .. fb + 7], .coefficient = gamma };
        terms[tb + 5] = .{ .factors = factors[fb + 7 .. fb + 8], .coefficient = Fp2.neg(Fp2.one) };

        // acc' - acc - p + q = 0
        factors[fb + 8] = .{ .column = .{ .index = acc, .offset = 1 } };
        factors[fb + 9] = .{ .column = .{ .index = acc } };
        factors[fb + 10] = .{ .column = .{ .index = p } };
        factors[fb + 11] = .{ .column = .{ .index = q } };
        terms[tb + 6] = .{ .factors = factors[fb + 8 .. fb + 9] };
        terms[tb + 7] = .{ .factors = factors[fb + 9 .. fb + 10], .coefficient = Fp2.neg(Fp2.one) };
        terms[tb + 8] = .{ .factors = factors[fb + 10 .. fb + 11], .coefficient = Fp2.neg(Fp2.one) };
        terms[tb + 9] = .{ .factors = factors[fb + 11 .. fb + 12] };

        // acc[0] = 0 (boundary_first)
        terms[tb + 10] = .{ .factors = factors[fb + 9 .. fb + 10] };

        const at = base.constraints.len + 4 * i;
        constraints[at + 0] = .{
            .name = "p·(beta·w + gamma) = 1",
            .scope = .composed,
            .terms = terms[tb + 0 .. tb + 3],
        };
        constraints[at + 1] = .{
            .name = "q·(beta·u + gamma) = 1",
            .scope = .composed,
            .terms = terms[tb + 3 .. tb + 6],
        };
        constraints[at + 2] = .{
            .name = "acc' - acc - p + q",
            .scope = .composed,
            .terms = terms[tb + 6 .. tb + 10],
        };
        constraints[at + 3] = .{
            .name = "acc[0] = 0",
            .scope = .boundary_first,
            .terms = terms[tb + 10 .. tb + 11],
        };
    }

    return .{
        .allocator = allocator,
        .constraints = constraints,
        .lookups = lookups,
        .terms = terms,
        .factors = factors,
        .system = .{ .constraints = constraints, .lookups = lookups },
    };
}

/// What extendTrace needs to see of a trace. Kept local so logup.zig does
/// not have to import root.zig (which imports this file).
pub const TraceView = struct {
    rows: usize,
    columns: []const []const Fp2,
};

/// A trace with the p/q/acc witness columns appended. Owns only the new
/// buffers; the base columns stay the caller's.
pub const Extended = struct {
    allocator: std.mem.Allocator,
    base: TraceView,
    extra: [][]Fp2,
    all: [][]const Fp2,
    rows: usize,

    pub fn view(self: *const Extended) TraceView {
        return .{ .rows = self.rows, .columns = self.all };
    }

    pub fn deinit(self: *Extended) void {
        for (self.extra) |c| self.allocator.free(c);
        self.allocator.free(self.extra);
        self.allocator.free(self.all);
        self.* = undefined;
    }
};

pub const ExtendError = error{ OutOfMemory, ZeroDenominator };

/// Fill p, q and acc for every lookup. The inverses are computed OFF the
/// trace domain (the prover may do any field arithmetic it likes); the AIR
/// then checks each of them pointwise.
pub fn extendTrace(
    allocator: std.mem.Allocator,
    trace: TraceView,
    lookups: []const LookupSpec,
    challenges: Challenges,
) ExtendError!Extended {
    const n = lookups.len;
    const extra = try allocator.alloc([]Fp2, 3 * n);
    errdefer allocator.free(extra);
    var made: usize = 0;
    errdefer for (extra[0..made]) |c| allocator.free(c);
    for (0..3 * n) |i| {
        const buf = try allocator.alloc(Fp2, trace.rows);
        @memset(buf, Fp2.zero);
        extra[i] = buf;
        made += 1;
    }

    const all = try allocator.alloc([]const Fp2, trace.columns.len + 3 * n);
    errdefer allocator.free(all);
    for (trace.columns, 0..) |src, i| all[i] = src;
    for (0..3 * n) |i| all[trace.columns.len + i] = extra[i];

    for (lookups, 0..) |l, i| {
        const w_col = trace.columns[l.value];
        const u_col = trace.columns[l.table];
        const p_col = extra[3 * i + 0];
        const q_col = extra[3 * i + 1];
        const acc_col = extra[3 * i + 2];
        const beta = challenges.betaFor();
        for (0..trace.rows) |r| {
            p_col[r] = try inverseOf(beta, w_col[r], challenges.gamma);
            q_col[r] = try inverseOf(beta, u_col[r], challenges.gamma);
        }
        // acc[0] = 0, acc[r+1] = acc[r] + p[r] - q[r], wrapping so the
        // cycle closes exactly (the AIR's composed constraint demands it).
        var acc = Fp2.zero;
        acc_col[0] = acc;
        for (0..trace.rows - 1) |r| {
            acc = acc.add(p_col[r]).sub(q_col[r]);
            acc_col[r + 1] = acc;
        }
    }
    return .{
        .allocator = allocator,
        .base = trace,
        .extra = extra,
        .all = all,
        .rows = trace.rows,
    };
}

fn inverseOf(beta: Fp2, x: Fp2, gamma: Fp2) ExtendError!Fp2 {
    const denom = beta.mul(x).add(gamma);
    if (denom.isZero()) return ExtendError.ZeroDenominator;
    return denom.inv() catch ExtendError.ZeroDenominator;
}

const testing = std.testing;

test "logup: a system gains 3 composed + 1 boundary constraint per lookup" {
    const a = testing.allocator;
    const base_terms = [_]Term{.{ .factors = &[_]Factor{.{ .column = .{ .index = 0 } }} }};
    const base_constraints = [_]Constraint{.{ .name = "base", .terms = &base_terms }};
    const base = System{ .constraints = &base_constraints };
    const ch = Challenges{ .beta = Fp2.fromRaw(7, 0), .gamma = Fp2.fromRaw(11, 0) };

    var sys = try buildSystem(a, base, &.{
        .{ .value = 0, .table = 1 },
        .{ .value = 2, .table = 3 },
    }, ch);
    defer sys.deinit();

    try testing.expectEqual(@as(usize, 1 + 8), sys.system.constraints.len);
    // 1 from the base + 3 per lookup.
    try testing.expectEqual(@as(usize, 1 + 6), sys.system.composedCount());
    try testing.expectEqual(@as(usize, 2), sys.system.lookups.len);
    // p, q, acc start right after the base's highest column.
    try testing.expectEqual(@as(u16, 4), sys.lookups[0].p);
    try testing.expectEqual(@as(u16, 5), sys.lookups[0].q);
    try testing.expectEqual(@as(u16, 6), sys.lookups[0].acc);
    try testing.expectEqual(@as(u16, 7), sys.lookups[1].p);
    // The inverse equations are quadratic; the accumulator is linear.
    try testing.expectEqual(@as(usize, 2), sys.system.maxDegree());
    // The challenges really landed in the coefficients.
    try testing.expect(ch.beta.eql(sys.terms[0].coefficient));
    try testing.expect(ch.gamma.eql(sys.terms[1].coefficient));
    try testing.expect(ch.beta.eql(sys.terms[3].coefficient));
    try testing.expect(ch.gamma.eql(sys.terms[4].coefficient));
}

test "logup: the new columns clear the inputs even when unreferenced" {
    const a = testing.allocator;
    // The base only mentions column 0, but the lookup's table lives at
    // column 9 and no constraint references it: the layout must still
    // start past 9, not past 0.
    const base_constraints = [_]Constraint{
        .{ .name = "base", .terms = &[_]Term{.{ .factors = &[_]Factor{.{ .column = .{ .index = 0 } }} }} },
    };
    const base = System{ .constraints = &base_constraints };
    const ch = Challenges{ .beta = Fp2.one, .gamma = Fp2.one };

    var sys = try buildSystem(a, base, &.{.{ .value = 0, .table = 9 }}, ch);
    defer sys.deinit();
    try testing.expectEqual(@as(u16, 10), sys.lookups[0].p);
    try testing.expectEqual(@as(u16, 11), sys.lookups[0].q);
    try testing.expectEqual(@as(u16, 12), sys.lookups[0].acc);
}

test "logup: the inverse witness satisfies its own constraint" {
    const a = testing.allocator;
    const base_constraints = [_]Constraint{};
    const base = System{ .constraints = &base_constraints };
    const ch = Challenges{ .beta = Fp2.fromRaw(3, 5), .gamma = Fp2.fromRaw(9, 2) };

    var sys = try buildSystem(a, base, &.{.{ .value = 0, .table = 1 }}, ch);
    defer sys.deinit();

    const rows = 4;
    const w = [_]Fp2{ Fp2.fromRaw(2, 1), Fp2.fromRaw(5, 0), Fp2.fromRaw(9, 4), Fp2.fromRaw(1, 1) };
    const u = [_]Fp2{ w[3], w[0], w[2], w[1] }; // a permutation of w
    const cols = [_][]const Fp2{ &w, &u };
    var ext = try extendTrace(a, .{ .rows = rows, .columns = &cols }, sys.system.lookups, ch);
    defer ext.deinit();

    // Every row: p·(beta·w + gamma) = 1, and the accumulator closes.
    const c = sys.system.constraints[0];
    for (0..rows) |r| {
        const pv = ext.extra[0][r];
        const denom = ch.beta.mul(w[r]).add(ch.gamma);
        try testing.expect(pv.mul(denom).sub(Fp2.one).isZero());
        _ = c;
    }
    try testing.expect(ext.extra[2][0].isZero());
    // The wrap row: acc[0] = acc[last] + p[last] - q[last].
    const last = rows - 1;
    const wrap = ext.extra[2][0].sub(ext.extra[2][last]).sub(ext.extra[0][last]).add(ext.extra[1][last]);
    try testing.expect(wrap.isZero());
}
