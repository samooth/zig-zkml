//! Minimal STARK adapter — bridges L3 AirGraph to L1 FRI.
//!
//! docs/BLUE_PRINT.md §5.1/§7.1: the trace is flattened row-major into Fp2
//! evaluations; FRI proves the evaluation polynomial has low degree.
//! Constraint verification at query points is the verifier's job
//! (implemented per-gadget in F2+).
//!
//! v1: monolithic per-layer proof. No cross-layer composition.

const std = @import("std");
const field = @import("../field.zig");
const fri = @import("../fri/root.zig");
const air = @import("../air/root.zig");
const tensor = @import("../tensor/root.zig");
const transcript = @import("../transcript.zig");
const domain = @import("../fri/domain.zig");

pub const Goldilocks = field.Goldilocks;
pub const Fp2 = fri.Fp2;

pub const Trace = struct {
    rows: usize,
    cols: usize,
    data: []Goldilocks, // row-major: data[row * cols + col]
};

pub const Proof = fri.Proof;
pub const Config = fri.Config;

/// Ceiling log2 for usize (comptime-friendly).
fn ceilLog2(n: usize) u6 {
    if (n <= 1) return 0;
    var log: u6 = 0;
    var v: usize = n - 1;
    while (v > 0) : (v >>= 1) {
        log += 1;
    }
    return log;
}

/// Flatten trace to Fp2 evaluations. Pads with zeros to the next
/// power-of-2 if trace.rows * trace.cols is not exact.
pub fn flattenToFp2(allocator: std.mem.Allocator, trace: Trace) ![]Fp2 {
    const raw_len = trace.rows * trace.cols;
    const log_n = ceilLog2(raw_len);
    const n: usize = @as(usize, 1) << log_n;
    const evals = try allocator.alloc(Fp2, n);
    for (0..n) |i| {
        if (i < raw_len) {
            evals[i] = Fp2{ .a = trace.data[i], .b = Fp2.zero.b };
        } else {
            evals[i] = Fp2.zero;
        }
    }
    return evals;
}

/// Prove a trace satisfying the given AirGraph. Returns a FRI proof.
pub fn prove(
    allocator: std.mem.Allocator,
    tr: anytype,
    trace: Trace,
    config: Config,
) !Proof {
    const evals = try flattenToFp2(allocator, trace);
    defer allocator.free(evals);
    return fri.prove(allocator, tr, evals, config);
}

/// Verify a FRI proof against the trace shape. Constraint checking
/// is delegated to the caller (per-gadget verifiers in F2+).
pub fn verify(
    _: std.mem.Allocator,
    proof: *const Proof,
    config: Config,
    tr: anytype,
) !bool {
    return fri.verify(tr, proof, config);
}

/// Map a query pair_index to (row, col) in the trace matrix.
pub fn queryToRowCol(pair_index: usize, cols: usize) struct { row: usize, col: usize } {
    const flat = pair_index * 2; // antipodal pair covers 2 positions
    return .{ .row = flat / cols, .col = flat % cols };
}

test "flattenToFp2 pads to power of 2" {
    const t = std.testing;
    const a = t.allocator;

    var data = [_]Goldilocks{ Goldilocks.one, Goldilocks.one, Goldilocks.one } ** 5;
    const trace = Trace{ .rows = 5, .cols = 1, .data = data[0..] };
    const evals = try flattenToFp2(a, trace);
    defer a.free(evals);

    const raw_len = 5;
    const log_n = ceilLog2(raw_len);
    try t.expectEqual(@as(usize, 1) << log_n, evals.len);
    try t.expect(evals.len >= raw_len);
    for (0..raw_len) |i| {
        try t.expect(evals[i].a.eql(data[i]));
    }
    for (raw_len..evals.len) |i| {
        try t.expect(evals[i].a.isZero());
    }
}

test "queryToRowCol maps correctly" {
    const t = std.testing;
    // cols=4: pair 0→flat 0→(0,0), pair 1→flat 2→(0,2), pair 2→flat 4→(1,0)
    try t.expectEqual(@as(usize, 0), queryToRowCol(0, 4).row);
    try t.expectEqual(@as(usize, 0), queryToRowCol(0, 4).col);
    try t.expectEqual(@as(usize, 0), queryToRowCol(1, 4).row);
    try t.expectEqual(@as(usize, 2), queryToRowCol(1, 4).col);
    try t.expectEqual(@as(usize, 1), queryToRowCol(2, 4).row);
    try t.expectEqual(@as(usize, 0), queryToRowCol(2, 4).col);
}

// TODO F2: end-to-end GEMM proof requires a proper STARK backend that
// checks constraints (running-sum, dequant ranges) at query points.
// The current flatten-and-FRI wrapper only proves low-degree of the
// flattened trace; real GEMM witness is NOT low-degree, so FRI rejects.
// See docs/BLUE_PRINT.md §5.2 (v1 monolithic AIR) and §11 (F2 milestone).

test "FRI direct: prove and verify simple polynomial" {
    const t = std.testing;
    const a = t.allocator;

    // Direct FRI test: degree-2 poly on 256-element domain.
    const log_n: u6 = 8;
    const dom = domain.Domain.init(log_n);
    const n = dom.size();
    var evals = try a.alloc(Fp2, n);
    defer a.free(evals);

    const c1 = Fp2.re(field.Goldilocks.fromU64(3));
    const c2 = Fp2.re(field.Goldilocks.fromU64(7));
    for (0..n) |i| {
        const x = dom.at(i);
        evals[i] = x.sqr().add(x.mul(c1)).add(c2);
    }

    var pt = transcript.Transcript.init("fri-direct");
    var proof = try fri.prove(a, &pt, evals, .{
        .log_domain = log_n,
        .log_final = 5,
        .log_residual_degree = 2,
        .num_queries = 8,
    });
    defer proof.deinit(a);

    var vt = transcript.Transcript.init("fri-direct");
    try t.expect(try fri.verify(&vt, &proof, .{
        .log_domain = log_n,
        .log_final = 5,
        .log_residual_degree = 2,
        .num_queries = 8,
    }));
}
