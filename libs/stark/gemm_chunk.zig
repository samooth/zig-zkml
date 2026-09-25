//! Chunked GEMM AIR (F2): 16 MACs per trace row.
//!
//! `libs/stark/gemm_air.zig` puts one MAC per row because `s' = s + a*b`
//! carries a single product. The gadget's real shape is 16 MACs per row
//! (`GemmGadget.macs_per_row`), and nothing about the IR prevents it: a
//! chunk's constraint is
//!
//!   s' = s + a₀b₀ + a₁b₁ + ... + a₁₅b₁₅
//!
//! a *sum* of 16 degree-2 products, still degree 2, still one LDE row. The
//! payoff is prover-side: a k=256 reduction goes from 512 trace rows to
//! 32, so the LDE, the Merkle commitment and FRI all shrink 16×.
//!
//! ## Why the closing row needs no boundary pins
//!
//! The domain is cyclic, so the last row must contribute exactly -C to
//! close the sum (see gemm_air.zig). This layout used to spend 32 boundary
//! constraints on it: `a₀[last] = 1`, `b₀[last] = -c[last]`, plus one zero
//! pin per unused slot (30 of them), because the recursion
//! `s[0] = s[last] + Σ_last` leaves `Σ_last` free and a prover could park a
//! product there to move the attested output.
//!
//! That whole apparatus is gone. The closing row is now EXEMPT from the
//! composed constraint (`System.transition_exemptions = 1`), so `Σ_last` is
//! simply not part of any equation, and the output is pinned by one
//! boundary: `s[last] = c[last]`. The sum is telescoping over rows
//! 0..last-1, all enforced, and the unused slots on the exempt row have no
//! equation to satisfy. Thirty-two fewer boundary constraints, and the row
//! is finally free of operand binding (see quant_binding.zig).
//!
//! Trace layout:
//!
//!   cols 0..15   a₀..a₁₅   the chunk's dequantized A elements
//!   cols 16..31  b₀..b₁₅   the chunk's B elements
//!   col  32      s          running sum, s[0] = 0
//!   col  33      c          claimed output element
//!
//! Operand binding lives in `chunk_binding.zig`; the chunked layout requires
//! k to be a multiple of `slots` and its rows to be exactly full chunks plus
//! the closing row, so no padding or ragged slot can hide a product.

const std = @import("std");
const tensor = @import("../tensor/root.zig");
const expr = @import("./expr.zig");

pub const Goldilocks = tensor.Goldilocks;
pub const Fp2 = expr.Fp2;
pub const System = expr.System;
pub const Constraint = expr.Constraint;
pub const Term = expr.Term;
pub const Factor = expr.Factor;
pub const ColumnRef = expr.ColumnRef;

/// MACs per trace row, matching `GemmGadget.macs_per_row`.
pub const slots: usize = 16;

pub const col_a_base: u16 = 0;
pub const col_b_base: u16 = @intCast(slots);
pub const col_s: u16 = @intCast(2 * slots);
pub const col_c: u16 = col_s + 1;
pub const column_count: usize = col_c + 1;

/// a_i for a slot, as a ColumnRef (no row offset).
pub fn colA(i: usize) u16 {
    return col_a_base + @as(u16, @intCast(i));
}
pub fn colB(i: usize) u16 {
    return col_b_base + @as(u16, @intCast(i));
}

// Container-scope storage: System holds pointers into these. Every array
// below is a container-scope const, and the slices inside them point at
// other container-scope consts, so nothing dangles (a local array inside
// a `blk:` would).

const kNegOne = Fp2.neg(Fp2.one);

const kSFactors: [2]Factor = .{
    .{ .column = .{ .index = col_s, .offset = 1 } },
    .{ .column = .{ .index = col_s } },
};

const kProdFactors: [slots][2]Factor = blk: {
    var arr: [slots][2]Factor = undefined;
    for (0..slots) |i| {
        arr[i] = .{ .{ .column = .{ .index = colA(i) } }, .{ .column = .{ .index = colB(i) } } };
    }
    break :blk arr;
};

const kRunningTerms: [2 + slots]Term = blk: {
    var arr: [2 + slots]Term = undefined;
    arr[0] = .{ .factors = kSFactors[0..1] };
    arr[1] = .{ .factors = kSFactors[1..2], .coefficient = kNegOne };
    for (0..slots) |i| {
        arr[2 + i] = .{ .factors = kProdFactors[i][0..2], .coefficient = kNegOne };
    }
    break :blk arr;
};

const kFirstRowTerms = [_]Term{.{ .factors = kSFactors[1..2] }};

/// s[last] - c[last]: the claim is read straight off the telescoped sum.
const kCloseCFactors = [_]Factor{.{ .column = .{ .index = col_c } }};
const kCloseOutputTerms = [_]Term{
    .{ .factors = kSFactors[1..2] },
    .{ .factors = kCloseCFactors[0..1], .coefficient = kNegOne },
};

const kConstraints = [_]Constraint{
    .{ .name = "s' - s - Σ aᵢbᵢ", .scope = .composed, .terms = &kRunningTerms },
    .{ .name = "s[0] = 0", .scope = .boundary_first, .terms = &kFirstRowTerms },
    .{ .name = "s[last] = c[last]", .scope = .boundary_last, .terms = &kCloseOutputTerms },
};

pub fn system(k: usize) BuildError!System {
    const rows = try rowsFor(k);
    return .{
        .constraints = &kConstraints,
        .trace_rows = rows,
        .transition_exemptions = 1,
    };
}

/// Data rows: one per full chunk of `slots` MACs.
pub fn chunkRowsFor(k: usize) usize {
    return k / slots;
}

/// Total trace rows: full chunk rows plus the synthetic closing row. The
/// domain must be a power of two, so k = slots * (2^m - 1).
pub fn rowsFor(k: usize) BuildError!usize {
    if (k == 0) return BuildError.EmptyReduction;
    if (k % slots != 0) return BuildError.RaggedReduction;
    const rows = chunkRowsFor(k) + 1;
    if (!std.math.isPowerOfTwo(rows)) return BuildError.UnsupportedTraceSize;
    return rows;
}

pub const Trace = struct {
    rows: usize,
    columns: [][]Fp2,

    pub fn deinit(self: *Trace, allocator: std.mem.Allocator) void {
        for (self.columns) |c| allocator.free(c);
        allocator.free(self.columns);
        self.* = undefined;
    }
};

pub const BuildError = error{
    OutOfMemory,
    EmptyReduction,
    ShortOperands,
    RaggedReduction,
    UnsupportedTraceSize,
};

/// Build an honest chunked trace for the output element
/// `sum(a[i] * b[i])`, claiming `claimed_output` as C.
pub fn buildTrace(
    allocator: std.mem.Allocator,
    a: []const Goldilocks,
    b: []const Goldilocks,
    claimed_output: Goldilocks,
) BuildError!Trace {
    if (a.len == 0 or a.len != b.len) return BuildError.ShortOperands;
    const rows = try rowsFor(a.len);
    const chunks = chunkRowsFor(a.len);

    const cols = try allocator.alloc([]Fp2, column_count);
    errdefer allocator.free(cols);
    var made: usize = 0;
    errdefer for (cols[0..made]) |c| allocator.free(c);
    for (0..column_count) |i| {
        const buf = try allocator.alloc(Fp2, rows);
        @memset(buf, Fp2.zero);
        cols[i] = buf;
        made += 1;
    }

    var total = Goldilocks.zero;
    for (0..chunks) |chunk| {
        for (0..slots) |i| {
            const idx = chunk * slots + i;
            cols[colA(i)][chunk] = Fp2.re(a[idx]);
            cols[colB(i)][chunk] = Fp2.re(b[idx]);
        }
        cols[col_s][chunk] = Fp2.re(total);
        for (0..slots) |i| {
            const idx = chunk * slots + i;
            total = total.add(a[idx].mul(b[idx]));
        }
    }

    // Closing row: exempt, so the slots are zero and only s/c matter.
    const last = rows - 1;
    cols[col_s][last] = Fp2.re(total);
    cols[col_c][last] = Fp2.re(claimed_output);

    return .{ .rows = rows, .columns = cols };
}

/// The value the AIR actually attests: the sum of the chunk's products.
pub fn traceOutput(trace: *const Trace) Goldilocks {
    return trace.columns[col_s][trace.rows - 1].a;
}

const testing = std.testing;

test "chunk: layout and row counts" {
    // k=240 -> 15 chunks -> 16 rows, already a power of two.
    try testing.expectEqual(@as(usize, 16), try rowsFor(240));
    try testing.expectEqual(@as(usize, 15), chunkRowsFor(240));
    // k=16 -> one chunk -> 2 rows, already a power of two.
    try testing.expectEqual(@as(usize, 2), try rowsFor(16));
    try testing.expectEqual(@as(usize, 1), chunkRowsFor(16));
    try testing.expectEqual(@as(usize, 34), column_count);
}

test "chunk: the system is one composed constraint of degree 2" {
    const sys = try system(240);
    try testing.expectEqual(@as(usize, 1), sys.composedCount());
    try testing.expectEqual(@as(usize, 2), sys.maxDegree());
    // Three constraints, not 34: the closing row is exempt from the
    // composed one, so the unused slots have no pin to satisfy and the
    // output is read off the sum by a single boundary.
    try testing.expectEqual(@as(usize, 3), sys.constraints.len);
    try testing.expectEqual(@as(?u16, col_c), sys.maxColumn());
    try testing.expectEqual(@as(usize, 1), sys.transition_exemptions);
}
