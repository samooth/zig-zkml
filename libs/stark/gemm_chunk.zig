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
//! ## Why the closing row needs 32 boundary constraints
//!
//! The domain is cyclic, so the last row must contribute exactly -C to
//! close the sum (see gemm_air.zig). With one a/b pair that is two pins:
//! `a₀[last] = 1`, `b₀[last] = -c[last]`. With 16 pairs the other 30 slots
//! are *witness* and must be pinned to zero, or the prover can absorb an
//! arbitrary product there and shift the attested output: the recursion
//! only says `s[0] = s[last] + Σ_last`, and `s[last]` already contains
//! `Σ_last`, so leaving it free lets the prover choose C. Hence one
//! `aᵢ[last] = 0` and one `bᵢ[last] = 0` per unused slot. The verifier
//! evaluates every boundary constraint of a scope against the single
//! last-row opening, so this costs openings, not queries.
//!
//! Trace layout:
//!
//!   cols 0..15   a₀..a₁₅   the chunk's dequantized A elements
//!   cols 16..31  b₀..b₁₅   the chunk's B elements
//!   col  32      s          running sum, s[0] = 0
//!   col  33      c          claimed output element
//!
//! Operand binding (the nibble/scale constraints of quant_binding.zig) is
//! NOT wired for this layout yet: the sound one-MAC-per-row path remains
//! the one to use for real proofs until the chunked binding lands. TODO
//! under F2 tracks it.

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

const kCloseAFactors = [_]Factor{.{ .column = .{ .index = colA(0) } }};
const kCloseATerms = [_]Term{
    .{ .factors = kCloseAFactors[0..1] },
    .{ .factors = &[_]Factor{.{ .constant = Fp2.one }}, .coefficient = kNegOne },
};
const kCloseBFactors = [_]Factor{.{ .column = .{ .index = colB(0) } }};
const kCloseCFactors = [_]Factor{.{ .column = .{ .index = col_c } }};
const kCloseBTerms = [_]Term{
    .{ .factors = kCloseBFactors[0..1] },
    .{ .factors = kCloseCFactors[0..1] },
};

/// One single-factor term per slot, per operand family: the `aᵢ[last] = 0`
/// and `bᵢ[last] = 0` pins each reference one product factor of `kProdFactors`.
const kSlotTermsA: [slots]Term = blk: {
    var arr: [slots]Term = undefined;
    for (0..slots) |i| arr[i] = .{ .factors = kProdFactors[i][0..1] };
    break :blk arr;
};
const kSlotTermsB: [slots]Term = blk: {
    var arr: [slots]Term = undefined;
    for (0..slots) |i| arr[i] = .{ .factors = kProdFactors[i][1..2] };
    break :blk arr;
};

const kUnusedA: [slots - 1]Constraint = blk: {
    var arr: [slots - 1]Constraint = undefined;
    for (0..slots - 1) |i| {
        arr[i] = .{
            .name = std.fmt.comptimePrint("a[{d}][last] = 0", .{i + 1}),
            .scope = .boundary_last,
            .terms = kSlotTermsA[i + 1 .. i + 2],
        };
    }
    break :blk arr;
};

const kUnusedB: [slots - 1]Constraint = blk: {
    var arr: [slots - 1]Constraint = undefined;
    for (0..slots - 1) |i| {
        arr[i] = .{
            .name = std.fmt.comptimePrint("b[{d}][last] = 0", .{i + 1}),
            .scope = .boundary_last,
            .terms = kSlotTermsB[i + 1 .. i + 2],
        };
    }
    break :blk arr;
};

const kConstraints = blk: {
    var arr: [4 + 2 * (slots - 1)]Constraint = undefined;
    arr[0] = .{ .name = "s' - s - Σ aᵢbᵢ", .scope = .composed, .terms = &kRunningTerms };
    arr[1] = .{ .name = "s[0] = 0", .scope = .boundary_first, .terms = &kFirstRowTerms };
    arr[2] = .{ .name = "a₀[last] = 1", .scope = .boundary_last, .terms = &kCloseATerms };
    var at: usize = 3;
    arr[at] = .{ .name = "b₀[last] = -c[last]", .scope = .boundary_last, .terms = &kCloseBTerms };
    at += 1;
    for (kUnusedA) |c| {
        arr[at] = c;
        at += 1;
    }
    for (kUnusedB) |c| {
        arr[at] = c;
        at += 1;
    }
    break :blk arr;
};

pub fn system() System {
    return .{ .constraints = &kConstraints };
}

/// Data rows: one per chunk of `slots` MACs.
pub fn chunkRowsFor(k: usize) usize {
    return (k + slots - 1) / slots;
}

/// Total trace rows: the chunk rows, the synthetic closing row, and enough
/// zero padding to reach a power of two.
pub fn rowsFor(k: usize) usize {
    const needed = chunkRowsFor(k) + 1;
    return std.math.ceilPowerOfTwo(usize, needed) catch needed;
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
    const rows = rowsFor(a.len);
    const chunks = chunkRowsFor(a.len);
    if (chunks + 1 > rows) return BuildError.EmptyReduction;

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
            if (idx >= a.len) continue;
            cols[colA(i)][chunk] = Fp2.re(a[idx]);
            cols[colB(i)][chunk] = Fp2.re(b[idx]);
        }
        cols[col_s][chunk] = Fp2.re(total);
        for (0..slots) |i| {
            const idx = chunk * slots + i;
            if (idx >= a.len) continue;
            total = total.add(a[idx].mul(b[idx]));
        }
    }

    // Padding chunks between the data and the closing row: all-zero
    // products, so the running sum holds still.
    var pad = chunks;
    while (pad < rows - 1) : (pad += 1) {
        cols[col_s][pad] = Fp2.re(total);
    }

    // Closing row: one cancelling product, every other slot zero.
    const last = rows - 1;
    cols[colA(0)][last] = Fp2.one;
    cols[colB(0)][last] = Fp2.re(Goldilocks.zero.sub(claimed_output));
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
    // k=256 -> 16 chunks -> 17 rows -> padded to 32.
    try testing.expectEqual(@as(usize, 32), rowsFor(256));
    try testing.expectEqual(@as(usize, 16), chunkRowsFor(256));
    // k=16 -> one chunk -> 2 rows, already a power of two.
    try testing.expectEqual(@as(usize, 2), rowsFor(16));
    try testing.expectEqual(@as(usize, 1), chunkRowsFor(16));
    try testing.expectEqual(@as(usize, 34), column_count);
}

test "chunk: the system is one composed constraint of degree 2" {
    const sys = system();
    try testing.expectEqual(@as(usize, 1), sys.composedCount());
    try testing.expectEqual(@as(usize, 2), sys.maxDegree());
    try testing.expectEqual(@as(usize, 4 + 2 * (slots - 1)), sys.constraints.len);
    // Every unused slot is pinned at the last row: without those the
    // prover could park a product there and move the attested output.
    try testing.expectEqual(@as(usize, 2 * (slots - 1)), @as(usize, sys.constraints.len) - 4);
    try testing.expectEqual(@as(?u16, col_c), sys.maxColumn());
}
