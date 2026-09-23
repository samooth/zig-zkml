//! Quantized-operand binding for the CHUNKED GEMM layout (F2).
//!
//! `quant_binding.zig` does this for one MAC per row. Here the same
//! binding is replicated over all 16 slots of a chunk, so the 16× smaller
//! trace is not a soundness regression: every one of the 32 operands in a
//! row is pinned to a raw 4-bit Q4_K nibble times its block scale.
//!
//! Column layout (on top of gemm_chunk's 34):
//!
//!   for slot i in 0..slots, 12 columns each:
//!     +0  nib_a        +1  nib_b
//!     +2  bits_a[0..4] +6  bits_b[0..4]
//!     +10 scale_a      +11 scale_b
//!
//! Everything is built in owned storage (the IR holds pointers): a
//! 26-bit-wide static table per slot would be unreadable, and 16 of them
//! worse.

const std = @import("std");
const expr = @import("./expr.zig");
const range = @import("./range.zig");
const chunk = @import("./gemm_chunk.zig");
const tensor = @import("../tensor/root.zig");

pub const Fp2 = expr.Fp2;
pub const System = expr.System;
pub const Constraint = expr.Constraint;
pub const Term = expr.Term;
pub const Factor = expr.Factor;
pub const Goldilocks = tensor.Goldilocks;

pub const gemm_columns: usize = chunk.column_count;
pub const per_slot: usize = 12;
pub const nibble_width: u8 = 4;

pub const column_count: usize = gemm_columns + chunk.slots * per_slot;

pub fn colNibA(i: usize) u16 {
    return @intCast(gemm_columns + i * per_slot);
}
pub fn colNibB(i: usize) u16 {
    return colNibA(i) + 1;
}
pub fn colBitsA(i: usize) u16 {
    return colNibA(i) + 2;
}
pub fn colBitsB(i: usize) u16 {
    return colNibA(i) + 6;
}
pub fn colScaleA(i: usize) u16 {
    return colNibA(i) + 10;
}
pub fn colScaleB(i: usize) u16 {
    return colNibA(i) + 11;
}

pub const BuildError = error{ OutOfMemory, BadWidth };

/// Chunked GEMM AIR + nibble range checks + dequantization equations for
/// all 2·slots operands per row. Owns every allocation the returned
/// System points into.
pub const BoundSystem = struct {
    inner: range.BuiltSystem,
    terms: []Term,
    factors: []Factor,

    pub fn system(self: BoundSystem) System {
        return self.inner.system;
    }

    pub fn deinit(self: *BoundSystem) void {
        // Read the allocator FIRST: range.BuiltSystem.deinit ends with
        // `self.* = undefined`, so asking it afterwards is a read of
        // poisoned memory.
        const gpa = self.inner.allocator;
        gpa.free(self.terms);
        gpa.free(self.factors);
        self.inner.deinit();
        self.* = undefined;
    }
};

pub fn buildSystem(allocator: std.mem.Allocator) BuildError!BoundSystem {
    const n_specs = 2 * chunk.slots;
    const specs = try allocator.alloc(range.Spec, n_specs);
    defer allocator.free(specs);
    for (0..chunk.slots) |i| {
        specs[2 * i] = .{ .column = colNibA(i), .bit_base = colBitsA(i), .width = nibble_width };
        specs[2 * i + 1] = .{ .column = colNibB(i), .bit_base = colBitsB(i), .width = nibble_width };
    }

    var inner = range.BuiltSystem.init(allocator, chunk.system(), specs) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.BadWidth => return error.BadWidth,
    };
    errdefer inner.deinit();

    // One dequantization equation per operand:
    //   op - (nib - 8)*scale = op - nib*scale + 8*scale
    // 3 terms, 4 factors each. These allocations outlive this function:
    // the returned System points into them, so BoundSystem owns them.
    const terms = try allocator.alloc(Term, 3 * n_specs);
    errdefer allocator.free(terms);
    const factors = try allocator.alloc(Factor, 4 * n_specs);
    errdefer allocator.free(factors);
    const dequant = try allocator.alloc(Constraint, n_specs);
    defer allocator.free(dequant);

    for (0..n_specs) |j| {
        const slot = j / 2;
        const is_a = j % 2 == 0;
        const op: u16 = if (is_a) chunk.colA(slot) else chunk.colB(slot);
        const nib: u16 = if (is_a) colNibA(slot) else colNibB(slot);
        const scale: u16 = if (is_a) colScaleA(slot) else colScaleB(slot);
        const base = 4 * j;

        factors[base + 0] = .{ .column = .{ .index = op } };
        factors[base + 1] = .{ .column = .{ .index = nib } };
        factors[base + 2] = .{ .column = .{ .index = scale } };
        factors[base + 3] = .{ .column = .{ .index = scale } };

        terms[3 * j + 0] = .{ .factors = factors[base + 0 .. base + 1] };
        terms[3 * j + 1] = .{
            .factors = factors[base + 1 .. base + 3],
            .coefficient = Fp2.neg(Fp2.one),
        };
        terms[3 * j + 2] = .{
            .factors = factors[base + 3 .. base + 4],
            .coefficient = Fp2.re(Goldilocks.fromU64(8)),
        };
        dequant[j] = .{
            .name = if (is_a) "a_slot = (nib_a - 8)*scale_a" else "b_slot = (nib_b - 8)*scale_b",
            .scope = .composed,
            .terms = terms[3 * j .. 3 * j + 3],
        };
    }

    // Append the dequantization equations to the merged constraint list.
    const n = inner.constraints.len;
    const grown = try allocator.alloc(Constraint, n + dequant.len);
    @memcpy(grown[0..n], inner.constraints);
    @memcpy(grown[n..], dequant);
    allocator.free(inner.constraints);
    inner.constraints = grown;
    inner.system = .{ .constraints = grown };
    return .{ .inner = inner, .terms = terms, .factors = factors };
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

pub const BindError = error{ OutOfMemory, NibbleOutOfRange, InconsistentOperands };

/// Extend a chunked GEMM trace with the quantization columns.
///
/// `nib_a`/`nib_b` hold one raw nibble per MAC and `scale_a`/`scale_b` one
/// block scale per MAC (a Q4_K block covers 256 elements, so within a
/// 16-MAC chunk the scale is constant — but taking it per MAC keeps this
/// correct for chunks that straddle a block boundary).
pub fn bindOperands(
    allocator: std.mem.Allocator,
    gemm_trace: *const chunk.Trace,
    nib_a: []const u8,
    scale_a: []const Goldilocks,
    nib_b: []const u8,
    scale_b: []const Goldilocks,
) BindError!Trace {
    const rows = gemm_trace.rows;
    const k = nib_a.len;
    if (nib_b.len != k or scale_a.len != k or scale_b.len != k) return BindError.InconsistentOperands;
    const chunks = chunk.chunkRowsFor(k);
    if (chunks + 1 > rows) return BindError.InconsistentOperands;

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
    for (gemm_trace.columns, 0..) |src, i| @memcpy(cols[i], src);

    const last = rows - 1;
    const eight = Goldilocks.fromU64(8);
    for (0..chunks) |r| {
        for (0..chunk.slots) |i| {
            const idx = r * chunk.slots + i;
            const live = idx < k;
            const na: u8 = if (live) nib_a[idx] else 8;
            const nb: u8 = if (live) nib_b[idx] else 8;
            if (na > 15 or nb > 15) return BindError.NibbleOutOfRange;
            bindSlot(cols, r, i, true, na, if (live) scale_a[idx] else Goldilocks.zero);
            bindSlot(cols, r, i, false, nb, if (live) scale_b[idx] else Goldilocks.zero);

            // The GEMM operand column must be exactly the dequantization
            // of (nibble, scale) — checked here so a mismatch is reported
            // as such instead of surfacing later as a bare AIR violation.
            if (live) {
                const want_a = Goldilocks.fromU64(na).sub(eight).mul(scale_a[idx]);
                const want_b = Goldilocks.fromU64(nb).sub(eight).mul(scale_b[idx]);
                if (!want_a.eql(gemm_trace.columns[chunk.colA(i)][r].a)) return BindError.InconsistentOperands;
                if (!want_b.eql(gemm_trace.columns[chunk.colB(i)][r].a)) return BindError.InconsistentOperands;
            }
        }
    }
    // Padding chunks: all-zero operands, satisfied by nibble 8 / scale 0,
    // but the bit columns must still agree with the nibble.
    for (chunks..last) |r| {
        for (0..chunk.slots) |i| {
            bindSlot(cols, r, i, true, 8, Goldilocks.zero);
            bindSlot(cols, r, i, false, 8, Goldilocks.zero);
        }
    }
    // Closing row: slot 0 carries the cancelling product (nibble 9 with
    // the scale derived from the synthetic operand), the rest are zero.
    {
        bindSlot(cols, last, 0, true, 9, gemm_trace.columns[chunk.colA(0)][last].a);
        bindSlot(cols, last, 0, false, 9, gemm_trace.columns[chunk.colB(0)][last].a);
        for (1..chunk.slots) |i| {
            bindSlot(cols, last, i, true, 8, Goldilocks.zero);
            bindSlot(cols, last, i, false, 8, Goldilocks.zero);
        }
    }

    return .{ .rows = rows, .columns = cols };
}

/// Write one operand's nibble, its bit expansion and its scale into row
/// `r`, slot `i`. Cannot fail: the caller range-checks the nibble.
fn bindSlot(cols: [][]Fp2, r: usize, i: usize, is_a: bool, nib: u8, scale: Goldilocks) void {
    const nib_col = if (is_a) colNibA(i) else colNibB(i);
    const bits_col = if (is_a) colBitsA(i) else colBitsB(i);
    const scale_col = if (is_a) colScaleA(i) else colScaleB(i);
    cols[nib_col][r] = Fp2.re(Goldilocks.fromU64(nib));
    cols[scale_col][r] = Fp2.re(scale);
    for (0..nibble_width) |bit| {
        const shift: u3 = @intCast(bit);
        cols[bits_col + bit][r] = Fp2.re(Goldilocks.fromU64((nib >> shift) & 1));
    }
}

const testing = std.testing;

test "chunk_binding: layout leaves no column gaps" {
    try testing.expectEqual(@as(usize, 226), column_count);
    try testing.expectEqual(@as(u16, 34), colNibA(0));
    try testing.expectEqual(@as(u16, 35), colNibB(0));
    try testing.expectEqual(@as(u16, 36), colBitsA(0));
    try testing.expectEqual(@as(u16, 40), colBitsB(0));
    try testing.expectEqual(@as(u16, 44), colScaleA(0));
    try testing.expectEqual(@as(u16, 45), colScaleB(0));
    // Slot 1 starts 12 columns later and does not overlap slot 0.
    try testing.expectEqual(colNibA(0) + 12, colNibA(1));
    try testing.expect(colScaleB(0) < colNibA(1));
}

test "chunk_binding: the system holds every operand's range check and equation" {
    const a = testing.allocator;
    var sys = try buildSystem(a);
    defer sys.deinit();
    const s = sys.system();
    // chunked AIR (4 + 30 boundary) + 32 range checks x (1 + 4) + 32
    // dequantization equations.
    try testing.expectEqual(@as(usize, 4 + 30 + 32 * 5 + 32), s.constraints.len);
    try testing.expectEqual(@as(usize, 1 + 32 * 5 + 32), s.composedCount());
    try testing.expectEqual(@as(usize, 2), s.maxDegree());
    try testing.expectEqual(@as(?u16, @intCast(column_count - 1)), s.maxColumn());
}
