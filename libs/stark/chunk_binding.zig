//! Quantized-operand binding for the CHUNKED GEMM layout (F2).
//!
//! `quant_binding.zig` does this for one MAC per row. Here the same
//! binding is replicated over all 16 slots of a chunk, so the 16× smaller
//! trace is not a soundness regression: every one of the 32 operands in a
//! row is pinned to a raw 4-bit Q4_K nibble times its block scale.
//!
//! Column layout (on top of gemm_chunk's 34):
//!
//!   for slot i in 0..slots, 41 columns each:
//!     +0  nib_a        +1  nib_b
//!     +2  bits_a[0..4] +6  bits_b[0..4]
//!     +10 scale_a      +11 scale_b
//!     +12 mant_a[0..10]        +22 sel_a[0..16]
//!     +38 shift_a     +39 out_a  +40 sign_a
//!     (the same block for b, starting at +0 of the next half)
//!
//! The five blocks after the scales are the scale-provenance gadget from
//! `scale_air.zig`: each operand's scale is pinned to the image of
//! `fp16ToFixedQ4_22`, not left as a free field element.
//!
//! Everything is built in owned storage (the IR holds pointers): a
//! 26-bit-wide static table per slot would be unreadable, and 16 of them
//! worse.

const std = @import("std");
const expr = @import("./expr.zig");
const range = @import("./range.zig");
const chunk = @import("./gemm_chunk.zig");
const scale_air = @import("./scale_air.zig");
const air_builder = @import("./air_builder.zig");
const tensor = @import("../tensor/root.zig");

pub const Fp2 = expr.Fp2;
pub const System = expr.System;
pub const Constraint = expr.Constraint;
pub const Term = expr.Term;
pub const Factor = expr.Factor;
pub const Goldilocks = tensor.Goldilocks;

pub const gemm_columns: usize = chunk.column_count;
/// nibble, bits, scale, then the provenance gadget: 5 mantissa bits
/// columns of 10, 16 selector columns, and 3 singles, per operand.
pub const provenance_per_operand: u16 = 10 + 16 + 3;
pub const per_operand: u16 = 12 + provenance_per_operand;
pub const per_slot: u16 = 2 * per_operand;
pub const nibble_width: u8 = 4;

pub const column_count: usize = gemm_columns + chunk.slots * per_slot;

pub fn colNibA(i: usize) u16 {
    return @intCast(gemm_columns + i * per_slot);
}
pub fn colNibB(i: usize) u16 {
    return colNibA(i) + @as(u16, 1);
}
pub fn colBitsA(i: usize) u16 {
    return colNibA(i) + @as(u16, 2);
}
pub fn colBitsB(i: usize) u16 {
    return colNibA(i) + @as(u16, 6);
}
pub fn colScaleA(i: usize) u16 {
    return colNibA(i) + @as(u16, 10);
}
pub fn colScaleB(i: usize) u16 {
    return colNibA(i) + @as(u16, 11);
}
pub fn colMantA(i: usize) u16 {
    return colNibA(i) + @as(u16, 12);
}

pub fn colSelA(i: usize) u16 {
    return colNibA(i) + @as(u16, 22);
}
pub fn colShiftA(i: usize) u16 {
    return colNibA(i) + @as(u16, 38);
}
pub fn colOutA(i: usize) u16 {
    return colNibA(i) + @as(u16, 39);
}
pub fn colSignA(i: usize) u16 {
    return colNibA(i) + @as(u16, 40);
}
pub fn colMantB(i: usize) u16 {
    return colNibA(i) + @as(u16, 12) + per_operand;
}
pub fn colSelB(i: usize) u16 {
    return colNibA(i) + @as(u16, 22) + per_operand;
}
pub fn colShiftB(i: usize) u16 {
    return colNibA(i) + @as(u16, 38) + per_operand;
}
pub fn colOutB(i: usize) u16 {
    return colNibA(i) + @as(u16, 39) + per_operand;
}
pub fn colSignB(i: usize) u16 {
    return colNibA(i) + @as(u16, 40) + per_operand;
}

/// Provenance gadget layout for one operand of one slot.
pub fn cfgA(i: usize) scale_air.Config {
    return .{
        .scale = colScaleA(i),
        .mant_base = colMantA(i),
        .sel_base = colSelA(i),
        .shift_col = colShiftA(i),
        .out_col = colOutA(i),
        .sign_col = colSignA(i),
    };
}
pub fn cfgB(i: usize) scale_air.Config {
    return .{
        .scale = colScaleB(i),
        .mant_base = colMantB(i),
        .sel_base = colSelB(i),
        .shift_col = colShiftB(i),
        .out_col = colOutB(i),
        .sign_col = colSignB(i),
    };
}

pub const BuildError = error{ OutOfMemory, BadWidth, InvalidReductionLength };

/// Chunked GEMM AIR + nibble range checks + dequantization equations for
/// all 2·slots operands per row. Owns every allocation the returned
/// System points into.
pub const BoundSystem = struct {
    inner: range.BuiltSystem,
    terms: []Term,
    factors: []Factor,
    /// The provenance gadgets' own terms and factors; the merged
    /// constraint slice holds copies that point into them.
    gadget: air_builder.Owned,

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
        self.gadget.deinit();
        self.* = undefined;
    }
};

pub fn buildSystem(allocator: std.mem.Allocator, k: usize) BuildError!BoundSystem {
    const n_specs = 2 * chunk.slots;
    const specs = try allocator.alloc(range.Spec, n_specs);
    defer allocator.free(specs);
    for (0..chunk.slots) |i| {
        specs[2 * i] = .{ .column = colNibA(i), .bit_base = colBitsA(i), .width = nibble_width };
        specs[2 * i + 1] = .{ .column = colNibB(i), .bit_base = colBitsB(i), .width = nibble_width };
    }

    const base_system = chunk.system(k) catch return BuildError.InvalidReductionLength;
    var inner = range.BuiltSystem.init(allocator, base_system, specs) catch |e| switch (e) {
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

    // Scale provenance for all 2·slots operands, built through the
    // Builder because the gadget needs a comptime-known layout.
    var b = air_builder.Builder.init(allocator);
    defer b.deinit();
    for (0..chunk.slots) |i| {
        try scale_air.build(&b, cfgA(i));
        try scale_air.build(&b, cfgB(i));
    }
    const n_gadget = b.count();
    std.debug.assert(n_gadget == n_specs * (1 + scale_air.shift_count + 4));

    // Append the dequantization equations and the gadgets.
    const n = inner.constraints.len;
    const grown = try allocator.alloc(Constraint, n + dequant.len + n_gadget);
    @memcpy(grown[0..n], inner.constraints);
    @memcpy(grown[n .. n + dequant.len], dequant);
    const frozen = try air_builder.freeze(allocator, &b, 1);
    @memcpy(grown[n + dequant.len ..], frozen.constraints);
    allocator.free(inner.constraints);
    inner.constraints = grown;
    inner.system = inner.system.replaceConstraints(grown);
    return .{ .inner = inner, .terms = terms, .factors = factors, .gadget = frozen };
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

pub const BindError = error{
    OutOfMemory,
    NibbleOutOfRange,
    InconsistentOperands,
    UnsupportedReductionLength,
    PaddedTrace,
    /// An fp16 pattern no q4.22 scale can come from.
    BadScale,
};

/// Extend a chunked GEMM trace with the quantization columns.
///
/// `nib_a`/`nib_b` hold one raw nibble per MAC and `scale_a`/`scale_b` the
/// raw fp16 scale PATTERN per MAC (a Q4_K block covers 256 elements, so
/// within a 16-MAC chunk the scale is constant — but taking it per MAC
/// keeps this correct for chunks that straddle a block boundary).
pub fn bindOperands(
    allocator: std.mem.Allocator,
    gemm_trace: *const chunk.Trace,
    nib_a: []const u8,
    scale_a: []const u16,
    nib_b: []const u8,
    scale_b: []const u16,
) BindError!Trace {
    const rows = gemm_trace.rows;
    const k = nib_a.len;
    if (nib_b.len != k or scale_a.len != k or scale_b.len != k) return BindError.InconsistentOperands;
    const expected_rows = chunk.rowsFor(k) catch return BindError.UnsupportedReductionLength;
    if (rows != expected_rows) return BindError.PaddedTrace;
    const chunks = chunk.chunkRowsFor(k);

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

    const eight = Goldilocks.fromU64(8);
    for (0..chunks) |r| {
        for (0..chunk.slots) |i| {
            const idx = r * chunk.slots + i;
            const na = nib_a[idx];
            const nb = nib_b[idx];
            if (na > 15 or nb > 15) return BindError.NibbleOutOfRange;
            bindSlot(cols, r, i, true, na, scale_a[idx]) catch return BindError.BadScale;
            bindSlot(cols, r, i, false, nb, scale_b[idx]) catch return BindError.BadScale;

            const s_a = scaleFromFp16(scale_a[idx]) catch return BindError.BadScale;
            const s_b = scaleFromFp16(scale_b[idx]) catch return BindError.BadScale;
            const want_a = Goldilocks.fromU64(na).sub(eight).mul(s_a);
            const want_b = Goldilocks.fromU64(nb).sub(eight).mul(s_b);
            if (!want_a.eql(gemm_trace.columns[chunk.colA(i)][r].a)) return BindError.InconsistentOperands;
            if (!want_b.eql(gemm_trace.columns[chunk.colB(i)][r].a)) return BindError.InconsistentOperands;
        }
    }
    // The closing row is exempt from every composed constraint (see
    // gemm_chunk.zig): its slots are witness with nothing to satisfy, and
    // the binder needs no scale there. Leaving the columns zero is exactly
    // as valid as any other filling — and it is the only filling that does
    // not claim a synthetic operand is a dequantized one.
    return .{ .rows = rows, .columns = cols };
}

/// Write one operand's nibble, its bit expansion, its scale and the
/// provenance witness into row `r`, slot `i`. Fails only on an fp16
/// pattern that is not a usable q4.22 scale; the caller range-checks the
/// nibble.
fn bindSlot(
    cols: [][]Fp2,
    r: usize,
    i: usize,
    is_a: bool,
    nib: u8,
    bits: u16,
) BindError!void {
    const cfg = if (is_a) cfgA(i) else cfgB(i);
    const nib_col = if (is_a) colNibA(i) else colNibB(i);
    const bits_col = if (is_a) colBitsA(i) else colBitsB(i);
    cols[nib_col][r] = Fp2.re(Goldilocks.fromU64(nib));
    for (0..nibble_width) |bit| {
        const shift: u3 = @intCast(bit);
        cols[bits_col + bit][r] = Fp2.re(Goldilocks.fromU64((nib >> shift) & 1));
    }
    scale_air.writeWitness(
        cols,
        r,
        cfg.mant_base,
        cfg.sel_base,
        cfg.shift_col,
        cfg.out_col,
        cfg.sign_col,
        bits,
    ) catch return BindError.BadScale;
    cols[cfg.scale][r] = Fp2.re(scaleFromFp16(bits) catch return BindError.BadScale);
}

/// The single place an fp16 pattern becomes the field element the
/// dequantization equation consumes.
pub fn scaleFromFp16(bits: u16) tensor.Fp16Error!Goldilocks {
    return Goldilocks.fromU64(try tensor.fp16ToFixedQ4_22(bits));
}

const testing = std.testing;

test "chunk_binding: layout leaves no column gaps" {
    try testing.expectEqual(@as(usize, 1346), column_count);
    try testing.expectEqual(@as(u16, 34), colNibA(0));
    try testing.expectEqual(@as(u16, 35), colNibB(0));
    try testing.expectEqual(@as(u16, 36), colBitsA(0));
    try testing.expectEqual(@as(u16, 40), colBitsB(0));
    try testing.expectEqual(@as(u16, 44), colScaleA(0));
    try testing.expectEqual(@as(u16, 45), colScaleB(0));
    // The provenance block for A follows its scale, and B's whole block
    // (nibble..sign, 41 columns) follows A's.
    try testing.expectEqual(@as(u16, 46), colMantA(0));
    try testing.expectEqual(@as(u16, 56), colSelA(0));
    try testing.expectEqual(@as(u16, 72), colShiftA(0));
    try testing.expectEqual(@as(u16, 73), colOutA(0));
    try testing.expectEqual(@as(u16, 74), colSignA(0));
    try testing.expectEqual(@as(u16, 87), colMantB(0));
    try testing.expectEqual(@as(u16, 115), colSignB(0));
    // Every slot is 82 columns and none of them overlap.
    try testing.expectEqual(@as(usize, 82), per_slot);
    try testing.expectEqual(colNibA(0) + per_slot, colNibA(1));
    try testing.expect(colSignB(0) < colNibA(1));
}

test "chunk_binding: the system holds every operand's range check and equation" {
    const a = testing.allocator;
    var sys = try buildSystem(a, 240);
    defer sys.deinit();
    const s = sys.system();
    // chunked AIR (3: one composed, two boundary — the closing row is
    // exempt, so its 30 unused slots need no pins) + 32 range checks x
    // (1 + 4) + 32 dequantization equations + 32 provenance gadgets x 21.
    try testing.expectEqual(@as(usize, 3 + 32 * 5 + 32 + 32 * 21), s.constraints.len);
    try testing.expectEqual(@as(usize, 1 + 32 * 5 + 32 + 32 * 21), s.composedCount());
    try testing.expectEqual(@as(usize, 2), s.maxDegree());
    try testing.expectEqual(@as(usize, 1), s.transition_exemptions);
    try testing.expectEqual(@as(?u16, @intCast(column_count - 1)), s.maxColumn());
}
