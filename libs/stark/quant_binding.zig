//! Q4_K binding for the GEMM operands (F2): range checks plus the
//! dequantization equation.
//!
//! The GEMM AIR proves `C = A·B` over the trace's own a/b columns. Without
//! a binding to the quantized weights, a prover could choose any operands
//! it liked and still produce a valid proof. This adds the missing half for
//! a Q4_K-style symmetric block:
//!
//!   every row:  a = (nib_a - 8) * scale_a
//!                b = (nib_b - 8) * scale_b
//!                nib_a in [0, 15],  nib_b in [0, 15]   (GGML raw - 8)
//!
//! The nibble bounds come from `libs/stark/range.zig` (bit decomposition
//! plus `bit*(bit-1) = 0`), so they compose with the existing quadratic IR.
//!
//! ## What the scale column does and does not buy
//!
//! The scale is a witness column constrained only by the equation above.
//! That binds the *representation* — each operand is a raw 4-bit nibble
//! times one scale — but NOT the scale's provenance. LogUp itself now
//! exists (logup.zig) and could express the lookup; what is missing is
//! PINNING the table: a lookup argument proves w is a permutation of u, and
//! `u` is still witness, so a prover can set u = w. Pinning needs the table
//! in a preprocessed trace whose commitment root is a public input — the
//! multi-trace plumbing of F3. Until then an operand with a *valid* nibble
//! but a scale no fp16 could produce is still provable, and
//! `quant_test.zig` pins that boundary as the known gap.
//!
//! There is a cheaper route for fp16 specifically, and it is the planned
//! one: `fp16ToFixedQ4_22` is `(1024 + m)·2^(e+12)`, a variable shift, i.e.
//! a barrel shifter, i.e. `out = sel·a + (1-sel)·b` — degree 2 and already
//! expressible. That closes the gap without any public input at all.
//!
//! Column layout (on top of gemm_air's four):
//!
//!   4  nib_a       unsigned Q4_K nibble of the A operand
//!   5  nib_b       same for B
//!   6..9   bits_a  binary expansion of nib_a
//!   10..13 bits_b  binary expansion of nib_b
//!   14 scale_a
//!   15 scale_b
//!   16..25  mant_a   fp16 mantissa of scale_a
//!   26..41  sel_a    one-hot shift selector for scale_a
//!   42      shift_a  the selected power of two
//!   43      out_a    unsigned q4.22 magnitude
//!   44      sign_a   fp16 sign bit
//!   45..73  the same five blocks for scale_b
//!
//! Rows that are not MAC rows no longer exist: the trace is exactly the
//! MAC rows plus the synthetic closing row, and the closing row is exempt
//! from every composed constraint (see gemm_air.zig), so it carries no
//! binding at all. That exemption is what makes a per-row scale-provenance
//! constraint possible: previously the closing row had to be a
//! dequantized operand of `(1, -C)`, and no fp16 produces `-C`.

const std = @import("std");
const expr = @import("./expr.zig");
const range = @import("./range.zig");
const gemm_air = @import("./gemm_air.zig");
const scale_air = @import("./scale_air.zig");
const air_builder = @import("./air_builder.zig");
const tensor = @import("../tensor/root.zig");

pub const Fp2 = expr.Fp2;
pub const System = expr.System;
pub const Constraint = expr.Constraint;
pub const Term = expr.Term;
pub const Factor = expr.Factor;
pub const Goldilocks = tensor.Goldilocks;

pub const col_nib_a: u16 = 4;
pub const col_nib_b: u16 = 5;
pub const col_bits_a: u16 = 6; // 6..9
pub const col_bits_b: u16 = 10; // 10..13
pub const col_scale_a: u16 = 14;
pub const col_scale_b: u16 = 15;
pub const col_mant_a: u16 = 16; // 16..25
pub const col_sel_a: u16 = 26; // 26..41
pub const col_shift_a: u16 = 42;
pub const col_out_a: u16 = 43;
pub const col_sign_a: u16 = 44;
pub const col_mant_b: u16 = 45; // 45..54
pub const col_sel_b: u16 = 55; // 55..70
pub const col_shift_b: u16 = 71;
pub const col_out_b: u16 = 72;
pub const col_sign_b: u16 = 73;
pub const column_count: usize = 74;

pub const nibble_width: u8 = 4;

/// The provenance gadget for one side: which columns it owns.
pub const scale_cfg_a: scale_air.Config = .{
    .scale = col_scale_a,
    .mant_base = col_mant_a,
    .sel_base = col_sel_a,
    .shift_col = col_shift_a,
    .out_col = col_out_a,
    .sign_col = col_sign_a,
};
pub const scale_cfg_b: scale_air.Config = .{
    .scale = col_scale_b,
    .mant_base = col_mant_b,
    .sel_base = col_sel_b,
    .shift_col = col_shift_b,
    .out_col = col_out_b,
    .sign_col = col_sign_b,
};

/// Container-scope storage: System holds pointers into these.
///
///   a - (nib_a - 8)*scale_a = a - nib_a*scale_a + 8*scale_a
///
/// Note the last term is `8*scale_a`, NOT the constant 8: the bias is
/// inside the product with the scale.
const kAFactors = [_]Factor{.{ .column = .{ .index = gemm_air.col_a } }};
const kNibScaleA = [_]Factor{
    .{ .column = .{ .index = col_nib_a } },
    .{ .column = .{ .index = col_scale_a } },
};
const kEightScaleA = [_]Factor{.{ .column = .{ .index = col_scale_a } }};
const kDequantATerms = [_]Term{
    .{ .factors = &kAFactors },
    .{ .factors = &kNibScaleA, .coefficient = Fp2.neg(Fp2.one) },
    .{ .factors = &kEightScaleA, .coefficient = Fp2.re(Goldilocks.fromU64(8)) },
};

const kBFactors = [_]Factor{.{ .column = .{ .index = gemm_air.col_b } }};
const kNibScaleB = [_]Factor{
    .{ .column = .{ .index = col_nib_b } },
    .{ .column = .{ .index = col_scale_b } },
};
const kEightScaleB = [_]Factor{.{ .column = .{ .index = col_scale_b } }};
const kDequantBTerms = [_]Term{
    .{ .factors = &kBFactors },
    .{ .factors = &kNibScaleB, .coefficient = Fp2.neg(Fp2.one) },
    .{ .factors = &kEightScaleB, .coefficient = Fp2.re(Goldilocks.fromU64(8)) },
};

const kDequantConstraints = [_]Constraint{
    .{ .name = "a = (nib_a - 8)*scale_a", .scope = .composed, .terms = &kDequantATerms },
    .{ .name = "b = (nib_b - 8)*scale_b", .scope = .composed, .terms = &kDequantBTerms },
};

/// The range checks the operand columns need.
pub const range_specs = [_]range.Spec{
    .{ .column = col_nib_a, .bit_base = col_bits_a, .width = nibble_width },
    .{ .column = col_nib_b, .bit_base = col_bits_b, .width = nibble_width },
};

pub const BuildError = error{ OutOfMemory, BadWidth, InvalidReductionLength };

/// Owns the merged system: GEMM AIR + nibble range checks + dequant
/// equations + scale provenance. `buildSystem(k)` stays valid until
/// `deinit`.
pub const BoundSystem = struct {
    inner: range.BuiltSystem,
    /// The gadget's own terms and factors. The merged constraint slice
    /// holds COPIES of its Constraint structs, and each of those points
    /// into these buffers — dropping them would leave the system pointing
    /// at freed memory. This is the same ownership shape as
    /// `range.BuiltSystem.checks`.
    gadget: air_builder.Owned,

    pub fn system(self: BoundSystem) System {
        return self.inner.system;
    }

    pub fn deinit(self: *BoundSystem) void {
        self.inner.deinit();
        self.gadget.deinit();
    }
};

/// Constraints the scale-provenance gadget adds, per side. 1 sign boolean +
/// 16 selector booleans + one-hot + `M = 2^s` + the shifted significand +
/// the signed magnitude = 21.
const kProvenancePerSide: usize = 1 + scale_air.shift_count + 4;

/// GEMM AIR + dequantization equations + nibble range checks + scale
/// provenance.
pub fn buildSystem(allocator: std.mem.Allocator, k: usize) BuildError!BoundSystem {
    const base = gemm_air.system(k) catch return BuildError.InvalidReductionLength;
    var inner = range.BuiltSystem.init(allocator, base, &range_specs) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.BadWidth => return error.BadWidth,
    };
    errdefer inner.deinit();

    // Scale provenance is built through the Builder (it needs a
    // comptime-known layout), then appended to the merged slice.
    var b = air_builder.Builder.init(allocator);
    defer b.deinit();
    try scale_air.build(&b, scale_cfg_a);
    try scale_air.build(&b, scale_cfg_b);
    const n_gadget = b.count();
    std.debug.assert(n_gadget == 2 * kProvenancePerSide);

    // Append the dequantization equations and the gadget. BuiltSystem
    // allocates the merged slice once; growing it here means reallocating
    // that one slice.
    const n = inner.constraints.len;
    const grown = try allocator.alloc(Constraint, n + kDequantConstraints.len + n_gadget);
    @memcpy(grown[0..n], inner.constraints);
    @memcpy(grown[n .. n + kDequantConstraints.len], &kDequantConstraints);
    // `freeze` turns the builder's index-based recording into slices over
    // its own buffers, replicated `rows` times. One row is the compact
    // form; BuiltSystem replicates them itself. Ownership moves to the
    // returned BoundSystem.
    const frozen = try air_builder.freeze(allocator, &b, 1);
    @memcpy(grown[n + kDequantConstraints.len ..], frozen.constraints);
    allocator.free(inner.constraints);
    inner.constraints = grown;
    inner.system = inner.system.replaceConstraints(grown);
    return .{ .inner = inner, .gadget = frozen };
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
    /// An fp16 pattern no q4.22 scale can come from (subnormal, too
    /// small, too large, inf or NaN).
    BadScale,
};

/// The seam where raw model bytes become a prover input.
///
/// The caller passes the raw fp16 PATTERN per MAC, not a field element: the
/// AIR needs mantissa, exponent and sign separately to prove the scale is
/// the image of `fp16ToFixedQ4_22`, and a `Goldilocks` would throw that
/// provenance away before the witness is built. `scaleFromFp16` stays the
/// single place where a pattern becomes the field element the dequantization
/// equation consumes.
pub fn scaleFromFp16(bits: u16) tensor.Fp16Error!Goldilocks {
    return Goldilocks.fromU64(try tensor.fp16ToFixedQ4_22(bits));
}

/// Extend a GEMM trace with the quantization columns.
///
/// `nib_a` / `nib_b` hold one raw Q4_K nibble (0..15) per MAC, and
/// `scale_a` / `scale_b` one block scale PER MAC — a reduction that spans
/// several 256-element Q4_K blocks has a different scale in each, so a
/// single scalar would refuse honest multi-block witnesses. The GEMM
/// operand columns must equal `(nibble - 8) * scale` exactly, or the
/// trace is not a dequantization of anything and is refused here (the AIR
/// would reject it too, but later and less legibly).
pub fn bindOperands(
    allocator: std.mem.Allocator,
    gemm_trace: *const gemm_air.Trace,
    nib_a: []const u8,
    scale_a: []const u16,
    nib_b: []const u8,
    scale_b: []const u16,
) BindError!Trace {
    const rows = gemm_trace.rows;
    const real = gemm_air.realRowsFor(nib_a.len);
    if (nib_b.len != nib_a.len or scale_a.len != nib_a.len or scale_b.len != nib_a.len) {
        return BindError.InconsistentOperands;
    }
    const expected_rows = gemm_air.rowsFor(real) catch return BindError.UnsupportedReductionLength;
    if (rows != expected_rows) return BindError.PaddedTrace;

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
    // Carry the GEMM columns through unchanged.
    for (gemm_trace.columns, 0..) |src, i| @memcpy(cols[i], src);

    for (0..rows) |r| {
        if (r >= real) {
            // The closing row is exempt from every composed constraint, so
            // its binding columns are witness with nothing to satisfy. It
            // is not a MAC row: the operands there are synthetic, and a
            // scale there would have to be the image of an fp16 with no
            // reason to exist. Leave the columns zero.
            continue;
        }
        const nib_a_row: u8 = nib_a[r];
        const nib_b_row: u8 = nib_b[r];
        if (nib_a_row > 15 or nib_b_row > 15) return BindError.NibbleOutOfRange;

        cols[col_nib_a][r] = Fp2.re(Goldilocks.fromU64(nib_a_row));
        cols[col_nib_b][r] = Fp2.re(Goldilocks.fromU64(nib_b_row));
        for (0..nibble_width) |bit| {
            const shift: u3 = @intCast(bit);
            cols[col_bits_a + bit][r] = Fp2.re(Goldilocks.fromU64((nib_a_row >> shift) & 1));
            cols[col_bits_b + bit][r] = Fp2.re(Goldilocks.fromU64((nib_b_row >> shift) & 1));
        }

        // Scale provenance: witness the gadget from the fp16 pattern, then
        // take the scale the dequantization equation uses from the SAME
        // function. Deriving the field element from the gadget's own `out`
        // instead would let the witness and the constraint disagree.
        scale_air.writeWitness(
            cols,
            r,
            col_mant_a,
            col_sel_a,
            col_shift_a,
            col_out_a,
            col_sign_a,
            scale_a[r],
        ) catch return BindError.BadScale;
        scale_air.writeWitness(
            cols,
            r,
            col_mant_b,
            col_sel_b,
            col_shift_b,
            col_out_b,
            col_sign_b,
            scale_b[r],
        ) catch return BindError.BadScale;

        const scale_a_row = scaleFromFp16(scale_a[r]) catch return BindError.BadScale;
        const scale_b_row = scaleFromFp16(scale_b[r]) catch return BindError.BadScale;
        cols[col_scale_a][r] = Fp2.re(scale_a_row);
        cols[col_scale_b][r] = Fp2.re(scale_b_row);

        const a_val = gemm_trace.columns[gemm_air.col_a][r].a;
        const b_val = gemm_trace.columns[gemm_air.col_b][r].a;
        const eight = Goldilocks.fromU64(8);
        const want_a = Goldilocks.fromU64(nib_a_row).sub(eight).mul(scale_a_row);
        const want_b = Goldilocks.fromU64(nib_b_row).sub(eight).mul(scale_b_row);
        if (!want_a.eql(a_val) or !want_b.eql(b_val)) return BindError.InconsistentOperands;
    }
    return .{ .rows = rows, .columns = cols };
}
