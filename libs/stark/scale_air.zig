//! Scale provenance: prove a Q4_K scale column is the image of an fp16,
//! in the quadratic IR, with no lookup and no preprocessed trace.
//!
//! `tensor.fp16ToFixedQ4_22` maps a finite fp16 to `±(1024 + m)·2^s` with
//! `m < 2^10` and `s ∈ [0, 15]` (the exponent range that keeps the value in
//! `[2^-12, 2^4)`). quant_binding already proves the dequantization
//! equation, but the scale itself was witness, so a fabricated scale proved.
//! This gadget pins the scale to that image.
//!
//! ## Why a one-hot shift selector and not the barrel shifter
//!
//! The natural encoding of `·2^s` is a barrel shifter, and `barrel.zig`
//! exists for it. Measured, it is the wrong tool here: a 26-bit vector with
//! a 4-bit amount costs 131 composed constraints and 157 columns PER SCALE
//! (see the plan's F2 notes), against a 16-column trace today. The shift is
//! only 4 bits wide, so selecting it costs less than shifting by it.
//!
//! The gadget is three degree-2 constraints plus the booleanity of the
//! selector bits:
//!
//!   sel_s boolean,      sum_s sel_s = 1        exactly one s, so M = 2^s
//!   M = sum_s 2^s·sel_s                        linear
//!   out = (2^10 + m)·M                        one quadratic
//!
//! with the 10 mantissa bits boolean (reusing the range gadget's shape) and
//! the sign as one more boolean plus `scale = ±out`. That is ~30 composed
//! constraints and ~29 columns per scale, a fifth of the barrel's, with no
//! new dependency and nothing to precompute.
//!
//! ## What this does and does not pin
//!
//! The scale is now in the image of the fp16 scale map: no fabricated field
//! element can pass, and every valid fp16 has a witness. It does NOT pin
//! WHICH fp16 the model used — that is the weight commitment, which is F3's
//! job (public inputs over a second trace). This gadget makes the scale a
//! *dequantizable* value rather than an arbitrary one.

const std = @import("std");
const expr = @import("expr.zig");
const bld = @import("air_builder.zig");
const tensor = @import("../tensor/root.zig");

pub const Fp2 = expr.Fp2;
pub const Builder = bld.Builder;
pub const LinTerm = bld.LinTerm;
pub const Goldilocks = tensor.Goldilocks;

pub const GadgetError = bld.BuildError;

/// Mantissa width of the significand `1024 + m`, and the number of mantissa
/// bits the caller supplies. fp16 has a 10-bit mantissa.
pub const mant_bits: u16 = 10;
/// Number of admissible shifts: s ∈ [0, 15] for a q4.22 fp16 scale.
pub const shift_bits: u16 = 4;
pub const shift_count: u16 = 1 << shift_bits;
/// The implicit leading bit: a significand is always `2^10 + m`.
pub const implicit_bit: u64 = 1 << mant_bits;

pub const Config = struct {
    /// The scale column this gadget constrains (`col_scale_a`/`_b`).
    scale: u16,
    /// Mantissa bits, LSB first.
    mant_base: u16,
    /// One-hot shift selector, LSB first.
    sel_base: u16,
    /// The reconstructed `M = 2^s`.
    shift_col: u16,
    /// The unsigned magnitude `(2^10 + m)·2^s`.
    out_col: u16,
    /// The sign bit; 0 = positive, 1 = negative (field negation).
    sign_col: u16,
};

pub fn cname(comptime what: []const u8) []const u8 {
    return comptime std.fmt.comptimePrint("scale: {s}", .{what});
}

/// Emit the gadget's constraints into `b` for a RUNTIME layout.
///
/// The per-slot chunked layout has 32 different column bases, so the
/// constraints cannot be built as comptime constants there. Recording
/// indices while building and resolving them at freeze time (see
/// `air_builder`) is exactly what the Builder is for.
pub fn build(b: *Builder, cfg: Config) GadgetError!void {
    // The sign is a bit of an fp16 pattern, so it is boolean like any other.
    try b.boolean(cname("sign is boolean"), cfg.sign_col);

    for (0..shift_count) |s| {
        try b.boolean(cname("shift selector bit is boolean"), cfg.sel_base + @as(u16, @intCast(s)));
    }

    // sum_s sel_s - 1 = 0: with every sel_s boolean, exactly one is 1.
    {
        var terms: [shift_count + 1]LinTerm = undefined;
        var n: usize = 0;
        for (0..shift_count) |s| {
            terms[n] = .{ .factors = try b.one(cfg.sel_base + @as(u16, @intCast(s))) };
            n += 1;
        }
        terms[n] = .{ .factors = try b.constant(bld.kNegOne) };
        try b.lin(cname("exactly one shift is selected"), .composed, &terms);
    }

    // M = sum_s 2^s·sel_s. Linear, and forced to a power of two by the two
    // constraints above, which is what makes `·M` a variable shift.
    {
        var terms: [shift_count + 1]LinTerm = undefined;
        var n: usize = 0;
        for (0..shift_count) |s| {
            terms[n] = .{
                .factors = try b.one(cfg.sel_base + @as(u16, @intCast(s))),
                .coefficient = g(@as(u64, 1) << @intCast(s)),
            };
            n += 1;
        }
        terms[n] = .{ .factors = try b.one(cfg.shift_col), .coefficient = bld.kNegOne };
        try b.lin(cname("shift column is the selected power of two"), .composed, &terms);
    }

    // out - (2^10 + m)·M = 0, written out so it stays degree 2: each
    // mantissa bit times M is ONE term, and the implicit leading bit rides
    // as a constant times M. A single term holding all of them would be
    // their product, not their sum.
    //
    //   out - 2^10·M - sum_j 2^j·(m_j·M) = 0
    {
        var terms: [mant_bits + 2]LinTerm = undefined;
        var n: usize = 0;
        terms[n] = .{ .factors = try b.one(cfg.out_col) };
        n += 1;
        terms[n] = .{
            .factors = try b.one(cfg.shift_col),
            .coefficient = Fp2.neg(g(implicit_bit)),
        };
        n += 1;
        for (0..mant_bits) |j| {
            terms[n] = .{
                .factors = try b.pair(cfg.mant_base + @as(u16, @intCast(j)), cfg.shift_col),
                .coefficient = Fp2.neg(g(@as(u64, 1) << @intCast(j))),
            };
            n += 1;
        }
        try b.lin(cname("out is the shifted significand"), .composed, &terms);
    }

    // scale = ±out, as `scale - out + 2·sign·out = 0`: one quadratic, and
    // the negation is the field's (p - x), which is what scaleFromFp16
    // returns for a negative fp16.
    try b.lin(cname("scale is the signed magnitude"), .composed, &.{
        .{ .factors = try b.one(cfg.scale) },
        .{ .factors = try b.one(cfg.out_col), .coefficient = bld.kNegOne },
        .{ .factors = try b.pair(cfg.sign_col, cfg.out_col), .coefficient = g(2) },
    });
}

fn g(v: u64) Fp2 {
    return Fp2.re(Goldilocks.fromU64(v));
}

/// Fill the gadget's columns for one row from an fp16 scale.
///
/// This is the witness side of the same three constraints, computed rather
/// than read back: `M` is the selected power of two, `out` the shifted
/// significand, and the scale the signed magnitude. `bits` is the raw fp16
/// pattern, so the caller cannot accidentally witness a scale the reference
/// would not produce — the point of the gadget.
pub fn writeWitness(
    columns: [][]Fp2,
    r: usize,
    mant_base: u16,
    sel_base: u16,
    shift_col: u16,
    out_col: u16,
    sign_col: u16,
    bits: u16,
) tensor.Fp16Error!void {
    const f = tensor.Fp16.fromBits(bits);
    if (f.isInfOrNan()) return error.ScaleNotFinite;
    const e: i32 = @as(i32, f.exp_raw) - 15;
    if (e > 3) return error.ScaleTooLarge;
    if (e < -12) return error.ScaleTooSmall;

    const m: u64 = f.mantissa;
    for (0..mant_bits) |j| {
        columns[mant_base + @as(u16, @intCast(j))][r] =
            g((m >> @intCast(j)) & 1);
    }
    const s: u16 = @intCast(e + 12);
    for (0..shift_count) |i| {
        columns[sel_base + @as(u16, @intCast(i))][r] = g(if (i == s) 1 else 0);
    }
    columns[shift_col][r] = g(@as(u64, 1) << @intCast(s));
    columns[out_col][r] = g((implicit_bit + m) << @intCast(s));
    columns[sign_col][r] = g(if (f.negative) 1 else 0);
}
