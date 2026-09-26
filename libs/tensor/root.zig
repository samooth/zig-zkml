//! Quantization schemes and tensor types for zkML gadgets.
//!
//! docs/BLUE_PRINT.md §4.2: each scheme declares a comptime magnitude bound M
//! and the corresponding range proof bits. The AIR generates range proofs
//! by LogUp on bytes (§4.3).
//!
//! Fixed-point note: fp16 block scales are dequantized EXACTLY as q4.22
//! (4 integer + 22 fractional bits): a positive normal fp16 in
//! [2^-12, 2^4) with a 10-bit mantissa needs up to 12+10 = 22 fractional
//! bits. q8.16 would truncate and break the exact-arithmetic contract (§3).

const std = @import("std");
const field = @import("../field.zig");

pub const Goldilocks = field.Goldilocks;

/// Fractional bits of the scale fixed-point format (§4.2: q4.22).
pub const scale_frac_bits: u6 = 22;
/// Total bits used by a q4.22 value: |x| < 2^4 · 2^22 = 2^26.
pub const scale_total_bits: u6 = 26;

/// Parsed IEEE 754 half-precision (binary16).
pub const Fp16 = struct {
    negative: bool,
    /// Raw exponent field [0, 31].
    exp_raw: u5,
    /// Raw mantissa field [0, 1023].
    mantissa: u10,

    pub const bits: u16 = 16;

    pub fn fromBits(v: u16) Fp16 {
        return .{
            .negative = (v >> 15) != 0,
            .exp_raw = @intCast((v >> 10) & 0x1f),
            .mantissa = @intCast(v & 0x3ff),
        };
    }

    pub fn isNormal(self: Fp16) bool {
        return self.exp_raw != 0;
    }

    pub fn isInfOrNan(self: Fp16) bool {
        return self.exp_raw == 0x1f;
    }
};

pub const Fp16Error = error{
    /// ±inf / NaN scales are meaningless for quantization and MUST fail
    /// (scales are witness data — never silently misinterpret, §3).
    ScaleNotFinite,
    /// |scale| ≥ 2^4 overflows the q4.22 fixed-point budget (§4.2).
    ScaleTooLarge,
    /// |scale| < 2^-12 (subnormal, zero, or small normal): not exactly
    /// representable in q4.22 — the dequant contract requires exactness,
    /// so it must fail instead of silently truncating.
    ScaleTooSmall,
};

/// Dequantize an fp16 into q4.22 fixed point (exact).
///
/// value = (-1)^sign · 2^(e-15) · (1 + m/1024)  with e ∈ [1, 30],
/// m ∈ [0, 1023]. In q4.22: value·2^22 = (1024 + m)·2^(e+12), which is
/// an integer exactly when e ≥ −12 and below the 2^26 budget when
/// e ≤ 3 (value < 16 = 2^4). The accepted range is therefore
/// |scale| ∈ [2^-12, 2^4) — matching docs/BLUE_PRINT.md §4.2's budget note.
pub fn fp16ToFixedQ4_22(v: u16) Fp16Error!u64 {
    const f = Fp16.fromBits(v);
    if (f.isInfOrNan()) return error.ScaleNotFinite;

    const e: i32 = @as(i32, f.exp_raw) - 15;
    if (e > 3) return error.ScaleTooLarge; // value ≥ 2^4
    if (e < -12) return error.ScaleTooSmall; // subnormal/zero/small: inexact

    // value·2^22 = (1024 + m) · 2^(e+12), shift ∈ [0, 15] — always exact.
    const significand: u64 = 1024 + @as(u64, f.mantissa); // [1024, 2047]
    const shift: u6 = @intCast(e + 12);
    const magnitude: u64 = significand << shift; // < 2^11·2^15 = 2^26

    if (f.negative) {
        // Field negation, NOT two's complement: -x ≡ p - |x| (mod p).
        // (Two's complement via fromU64 would be off by 2^64 mod p = 8.)
        return Goldilocks.p - magnitude;
    }
    return magnitude;
}

/// Quantization scheme — part of the TYPE and of the statement (§6.1).
/// The magnitude bound M is comptime and the AIR generates the range proof.
///
/// STABILITY: these ordinals are serialized into statements
/// (`@intFromEnum` in statement/root.zig). Never reorder or renumber —
/// only append new variants (and bump the statement version then).
pub const Scheme = enum(u8) {
    int8_symmetric = 0,
    int4_gguf_q4_k = 1,
    int4_q8_0 = 2,
    fp8_e4m3 = 3,
    mxfp8_e4m3 = 4,
    fixed_q16_16 = 5,
    fixed_q8_8 = 6,

    /// Magnitude bound of the represented value (§4.2).
    pub fn magnitudeBound(comptime self: Scheme) usize {
        return switch (self) {
            .int8_symmetric, .fp8_e4m3, .mxfp8_e4m3 => 1 << 7,
            .int4_gguf_q4_k, .int4_q8_0 => 1 << 4,
            .fixed_q16_16 => 1 << 15,
            .fixed_q8_8 => 1 << 8,
        };
    }

    /// Bits per range proof (§4.3: LogUp on bytes).
    pub fn rangeBits(comptime self: Scheme) u8 {
        return switch (self) {
            .int8_symmetric, .fp8_e4m3, .mxfp8_e4m3 => 8,
            .int4_gguf_q4_k, .int4_q8_0 => 4,
            .fixed_q16_16 => 32,
            .fixed_q8_8 => 16,
        };
    }

    pub fn name(comptime self: Scheme) []const u8 {
        return switch (self) {
            .int8_symmetric => "int8_sym",
            .int4_gguf_q4_k => "q4_k",
            .int4_q8_0 => "q8_0",
            .fp8_e4m3 => "fp8_e4m3",
            .mxfp8_e4m3 => "mxfp8_e4m3",
            .fixed_q16_16 => "q16.16",
            .fixed_q8_8 => "q8.8",
        };
    }
};

/// Quantized tensor — a 2D matrix of quantized values with optional scales.
/// The scheme is part of the type (docs/BLUE_PRINT.md §7.2: "parte del TIPO").
pub fn QuantTensor(comptime scheme: Scheme) type {
    return struct {
        const Self = @This();

        /// Scheme carried by the type — distinguishes QuantTensor(.int8_symmetric)
        /// from QuantTensor(.fp8_e4m3) and enables runtime introspection.
        pub const quant_scheme: Scheme = scheme;

        rows: usize,
        cols: usize,
        data: []const u8,
        /// fp16 bits (GGML) according to scheme; empty for scale-less schemes.
        scales: []const u16,
        zeropoints: ?[]const u8 = null,

        pub fn shape(self: Self) struct { usize, usize } {
            return .{ self.rows, self.cols };
        }

        pub fn numel(self: Self) usize {
            return self.rows * self.cols;
        }
    };
}

/// Signed int4 nibble value (GGML convention): even index = low nibble.
pub fn q4Nibble(byte: u8, i: usize) i8 {
    const raw: u4 = if (i % 2 == 0)
        @intCast(byte & 0x0f)
    else
        @intCast((byte >> 4) & 0x0f);
    return @as(i8, raw) - 8; // [-8, 7]
}

/// Dequantize one Q4_K block (256 nibbles + per-block fp16 scale) to
/// Goldilocks elements in q4.22-scaled representation.
///
/// NOTE (v1 simplification): real GGML Q4_K super-blocks carry TWO fp16
/// values (d, m) plus 8 six-bit sub-scales per 256-element block. This
/// implements the symmetric single-scale variant (nibble − 8)·d used to
/// validate the stack; the full d/m format lands with the F2 gadget work.
pub fn dequantQ4K(nibbles: *const [128]u8, scale_fp16: u16) Fp16Error![256]Goldilocks {
    const s = try fp16ToFixedQ4_22(scale_fp16);
    const scale_g = Goldilocks.fromU64(s);
    var out: [256]Goldilocks = undefined;
    for (0..256) |i| {
        const v = q4Nibble(nibbles[i / 2], i);
        out[i] = Goldilocks.fromI64(v).mul(scale_g);
    }
    return out;
}

test "scheme magnitude bounds" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 128), Scheme.int8_symmetric.magnitudeBound());
    try t.expectEqual(@as(usize, 16), Scheme.int4_gguf_q4_k.magnitudeBound());
    try t.expectEqual(@as(usize, 32768), Scheme.fixed_q16_16.magnitudeBound());
    try t.expectEqual(@as(usize, 256), Scheme.fixed_q8_8.magnitudeBound());
}

test "scheme range bits" {
    const t = std.testing;
    try t.expectEqual(@as(u8, 8), Scheme.int8_symmetric.rangeBits());
    try t.expectEqual(@as(u8, 4), Scheme.int4_gguf_q4_k.rangeBits());
    try t.expectEqual(@as(u8, 32), Scheme.fixed_q16_16.rangeBits());
    try t.expectEqual(@as(u8, 16), Scheme.fixed_q8_8.rangeBits());
}

test "quant tensor shape and scheme type tag" {
    const t = std.testing;
    const T = QuantTensor(.int8_symmetric);
    const ten = T{ .rows = 3, .cols = 4, .data = &.{}, .scales = &.{} };
    const s = ten.shape();
    try t.expectEqual(@as(usize, 3), s[0]);
    try t.expectEqual(@as(usize, 4), s[1]);
    try t.expectEqual(@as(usize, 12), ten.numel());
    // The scheme is carried by the type.
    try t.expectEqual(Scheme.int8_symmetric, T.quant_scheme);
    const T2 = QuantTensor(.fp8_e4m3);
    try t.expectEqual(Scheme.fp8_e4m3, T2.quant_scheme);
}

test "fp16 to q4.22 exact dequant" {
    const t = std.testing;

    // 1.0 (0x3C00) → 2^22.
    try t.expectEqual(@as(u64, 1 << 22), try fp16ToFixedQ4_22(0x3C00));
    // 0.5 (0x3800) → 2^21.
    try t.expectEqual(@as(u64, 1 << 21), try fp16ToFixedQ4_22(0x3800));
    // 1.5 = 2^0·(1 + 512/1024) → (1024+512)·2^12.
    try t.expectEqual(@as(u64, 1536 << 12), try fp16ToFixedQ4_22(0x3E00));
    // 8.0 (0x4800) → 8·2^22 — the largest power-of-two in budget.
    try t.expectEqual(@as(u64, 8 << 22), try fp16ToFixedQ4_22(0x4800));
    // 14.0 (0x4B00) = 7·2·2^22 → (1024+0)·2^(1+12)·7/8... 14 = 7·2 →
    // (1.75·8)·2^22: 14·2^22 = 58_720_256.
    try t.expectEqual(@as(u64, 14 << 22), try fp16ToFixedQ4_22(0x4B00));
    // Smallest exact scale: 2^-12 (0x0C00) → (1024)·2^0 = 1024.
    try t.expectEqual(@as(u64, 1024), try fp16ToFixedQ4_22(0x0C00));
    // Smallest with full mantissa: 2^-12·(2047/1024) → 2047·2^0.
    // 0x0FFF: exp=3 (e=-12), mantissa=1023.
    try t.expectEqual(@as(u64, 2047), try fp16ToFixedQ4_22(0x0FFF));
    // Negative 1.0 → p − 2^22 (field negation, not two's complement).
    const neg = try fp16ToFixedQ4_22(0xBC00);
    try t.expectEqual(Goldilocks.p - (1 << 22), neg);
    const neg_g = Goldilocks.fromU64(neg);
    const pos_g = Goldilocks.fromU64(1 << 22);
    // neg + pos ≡ 0 (mod p) — the field checks the sign encoding.
    try t.expect(neg_g.add(pos_g).isZero());
}

test "fp16 rejects out-of-budget scales" {
    const t = std.testing;
    // +inf (0x7C00), -inf (0xFC00), NaN (0x7E00).
    try t.expectError(error.ScaleNotFinite, fp16ToFixedQ4_22(0x7C00));
    try t.expectError(error.ScaleNotFinite, fp16ToFixedQ4_22(0xFC00));
    try t.expectError(error.ScaleNotFinite, fp16ToFixedQ4_22(0x7E00));
    // 16.0 (0x4C00): e = 4 → ≥ 2^4, out of the q4.22 budget.
    try t.expectError(error.ScaleTooLarge, fp16ToFixedQ4_22(0x4C00));
    // 2^-13 (0x0A00): e = -13 → inexact in q4.22 (would need 23 frac bits).
    try t.expectError(error.ScaleTooSmall, fp16ToFixedQ4_22(0x0A00));
    // Subnormal (exp field 0, mantissa != 0) and ±zero.
    try t.expectError(error.ScaleTooSmall, fp16ToFixedQ4_22(0x00FF));
    try t.expectError(error.ScaleTooSmall, fp16ToFixedQ4_22(0x0000));
    try t.expectError(error.ScaleTooSmall, fp16ToFixedQ4_22(0x8000));
}

test "dequant q4k basic" {
    const t = std.testing;
    var nibbles: [128]u8 = [_]u8{0} ** 128;
    // nibble value 8 → 8 − 8 = 0 → zero regardless of scale.
    nibbles[0] = 0x08;
    const result = try dequantQ4K(&nibbles, 0x3C00);
    try t.expect(result[0].isZero());

    // nibble value 9 → +1 · scale 1.0 (q4.22) = 2^22.
    nibbles[0] = 0x09;
    const r2 = try dequantQ4K(&nibbles, 0x3C00);
    try t.expectEqual(@as(u64, 1 << 22), r2[0].toU64());

    // nibble value 0 → −8 · scale 0.5 (q4.22 = 2^21) = −2^24 ≡ p − 2^24.
    nibbles[0] = 0x00;
    const r3 = try dequantQ4K(&nibbles, 0x3800);
    const expected = Goldilocks.fromI64(-8).mul(Goldilocks.fromU64(1 << 21));
    try t.expectEqual(expected.toU64(), r3[0].toU64());

    // Malformed scale must propagate the error.
    try t.expectError(error.ScaleNotFinite, dequantQ4K(&nibbles, 0x7C00));
}
