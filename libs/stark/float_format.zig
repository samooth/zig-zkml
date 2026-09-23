//! Binary floating-point format parameters (F2, S2).
//!
//! The fp16 AIR in fp16_air.zig was written for one format with the
//! widths spelled out. "Works for any quantized or unquantized model"
//! needs bf16 and fp8 too, and the STRUCTURE is identical across them —
//! only the field widths and the exponent bias change. This module is
//! that description, so the AIR can be written once and instantiated per
//! format.
//!
//! The formats that actually matter for inference weights and activations:
//!
//!   | format    | exp | mant | bias | normal range          | used by            |
//!   |-----------|-----|------|------|-----------------------|--------------------|
//!   | binary16  |   5 |   10 |   15 | 2^−14 .. (2−2^−10)·2^15 | fp16 tensors       |
//!   | bfloat16  |   8 |    7 |  127 | 2^−126 .. ~2^128      | bf16 activations   |
//!   | fp8 e4m3  |   4 |    3 |    7 | 2^−6 .. 448           | fp8 weights (MX)   |
//!   | fp8 e5m2  |   2 |    2 |   15 | 2^−14 .. 57344        | fp8 scales (MX)    |
//!
//! ## Encoding, uniformly
//!
//!   sign     1 bit, the top bit
//!   exponent E bits, bias `bias`
//!   mantissa M bits
//!
//!   E = 0          subnormal or zero: value = (−1)^s · m · 2^(min_shift)
//!   1 ≤ E ≤ max−1  normal: value = (−1)^s · (2^M + m) · 2^(E − bias − M)
//!   E = emax       infinity (m = 0) or NaN (m != 0)
//!
//! where `min_shift = 1 − bias − M` (the value of the smallest subnormal is
//! 2^(min_shift)) and `emax = 2^E_bits − 1`. For binary16: emax = 31,
//! min_shift = 1 − 15 − 10 = −24 ✓. For bfloat16: emax = 255,
//! min_shift = 1 − 127 − 7 = −133 ✓.
//!
//! ## Why this is not a table of special cases
//!
//! Every derived quantity the AIR needs is a function of (E_bits, M_bits,
//! bias) alone: the significand width, the product width, the kept width,
//! which product bit decides normalisation, and where the round and sticky
//! bits sit. That is why one implementation covers all four.

const std = @import("std");

pub const Format = struct {
    name: []const u8,
    exp_bits: u8,
    mant_bits: u8,
    bias: i32,
    /// Formats with no infinities (e.g. MXFP8's UE8M0 scale) set this and
    /// treat the all-ones exponent as a normal number.
    has_inf_nan: bool = true,

    /// The value is `significand · 2^shift`, with the significand
    /// normalised to [2^M, 2^(M+1)) for normals.
    pub fn mantImplicit(self: Format) u16 {
        return @intCast(@as(u64, 1) << @intCast(self.mant_bits));
    }

    /// All-ones exponent: infinity / NaN, or a normal when `has_inf_nan`
    /// is false.
    pub fn emax(self: Format) u16 {
        return @intCast((@as(u64, 1) << @intCast(self.exp_bits)) - 1);
    }

    /// Largest normal exponent field (one below emax when there are
    /// infinities).
    pub fn e_normal_max(self: Format) u16 {
        return if (self.has_inf_nan) self.emax() - 1 else self.emax();
    }

    /// Value of the smallest subnormal: 2^(min_shift).
    pub fn minShift(self: Format) i32 {
        return 1 - self.bias - @as(i32, self.mant_bits);
    }

    /// Value of the smallest normal: 2^(minShift + mant_bits) = 2^(1 − bias).
    pub fn minNormalShift(self: Format) i32 {
        return 1 - self.bias;
    }

    /// Significand width including the implicit leading one.
    pub fn sigBits(self: Format) u8 {
        return self.mant_bits + 1;
    }

    /// Width of the EXACT product of two significands. Since the AIR
    /// decomposes the product into bits, this is the number of columns and
    /// booleanity constraints it costs.
    pub fn productBits(self: Format) u8 {
        return 2 * self.sigBits();
    }

    /// How many bits of the product survive rounding: `sigBits` when the
    /// product needs the full width, one less when it does not. The AIR
    /// picks between them with a single witness bit (see fp16_air.zig).
    pub fn keptHigh(self: Format) u8 {
        return self.sigBits();
    }
    pub fn keptLow(self: Format) u8 {
        return self.sigBits() - 1;
    }

    /// The product bit that decides normalisation: the product of two
    /// significands in [2^M, 2^(M+1)) is in [2^(2M), 2^(2M+2)), so bit
    /// (2M+1) being set is exactly "the product is at least 2^(2M+1)".
    pub fn normBit(self: Format) u8 {
        return 2 * @as(u8, self.mant_bits) + 1;
    }

    pub fn byteWidth(self: Format) u16 {
        return 1 + @as(u16, self.exp_bits) + self.mant_bits;
    }

    /// Extract the sign / exponent / mantissa fields of a pattern.
    pub fn parts(self: Format, bits: u16) Parts {
        const width: u4 = @intCast(self.byteWidth() - 1);
        const mant: u4 = @intCast(self.mant_bits);
        return .{
            .sign = @truncate((bits >> width) & 1),
            .exponent = @truncate((bits >> mant) & self.emax()),
            .mantissa = @truncate(bits & (@as(u16, 1) << mant) - 1),
        };
    }

    pub fn pack(self: Format, p: Parts) u16 {
        const width: u4 = @intCast(self.byteWidth() - 1);
        const mant: u4 = @intCast(self.mant_bits);
        return (@as(u16, p.sign) << width) |
            (@as(u16, p.exponent) << mant) |
            p.mantissa;
    }

    pub const Parts = struct {
        sign: u1,
        exponent: u16,
        mantissa: u16,
    };
};

pub const binary16 = Format{
    .name = "binary16",
    .exp_bits = 5,
    .mant_bits = 10,
    .bias = 15,
};

pub const bfloat16 = Format{
    .name = "bfloat16",
    .exp_bits = 8,
    .mant_bits = 7,
    .bias = 127,
};

pub const fp8_e4m3 = Format{
    .name = "fp8_e4m3",
    .exp_bits = 4,
    .mant_bits = 3,
    .bias = 7,
};

/// E5M2 means FIVE exponent bits and TWO mantissa bits (1+5+2 = 8). It
/// was briefly defined with 2 exponent bits, which is a different — and
/// unrepresentable — format: with a 2-bit field and bias 15 the maximum
/// is 2^−12, so not even 1.0 exists.
pub const fp8_e5m2 = Format{
    .name = "fp8_e5m2",
    .exp_bits = 5,
    .mant_bits = 2,
    .bias = 15,
};

/// The formats the F2 float AIR is instantiated for, in the order the
/// tests exercise them.
pub const all = [_]Format{ binary16, bfloat16, fp8_e4m3, fp8_e5m2 };

const testing = std.testing;

test "format: derived widths for binary16 match the spike's constants" {
    try testing.expectEqual(@as(u16, 1024), binary16.mantImplicit());
    try testing.expectEqual(@as(u16, 31), binary16.emax());
    try testing.expectEqual(@as(u16, 30), binary16.e_normal_max());
    try testing.expectEqual(@as(i32, -24), binary16.minShift());
    try testing.expectEqual(@as(i32, -14), binary16.minNormalShift());
    try testing.expectEqual(@as(u8, 11), binary16.sigBits());
    try testing.expectEqual(@as(u8, 22), binary16.productBits());
    try testing.expectEqual(@as(u8, 21), binary16.normBit());
    try testing.expectEqual(@as(u16, 16), binary16.byteWidth());
}

test "format: the same structure holds for the other three" {
    try testing.expectEqual(@as(u16, 128), bfloat16.mantImplicit());
    try testing.expectEqual(@as(u16, 255), bfloat16.emax());
    try testing.expectEqual(@as(i32, -133), bfloat16.minShift());
    try testing.expectEqual(@as(u8, 8), bfloat16.sigBits());
    try testing.expectEqual(@as(u8, 16), bfloat16.productBits());
    try testing.expectEqual(@as(u16, 16), bfloat16.byteWidth());

    // fp8 e4m3: exponent 4 bits, so emax = 15 and the largest normal is
    // 14 (448 = 1.75 · 2^8).
    try testing.expectEqual(@as(u16, 8), fp8_e4m3.mantImplicit());
    try testing.expectEqual(@as(u16, 15), fp8_e4m3.emax());
    try testing.expectEqual(@as(i32, -9), fp8_e4m3.minShift());
    try testing.expectEqual(@as(i32, -6), fp8_e4m3.minNormalShift());
    try testing.expectEqual(@as(u8, 4), fp8_e4m3.sigBits());
    try testing.expectEqual(@as(u8, 8), fp8_e4m3.productBits());
    try testing.expectEqual(@as(u16, 8), fp8_e4m3.byteWidth());

    // fp8 e5m2: the only 2-bit exponent format, bias 15 like binary16.
    try testing.expectEqual(@as(u16, 4), fp8_e5m2.mantImplicit());
    try testing.expectEqual(@as(u16, 31), fp8_e5m2.emax());
    try testing.expectEqual(@as(i32, -16), fp8_e5m2.minShift());
    try testing.expectEqual(@as(u8, 3), fp8_e5m2.sigBits());
    try testing.expectEqual(@as(u8, 6), fp8_e5m2.productBits());
}

test "format: pack/parts round-trips every pattern of the small formats" {
    inline for (.{ fp8_e4m3, fp8_e5m2 }) |f| {
        const mask: u16 = @intCast((@as(u64, 1) << @intCast(f.byteWidth())) - 1);
        var v: u16 = 0;
        while (true) : (v += 1) {
            const p = f.parts(v & mask);
            try testing.expectEqual(v & mask, f.pack(p));
            if (v & mask == mask) break;
        }
    }
}
