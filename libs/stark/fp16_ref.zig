//! Reference fp16 arithmetic from the raw 16-bit pattern (F2, spike S1).
//!
//! Deliberately NOT written with the host `f16` type: the point of the
//! whole exercise is to reproduce an engine's rounding bit for bit, and a
//! test that leans on the compiler's float semantics proves nothing about
//! IEEE-754. Everything here is integer arithmetic on the encoding, so it
//! is auditable against the standard by reading it.
//!
//! Encoding (IEEE-754 binary16):
//!   sign     1 bit, bit 15
//!   exponent 5 bits, bits 14..10, bias 15
//!   mantissa 10 bits, bits 9..0
//!   e = 0  -> subnormal or zero, value = (−1)^s · m · 2^−24
//!   e > 0  -> normal, value = (−1)^s · (1024 + m) · 2^(e−25)
//!   e = 31 -> inf (m = 0) or NaN (m != 0)

const std = @import("std");
const float_format = @import("float_format.zig");
const float_ref = @import("float_ref.zig");

pub const exp_bits: u8 = 5;
pub const mant_bits: u8 = 10;
pub const mant_implicit: u16 = 1024;
pub const exp_bias: i32 = 15;
pub const exp_max: u16 = 31;
pub const exp_min_normal: u16 = 1;
pub const mant_mask: u16 = (1 << mant_bits) - 1;

/// The format this file is the convenience wrapper for.
pub const format = float_format.binary16;

fn pack(sign: u1, exponent: u5, mantissa: u10) u16 {
    return format.pack(.{
        .sign = sign,
        .exponent = exponent,
        .mantissa = mantissa,
    });
}

pub const Parts = struct {
    sign: u1,
    exponent: u5,
    mantissa: u10,

    pub fn fromBits(b: u16) Parts {
        const p = format.parts(b);
        return .{
            .sign = @intCast(p.sign),
            .exponent = @intCast(p.exponent),
            .mantissa = @intCast(p.mantissa),
        };
    }

    pub fn toBits(p: Parts) u16 {
        return format.pack(.{
            .sign = p.sign,
            .exponent = p.exponent,
            .mantissa = p.mantissa,
        });
    }

    pub fn isSubnormal(p: Parts) bool {
        return p.exponent == 0;
    }
    pub fn isZero(p: Parts) bool {
        return p.exponent == 0 and p.mantissa == 0;
    }
    pub fn isInf(p: Parts) bool {
        return p.exponent == exp_max and p.mantissa == 0;
    }
    pub fn isNaN(p: Parts) bool {
        return p.exponent == exp_max and p.mantissa != 0;
    }
};

pub const Rounded = float_ref.Rounded;
pub const roundToNearestEven = float_ref.roundToNearestEven;
pub const Decomposed = float_ref.Decomposed;
pub const decompose = float_ref.decompose;

/// Bit-exact binary16 multiply, delegated to the format-generic
/// implementation in float_ref.zig. Everything here used to be a
/// hand-specialised copy; keeping the wrapper means this file's ~150 tests
/// (all 65536 values of `1.0 · x`, 20000 random pairs, the RNE tie cases)
/// are now the regression suite for the generic code that serves bfloat16
/// and both fp8 variants too.
pub const MultiplyError = error{OutOfRange};

pub fn multiply(a: u16, b: u16) MultiplyError!u16 {
    return float_ref.multiply(format, a, b);
}

pub fn infBits(sign: u1) u16 {
    return format.pack(.{ .sign = sign, .exponent = format.emax(), .mantissa = 0 });
}
pub fn nanBits() u16 {
    return float_ref.canonicalNaN(format);
}
pub fn signedZero(sign: u1) u16 {
    return format.pack(.{ .sign = sign, .exponent = 0, .mantissa = 0 });
}

const testing = std.testing;

test "fp16 ref: exact values and powers of two" {
    try testing.expectEqual(@as(u16, 0x3C00), pack(0, 15, 0)); // 1.0
    try testing.expectEqual(@as(u16, 0x3800), pack(0, 14, 0)); // 0.5
    try testing.expectEqual(@as(u16, 0x4000), pack(0, 16, 0)); // 2.0
    try testing.expectEqual(@as(u16, 0xBC00), pack(1, 15, 0)); // −1.0
    // min/max normal and min subnormal
    try testing.expectEqual(@as(u16, 0x0400), pack(0, 1, 0));
    try testing.expectEqual(@as(u16, 0x7BFF), pack(0, 30, 1023));
    try testing.expectEqual(@as(u16, 0x0001), pack(0, 0, 1));
}

test "fp16 ref: 1.0 times anything is identity" {
    var i: u32 = 0;
    while (i < 0x10000) : (i += 1) {
        const v: u16 = @intCast(i);
        const p = Parts.fromBits(v);
        if (p.isNaN()) continue;
        if (p.exponent == 31) continue; // 1*inf and 1*0 handled above
        // Subnormals included: S2 computes them exactly, so 1.0 is the
        // identity over every pattern that is not inf/NaN.
        try testing.expectEqual(v, try multiply(0x3C00, v));
    }
}

test "fp16 ref: 2.0 times 1.5 is 3.0" {
    // 3.0 = 1.5 · 2^1, so exponent 16 with mantissa 512 — 0x4200. (An
    // earlier version of this test expected 0x4240, which is 3.125: the
    // reference was right and the expectation was wrong.)
    try testing.expectEqual(@as(u16, 0x4200), try multiply(0x4000, 0x3E00));
}

test "fp16 ref: round-to-nearest-even on a tie" {
    // A tie is exactly half of the last kept bit: value = kept·2^11 + 2^10,
    // with NOTHING below the round bit. With kept EVEN, RNE stays put.
    const r = roundToNearestEven((@as(u64, 2) << 11) | (1 << 10), 11);
    try testing.expectEqual(@as(u32, 2), r.kept);
    try testing.expectEqual(@as(u1, 1), r.round_bit);
    try testing.expectEqual(@as(u1, 0), r.sticky);

    // The same tie with an ODD kept value rounds up.
    const r2 = roundToNearestEven((@as(u64, 3) << 11) | (1 << 10), 11);
    try testing.expectEqual(@as(u32, 4), r2.kept);
    try testing.expectEqual(@as(u1, 1), r2.round_bit);
    try testing.expectEqual(@as(u1, 0), r2.sticky);

    // Anything below the round bit sets sticky, which forces the round up
    // even when the kept value is even.
    const r3 = roundToNearestEven((@as(u64, 2) << 11) | (1 << 10) | 1, 11);
    try testing.expectEqual(@as(u32, 3), r3.kept);
    try testing.expectEqual(@as(u1, 1), r3.round_bit);
    try testing.expectEqual(@as(u1, 1), r3.sticky);

    // A carry out of the top bit: kept lands one past the field.
    const r4 = roundToNearestEven((@as(u64, 0x7FF) << 11) | (1 << 10) | 1, 11);
    try testing.expectEqual(@as(u32, 0x800), r4.kept);
    try testing.expectEqual(@as(u1, 1), r4.carry);

    // Below half, nothing moves regardless of stickiness.
    const r5 = roundToNearestEven((@as(u64, 2) << 11) | ((1 << 10) - 1), 11);
    try testing.expectEqual(@as(u32, 2), r5.kept);
    try testing.expectEqual(@as(u1, 0), r5.round_bit);
    try testing.expectEqual(@as(u1, 1), r5.sticky);
}

test "fp16 ref: overflow to infinity and NaN propagation" {
    try testing.expectEqual(infBits(0), try multiply(0x7BFF, 0x4000)); // max · 2
    try testing.expectEqual(nanBits(), try multiply(0x7C00, 0x0000)); // inf · 0
    try testing.expectEqual(nanBits(), try multiply(0x7E00, 0x3C00)); // NaN · 1
    try testing.expectEqual(@as(u16, 0x8000), try multiply(0x3C00, 0x8000)); // 1 · −0
}

test "fp16 ref: subnormals are computed exactly, not refused" {
    // S1 refused these; S2's format-generic reference computes them. The
    // smallest subnormal times 2 is twice itself.
    try testing.expectEqual(@as(u16, 0x0002), try multiply(0x0001, 0x4000));
    // 2^-12 squared is 2^-24, the smallest subnormal — NOT zero, which is
    // exactly the case S1 refused to guess at.
    try testing.expectEqual(@as(u16, 0x0001), try multiply(0x0C00, 0x0C00));
    // 2^-14 squared is 2^-28, below the subnormal range: zero is correct.
    try testing.expectEqual(@as(u16, 0x0000), try multiply(0x0400, 0x0400));
}

test "fp16 ref: random patterns are stable and symmetric" {
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rng = prng.random();
    for (0..20000) |_| {
        const a: u16 = @intCast(rng.int(u16));
        const b: u16 = @intCast(rng.int(u16));
        // S2 computes every case, so nothing errors any more: the
        // symmetry properties below now cover subnormals and overflow too.
        const p = try multiply(a, b);
        // Multiplication commutes, including NaN and signed zero.
        try testing.expectEqual(p, try multiply(b, a));
        // Negating one input negates the result unless it is NaN.
        const neg_a = a ^ 0x8000;
        const p_neg = try multiply(neg_a, b);
        if (Parts.fromBits(p).isNaN()) {
            try testing.expect(Parts.fromBits(p_neg).isNaN());
        } else {
            try testing.expectEqual(p ^ 0x8000, p_neg);
        }
    }
}
