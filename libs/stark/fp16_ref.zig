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

pub const exp_bits: u8 = 5;
pub const mant_bits: u8 = 10;
pub const mant_implicit: u16 = 1 << mant_bits; // 1024
pub const exp_bias: i32 = 15;
pub const exp_max: u16 = 31;
pub const exp_min_normal: u16 = 1;
pub const mant_mask: u16 = (1 << mant_bits) - 1;

pub const Parts = struct {
    sign: u1,
    exponent: u5,
    mantissa: u10,

    pub fn fromBits(bits: u16) Parts {
        return .{
            .sign = @intCast((bits >> 15) & 1),
            .exponent = @intCast((bits >> mant_bits) & (exp_max)),
            .mantissa = @intCast(bits & mant_mask),
        };
    }

    pub fn toBits(p: Parts) u16 {
        return (@as(u16, p.sign) << 15) | (@as(u16, p.exponent) << mant_bits) | p.mantissa;
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

/// Integer significand: the 11-bit `1024 + m` for a normal, the raw `m`
/// for a subnormal or zero, and 0 for inf/NaN. The exponent the value is
/// anchored to is returned alongside so the caller can reconstruct
/// (−1)^s · sig · 2^shift without ever leaving integer arithmetic.
pub const Decomposed = struct {
    significand: u16,
    /// value = (−1)^sign · significand · 2^shift
    shift: i32,
    special: enum { finite, zero, inf, nan },
};

pub fn decompose(bits: u16) Decomposed {
    const p = Parts.fromBits(bits);
    if (p.isNaN()) return .{ .significand = 0, .shift = 0, .special = .nan };
    if (p.isInf()) return .{ .significand = 0, .shift = 0, .special = .inf };
    if (p.isZero()) return .{ .significand = 0, .shift = 0, .special = .zero };
    if (p.exponent == 0) {
        // Subnormal: value = m · 2^−24.
        return .{ .significand = p.mantissa, .shift = -24, .special = .finite };
    }
    return .{
        .significand = mant_implicit + p.mantissa,
        .shift = @as(i32, p.exponent) - exp_bias - mant_bits,
        .special = .finite,
    };
}

/// Round a 32-bit significand down to `keep` bits, round-to-nearest-even.
/// Returns the kept value plus the round and sticky decisions, so a prover
/// has exactly the two boolean witnesses the AIR will check.
pub const Rounded = struct {
    kept: u32,
    round_bit: u1,
    sticky: u1,
    /// Carry out of the kept field: 0, or 1 when rounding pushed it past
    /// its top bit. Wider than a bit because a caller may round a value
    /// whose kept part is already at the field maximum.
    carry: u2,
};

/// Round `value` to `keep` low bits, round-to-nearest-even.
///
/// The dropped part is bits 0..keep-1, so the ROUND bit is bit keep−1 and
/// the STICKY bit summarises bits 0..keep−2. Getting this off by one turns
/// a tie into a sticky value and silently rounds the wrong way, which is
/// exactly the bug this function had for one iteration.
pub fn roundToNearestEven(value: u64, keep: u8) Rounded {
    std.debug.assert(keep > 0 and keep < 64);
    const kept = @as(u32, @intCast(value >> @intCast(keep)));
    const sticky_mask = (@as(u64, 1) << @intCast(keep - 1)) - 1;
    const round_bit: u1 = @intCast((value >> @intCast(keep - 1)) & 1);
    const dropped = value & sticky_mask;
    const sticky: u1 = if (dropped == 0) 0 else 1;
    // RNE: increment iff round and (sticky or an odd kept value).
    const lsb = kept & 1;
    const increment: u32 = if (round_bit == 1 and (sticky == 1 or lsb == 1)) 1 else 0;
    const sum = kept + increment;
    return .{
        .kept = sum,
        .round_bit = round_bit,
        .sticky = sticky,
        .carry = @intCast(sum >> @intCast(keep)),
    };
}

pub const MultiplyError = error{
    /// S1 scope: subnormal inputs are rejected, never approximated. A
    /// subnormal significand is below 2^10, so the normal-case
    /// normalisation below does not apply to it; returning a wrong answer
    /// silently would be the one unforgivable bug in a rounding
    /// implementation, so it errors instead. Handled in S2.
    SubnormalInput,
    /// The exact result is subnormal but NON-ZERO (between 2^−24 and
    /// 2^−14). The S1 path can only produce normals and zero, so it would
    /// have to return a plausible-looking zero — refused instead. Handled
    /// in S2 together with SubnormalInput.
    SubnormalResult,
};

/// Bit-exact binary16 multiply: the engine's rounding, reproduced.
///
/// Scope of S1: normal × normal, with the full special-value set (signed
/// zero, inf, NaN) handled exactly. Subnormal INPUTS error out rather
/// than approximating; subnormal RESULTS currently round to zero or to the
/// smallest normal, which S2 replaces with the real denormal path.
pub fn multiply(a: u16, b: u16) MultiplyError!u16 {
    const da = decompose(a);
    const db = decompose(b);
    if (da.special == .finite and da.significand < mant_implicit) return error.SubnormalInput;
    if (db.special == .finite and db.significand < mant_implicit) return error.SubnormalInput;

    switch (da.special) {
        .nan => return nanBits(),
        .inf => {
            if (db.special == .zero) return nanBits(); // inf * 0
            return infBits(Parts.fromBits(a).sign ^ Parts.fromBits(b).sign);
        },
        .zero => {
            if (db.special == .inf) return nanBits();
            return signedZero(Parts.fromBits(a).sign ^ Parts.fromBits(b).sign);
        },
        .finite => {},
    }
    switch (db.special) {
        .nan => return nanBits(),
        .inf => return infBits(Parts.fromBits(a).sign ^ Parts.fromBits(b).sign),
        .zero => return signedZero(Parts.fromBits(a).sign ^ Parts.fromBits(b).sign),
        .finite => {},
    }

    const sign = Parts.fromBits(a).sign ^ Parts.fromBits(b).sign;
    // The exact product is da.significand · db.significand · 2^(da.shift +
    // db.shift) — an exact integer, never rounded here.
    const product: u32 = @as(u32, da.significand) * @as(u32, db.significand);
    const exact_shift: i32 = da.shift + db.shift;

    // Normalize. A finite non-zero fp16 is (1024 + m) · 2^(e−25), so each
    // significand is in [2^10, 2^11) and the product P is in [2^20, 2^22).
    // The result must be a significand back in [2^10, 2^11), so P is
    // rounded to ELEVEN bits when P >= 2^21 and to TEN when it is not —
    // that one bit of data dependence is the whole normalization, and it
    // is why the AIR needs a normalisation shift witness.
    std.debug.assert(product >= (1 << 20) and product < (1 << 22));
    const keep_bits: u8 = if (product >= (1 << 21)) 11 else 10;

    const r = roundToNearestEven(product, keep_bits);
    var kept = r.kept;
    var carry: u1 = 0;
    // The normalised significand field is [1024, 2048) whichever
    // keep_bits was used, so the carry threshold is ALWAYS 2048 — not
    // 1 << keep_bits, which fires on an already-normal value when
    // keep_bits is 10. (That bug broke `1.0 · min_normal`.)
    if (kept == (@as(u32, mant_implicit) << 1)) {
        // Rounding carried out of the top bit: the significand becomes
        // exactly 1.0 and the exponent steps up by one.
        kept = @as(u32, mant_implicit);
        carry = 1;
    }
    std.debug.assert(kept >= mant_implicit and kept < (@as(u32, mant_implicit) << 1));
    const result_shift: i32 = exact_shift + @as(i32, keep_bits) + @as(i32, carry);
    const exponent: i32 = result_shift + mant_bits + exp_bias;
    if (exponent >= @as(i32, exp_max)) return infBits(sign);
    if (exponent == 0) {
        // Exactly 2^−24: the smallest subnormal, which S1 cannot encode.
        return error.SubnormalResult;
    }
    if (exponent < 0) {
        // Below 2^−24 the result is genuinely zero — that IS the correct
        // answer, and unlike the band above it needs no denormal path.
        if (result_shift + mant_bits >= -24) return error.SubnormalResult;
        return signedZero(sign);
    }
    return pack(sign, @intCast(exponent), @as(u10, @intCast(kept - mant_implicit)));
}

fn pack(sign: u1, exponent: u5, mantissa: u10) u16 {
    if (exponent >= exp_max) {
        return if (mantissa == 0) infBits(sign) else nanBits();
    }
    return (Parts{ .sign = sign, .exponent = exponent, .mantissa = mantissa }).toBits();
}

pub fn infBits(sign: u1) u16 {
    return (@as(u16, sign) << 15) | (@as(u16, exp_max) << mant_bits);
}
pub fn nanBits() u16 {
    return (@as(u16, exp_max) << mant_bits) | 1;
}
pub fn signedZero(sign: u1) u16 {
    return @as(u16, sign) << 15;
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
        if (p.exponent == 0) {
            // Subnormal inputs are S2; they must ERROR, not approximate.
            if (p.mantissa != 0) {
                try testing.expectError(error.SubnormalInput, multiply(0x3C00, v));
            }
            continue;
        }
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

test "fp16 ref: subnormals error instead of approximating" {
    // 0x0001 is the smallest subnormal (exponent field 0); multiplying it
    // needs the denormal path, so the honest answer until S2 lands is an
    // error, not a plausible-looking number.
    try testing.expectError(error.SubnormalInput, multiply(0x0001, 0x4000));
    try testing.expectError(error.SubnormalInput, multiply(0x0400, 0x0001));

    // A subnormal RESULT from two normals: 2^−12 · 2^−12 = 2^−24 is the
    // smallest subnormal and NOT zero, so the S1 path must refuse rather
    // than return 0.
    try testing.expectError(error.SubnormalResult, multiply(0x1000, 0x1000));
    // But (2^−14)² = 2^−28 is genuinely below the subnormal range, so
    // zero is the CORRECT answer there and must be produced.
    try testing.expectEqual(@as(u16, 0x0000), try multiply(0x0400, 0x0400));
}

test "fp16 ref: random patterns are stable and symmetric" {
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rng = prng.random();
    for (0..20000) |_| {
        const a: u16 = @intCast(rng.int(u16));
        const b: u16 = @intCast(rng.int(u16));
        const p = multiply(a, b) catch |e| switch (e) {
            error.SubnormalInput => {
                // A NORMAL input must never fall into the subnormal-input
                // path: that would mean the guard is wrong.
                try testing.expect(
                    Parts.fromBits(a).exponent == 0 or Parts.fromBits(b).exponent == 0,
                );
                continue;
            },
            error.SubnormalResult => {
                // Reaching here means both operands are finite and
                // non-zero, and subnormal inputs already errored above, so
                // both must be NORMAL with a product in the subnormal band
                // (e.g. 2^−14 · 2^−11 = 2^−25).
                try testing.expect(Parts.fromBits(a).exponent >= 1);
                try testing.expect(Parts.fromBits(b).exponent >= 1);
                continue;
            },
        };
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
