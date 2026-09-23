//! Reference binary floating-point multiply, generic over the format (F2, S2).
//!
//! This is fp16_ref.zig's algorithm lifted from "binary16 with 5 and 10
//! spelled out" to "any (exp_bits, mant_bits, bias)" — the four formats in
//! float_format.zig differ only in those three numbers, and the structure
//! of IEEE-754 rounding does not change with them.
//!
//! `fp16_ref.zig` delegates here, so the ~150 existing binary16 tests are
//! now the test suite for THIS code. That is the safety net: a refactor
//! that breaks bfloat16 or fp8 cannot do it without also breaking fp16.
//!
//! Everything is integer arithmetic on the encoding. No host float type is
//! involved, because the whole point is to reproduce what an ENGINE does,
//! and a test that leans on the compiler's float semantics proves nothing
//! about IEEE-754.

const std = @import("std");
const fmt_lib = @import("float_format.zig");

pub const Format = fmt_lib.Format;
pub const Error = error{OutOfRange};

/// Round `value` to `keep` low bits, round-to-nearest-even.
///
/// The dropped part is bits 0..keep-1, so the ROUND bit is bit keep−1 and
/// the STICKY bit summarises bits 0..keep−2. Off by one here turns every
/// exact tie into a sticky value and silently rounds the wrong way.
pub const Rounded = struct {
    kept: u64,
    round_bit: u1,
    sticky: u1,
    /// Carry out of the kept field, as a count. Wide because a caller
    /// rounding at a small `keep` — the subnormal reduction rounds the
    /// significand itself — can carry several bits at once.
    carry: u32,
};

pub fn roundToNearestEven(value: u64, keep: u8) Rounded {
    std.debug.assert(keep > 0 and keep < 64);
    const kept = value >> @intCast(keep);
    const sticky_mask = (@as(u64, 1) << @intCast(keep - 1)) - 1;
    const round_bit: u1 = @intCast((value >> @intCast(keep - 1)) & 1);
    const dropped = value & sticky_mask;
    const sticky: u1 = if (dropped == 0) 0 else 1;
    // RNE: increment iff round and (sticky or an odd kept value).
    const lsb = kept & 1;
    const increment: u64 = if (round_bit == 1 and (sticky == 1 or lsb == 1)) 1 else 0;
    const sum = kept + increment;
    return .{
        .kept = sum,
        .round_bit = round_bit,
        .sticky = sticky,
        .carry = @intCast(sum >> @intCast(keep)),
    };
}

/// The value of a pattern as `sign · significand · 2^shift`, with the
/// significand NORMALISED to [2^M, 2^(M+1)) even for subnormal inputs —
/// that normalisation is what makes one code path serve every case.
pub const Decomposed = struct {
    sign: u1,
    significand: u32,
    shift: i32,
    special: enum { finite, zero, inf, nan },
};

pub fn decompose(f: Format, bits: u16) Decomposed {
    const p = f.parts(bits);
    if (f.has_inf_nan and p.exponent == f.emax()) {
        return .{
            .sign = p.sign,
            .significand = 0,
            .shift = 0,
            .special = if (p.mantissa == 0) .inf else .nan,
        };
    }
    if (p.exponent == 0) {
        if (p.mantissa == 0) {
            return .{ .sign = p.sign, .significand = 0, .shift = 0, .special = .zero };
        }
        // Subnormal: m · 2^minShift, rewritten as a normalised significand
        // by shifting the mantissa left until its top bit lands at
        // position M. That single normalisation is what lets one code path
        // serve normals and subnormals alike — and it is the reason the AIR
        // needs no separate subnormal branch for the INPUT side.
        const m: u32 = p.mantissa;
        const top_bit: i32 = 31 - @as(i32, @intCast(@clz(m)));
        const shift_left: u5 = @intCast(@as(i32, f.mant_bits) - top_bit);
        return .{
            .sign = p.sign,
            .significand = m << shift_left,
            .shift = f.minShift() - @as(i32, shift_left),
            .special = .finite,
        };
    }
    return .{
        .sign = p.sign,
        .significand = @as(u32, f.mantImplicit()) + p.mantissa,
        .shift = @as(i32, p.exponent) - f.bias - @as(i32, f.mant_bits),
        .special = .finite,
    };
}

/// Bit-exact multiply. Handles zero, infinity, NaN, normals and
/// subnormals on both sides, and produces every representable result
/// including subnormals and overflow to infinity.
pub fn multiply(f: Format, a: u16, b: u16) !u16 {
    const da = decompose(f, a);
    const db = decompose(f, b);
    const sign = da.sign ^ db.sign;

    // Special-value matrix.
    if (da.special == .nan or db.special == .nan) return canonicalNaN(f);
    if (da.special == .inf or db.special == .inf) {
        if (da.special == .zero or db.special == .zero) return canonicalNaN(f);
        return packSpecial(f, sign, .inf);
    }
    if (da.special == .zero or db.special == .zero) return packSpecial(f, sign, .zero);

    // Exact product: an integer, never rounded here.
    const product: u64 = @as(u64, da.significand) * @as(u64, db.significand);
    const exact_shift: i32 = da.shift + db.shift;
    const implicit: u64 = f.mantImplicit();
    // Both significands are in [2^M, 2^(M+1)), so the product is in
    // [2^(2M), 2^(2M+2)) and the result's significand width is M+1 kept
    // bits when the product reaches 2^(2M+1), and M otherwise.
    std.debug.assert(product >= (implicit * implicit));
    const keep: u8 = if (product >= (implicit << @intCast(f.normBit() - @as(u8, f.mant_bits)))) f.keptHigh() else f.keptLow();

    const r = roundToNearestEven(product, keep);
    var kept = r.kept;
    var carry: u1 = 0;
    if (kept == (implicit << 1)) {
        kept = implicit;
        carry = 1;
    }
    // value = kept · 2^(exact_shift + keep + carry)
    const result_shift: i32 = exact_shift + @as(i32, keep) + @as(i32, carry);

    // Re-attach the exponent. The result's value is
    // significand · 2^(shift) with significand in [implicit, 2·implicit),
    // so the biased exponent field is `shift + mant_bits + bias`.
    const exponent: i32 = result_shift + @as(i32, f.mant_bits) + f.bias;
    // Compare against the largest NORMAL exponent field. When the format
    // has infinities that is emax - 1, because emax itself encodes
    // inf/NaN: checking against emax let a rounded overflow fall through
    // and pack exponent = emax with a non-zero mantissa, i.e. a NaN where
    // the answer is infinity.
    const limit: i32 = @as(i32, f.e_normal_max());
    if (exponent > limit) {
        // Overflow: infinity where the format has one, otherwise the
        // largest finite value (saturation is the only option left).
        if (f.has_inf_nan) return packSpecial(f, sign, .inf);
        return f.pack(.{
            .sign = sign,
            .exponent = @intCast(limit),
            .mantissa = @intCast((@as(u32, 1) << @intCast(f.mant_bits)) - 1),
        });
    }
    if (exponent > 0) {
        return f.pack(.{
            .sign = sign,
            .exponent = @intCast(exponent),
            .mantissa = @intCast(kept - implicit),
        });
    }
    // exponent <= 0: the result is subnormal or zero. The subnormal
    // encoding is m · 2^minShift with m < implicit, so the kept significand
    // has to shift right by (1 − exponent) — the range reduction, and the
    // one part of this operation that is NOT a single rounding step.
    // Compute the reduction wide and clamp BEFORE narrowing: with a small
    // exponent field (fp8 e4m3 has 4) a tiny result needs a shift far past
    // 15, and @intCast to u4 panics instead of flushing to zero.
    const reduction: i32 = 1 - exponent;
    if (reduction > @as(i32, f.keptHigh()) + 2) {
        // Below half of the smallest subnormal: flushes to zero, which is
        // what round-to-nearest-even says.
        return packSpecial(f, sign, .zero);
    }
    const reduced = roundToNearestEven(kept, @intCast(reduction));
    if (reduced.kept >= implicit) {
        // Rounding carried into the smallest normal.
        return f.pack(.{ .sign = sign, .exponent = 1, .mantissa = 0 });
    }
    return f.pack(.{
        .sign = sign,
        .exponent = 0,
        .mantissa = @intCast(reduced.kept),
    });
}

fn packSpecial(f: Format, sign: u1, which: enum { zero, inf }) u16 {
    return switch (which) {
        .zero => f.pack(.{ .sign = sign, .exponent = 0, .mantissa = 0 }),
        .inf => f.pack(.{ .sign = sign, .exponent = f.emax(), .mantissa = 0 }),
    };
}

/// The canonical NaN of a format: all-ones exponent with the mantissa's
/// top bit set, which every IEEE-754 producer recognises and no arithmetic
/// operation turns into a number.
pub fn canonicalNaN(f: Format) u16 {
    return f.pack(.{
        .sign = 0,
        .exponent = f.emax(),
        .mantissa = @as(u16, 1) << (@as(u4, @intCast(f.mant_bits)) - 1),
    });
}

const testing = std.testing;

test "float ref: binary16 matches the spike's answers exactly" {
    const f = fmt_lib.binary16;
    try testing.expectEqual(@as(u16, 0x3E00), try multiply(f, 0x3C00, 0x3E00)); // 1 · 1.5
    try testing.expectEqual(@as(u16, 0x4200), try multiply(f, 0x4000, 0x3E00)); // 2 · 1.5 = 3
    try testing.expectEqual(@as(u16, 0x3C00), try multiply(f, 0x4000, 0x3800)); // 2 · 0.5
    try testing.expectEqual(@as(u16, 0x3E00), try multiply(f, 0x3C00, 0x3E00));
    try testing.expectEqual(@as(u16, 0x7C00), try multiply(f, 0x7BFF, 0x4000)); // overflow
    // 0x1000 is exponent field 4, i.e. 2^-11, NOT 2.0 — reading the bit
    // layout wrong is how this test first "failed" against a correct
    // implementation.
    try testing.expectEqual(@as(u16, 0x0004), try multiply(f, 0x1000, 0x1000)); // 2^-22, subnormal
    try testing.expectEqual(@as(u16, 0x0001), try multiply(f, 0x0C00, 0x0C00)); // 2^-24 = min subnormal
    try testing.expectEqual(@as(u16, 0x0000), try multiply(f, 0x0400, 0x0400)); // underflow
    try testing.expectEqual(@as(u16, 0x8000), try multiply(f, 0x3C00, 0x8000)); // 1 · −0
    try testing.expectEqual(@as(u16, 0xFC00), try multiply(f, 0x7C00, 0xBC00)); // inf · −1
    try testing.expectEqual(canonicalNaN(f), try multiply(f, 0x7C00, 0x0000)); // inf · 0
    try testing.expectEqual(canonicalNaN(f), try multiply(f, 0x7E00, 0x3C00)); // NaN · 1
    try testing.expectEqual(@as(u16, 0x0002), try multiply(f, 0x0001, 0x4000)); // subnormal · 2
}

test "float ref: 1.0 is the identity over every binary16 pattern" {
    const f = fmt_lib.binary16;
    var i: u32 = 0;
    while (i < 0x10000) : (i += 1) {
        const v: u16 = @intCast(i);
        const p = f.parts(v);
        if (p.exponent == f.emax()) continue; // 1·inf and 1·NaN are special
        try testing.expectEqual(v, try multiply(f, 0x3C00, v));
    }
}

test "float ref: bfloat16 — 1.0 identity and a few exact products" {
    const f = fmt_lib.bfloat16;
    // bf16 1.0 is 0x3F80, 2.0 is 0x4000, 1.5 is 0x3FC0.
    try testing.expectEqual(@as(u16, 0x3FC0), try multiply(f, 0x3F80, 0x3FC0));
    try testing.expectEqual(@as(u16, 0x4040), try multiply(f, 0x4000, 0x3FC0)); // 2 · 1.5 = 3
    try testing.expectEqual(@as(u16, 0x4000), try multiply(f, 0x4000, 0x3F80)); // 2 · 1 = 2
    // Largest bf16 times 1.0 stays put; times 2 overflows to infinity.
    try testing.expectEqual(@as(u16, 0x7F7F), try multiply(f, 0x7F7F, 0x3F80));
    try testing.expectEqual(@as(u16, 0x7F80), try multiply(f, 0x7F7F, 0x4000));
    var i: u32 = 0;
    while (i < 0x10000) : (i += 1) {
        const v: u16 = @intCast(i);
        const p = f.parts(v);
        if (p.exponent == f.emax()) continue;
        try testing.expectEqual(v, try multiply(f, 0x3F80, v));
    }
}

test "float ref: fp8 e4m3 — 1.0 identity, overflow, subnormals" {
    const f = fmt_lib.fp8_e4m3;
    // 1.0 is 0x38, 2.0 is 0x40. The largest NORMAL is 0x77 (240) because
    // exponent 15 is reserved: 0x78 is infinity and 0x79+ are NaN, so
    // 0x7E — which I first wrote here — is a NaN pattern, not a number.
    try testing.expectEqual(@as(u16, 0x38), try multiply(f, 0x38, 0x38));
    try testing.expectEqual(@as(u16, 0x40), try multiply(f, 0x40, 0x38));
    try testing.expectEqual(@as(u16, 0x77), try multiply(f, 0x77, 0x38));
    try testing.expectEqual(@as(u16, 0x78), try multiply(f, 0x77, 0x40)); // 240 · 2 overflows
    try testing.expectEqual(@as(u16, 0x00), try multiply(f, 0x08, 0x08)); // min subnormal² = 0
    try testing.expectEqual(@as(u16, 0x0002), try multiply(f, 0x18, 0x18)); // 2^-4 squared = 2·min_subnormal
    var i: u32 = 0;
    while (i < 0x100) : (i += 1) {
        const v: u16 = @intCast(i);
        const p = f.parts(v);
        if (p.exponent == f.emax()) continue;
        try testing.expectEqual(v, try multiply(f, 0x38, v));
    }
}

test "float ref: fp8 e5m2 — the narrowest exponent field" {
    const f = fmt_lib.fp8_e5m2;
    // 1.0 is 0x3C (exponent 15, bias 15, mantissa 0), max normal 0x7B
    // (57344), infinity 0x7C.
    try testing.expectEqual(@as(u16, 0x3C), try multiply(f, 0x3C, 0x3C));
    try testing.expectEqual(@as(u16, 0x40), try multiply(f, 0x40, 0x3C)); // 2 · 1
    try testing.expectEqual(@as(u16, 0x7C), try multiply(f, 0x7B, 0x40)); // overflow
    try testing.expectEqual(@as(u16, 0x0001), try multiply(f, 0x1C, 0x1C)); // 2^-8 squared = min subnormal
    var i: u32 = 0;
    while (i < 0x100) : (i += 1) {
        const v: u16 = @intCast(i);
        const p = f.parts(v);
        if (p.exponent == f.emax()) continue;
        try testing.expectEqual(v, try multiply(f, 0x3C, v));
    }
}

test "float ref: multiplication commutes and negates symmetrically" {
    inline for (fmt_lib.all) |f| {
        // 1.0 in any format: exponent field = bias, mantissa 0.
        const limit: u32 = @intCast(@as(u64, 1) << @intCast(f.byteWidth()));
        const sign_bit: u16 = @intCast(@as(u64, 1) << @intCast(f.byteWidth() - 1));
        var prng = std.Random.DefaultPrng.init(0xC0FFEE);
        const rng = prng.random();
        for (0..20000) |_| {
            const a: u16 = @intCast(rng.int(u16) & (limit - 1));
            const b: u16 = @intCast(rng.int(u16) & (limit - 1));
            const p = try multiply(f, a, b);
            try testing.expectEqual(p, try multiply(f, b, a));
            const p_neg = try multiply(f, a ^ sign_bit, b);
            if (f.parts(p).exponent == f.emax() and f.parts(p).mantissa != 0) {
                // NaN stays NaN regardless of the sign.
                try testing.expectEqual(f.parts(p_neg).exponent, f.emax());
                try testing.expect(f.parts(p_neg).mantissa != 0);
            } else {
                try testing.expectEqual(p ^ sign_bit, p_neg);
            }
        }
    }
}

test "float ref: roundToNearestEven still behaves" {
    const r = roundToNearestEven((@as(u64, 2) << 11) | (1 << 10), 11);
    try testing.expectEqual(@as(u64, 2), r.kept);
    try testing.expectEqual(@as(u1, 1), r.round_bit);
    try testing.expectEqual(@as(u1, 0), r.sticky);
}
