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
    // encoding is m · 2^minShift, and m is the EXACT product placed on the
    // subnormal grid — ONE rounding, at the grid the answer is written to.
    //
    // The previous version rounded `kept` a second time, which is double
    // rounding: whenever the first rounding lands exactly on a midpoint of
    // the second, the two disagree. That is not a corner case — it is
    // 6_459_545 of the normal x normal binary16 pairs whose result is
    // subnormal. 0x0401 · 0x18FF is one: the exact value is
    // 2.500486 · 2^-24, so IEEE-754 says mantissa 3, and the double
    // rounding said 2. numpy agrees with 3.
    const scale: i32 = f.minShift() - exact_shift;
    // The product is below 2^(2M+2), so a scale past 2M+2 is under half
    // the smallest subnormal and flushes to zero. Compute it wide and
    // clamp BEFORE narrowing: with a small exponent field (fp8 e4m3 has
    // 4) a tiny result needs a shift far past 15, and @intCast to u4
    // panics instead of flushing to zero.
    if (scale > @as(i32, f.mant_bits) * 2 + 2) {
        return packSpecial(f, sign, .zero);
    }
    std.debug.assert(scale >= 1);
    const reduced = roundToNearestEven(product, @intCast(scale));
    if (reduced.kept >= implicit) {
        // The rounding carried into the smallest normal.
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
/// IEEE-754 addition, ONE rounding, for the cross-check of §3.1 in
/// docs/BLUE_PRINT.md: the native fast path accumulates in fp32 and the test
/// that compares it against the proof's contract needs a reference that
/// rounds exactly once, in the same order, per step.
///
/// It is a HOST reference, not an AIR. That is a measured decision, not a
/// shortcut: proving this in the IR costs several hundred constraints per
/// add (the alignment shift alone is a 24-bit, 7-stage barrel), and sixteen
/// adds per row would need an alpha array two orders of magnitude larger
/// than the verifier keeps on its stack. The contract the proof verifies is
/// the EXACT field sum — see the cost measurement in docs/BLUE_PRINT.md.
///
/// The same discipline as `multiply`: the exact sum is an integer, and it
/// is rounded ONCE, at the end, into the target grid. Nothing here rounds
/// twice, which is the bug the multiply reference had.
pub fn add(f: Format, a: u16, b: u16) !u16 {
    const da = decompose(f, a);
    const db = decompose(f, b);
    if (da.special == .nan or db.special == .nan) return canonicalNaN(f);
    if (da.special == .inf and db.special == .inf) {
        if (da.sign == db.sign) return packSpecial(f, 0, .inf);
        return canonicalNaN(f); // inf + (-inf) is the invalid operation
    }
    if (da.special == .inf) return packSpecial(f, da.sign, .inf);
    if (db.special == .inf) return packSpecial(f, db.sign, .inf);

    // Zeros: the sign of an exact zero sum is -0 only when both are -0,
    // and a zero plus a non-zero takes the other one's sign.
    if (da.special == .zero and db.special == .zero) {
        const sign: u1 = if (da.sign == 1 and db.sign == 1) 1 else 0;
        return packSpecial(f, sign, .zero);
    }
    if (da.special == .zero) return b;
    if (db.special == .zero) return a;

    // Exact sum as ONE integer plus one common shift. Each operand is
    // `significand · 2^shift`, so the common scale is the SMALLER shift and
    // the bigger operand is scaled UP by the gap — exact, no bits lost, and
    // no rounding yet. (Scaling the smaller one up instead, which looks
    // equivalent, is not: the two terms would end up on different shifts
    // and the sum would be off by the gap. The min normal plus the min
    // subnormal is the case that says so.)
    const sign: u1 = da.sign ^ db.sign;
    const big = if (da.shift > db.shift) da else db;
    const small = if (da.shift > db.shift) db else da;
    const gap: u32 = @intCast(big.shift - small.shift);
    // u128 cannot overflow here: significand < 2^(M+1) and gap <= 2·emax,
    // and 2^(M+1+2·emax) is far inside 128 bits for every format.
    const big_scaled: u128 = @as(u128, big.significand) << @intCast(gap);
    const small_scaled: u128 = small.significand;
    const magnitude: u128 = if (sign == 0)
        big_scaled + small_scaled
    else if (big_scaled >= small_scaled)
        big_scaled - small_scaled
    else
        small_scaled - big_scaled;
    if (magnitude == 0) return packSpecial(f, 0, .zero); // exact cancellation

    // The sign of a SUM is the xor; the sign of a DIFFERENCE belongs to
    // whichever operand is larger in magnitude — and "larger" here is the
    // comparison above, not the one that picked `big` (which picks by
    // exponent, and a subtraction can go the other way). Returning the xor
    // unconditionally made `1+ulp + (-(2^-11 + tail))` come out as -1.0.
    const result_sign: u1 = if (sign == 0) da.sign else if (big_scaled >= small_scaled) big.sign else small.sign;

    return roundInto(f, result_sign, magnitude, small.shift);
}

/// round(m / 2^k) for k >= 0, RNE, with the sticky taken from the bits
/// below the round bit. One rounding, at the end, always.
fn divRoundPow2(m: u128, k: u16) u64 {
    if (k == 0) return @intCast(m);
    var q: u64 = @intCast(m >> @intCast(k));
    const round_bit: u64 = @intCast((m >> @intCast(k - 1)) & 1);
    const rest_mask: u128 = if (k <= 1) 0 else (@as(u128, 1) << @intCast(k - 1)) - 1;
    const sticky: u64 = if (m & rest_mask == 0) 0 else 1;
    if (round_bit == 1 and (sticky == 1 or q & 1 == 1)) q += 1;
    return q;
}

/// Round an exact magnitude `m · 2^shift` (m's top bit set) once into the
/// format's grid.
fn roundInto(f: Format, sign: u1, m: u128, shift: i32) u16 {
    std.debug.assert(m > 0);
    const bits: u16 = 128 - @as(u16, @intCast(@clz(m)));
    const keep: u16 = f.keptLow(); // M
    // The field the answer lands in, before rounding. A value above the
    // grid overflows to infinity, exactly as the multiply does.
    const e_field: i32 = shift + @as(i32, @intCast(bits)) + f.bias - 1;
    const max_field: i32 = if (f.has_inf_nan) @as(i32, f.emax()) - 1 else f.emax();
    // The subnormal grid's step, and the count of steps in the exact value.
    const sub_shift: i32 = 1 - f.bias - @as(i32, @intCast(f.mant_bits));
    const step_shift: i32 = shift - sub_shift;

    if (e_field >= 1) {
        // NORMAL: the kept field is the top M+1 bits, and a rounding carry
        // can push it to 2^(M+1), which is 1.0 of the next exponent.
        const drop: u16 = @as(u16, @intCast(bits)) - 1 - keep;
        var mant: u64 = divRoundPow2(m, drop);
        var field: i32 = e_field;
        if (mant == (@as(u64, 1) << @intCast(keep + 1))) {
            mant = @as(u64, 1) << @intCast(keep);
            field += 1;
        }
        if (field > max_field) return packSpecial(f, sign, .inf);
        return f.pack(.{
            .sign = sign,
            .exponent = @intCast(field),
            .mantissa = @intCast(mant - (@as(u64, 1) << @intCast(keep))),
        });
    }

    // SUBNORMAL or zero: the count of subnormal steps, rounded ONCE. When
    // the exact value is below the step this is a right shift, and the
    // sticky that decides a tie has to come from the bits it discarded.
    const count: u64 = if (step_shift >= 0) blk: {
        const exact: u128 = m << @intCast(step_shift);
        break :blk @intCast(exact);
    } else divRoundPow2(m, @intCast(-step_shift));
    if (count == 0) return packSpecial(f, sign, .zero);
    if (count == (@as(u64, 1) << @intCast(f.mant_bits))) {
        // Rounded all the way up: the min normal, whose mantissa is zero.
        return f.pack(.{ .sign = sign, .exponent = 1, .mantissa = 0 });
    }
    return f.pack(.{ .sign = sign, .exponent = 0, .mantissa = @intCast(count) });
}

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

// The subnormal-result branch used to round the already-rounded kept
// field a second time. That is double rounding, and it is wrong wherever
// the first rounding lands on a midpoint of the second: 6_459_545 of the
// normal x normal binary16 pairs whose result is subnormal. These three
// are the ones that pinned it, checked against IEEE-754 and numpy:
//
//   0x0401 · 0x18FF = 2.500486 · 2^-24  ->  mantissa 3
//   0x0401 · 0x1AFE = 3.499508 · 2^-24 ->  mantissa 3
//   0x0401 · 0x1C7F = 4.500484 · 2^-24 ->  mantissa 5
//
// The double rounding said 2, 4 and 4.
test "float ref: a subnormal result is rounded once, not twice" {
    const f = fmt_lib.binary16;
    try testing.expectEqual(@as(u16, 0x0003), try multiply(f, 0x0401, 0x18FF));
    try testing.expectEqual(@as(u16, 0x0003), try multiply(f, 0x0401, 0x1AFE));
    try testing.expectEqual(@as(u16, 0x0005), try multiply(f, 0x0401, 0x1C7F));
    // A subnormal result that rounds UP into the smallest normal must say
    // so with exponent 1, not with a mantissa that happens to look right.
    try testing.expectEqual(@as(u16, 0x0400), try multiply(f, 0x0401, 0x3BFF));
}

// The subnormal-result branch against an INDEPENDENT single rounding: the
// exact product as an integer, placed on the subnormal grid. Every normal
// x normal binary16 pair is 1.6 billion, too many for a test, so the
// window is where subnormal results actually live: the low exponent
// fields, with the significands sampled across their whole range.
test "float ref: every subnormal binary16 result matches one rounding" {
    const f = fmt_lib.binary16;
    var checked: usize = 0;
    var ea: u32 = 1;
    while (ea <= 8) : (ea += 1) {
        var mant: u32 = 0;
        while (mant <= 0x3FF) : (mant += 0x37) {
            const a: u16 = @intCast((ea << 10) | mant);
            var eb: u32 = 1;
            while (eb <= 8) : (eb += 1) {
                var mb: u32 = 0;
                while (mb <= 0x3FF) : (mb += 0x5B) {
                    const b: u16 = @intCast((eb << 10) | mb);
                    const got = try multiply(f, a, b);
                    const p = f.parts(got);
                    if (p.exponent != 0) continue; // only the subnormal results

                    const da = decompose(f, a);
                    const db = decompose(f, b);
                    const product: u64 = @as(u64, da.significand) * @as(u64, db.significand);
                    const scale = f.minShift() - (da.shift + db.shift);
                    if (scale > @as(i32, f.mant_bits) * 2 + 2) {
                        try testing.expectEqual(@as(u16, 0), got);
                    } else {
                        const one = roundToNearestEven(product, @intCast(scale));
                        const want: u16 = if (one.kept >= f.mantImplicit())
                            f.pack(.{ .sign = 0, .exponent = 1, .mantissa = 0 })
                        else
                            f.pack(.{ .sign = 0, .exponent = 0, .mantissa = @intCast(one.kept) });
                        if (want != got) {
                            std.debug.print("subnormal result: {x} · {x} gave {x}, one rounding says {x}\n", .{ a, b, got, want });
                            return error.DoubleRounding;
                        }
                    }
                    checked += 1;
                }
            }
        }
    }
    try testing.expect(checked > 1000);
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

/// One operand counted in subnormal steps: an integer part and an exact
/// remainder over 2^k.
const Steps = struct {
    q: u128,
    rem: u128,
    k: u16,
};

fn stepsOf(d: Decomposed, sub_shift: i32) Steps {
    const delta: i32 = d.shift - sub_shift;
    if (delta >= 0) return .{ .q = @as(u128, d.significand) << @intCast(delta), .rem = 0, .k = 0 };
    const k: u16 = @intCast(-delta);
    const mask: u128 = (@as(u128, 1) << @intCast(k)) - 1;
    return .{
        .q = @as(u128, d.significand) >> @intCast(k),
        .rem = @as(u128, d.significand) & mask,
        .k = k,
    };
}

/// An independent second opinion for the finite non-zero case, written the
/// other way round. `add` normalises the exact magnitude and rounds it once;
/// this one counts SUBNORMAL STEPS — each operand as an integer part plus
/// an exact remainder over a power of two — adds those, and rounds the
/// combined fraction ONCE. The two disagree if either rounds twice, which
/// is how the multiply reference's double-rounding bug was found.
fn addBySteps(f: Format, a: u16, b: u16) !u16 {
    const da = decompose(f, a);
    const db = decompose(f, b);
    if (da.special != .finite or db.special != .finite) return add(f, a, b);
    if (da.significand == 0 or db.significand == 0) return add(f, a, b);

    const sub_shift: i32 = 1 - f.bias - @as(i32, @intCast(f.mant_bits));
    const sa = stepsOf(da, sub_shift);
    const sb = stepsOf(db, sub_shift);
    const k: u16 = @max(sa.k, sb.k);
    const scale_a: u16 = k - sa.k;
    const scale_b: u16 = k - sb.k;
    // `q` already counts STEPS (an integer) and `rem` is the fraction of a
    // step over 2^k, so the sum is integer + fraction/2^k and the ONE
    // rounding is RNE on that fraction. Dividing a combined numerator by
    // 2^k instead would throw the integer part away — which is how the
    // first version of this function answered 0 for min_sub + min_sub.
    const sign_a: i128 = if (da.sign == 1) -1 else 1;
    const sign_b: i128 = if (db.sign == 1) -1 else 1;
    // The integer parts are counts of steps already — no scaling. Only the
    // REMAINDERS live at different resolutions, so only they are lifted to
    // the common denominator.
    const integer: i128 = sign_a * @as(i128, @intCast(sa.q)) + sign_b * @as(i128, @intCast(sb.q));
    const rest: i128 = sign_a * @as(i128, @intCast(sa.rem << @intCast(scale_a))) +
        sign_b * @as(i128, @intCast(sb.rem << @intCast(scale_b)));
    const denom: i128 = @as(i128, 1) << @intCast(k);
    var count: i128 = integer;
    const tail: i128 = if (rest < 0) -rest else rest;
    const half: i128 = @divTrunc(denom, 2);
    if (tail > half) {
        count += if (rest > 0) 1 else -1;
    } else if (tail == half and k > 0) {
        if (@mod(count, 2) != 0) count += if (rest > 0) 1 else -1;
    }
    if (count == 0) return packSpecial(f, if (da.sign == 1 and db.sign == 1) 1 else 0, .zero);
    const result_sign: u1 = if (count < 0) 1 else 0;
    const m: u128 = @intCast(if (count < 0) -count else count);

    // The count is in subnormal steps, so the value is m · 2^sub_shift and
    // the field follows from m's top bit. No rounding is left: the single
    // rounding already happened when the fraction was folded into `count`.
    const bits: u16 = 128 - @as(u16, @intCast(@clz(m)));
    const field: i32 = sub_shift + @as(i32, @intCast(bits - 1)) + f.bias;
    const max_field: i32 = if (f.has_inf_nan) @as(i32, f.emax()) - 1 else f.emax();
    if (field < 1) {
        return f.pack(.{ .sign = result_sign, .exponent = 0, .mantissa = @intCast(m) });
    }
    const keep: u16 = f.keptLow();
    const drop: u16 = bits - 1 - keep;
    var mant: u64 = divRoundPow2(m, drop);
    var e: i32 = field;
    if (mant == (@as(u64, 1) << @intCast(keep + 1))) {
        mant = @as(u64, 1) << @intCast(keep);
        e += 1;
    }
    if (e > max_field) return packSpecial(f, result_sign, .inf);
    return f.pack(.{
        .sign = result_sign,
        .exponent = @intCast(e),
        .mantissa = @intCast(mant - (@as(u64, 1) << @intCast(keep))),
    });
}

test "add: the cases a float test always forgets" {
    const f = fmt_lib.binary16;
    const one: u16 = 0x3C00; // 1.0
    const neg_one: u16 = 0xBC00;
    const zero: u16 = 0x0000;
    const neg_zero: u16 = 0x8000;
    const inf: u16 = 0x7C00;
    const nan: u16 = canonicalNaN(f);
    const min_normal: u16 = 0x0400;
    const min_sub: u16 = 0x0001;
    const max_norm: u16 = 0x7BFF;

    try testing.expectEqual(@as(u16, 0x4000), try add(f, one, one)); // 2.0
    try testing.expectEqual(@as(u16, 0x0000), try add(f, one, neg_one)); // exact cancellation -> +0
    try testing.expectEqual(neg_zero, try add(f, neg_zero, neg_zero)); // -0 + -0 = -0
    try testing.expectEqual(@as(u16, 0x0000), try add(f, neg_zero, zero)); // -0 + +0 = +0
    try testing.expectEqual(@as(u16, 0xBC00), try add(f, neg_one, zero)); // the sign of the non-zero
    try testing.expectEqual(inf, try add(f, inf, one));
    try testing.expectEqual(inf, try add(f, inf, neg_one)); // the infinity's sign, not the other's
    try testing.expectEqual(nan, try add(f, inf, neg_inf(f))); // the invalid operation
    try testing.expectEqual(nan, try add(f, nan, one));
    try testing.expectEqual(inf, try add(f, max_norm, max_norm)); // overflow
    // The min normal plus the min subnormal is EXACT: no rounding, no
    // error, and the second operand's scale is what makes it so. This pair
    // is also the one that caught the add's first version scaling the
    // smaller significand up instead of the bigger one.
    try testing.expectEqual(@as(u16, 0x0401), try add(f, min_normal, min_sub));
    try testing.expectEqual(@as(u16, 0x0000), try add(f, neg_min_sub(f), min_sub)); // exact cancellation
}

fn neg_inf(f: Format) u16 {
    return f.pack(.{ .sign = 1, .exponent = f.emax(), .mantissa = 0 });
}

fn neg_min_sub(f: Format) u16 {
    return f.pack(.{ .sign = 1, .exponent = 0, .mantissa = 1 });
}

test "add: a tie rounds to even, and only once" {
    const f = fmt_lib.binary16;
    // A tie at 1.0: 1.0 + 2^-11 sits exactly between 1.0 (even) and
    // 1.0 + ulp (odd), so RNE keeps 1.0.
    const two_pow_minus_11: u16 = f.pack(.{ .sign = 0, .exponent = @intCast(f.bias - 11), .mantissa = 0 });
    try testing.expectEqual(@as(u16, 0x3C00), try add(f, 0x3C00, two_pow_minus_11));
    // The same tie one ulp up: 1+ulp (odd) + 2^-11 is between 1+ulp and
    // 1+2ulp (even), so this one goes UP. Same bit pattern, opposite
    // answer: a rounding that only looked at the sticky would get this
    // wrong.
    try testing.expectEqual(@as(u16, 0x3C02), try add(f, 0x3C01, two_pow_minus_11));
    // The subnormal grid crossing into normal: the max subnormal is 1023
    // steps and twice that is 2046, which is the min normal plus 1022 —
    // exact, and the answer's exponent field has to become 1.
    try testing.expectEqual(@as(u16, 0x07FE), try add(f, 0x03FF, 0x03FF));
    // Just ABOVE the halfway point goes up, and just BELOW stays: the
    // same tie with the second operand negated, one ulp of difference in
    // the operand, opposite answers. A rounding that ignored the tail
    // would give the same answer for both.
    const just_over_half: u16 = f.pack(.{ .sign = 0, .exponent = @intCast(f.bias - 11), .mantissa = 1 });
    try testing.expectEqual(@as(u16, 0x3C02), try add(f, 0x3C01, just_over_half));
    // Subtracting the same quantity lands strictly BELOW the midpoint, so
    // the answer is 1.0 — the other neighbour of the tie above.
    const just_under_half: u16 = f.pack(.{ .sign = 1, .exponent = @intCast(f.bias - 11), .mantissa = 1 });
    try testing.expectEqual(@as(u16, 0x3C00), try add(f, 0x3C01, just_under_half));
}

test "add: agrees with a second implementation over a grid" {
    const f = fmt_lib.binary16;
    var i: u16 = 1;
    var checked: usize = 0;
    while (i < 0x7C00) : (i += 37) {
        var j: u16 = 1;
        while (j < 0x7C00) : (j += 41) {
            const got = try add(f, i, j);
            const other = try addBySteps(f, i, j);
            if (got != other) {
                std.debug.print("add {x} + {x}: {x} vs {x}\n", .{ i, j, got, other });
                return error.AddMismatch;
            }
            checked += 1;
        }
    }
    try testing.expect(checked > 1000);
}
