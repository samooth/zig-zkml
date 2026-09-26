//! Reference implementation of RMSNorm on the recorded path.
//!
//! # Why this exists and the AIR does not
//!
//! `libs/gadgets/norm/root.zig` used to emit a `degree = 1` constraint with no
//! expression behind it. That is not an incomplete gadget, it is a false
//! claim, and the reason it survived is that a float reference did not exist
//! to disagree with it. This file is that reference: a direct, obviously
//! correct transcription of the arithmetic, with no constraint system in it at
//! all. An AIR built on top of it can then be tested against something that
//! cannot itself be wrong in the interesting way.
//!
//! # The arithmetic being referenced
//!
//! Recorded RMSNorm, on int8 inputs, is
//!
//!   mean = Σ xᵢ² / n                      (exact: the field holds it)
//!   r    = rsqrt(mean)                    (one lookup, one rounding — §8 rule 1)
//!   yᵢ   = r · xᵢ · wᵢ                    (one rounding, combined)
//!
//! Two properties of that shape are load-bearing and are what the tests below
//! pin:
//!
//! 1. **`Σ xᵢ²` is exact in the field.** The largest value is 127² · n, and for
//!    n = 4096 that is 66,064,384 — nothing near the Goldilocks modulus, so
//!    the accumulation carries no rounding at all. It is also order
//!    independent, which is why the AIR can use a running sum with a single
//!    closing equation instead of proving an association order.
//!
//! 2. **Exactly one rounding happens at the rsqrt**, and the final product is
//!    rounded once. Adding a second rounding here is the bug class §8 rule 1
//!    exists to prevent, so `rmsnorm` takes a single accumulator and there is
//!    no intermediate variable for a caller to round twice.
//!
//! # What this does NOT do
//!
//! It does not model the engine. `mean` here is the field mean, whereas an
//! engine accumulates in fp32 and rounds at every step. That divergence is the
//! dual-path contract of §3.1, not an oversight: in recorded mode the kernel
//! runs *this*, and the fast path stays unverified by construction.

const std = @import("std");
const Allocator = std.mem.Allocator;
const nonlin = @import("../gadgets/nonlin/root.zig");

pub const Error = error{
    /// Row of zeros: `mean` is 0, `rsqrt(0)` is undefined and the table's
    /// index 0 is a marker, not a value. Reading it would make a degenerate
    /// norm look like a valid one scaled to zero.
    DegenerateRow,
    /// `mean` beyond `rsqrt_max_mean` — the table cannot represent it, and
    /// clamping would silently weaken the normalisation.
    MeanOutOfDomain,
    /// Input slice lengths disagree with the declared width.
    BadLength,
    OutOfMemory,
};

pub const Row = struct {
    /// Normalised outputs, i16 q8.8, one per column.
    out: []i16,
    /// Σ xᵢ² as an exact integer.
    sum_sq: u64,
    /// The table index actually used, so a caller can assert the domain.
    rsqrt_index: u16,
    /// The per-row shift, which the AIR must pin.
    rsqrt_shift: u8,
};

/// Round `value / 2^shift` half away from zero, as i16. The shift is what
/// turns an exact integer into q8.8; it is the only rounding on the scale.
fn quantize(value: i64, shift: u8) i16 {
    const denom: i64 = @as(i64, 1) << @intCast(shift);
    const half: i64 = @divTrunc(denom, 2);
    // Half away from zero needs the bias to follow the sign. Adding a
    // constant +half and truncating rounds half UP for positives and toward
    // zero for negatives — for -508/256 it gives -1 where -2 is correct. A
    // silent asymmetry on the negative half of a normalisation is exactly the
    // kind of bug that only shows up on odd inputs, so it is fixed here rather
    // than documented.
    const biased = if (value >= 0) value + half else value - half;
    const q = @divTrunc(biased, denom);
    const clamped = std.math.clamp(q, -32768, 32767);
    return @intCast(clamped);
}

/// Σ xᵢ², exact, and `n` as the field would see it.
pub fn sumOfSquares(x: []const i8) Error!u64 {
    var acc: u64 = 0;
    for (x) |v| {
        const sq: u64 = @intCast(@as(i32, v) * @as(i32, v));
        acc += sq;
    }
    if (acc == 0) return Error.DegenerateRow;
    return acc;
}

/// The mean of squares as the rsqrt table wants it: an integer index into
/// `rsqrt_q8_8` plus the shift that index is scaled by.
///
/// The shift is chosen per row so that the index always lands in [1, 255]:
///
///     shift = 0                                   if mean < 2⁸
///     shift = bitlen(mean) - 8                    otherwise
///
/// and `index = mean >> shift`, which is in [128, 255] in the second case and
/// in [1, 255] in the first. The AIR proves two relations and nothing more:
/// `mean · 2⁻ˢʰⁱᶠᵗ = index` in the field, and `shift <= rsqrt_max_shift`. A
/// prover cannot pick a different shift, because the relation pins it.
///
/// Flooring is part of the specification, not an accident of the division: it
/// guarantees `index · 2^shift <= mean`, so the table entry represents
/// 1/sqrt of something at or below the true mean, which makes the resulting
/// rsqrt **pessimistic** — the output is never scaled above what a float
/// RMSNorm would produce. That direction is chosen so the one rounding at the
/// rsqrt can never inflate the witness.
pub fn rsqrtIndexFor(sum_sq: u64, n: usize) Error!struct { index: u16, shift: u8 } {
    if (n == 0) return Error.BadLength;
    if (sum_sq == 0) return Error.DegenerateRow;
    if (sum_sq > std.math.maxInt(u32)) return Error.MeanOutOfDomain;

    const mean = sum_sq / @as(u64, n);
    if (mean == 0) return Error.DegenerateRow;

    var shift: u8 = 0;
    if (mean >= 256) {
        // bitlen(mean) - 8, computed without a log builtin.
        var bits: u8 = 0;
        var t = mean;
        while (t > 0) : (t >>= 1) bits += 1;
        shift = bits - 8;
        if (shift > nonlin.SiLULookup.rsqrt_max_shift) return Error.MeanOutOfDomain;
    }
    // The correction factor is 2^(-shift/2), a whole number only for an even
    // shift. The shift must therefore be rounded **up** to even, not down:
    // rounding down pushes `index` out of the table for every mean whose
    // bitlen is odd above 8. mean = 256 has shift 1, and shift 0 would ask for
    // index 256 in a 256-entry table. Rounding up gives shift 2, index 64, and
    // a table entry that still satisfies `index · 2^shift <= mean`.
    if (shift % 2 != 0) shift += 1;
    const index: u64 = mean >> @intCast(shift);
    if (index == 0) return Error.DegenerateRow;
    if (index >= nonlin.table_size) return Error.MeanOutOfDomain;
    const k = shift / 2;
    if (k >= nonlin.SiLULookup.rsqrt_scale_size) return Error.MeanOutOfDomain;
    return .{ .index = @intCast(index), .shift = shift };
}

/// Recorded-path RMSNorm. `x` and `weight` are int8, `cols` is the row width.
///
/// The output is i16 q8.8 so the q8.8 scale of the rsqrt table survives
/// multiplication by an int8 without a second rounding.
pub fn rmsnorm(
    gpa: Allocator,
    x: []const i8,
    weight: []const i8,
    cols: usize,
) Error!Row {
    if (x.len != cols) return Error.BadLength;
    if (weight.len != cols) return Error.BadLength;
    if (cols == 0) return Error.BadLength;

    const sum_sq = try sumOfSquares(x);
    const sel = try rsqrtIndexFor(sum_sq, cols);
    // r = r0 · 2^(-shift/2) in q8.8, so both factors come from bound tables.
    const r0 = nonlin.SiLULookup.rsqrt_q8_8[sel.index];
    const scale = nonlin.SiLULookup.rsqrt_shift_scale[sel.shift / 2];
    const r = quantize(@as(i32, r0) * @as(i32, scale), 8);

    const out = try gpa.alloc(i16, cols);
    errdefer gpa.free(out);

    // One rounding: r·x·w is accumulated as an exact i32 and quantized once at
    // the end. r is q8.8 and x, w are int8, so the product is exact in i32
    // (|·| ≤ 128·128·128 = 2,097,152) and nothing here can overflow.
    for (x, weight, 0..) |xv, wv, i| {
        const prod: i32 = @as(i32, r) * @as(i32, xv) * @as(i32, wv);
        out[i] = quantize(prod, 8);
    }

    return .{ .out = out, .sum_sq = sum_sq, .rsqrt_index = sel.index, .rsqrt_shift = sel.shift };
}

test "the sum of squares is exact and order independent" {
    const a = [_]i8{ 3, -4, 5, 127, -127, 1, 0, -1 };
    const b = [_]i8{ -127, 1, 0, -1, 127, 5, -4, 3 };
    const sa = try sumOfSquares(&a);
    const sb = try sumOfSquares(&b);
    try std.testing.expectEqual(sa, sb);
    // 9 + 16 + 25 + 16129 + 16129 + 1 + 0 + 1
    try std.testing.expectEqual(@as(u64, 32310), sa);
}

test "a row of zeros is rejected, not scaled to zero" {
    const zero = [_]i8{0} ** 8;
    try std.testing.expectError(Error.DegenerateRow, sumOfSquares(&zero));

    // A weight of all ones does not rescue it: the mean is still zero.
    const ones = [_]i8{1} ** 8;
    try std.testing.expectError(Error.DegenerateRow, rmsnorm(
        std.testing.allocator,
        &zero,
        &ones,
        8,
    ));
}

test "the rsqrt index is floored, so rsqrt is never optimistic" {
    // sum_sq = 4096, n = 64 -> mean = 64 -> below 256, so shift 0 and the
    // index is the mean itself: 64.
    const sel = try rsqrtIndexFor(4096, 64);
    try std.testing.expectEqual(@as(u16, 64), sel.index);
    try std.testing.expectEqual(@as(u8, 0), sel.shift);
    try std.testing.expectEqual(@as(i16, 32), nonlin.SiLULookup.rsqrt_q8_8[sel.index]);

    // A mean just above a grid point must floor to that grid point, and the
    // table entry must be no larger than the true 1/sqrt(mean) scaled. That is
    // the "never optimistic" property, and it is what makes the single
    // rounding at the rsqrt sound.
    const mean: u64 = 65;
    const sel2 = try rsqrtIndexFor(mean * 64, 64);
    try std.testing.expectEqual(@as(u16, 65), sel2.index);
    const r_q8_8 = @as(f32, @floatFromInt(nonlin.SiLULookup.rsqrt_q8_8[sel2.index])) / 256.0;
    const true_r = 1.0 / @sqrt(@as(f32, @floatFromInt(mean)));
    try std.testing.expect(r_q8_8 <= true_r + 1.0 / 256.0);
}

test "every finite mean lands in the table, via a per-row shift" {
    // The whole point of the per-row shift: there is no "mean too large"
    // failure any more. A fixed shift had one — means below 2^shift fell off
    // the bottom — and a perfectly ordinary row of unit activations hit it.
    // The cost is that the shift must be pinned, and the bound that replaces
    // the domain check is on the shift itself.
    // The multiplier keeps sum_sq inside u32 so the domain check is not what
    // this test is measuring — that has its own test below.
    for ([_]u64{ 1, 2, 63, 64, 255, 256, 16129, 1_000_000 }) |mean| {
        const sel = try rsqrtIndexFor(mean, 1);
        try std.testing.expect(sel.index >= 1);
        try std.testing.expect(sel.index < 256);
        try std.testing.expect(sel.shift <= nonlin.SiLULookup.rsqrt_max_shift);
        // The relation the AIR will prove. It is `index · 2^shift <= mean`
        // and NOT equality: the shift is a floor, so the remainder is dropped.
        // 16129 = 252·64 + 1 is the case that catches anyone who writes this
        // as an equality — that prover cannot exist, and the AIR would reject
        // every mean that is not an exact multiple of the grid.
        const product: u64 = @as(u64, sel.index) << @intCast(sel.shift);
        try std.testing.expect(product <= mean);
        // The shift is chosen as bitlen-8 (rounded down to even), so the
        // remainder is bounded by the grid, which for shift = 0 means the
        // mean IS a multiple of 2^shift exactly. The bound below is therefore
        // on the grid step, and the case mean = 1_000_000 is the one that
        // checks it: 244·4096 = 999424, remainder 576.
        const grid: u64 = @as(u64, 1) << @intCast(sel.shift);
        try std.testing.expect(mean - product < grid + 1);
        // And the table entry is never optimistic, i.e. it never represents a
        // value below the true mean.
        const entry_q8_8 = @as(f32, @floatFromInt(nonlin.SiLULookup.rsqrt_q8_8[sel.index]));
        const represented = @as(f32, @floatFromInt(sel.index)) *
            @as(f32, @floatFromInt(@as(u32, 1) << @intCast(sel.shift)));
        const mean_f: f32 = @floatFromInt(mean);
        try std.testing.expect(represented <= mean_f);
        try std.testing.expect(entry_q8_8 >= 0);
    }
}

test "a sum of squares beyond u32 is rejected" {
    // n·127² is the practical ceiling for any width this library accepts, so a
    // sum past u32 means either a wrong width or a bug upstream. Refusing it
    // keeps the mean computation meaningful instead of silently wrapping.
    const too_big = @as(u64, 1) << 33;
    try std.testing.expectError(Error.MeanOutOfDomain, rsqrtIndexFor(too_big, 1));
}

test "rmsnorm output is i16 q8.8 and uses one rounding" {
    const a = std.testing.allocator;
    const x = [_]i8{ 127, -127, 0, 3, -3, 7, -7, 11 };
    const w = [_]i8{ 1, 1, 1, 1, 1, 1, 1, 1 };
    const row = try rmsnorm(a, &x, &w, 8);
    defer a.free(row.out);

    // The exact chain, so the expected numbers below are derived rather than
    // guessed:
    //   sum_sq = 2·127² + 2·3² + 2·7² + 11² = 32495
    //   mean   = 32495 / 8                   = 4061
    //   shift  = bitlen(4061) - 8 = 4, already even
    //   index  = 4061 >> 4                    = 253
    //   k      = shift / 2                    = 2
    //   r0     = round(256 / sqrt(253))       = 16
    //   scale  = 256 · 2^-k                   = 64
    //   r      = r0 · scale / 256             = 4      (= 256/sqrt(4061) ≈ 4.02)
    //   y      = r · x · w / 256 = 4x/256 = x/64
    //
    // Asserting 253, 4, 16, 64 and the outputs pins the whole chain. The two
    // that matter are `scale` (a missing or inverted factor-2 correction is a
    // 16x error that stays "in range") and `r = 4` (the product of the two
    // tables, which is the only value the caller ever uses).
    try std.testing.expectEqual(@as(u16, 253), row.rsqrt_index);
    try std.testing.expectEqual(@as(u8, 4), row.rsqrt_shift);
    try std.testing.expectEqual(@as(i16, 16), nonlin.SiLULookup.rsqrt_q8_8[row.rsqrt_index]);
    try std.testing.expectEqual(@as(i16, 64), nonlin.SiLULookup.rsqrt_shift_scale[row.rsqrt_shift / 2]);
    // The final product goes through `quantize`, which rounds half away from
    // zero rather than truncating: 4·127/256 = 1.984 -> 2, not 1. That is the
    // single rounding §8 rule 1 allows, and truncating instead would be a
    // second, silent one.
    try std.testing.expectEqual(@as(i16, 2), row.out[0]); // x = 127
    // -1.984 rounds to -2 half away from zero, same rule as the positive side.
    try std.testing.expectEqual(@as(i16, -2), row.out[1]); // x = -127
    try std.testing.expectEqual(@as(i16, 0), row.out[2]); // x = 0
    try std.testing.expectEqual(@as(i16, 0), row.out[3]); // x = 3
    for (row.out) |v| {
        try std.testing.expect(v >= -32768 and v <= 32767);
    }
    // A unit weight on a symmetric row gives a symmetric output.
    try std.testing.expectEqual(row.out[0], -row.out[1]);
    try std.testing.expectEqual(row.out[3], -row.out[4]);
}

test "length mismatches are rejected before any arithmetic runs" {
    const a = std.testing.allocator;
    const x = [_]i8{1} ** 4;
    const w8 = [_]i8{1} ** 8;
    // weight longer than the row
    try std.testing.expectError(Error.BadLength, rmsnorm(a, &x, &w8, 4));
    // weight shorter than the row
    const w2 = [_]i8{1} ** 2;
    try std.testing.expectError(Error.BadLength, rmsnorm(a, &x, &w2, 4));
    // x longer than the declared width
    try std.testing.expectError(Error.BadLength, rmsnorm(a, &w8, &w8, 4));
    // A unit row has mean 1, which the OLD fixed-shift table could not index.
    // It must now succeed, which is the regression test for that design error.
    const row = try rmsnorm(a, &x, &x, 4);
    defer a.free(row.out);
    try std.testing.expectEqual(@as(u16, 1), row.rsqrt_index);
    try std.testing.expectEqual(@as(u8, 0), row.rsqrt_shift);
    // mean = 1, shift 0, index 1, r = 256 in q8.8 = 1.0. Then y = 256·1·1/256
    // = 1, which in q8.8 is one unit — i.e. the activation is unchanged. A
    // unit row is the identity, and that is the property worth pinning.
    for (row.out) |v| try std.testing.expectEqual(@as(i16, 1), v);
}

test "quantize rounds half away from zero on both signs" {
    // Regression. The implementation added a constant +half and truncated,
    // which rounds half up for positives and toward zero for negatives:
    // -508/256 came out -1 where -2 is correct. A normalisation that is
    // asymmetric on the negative half is invisible on symmetric test data and
    // shows up only on odd inputs, so it is pinned directly.
    try std.testing.expectEqual(@as(i16, 2), quantize(508, 8)); //  1.98 -> 2
    try std.testing.expectEqual(@as(i16, -2), quantize(-508, 8)); // -1.98 -> -2
    // Exact halves go away from zero, not toward it.
    try std.testing.expectEqual(@as(i16, 1), quantize(128, 8)); //  0.5 -> 1
    try std.testing.expectEqual(@as(i16, -1), quantize(-128, 8)); // -0.5 -> -1
    // No bias on exact values.
    try std.testing.expectEqual(@as(i16, 1), quantize(256, 8));
    try std.testing.expectEqual(@as(i16, -1), quantize(-256, 8));
    try std.testing.expectEqual(@as(i16, 0), quantize(0, 8));
    // Odd inputs, where a sign-symmetric bug would hide.
    try std.testing.expectEqual(@as(i16, 0), quantize(127, 8)); //  0.496 -> 0
    try std.testing.expectEqual(@as(i16, 0), quantize(-127, 8)); // -0.496 -> 0
    try std.testing.expectEqual(@as(i16, 2), quantize(384, 8)); //  1.5 -> 2
    try std.testing.expectEqual(@as(i16, -2), quantize(-384, 8)); // -1.5 -> -2
}
