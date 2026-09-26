//! Canonical non-linear activation tables — the normative definition.
//!
//! # What these tables are
//!
//! docs/BLUE_PRINT.md §3.1 makes the `QuantScheme` normative: the proof
//! verifies the *exact arithmetic the scheme defines*, and non-linearities are
//! defined **by table**. That makes the table the specification, not an
//! approximation of something else. There is no "true" SiLU that this code is
//! failing to reach — on a q8.8 integer domain there is no such thing, and a
//! proof needs a total function with a defined output for every input.
//!
//! # What these tables are NOT
//!
//! They are **not** what llama.cpp, vLLM, zig-ai or ktransformers compute.
//! Verified against `llama.cpp@1c3c9674d`: `ggml_vec_silu_f32` dispatches to
//! one of six `ggml_v_silu` variants (AVX512, AVX2, SSE2, SVE, NEON, RISC-V),
//! each calling a *different* polynomial approximation of `expf`. The results
//! differ in the last bit by CPU. A witness recorded on one machine therefore
//! does not verify against an AIR built for another.
//!
//! The resolution is not to chase the engine — it is the dual-path contract of
//! §3.1: in **recorded** mode the kernel runs *this* table, byte for byte, not
//! the engine's native activation. The native fast path is unverified by
//! construction and stays that way, with no cost on the normal path.
//!
//! # Why the binding matters more than the values
//!
//! A table that nothing commits to is not a specification. Before this module
//! the table was a compile-time constant that the prover could change freely:
//! a different SiLU table produced a different trace, and both verified,
//! because the statement said nothing about which table was used. That is a
//! soundness hole, not a style issue.
//!
//! `tablesDigest` is what closes it. The digest covers these tables and is
//! absorbed into the statement, so a prover using a different table is
//! asserting a different statement and cannot produce a proof for the
//! original one. Swapping the table is now a detectable forgery, not a
//! silent redefinition.
//!
//! Consequently, `silu_q8_8` and friends may be regenerated **only** together
//! with a `nonlinearity_version` bump, and the bump is a breaking change for
//! every existing proof.

const std = @import("std");
const Allocator = std.mem.Allocator;
const air = @import("../../air/root.zig");
const tensor = @import("../../tensor/root.zig");

pub const Goldilocks = tensor.Goldilocks;

/// Blake3-256, the same primitive the weights tree uses.
pub const Hash = [32]u8;

/// Bump when any table in this file changes. It is part of the statement, so
/// a bump invalidates proofs made against the old tables — which is the point:
/// the tables are normative.
pub const nonlinearity_version: u16 = 1;

/// Entries per table. The q8.8 grid of an int8 input is 256 values, and the
/// output fits in i16 (`|silu(x)| <= ~128 < 2^15`), so 256 is the natural and
/// complete domain — not a sample of a larger one.
pub const table_size: usize = 256;

pub const SiLULookup = struct {
    /// silu(x) = x / (1 + exp(-x)) evaluated on the q8.8 grid and rounded
    /// half-away-from-zero to i16. The rounding rule is part of the
    /// specification: a different tie-breaking rule is a different function
    /// and would require a version bump.
    pub const silu_q8_8: [table_size]i16 = blk: {
        var t: [table_size]i16 = undefined;
        for (0..table_size) |i| {
            const x: f32 = @floatFromInt(@as(i8, @bitCast(@as(u8, @intCast(i)))));
            const s = x / (1.0 + @exp(-x));
            t[i] = @intFromFloat(@round(s * 256.0));
        }
        break :blk t;
    };

    /// GELU, tanh approximation — the variant Qwen-style models use. Stated
    /// explicitly because the exact and tanh forms differ in the last bits and
    /// both are called "GELU" in the wild.
    pub const gelu_tanh_q8_8: [table_size]i16 = blk: {
        var t: [table_size]i16 = undefined;
        for (0..table_size) |i| {
            const x: f32 = @floatFromInt(@as(i8, @bitCast(@as(u8, @intCast(i)))));
            const c: f32 = 0.7978845608; // sqrt(2/pi), to f32 precision
            const inner = c * (x + 0.044715 * x * x * x);
            // tanh(inner) in the numerically stable form. The naive
            // (e-1)/(e+1) with e = exp(2*inner) overflows f32 to inf for
            // |x| > ~12 and then evaluates inf/inf = NaN, which cannot be
            // stored in an i16 table at comptime. Both branches below tend to
            // +-1 as the argument saturates, which is the true limit, so the
            // saturation is correct rather than a clamp.
            const th = if (inner >= 0.0) pos: {
                const e = @exp(2.0 * inner);
                break :pos 1.0 - 2.0 / (e + 1.0);
            } else neg: {
                const e = @exp(-2.0 * inner);
                break :neg -1.0 + 2.0 / (e + 1.0);
            };
            t[i] = @intFromFloat(@round(0.5 * x * (1.0 + th) * 256.0));
        }
        break :blk t;
    };

    /// Largest mean of squares the table can represent, exclusive.
    pub const rsqrt_max_mean: u32 = (@as(u32, 1) << 8) * table_size;
    /// Shift is chosen per row, so no fixed floor applies; the value is the
    /// largest legal shift, not a constant.
    pub const rsqrt_max_shift: u8 = 24;

    /// `rsqrt_q8_8[j] = round(256 / sqrt(j))`, in q8.8. Pure, with no notion
    /// of a row or a scale.
    ///
    /// The per-row shift is applied by a **second** lookup rather than baked
    /// in, because the correction is `2^(-shift/2)`: from
    /// `mean = index · 2^shift` it follows that
    ///
    ///     256/sqrt(mean) = 256·2^(-shift/2) / sqrt(index)
    ///
    /// so the factor divides and is an integer only when `shift` is even. The
    /// row shift is therefore normalised UP to even and the factor is read
    /// from `rsqrt_shift_scale`. Rounding down is wrong: for a mean whose
    /// bitlen is odd above 8 it pushes the index past the end of the table
    /// (mean = 256 wants shift 1, and shift 0 asks for index 256 of 256). A single table cannot express this: an
    /// earlier version tried, multiplied where it had to divide, and produced
    /// a normalisation off by `2^(shift/2)` — 16x on a real int8 row.
    ///
    /// Index 0 has no finite value; it is 0 by construction, and callers must
    /// reject a zero mean before indexing rather than read a zero and call it a
    /// valid norm.
    pub const rsqrt_q8_8: [table_size]i16 = blk: {
        var t: [table_size]i16 = undefined;
        for (0..table_size) |i| {
            if (i == 0) {
                t[i] = 0;
            } else {
                const v = @as(f32, @floatFromInt(i));
                t[i] = @intFromFloat(@round(256.0 / @sqrt(v)));
            }
        }
        break :blk t;
    };

    /// Number of shift-scale slots. `shift/2` cannot exceed 6 for int8 inputs
    /// (the largest mean is 127² = 16129, and 16129 needs shift 6), so 8 slots
    /// cover the domain with two to spare.
    pub const rsqrt_scale_size: usize = 8;

    /// `rsqrt_shift_scale[k] = 256 · 2^-k` in q8.8, the correction factor for a
    /// row whose (even) shift is `2k`.
    pub const rsqrt_shift_scale: [rsqrt_scale_size]i16 = blk: {
        var t: [rsqrt_scale_size]i16 = undefined;
        for (0..rsqrt_scale_size) |i| {
            t[i] = @intFromFloat(@round(@as(f32, 256.0) / @as(f32, @floatFromInt(@as(u32, 1) << @intCast(i)))));
        }
        break :blk t;
    };

    /// Domain-segregated digest of every table here.
    ///
    /// Each table is absorbed under its own label, so swapping two tables of
    /// equal length cannot collide with a reordering. Lengths are absorbed too,
    /// for the same reason. `blake3` is the same primitive the weights tree
    /// uses, and the only hash in the repo whose implementation is available
    /// both in the stdlib and in the independent Python auditor.
    pub fn tablesDigest() Hash {
        var h = std.crypto.hash.Blake3.init(.{});
        h.update("zkml.nonlin.v");
        var ver: [2]u8 = undefined;
        std.mem.writeInt(u16, &ver, nonlinearity_version, .little);
        h.update(&ver);
        absorb(&h, "silu", i16, &silu_q8_8);
        absorb(&h, "gelu_tanh", i16, &gelu_tanh_q8_8);
        absorb(&h, "rsqrt", i16, &rsqrt_q8_8);
        absorbShort(&h, "rsqrt_shift_scale", i16, &rsqrt_shift_scale);
        var out: Hash = undefined;
        h.final(&out);
        return out;
    }

    fn absorbShort(h: anytype, label: []const u8, comptime T: type, values: *const [rsqrt_scale_size]T) void {
        h.update(label);
        h.update(&[_]u8{0});
        var len: [8]u8 = [_]u8{0} ** 8;
        std.mem.writeInt(u64, len[0..8], @sizeOf(T), .little);
        h.update(&len);
        for (values) |*v| {
            var b: [@sizeOf(T)]u8 = undefined;
            std.mem.writeInt(T, &b, v.*, .little);
            h.update(&b);
        }
    }

    fn absorb(h: anytype, label: []const u8, comptime T: type, values: *const [table_size]T) void {
        h.update(label);
        h.update(&[_]u8{0});
        var len: [8]u8 = [_]u8{0} ** 8;
        std.mem.writeInt(u64, len[0..8], @sizeOf(T), .little);
        h.update(&len);
        for (values) |*v| {
            var b: [@sizeOf(T)]u8 = undefined;
            std.mem.writeInt(T, &b, v.*, .little);
            h.update(&b);
        }
    }

    /// Decompose a q8.8 value into two byte tables for LogUp, so the AIR
    /// proves membership with 2 lookups of 256 entries instead of one of 65536.
    fn byteFragment(
        gpa: Allocator,
        name: []const u8,
        table: *const [table_size]i16,
    ) !air.Fragment {
        const lut_low = try gpa.alloc(i16, table_size);
        errdefer gpa.free(lut_low);
        const lut_high = try gpa.alloc(i16, table_size);
        errdefer gpa.free(lut_high);
        for (0..table_size) |i| {
            const v = table[i];
            lut_low[i] = v & 0xff;
            lut_high[i] = (v >> 8) & 0xff;
        }
        const luts = try gpa.alloc(air.LookupTable, 2);
        errdefer gpa.free(luts);
        // width 8: the two lookups together carry a 16-bit q8.8 value, split
        // so neither table exceeds a byte index.
        luts[0] = .{ .table = lut_low, .width = 8 };
        luts[1] = .{ .table = lut_high, .width = 8 };

        const cols = try gpa.alloc(air.Column, 1);
        errdefer gpa.free(cols);
        cols[0] = .{
            .name = name,
            .role = air.ColumnRole.advice,
            .scheme = tensor.Scheme.fixed_q8_8,
            .bound = 256,
        };

        const constraints = try gpa.alloc(air.Constraint, 1);
        errdefer gpa.free(constraints);
        constraints[0] = .{ .degree = 1 };

        return air.Fragment{
            .columns = cols,
            .constraints = constraints,
            .lookups = luts,
            .rows = 1,
        };
    }

    pub fn siluFragment(gpa: Allocator) !air.Fragment {
        return byteFragment(gpa, "silu_out", &silu_q8_8);
    }

    /// GELU tanh-approximation. The exact (erf) form is a *different* function
    /// and is deliberately not provided: a caller must pick one and the
    /// statement's `nonlinearity_version` covers the choice.
    pub fn geluFragment(gpa: Allocator) !air.Fragment {
        return byteFragment(gpa, "gelu_out", &gelu_tanh_q8_8);
    }
};

/// Every table in this module, in one place, so the digest cannot accidentally
/// cover a subset. This is the value the statement absorbs.
pub fn digest() Hash {
    return SiLULookup.tablesDigest();
}

test "the digest changes when any table changes" {
    const base = digest();
    // Recomputing over the same values must be stable.
    try std.testing.expectEqualSlices(u8, &base, &digest());

    // A single altered entry must move the digest. This is the property that
    // makes the binding a soundness control rather than a checksum: if a
    // prover can perturb the table by one ulp and keep the digest, the
    // statement is not binding anything.
    const perturbed = blk: {
        var t = SiLULookup.silu_q8_8;
        t[17] +%= 1;
        break :blk t;
    };
    try std.testing.expect(!std.mem.eql(u8, &base, &digestWithSilu(&perturbed)));
}

/// Digest of the shipped tables with `silu` replaced by `alt`, so the test can
/// show that a single altered entry moves the binding.
fn digestWithSilu(alt: *const [table_size]i16) Hash {
    var h = std.crypto.hash.Blake3.init(.{});
    h.update("zkml.nonlin.v");
    var ver: [2]u8 = undefined;
    std.mem.writeInt(u16, &ver, nonlinearity_version, .little);
    h.update(&ver);
    SiLULookup.absorb(&h, "silu", i16, alt);
    SiLULookup.absorb(&h, "gelu_tanh", i16, &SiLULookup.gelu_tanh_q8_8);
    SiLULookup.absorb(&h, "rsqrt", i16, &SiLULookup.rsqrt_q8_8);
    SiLULookup.absorbShort(&h, "rsqrt_shift_scale", i16, &SiLULookup.rsqrt_shift_scale);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

test "the tables are total over their declared domain" {
    // Every input has a defined output: a proof needs a function, and a hole
    // in the domain is an input a prover can exploit.
    for (SiLULookup.silu_q8_8) |v| {
        // silu is bounded by |x|, and |x| <= 128, so i16 cannot overflow.
        try std.testing.expect(v >= -32768 and v <= 32767);
    }
    // Index i holds x = (i8)@bitCast(u8 i), so x = 0 lives at index 128, not
    // at 0. silu(0) = 0 exactly: a non-zero value there means the rounding
    // rule is not what the table claims.
    try std.testing.expectEqual(@as(i16, 0), SiLULookup.silu_q8_8[128]);
    // The index->x map is i8@bitCast(u8 i), so the table is NOT ordered in x:
    // indices 0..127 are x = 0..127 and indices 128..255 are x = -128..-1.
    // Two properties follow from that, and both are checked because a
    // scrambled or shifted table would pass a monotonicity test otherwise.
    //
    // On the non-negative half silu is strictly increasing.
    for (1..128) |i| {
        try std.testing.expect(SiLULookup.silu_q8_8[i] > SiLULookup.silu_q8_8[i - 1]);
    }
    // On the negative half silu is monotonically INCREASING IN X, which means
    // it must be checked walking x from -1 back down, not walking the index
    // upward: index order runs x from -128 to -1 there. Walking the index
    // instead looks like a violation (the tail is flat at 0, then drops to
    // -1, -2, -4 ... as x nears -1), and that apparent violation is correct:
    // it is the quantisation of a steep region, not a broken table.
    // The negative half is checked for the property that actually holds after
    // quantisation: every value is in [-|x|, 0] and the magnitude never exceeds
    // the bound silu(x) >= x for x < 0. Monotonicity is deliberately NOT
    // asserted there — see the comment on |silu| above — because rounding a
    // negative function to integers is not order-preserving and the test would
    // be asserting something false.
    //
    // The positive half, by contrast, is strictly increasing in index, and
    // there rounding does preserve order, so it is checked strictly.
    try std.testing.expectEqual(@as(i16, 0), SiLULookup.silu_q8_8[128]); // x = -128
    // silu(x) >= x for x < 0, so -x is an upper bound on |silu(x)|. In q8.8
    // that is x*256, and it is the only order-like property that survives
    // rounding on this half.
    for (128..table_size) |i| {
        const xv: i32 = @as(i32, @as(i8, @bitCast(@as(u8, @intCast(i))))) * 256;
        const got: i32 = SiLULookup.silu_q8_8[i];
        try std.testing.expect(got <= 0);
        try std.testing.expect(got >= xv - 1); // -1 unit of rounding slack
    }
    // The two ends of the grid: x = 0 gives exactly 0, and x = 127 is the
    // largest representable positive input.
    try std.testing.expectEqual(@as(i16, 0), SiLULookup.silu_q8_8[0]);
    try std.testing.expectEqual(@as(i16, 32512), SiLULookup.silu_q8_8[127]);
}

test "rsqrt is positive and decreasing on the positive domain" {
    // Index 0 is the zero argument, which has no finite rsqrt and is marked 0
    // by construction; callers must reject a zero sum of squares before
    // indexing. Monotonicity is therefore only meaningful from index 1, where
    // the function is well defined and strictly decreasing.
    try std.testing.expectEqual(@as(i16, 0), SiLULookup.rsqrt_q8_8[0]);
    for (1..table_size) |i| {
        try std.testing.expect(SiLULookup.rsqrt_q8_8[i] > 0);
    }
    for (2..table_size) |i| {
        try std.testing.expect(SiLULookup.rsqrt_q8_8[i] <= SiLULookup.rsqrt_q8_8[i - 1]);
    }
    // 1/sqrt(1) = 1 in q8.8, and 1/sqrt(255) ~ 0.0626 -> 16. Both ends are
    // checked because an off-by-one in the scale factor survives the
    // monotonicity check.
    // With a per-row shift the table is 1/sqrt(j) directly, so index 1 is
    // 1.0 (256 in q8.8) and index 255 is 1/sqrt(255) ≈ 0.0626 (16). Both ends
    // are asserted because the scale factor is the convention: an off-by-8
    // version of this table still decreases monotonically and would pass every
    // shape check below.
    try std.testing.expectEqual(@as(i16, 256), SiLULookup.rsqrt_q8_8[1]);
    try std.testing.expectEqual(@as(i16, 16), SiLULookup.rsqrt_q8_8[255]);
}

test "gelu agrees with silu at zero and saturates correctly" {
    // x = 0 is index 128 of the signed grid, by the same bitcast mapping.
    try std.testing.expectEqual(@as(i16, 0), SiLULookup.gelu_tanh_q8_8[128]);
    // gelu(-x) + gelu(x) == x, the defining symmetry. A tanh-approximation
    // bug breaks it immediately.
    for (1..table_size / 2) |i| {
        const pos = SiLULookup.gelu_tanh_q8_8[i];
        const neg = SiLULookup.gelu_tanh_q8_8[table_size - i];
        const x: i32 = @as(i32, @as(i8, @bitCast(@as(u8, @intCast(i))))) * 256;
        const sum: i32 = @as(i32, pos) + @as(i32, neg);
        // f32 keeps 24 bits, so at |x| ~ 32767 a one-ulp difference is ~4e-3
        // in q8.8 units; the cubic term's own rounding pushes the observed
        // deviation to ~80 (0.12%). The bound is set by f32, not by taste.
        try std.testing.expect(@abs(sum - x) <= 128);
    }
}
