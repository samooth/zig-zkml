//! Symmetric-quantization binding for the GEMM operands (F2): range checks,
//! the dequantization equation, and scale provenance.
//!
//! The GEMM AIR proves `C = A·B` over the trace's own a/b columns. Without
//! a binding to the quantized weights, a prover could choose any operands
//! it liked and still produce a valid proof. This adds the missing half for
//! a GGML-style symmetric block of quant width `w`:
//!
//!   every row:  a = (q_a - 8) * scale_a
//!                b = (q_b - 8) * scale_b
//!                q_a in [0, 2^w),  q_b in [0, 2^w)   (GGML raw - 8)
//!
//! The quant bounds come from `libs/stark/range.zig` (bit decomposition plus
//! `bit*(bit-1) = 0`), so they compose with the existing quadratic IR.
//!
//! `SymmetricBinding(w)` is a comptime template, instantiated as `Q4_0` and
//! `Q8_0`. The format is explicit at every call site rather than inferred
//! from the data: a 4-bit and an 8-bit block share the equation above and
//! differ only in the range decomposition, and mixing them up would attest
//! a tensor no loader can produce.
//!
//! ## Scale provenance
//!
//! The equation alone binds the *representation* — each operand is a raw
//! quant value times one scale — but not the scale's provenance, so a
//! fabricated field element used to prove. That is closed for the formats
//! whose scale is an fp16 in q4.22 (`fp16ToFixedQ4_22`): `scale_air` pins
//! `u = ±(1024 + m)·2^s` with a 4-bit one-hot selector, 21 composed
//! constraints and 29 columns per scale.
//!
//! The barrel shifter was the original plan and was measured first: 131
//! composed constraints and 157 columns for the same job. The shift is only
//! four bits wide, so selecting it is cheaper than shifting by it.
//!
//! `provenance` is a template flag because the answer is not the same for
//! every format. Q8_0's scale is NOT an fp16 in q4.22 and the tensor layer
//! has no Q8_0 dequantizer yet, so applying this gadget there would attest
//! a scale convention Q8_0 does not use. `Q8_0` therefore runs without it:
//! it proves the representation, not the scale's provenance. That gap is
//! deliberate and documented, not an oversight — see the plan under F2.
//!
//! Column layout (on top of gemm_air's four), for quant width `w`:
//!
//!   4          q_a        unsigned quant value of the A operand
//!   5          q_b        same for B
//!   6..6+w-1   bits_a     binary expansion of q_a
//!   ..6+2w-1   bits_b     binary expansion of q_b
//!   +2w        scale_a
//!   +2w+1      scale_b
//!   then, with provenance: 10 mantissa bits, 16 shift selectors, and
//!   shift/out/sign for each of the two scales.
//!
//! Rows that are not MAC rows no longer exist: the trace is exactly the MAC
//! rows plus the synthetic closing row, and the closing row is exempt from
//! every composed constraint (see gemm_air.zig), so it carries no binding at
//! all. That exemption is what makes a per-row scale-provenance constraint
//! possible: previously the closing row had to be a dequantized operand of
//! `(1, -C)`, and no fp16 produces `-C`.

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

pub const BuildError = error{ OutOfMemory, BadWidth, InvalidReductionLength };

pub const BindError = error{
    OutOfMemory,
    /// A quant value outside [0, 2^width) for this format.
    QuantOutOfRange,
    InconsistentOperands,
    UnsupportedReductionLength,
    PaddedTrace,
    /// An fp16 pattern no q4.22 scale can come from (subnormal, too small,
    /// too large, inf or NaN). Only reachable when `provenance` is on.
    BadScale,
};

/// The seam where raw model bytes become a prover input.
///
/// The caller passes the raw fp16 PATTERN per MAC, not a field element: with
/// provenance on, the AIR needs mantissa, exponent and sign separately to
/// prove the scale is the image of `fp16ToFixedQ4_22`, and a `Goldilocks`
/// would throw that away before the witness is built. This stays the single
/// place where a pattern becomes the element the dequant equation consumes.
pub fn scaleFromFp16(bits: u16) tensor.Fp16Error!Goldilocks {
    return Goldilocks.fromU64(try tensor.fp16ToFixedQ4_22(bits));
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

/// A symmetric quantization scheme of width `quant_width`.
///
/// `provenance` turns on the fp16 scale gadget. It is a template parameter
/// rather than a runtime choice because the answer differs per format: Q8_0's
/// scale is not an fp16 in q4.22, so proving it with this gadget would attest
/// a convention Q8_0 does not use. See the module docs.
pub fn SymmetricBinding(comptime quant_width: u8, comptime provenance: bool) type {
    return struct {
        pub const width: u8 = quant_width;
        pub const max_quant: u8 = @intCast((@as(u16, 1) << quant_width) - 1);

        pub const col_q_a: u16 = 4;
        pub const col_q_b: u16 = 5;
        pub const col_bits_a: u16 = 6;
        pub const col_bits_b: u16 = col_bits_a + @as(u16, quant_width);
        pub const col_scale_a: u16 = col_bits_b + @as(u16, quant_width);
        pub const col_scale_b: u16 = col_scale_a + 1;
        const gadget_base: u16 = col_scale_b + 1;
        pub const column_count: usize = if (provenance)
            @as(usize, gadget_base) + 2 * (10 + scale_air.shift_count + 3)
        else
            @as(usize, gadget_base);

        const kAFactors = [_]Factor{.{ .column = .{ .index = gemm_air.col_a } }};
        const kQuantScaleA = [_]Factor{
            .{ .column = .{ .index = col_q_a } },
            .{ .column = .{ .index = col_scale_a } },
        };
        const kEightScaleA = [_]Factor{.{ .column = .{ .index = col_scale_a } }};
        const kDequantATerms = [_]Term{
            .{ .factors = &kAFactors },
            .{ .factors = &kQuantScaleA, .coefficient = Fp2.neg(Fp2.one) },
            .{ .factors = &kEightScaleA, .coefficient = Fp2.re(Goldilocks.fromU64(8)) },
        };

        const kBFactors = [_]Factor{.{ .column = .{ .index = gemm_air.col_b } }};
        const kQuantScaleB = [_]Factor{
            .{ .column = .{ .index = col_q_b } },
            .{ .column = .{ .index = col_scale_b } },
        };
        const kEightScaleB = [_]Factor{.{ .column = .{ .index = col_scale_b } }};
        const kDequantBTerms = [_]Term{
            .{ .factors = &kBFactors },
            .{ .factors = &kQuantScaleB, .coefficient = Fp2.neg(Fp2.one) },
            .{ .factors = &kEightScaleB, .coefficient = Fp2.re(Goldilocks.fromU64(8)) },
        };

        const kDequantConstraints = [_]Constraint{
            .{ .name = "a = (q_a - 8)*scale_a", .scope = .composed, .terms = &kDequantATerms },
            .{ .name = "b = (q_b - 8)*scale_b", .scope = .composed, .terms = &kDequantBTerms },
        };

        pub const range_specs = [_]range.Spec{
            .{ .column = col_q_a, .bit_base = col_bits_a, .width = quant_width },
            .{ .column = col_q_b, .bit_base = col_bits_b, .width = quant_width },
        };

        /// Constraints the provenance gadget adds per side: 1 sign boolean +
        /// 16 selector booleans + one-hot + `M = 2^s` + the shifted
        /// significand + the signed magnitude = 21.
        const kProvenancePerSide: usize = 1 + scale_air.shift_count + 4;

        /// The gadget's layout for one side. `offset` is measured from
        /// `gadget_base`, which is what keeps the two blocks from landing
        /// on top of the quant and bit columns.
        fn gadgetCfg(comptime offset: u16) scale_air.Config {
            const mant: u16 = gadget_base + offset;
            const sel: u16 = mant + @as(u16, 10);
            const shift: u16 = sel + @as(u16, scale_air.shift_count);
            return .{
                .scale = if (offset == 0) col_scale_a else col_scale_b,
                .mant_base = mant,
                .sel_base = sel,
                .shift_col = shift,
                .out_col = shift + 1,
                .sign_col = shift + 2,
            };
        }
        pub const gadget_a: scale_air.Config = gadgetCfg(0);
        pub const gadget_b: scale_air.Config = gadgetCfg(@as(u16, 10 + scale_air.shift_count + 3));

        /// Owns the merged system. The gadget's `Owned` is a field because
        /// the merged slice holds COPIES of its Constraint structs that point
        /// into its buffers; dropping it would leave the system pointing at
        /// freed memory. Same ownership shape as `range.BuiltSystem.checks`.
        pub const BoundSystem = struct {
            inner: range.BuiltSystem,
            gadget: if (provenance) air_builder.Owned else void,

            pub fn system(self: BoundSystem) System {
                return self.inner.system;
            }

            pub fn deinit(self: *BoundSystem) void {
                self.inner.deinit();
                if (provenance) self.gadget.deinit();
            }
        };

        /// GEMM AIR + dequantization equations + range checks (+ provenance).
        ///
        /// `k` is not optional: it is what pins the trace's shape through
        /// `gemm_air.system(k)`, and `replaceConstraints` is what carries
        /// `trace_rows` and `transition_exemptions` into the merged system.
        /// Both were lost when this binding was generalised from a
        /// hard-coded Q4_K one, which called `gemm_air.system()` with no
        /// argument and rebuilt the System from scratch.
        pub fn buildSystem(allocator: std.mem.Allocator, k: usize) BuildError!BoundSystem {
            const base = gemm_air.system(k) catch return BuildError.InvalidReductionLength;
            var inner = range.BuiltSystem.init(allocator, base, &range_specs) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.BadWidth => return error.BadWidth,
            };
            errdefer inner.deinit();

            var n_extra = kDequantConstraints.len;
            var b = air_builder.Builder.init(allocator);
            defer b.deinit();
            if (provenance) {
                try scale_air.build(&b, gadget_a);
                try scale_air.build(&b, gadget_b);
                std.debug.assert(b.count() == 2 * kProvenancePerSide);
                n_extra += b.count();
            }

            const n = inner.constraints.len;
            const grown = try allocator.alloc(Constraint, n + n_extra);
            @memcpy(grown[0..n], inner.constraints);
            @memcpy(grown[n .. n + kDequantConstraints.len], &kDequantConstraints);
            if (provenance) {
                const frozen = try air_builder.freeze(allocator, &b, 1);
                @memcpy(grown[n + kDequantConstraints.len ..], frozen.constraints);
                allocator.free(inner.constraints);
                inner.constraints = grown;
                inner.system = inner.system.replaceConstraints(grown);
                return .{ .inner = inner, .gadget = frozen };
            }
            allocator.free(inner.constraints);
            inner.constraints = grown;
            inner.system = inner.system.replaceConstraints(grown);
            return .{ .inner = inner, .gadget = {} };
        }

        /// Extend a GEMM trace with this format's quantization columns.
        ///
        /// `q_a` / `q_b` hold one raw quant value per MAC, and `scale_a` /
        /// `scale_b` the raw fp16 scale PATTERN per MAC: a reduction spanning
        /// several blocks has a different scale in each, so a single scalar
        /// would refuse honest multi-block witnesses. The GEMM operand
        /// columns must equal `(q - 8) * scale` exactly, or the trace is not
        /// a dequantization of anything.
        pub fn bindOperands(
            allocator: std.mem.Allocator,
            gemm_trace: *const gemm_air.Trace,
            q_a: []const u8,
            scale_a: []const u16,
            q_b: []const u8,
            scale_b: []const u16,
        ) BindError!Trace {
            const rows = gemm_trace.rows;
            const real = gemm_air.realRowsFor(q_a.len);
            if (q_b.len != q_a.len or scale_a.len != q_a.len or scale_b.len != q_a.len) {
                return BindError.InconsistentOperands;
            }
            // Shape: the trace must be exactly the MAC rows plus the
            // closing row, with nothing padded in between.
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
            for (gemm_trace.columns, 0..) |src, i| @memcpy(cols[i], src);

            for (0..rows) |r| {
                if (r >= real) {
                    // The closing row is exempt from every composed
                    // constraint, so its columns are witness with nothing
                    // to satisfy. It is not a MAC row: the operands there
                    // are synthetic, and a scale there would have to be the
                    // image of an fp16 with no reason to exist. Zero.
                    continue;
                }
                const qa: u8 = q_a[r];
                const qb: u8 = q_b[r];
                if (qa > max_quant or qb > max_quant) return BindError.QuantOutOfRange;

                cols[col_q_a][r] = Fp2.re(Goldilocks.fromU64(qa));
                cols[col_q_b][r] = Fp2.re(Goldilocks.fromU64(qb));
                for (0..quant_width) |bit| {
                    const bit_index: u16 = @intCast(bit);
                    const shift: u3 = @intCast(bit);
                    cols[col_bits_a + bit_index][r] = Fp2.re(Goldilocks.fromU64((qa >> shift) & 1));
                    cols[col_bits_b + bit_index][r] = Fp2.re(Goldilocks.fromU64((qb >> shift) & 1));
                }

                const sa = scaleFromFp16(scale_a[r]) catch return BindError.BadScale;
                const sb = scaleFromFp16(scale_b[r]) catch return BindError.BadScale;
                cols[col_scale_a][r] = Fp2.re(sa);
                cols[col_scale_b][r] = Fp2.re(sb);
                if (provenance) {
                    // Witness the gadget from the fp16 pattern, then take
                    // the scale the equation uses from the SAME function:
                    // deriving it from the gadget's own `out` would let the
                    // witness and the constraint disagree.
                    scale_air.writeWitness(
                        cols,
                        r,
                        gadget_a.mant_base,
                        gadget_a.sel_base,
                        gadget_a.shift_col,
                        gadget_a.out_col,
                        gadget_a.sign_col,
                        scale_a[r],
                    ) catch return BindError.BadScale;
                    scale_air.writeWitness(
                        cols,
                        r,
                        gadget_b.mant_base,
                        gadget_b.sel_base,
                        gadget_b.shift_col,
                        gadget_b.out_col,
                        gadget_b.sign_col,
                        scale_b[r],
                    ) catch return BindError.BadScale;
                }

                const a_val = gemm_trace.columns[gemm_air.col_a][r].a;
                const b_val = gemm_trace.columns[gemm_air.col_b][r].a;
                const eight = Goldilocks.fromU64(8);
                const want_a = Goldilocks.fromU64(qa).sub(eight).mul(sa);
                const want_b = Goldilocks.fromU64(qb).sub(eight).mul(sb);
                if (!want_a.eql(a_val) or !want_b.eql(b_val)) return BindError.InconsistentOperands;
            }
            return .{ .rows = rows, .columns = cols };
        }
    };
}

/// 4-bit symmetric block, GGML raw minus 8. Its scale IS an fp16 in
/// q4.22, so the provenance gadget applies.
pub const Q4_0 = SymmetricBinding(4, true);
/// 8-bit symmetric block. KNOWN GAP: no scale provenance, because Q8_0's
/// scale is not an fp16 in q4.22 and the tensor layer has no Q8_0
/// dequantizer yet. This binding proves the representation, not the scale's
/// origin. Tracked under F2 in the plan.
pub const Q8_0 = SymmetricBinding(8, false);
