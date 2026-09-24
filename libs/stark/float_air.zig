//! Bit-exact fp16 multiply in the STARK (F2, spike S1).
//!
//! This is the spike that decides whether the architecture works. If
//! IEEE-754 rounding is not expressible in this IR, "works for any model"
//! is false and the plan changes again. The answer the spike found: it IS
//! expressible, at **108 composed constraints per multiply** — which is
//! affordable precisely because the fingerprint statement touches O(m+n)
//! values instead of O(mnk) (TODO, ROADMAP S3).
//!
//! ## The idea
//!
//! The product of two binary16 significands is an EXACT integer: each is
//! 11 bits, so A·B is 21 bits with no rounding anywhere. The only rounding
//! in the whole operation is the step down to the output's 11 bits, and
//! IEEE-754 round-to-nearest-even is decided by exactly two booleans —
//! the round bit and the sticky bit. Everything else is linear bookkeeping
//! over the bit columns that already exist.
//!
//! The one piece of genuine data dependence is NORMALISATION: the product
//! of two significands in [2^10, 2^11) lands in [2^20, 2^22), which is
//! 11 kept bits when it is >= 2^21 and 10 when it is not. That single bit
//! is a witness column (`norm`), and both the round/sticky selection and
//! the kept bits are 2-way muxes over it — degree 2, not a barrel shifter.
//! A barrel shifter only becomes necessary for denormal range reduction
//! (S2), which is why this is affordable here.
//!
//! ## Constraint inventory (per multiply)
//!
//!   48  booleanity of the two input patterns and the output pattern
//!    2  input significands from the mantissa bits
//!    2  the exponent values (a 5-bit sum cannot be multiplied directly)
//!    4  the inputs are neither subnormal (exp != 0) nor inf/NaN (exp != 31)
//!    1  the exact product, ONE degree-2 constraint
//!   23  the product's own bit decomposition (reconstruction + booleanity)
//!    1  norm = bit 21 of the product
//!    2  round-bit mux and the sticky mux
//!    6  two sticky ORs, via the "sum is non-zero" inverse trick
//!   11  kept-bit muxes
//!    4  or_sl and the increment, same inverse trick
//!    1  carry booleanity
//!    2  the output exponent and mantissa sums
//!    1  the output sign
//!
//! ## Scope
//!
//! The column layout, for one format. Everything the AIR needs is derived
//! from (exp_bits, mant_bits, bias) — see float_format.zig for why.

const std = @import("std");
const expr = @import("./expr.zig");
const range = @import("./range.zig");
const tensor = @import("../tensor/root.zig");
const float_ref = @import("./float_ref.zig");
const fmt_lib = @import("./float_format.zig");

pub const Fp2 = expr.Fp2;
pub const Goldilocks = tensor.Goldilocks;
pub const System = expr.System;
pub const Constraint = expr.Constraint;
pub const Term = expr.Term;
pub const Factor = expr.Factor;
pub const Scope = expr.Scope;
pub const Format = fmt_lib.Format;

/// The trace layout for one format. Column-major: every operand is a ROW,
/// so all the copies of a constraint below are identical and share storage.
pub fn Layout(comptime f: Format) type {
    return struct {
        const width: u16 = f.byteWidth();
        const prod_bits: u16 = f.productBits();
        const kept: u16 = f.sigBits();
        const mant: u16 = f.mant_bits;
        const exps: u16 = f.exp_bits;

        pub const col_a_bits: u16 = 0;
        pub const col_b_bits: u16 = col_a_bits + width;
        pub const col_a_sig: u16 = col_b_bits + width;
        pub const col_b_sig: u16 = col_a_sig + 1;
        pub const col_a_exp_raw: u16 = col_b_sig + 1;
        pub const col_b_exp_raw: u16 = col_a_exp_raw + 1;
        pub const col_product: u16 = col_b_exp_raw + 1;
        pub const col_p_bits: u16 = col_product + 1;
        pub const col_norm: u16 = col_p_bits + prod_bits;
        pub const col_round: u16 = col_norm + 1;
        pub const col_sticky_hi_sum: u16 = col_round + 1;
        pub const col_sticky_hi_inv: u16 = col_sticky_hi_sum + 1;
        pub const col_sticky_lo_sum: u16 = col_sticky_hi_inv + 1;
        pub const col_sticky_lo_inv: u16 = col_sticky_lo_sum + 1;
        pub const col_sticky_hi: u16 = col_sticky_lo_inv + 1;
        pub const col_sticky_lo: u16 = col_sticky_hi + 1;
        pub const col_sticky: u16 = col_sticky_lo + 1;
        pub const col_kept: u16 = col_sticky + 1;
        pub const col_or_sl_sum: u16 = col_kept + kept;
        pub const col_or_sl_inv: u16 = col_or_sl_sum + 1;
        pub const col_or_sl: u16 = col_or_sl_inv + 1;
        pub const col_inc: u16 = col_or_sl + 1;
        pub const col_carry: u16 = col_inc + 1;
        pub const col_c_bits: u16 = col_carry + 1;
        // Overflow. When the rounded result's exponent reaches the all-ones
        // field the answer is INFINITY, whose mantissa is zero — while the
        // arithmetic below still produces the correctly rounded
        // significand. So the output keeps one set of bits, the flag
        // zeroes their mantissa, and the mantissa equation is switched off
        // exactly when the flag is set. The exponent needs no switch: an
        // overflow's exponent IS the all-ones field, so the same equation
        // holds either way.
        pub const col_c_mant_val: u16 = col_c_bits + width;
        pub const col_diff: u16 = col_c_mant_val + 1;
        pub const col_not_overflow: u16 = col_diff + 1;
        pub const col_path_on: u16 = col_not_overflow + 1;
        pub const col_d0: u16 = col_path_on + 1;
        pub const col_d0_gap: u16 = col_d0 + 1;
        pub const col_c_exp_val: u16 = col_d0_gap + 1;
        pub const col_overflow: u16 = col_c_exp_val + 1;

        /// Classification of the two inputs. The AIR no longer ASSERTS that
        /// they are normal — it classifies them, and the selector below
        /// picks the answer. A subnormal input matches no class, which is
        /// what makes it unsatisfiable rather than silently wrong.
        pub const col_a_mant_val: u16 = col_overflow + 1;
        pub const col_a_exp_zero: u16 = col_a_mant_val + 1;
        pub const col_a_mant_zero: u16 = col_a_exp_zero + 1;
        pub const col_a_exp_max: u16 = col_a_mant_zero + 1;
        pub const col_a_zero_inv: u16 = col_a_exp_max + 1;
        pub const col_a_mant_inv: u16 = col_a_zero_inv + 1;
        pub const col_a_max_inv: u16 = col_a_mant_inv + 1;
        pub const col_a_gap: u16 = col_a_max_inv + 1;
        pub const col_b_mant_val: u16 = col_a_gap + 1;
        pub const col_b_exp_zero: u16 = col_b_mant_val + 1;
        pub const col_b_mant_zero: u16 = col_b_exp_zero + 1;
        pub const col_b_exp_max: u16 = col_b_mant_zero + 1;
        pub const col_b_zero_inv: u16 = col_b_exp_max + 1;
        pub const col_b_mant_inv: u16 = col_b_zero_inv + 1;
        pub const col_b_max_inv: u16 = col_b_mant_inv + 1;
        pub const col_b_gap: u16 = col_b_max_inv + 1;

        /// The classes, per operand. `is_normal` is the one the arithmetic
        /// path uses; the other three feed the selector.
        pub const col_a_is_zero: u16 = col_b_gap + 1;
        pub const col_a_is_inf: u16 = col_a_is_zero + 1;
        pub const col_a_is_nan: u16 = col_a_is_inf + 1;
        pub const col_a_is_normal: u16 = col_a_is_nan + 1;
        pub const col_b_is_zero: u16 = col_a_is_normal + 1;
        pub const col_b_is_inf: u16 = col_b_is_zero + 1;
        pub const col_b_is_nan: u16 = col_b_is_inf + 1;
        pub const col_b_is_normal: u16 = col_b_is_nan + 1;

        /// The sanitised values the arithmetic path reads, so it stays
        /// satisfiable whatever the inputs are.
        pub const col_a_sig_eff: u16 = col_b_is_normal + 1;
        pub const col_a_exp_eff: u16 = col_a_sig_eff + 1;
        pub const col_b_sig_eff: u16 = col_a_exp_eff + 1;
        pub const col_b_exp_eff: u16 = col_b_sig_eff + 1;

        /// The selected class: exactly one of nan / inf / zero / normal.
        pub const col_nan_any: u16 = col_b_exp_eff + 1;
        pub const col_bad_pair: u16 = col_nan_any + 1;
        pub const col_s_nan: u16 = col_bad_pair + 1;
        pub const col_s_inf: u16 = col_s_nan + 1;
        pub const col_s_zero: u16 = col_s_inf + 1;
        pub const col_s_normal: u16 = col_s_zero + 1;

        /// The ANSWER's bits, selected from the arithmetic path's bits and
        /// the three special patterns.
        pub const col_out: u16 = col_s_normal + 1;
        pub const column_count: usize = col_out + width;

        pub inline fn aBit(i: u16) u16 {
            return col_a_bits + i;
        }
        pub inline fn bBit(i: u16) u16 {
            return col_b_bits + i;
        }
        /// The RAW exponent field value, which the classifier reads. The
        /// reconstruction constraints below are on the sanitised columns.
        pub inline fn aExpVal() u16 {
            return col_a_exp_raw;
        }
        pub inline fn bExpVal() u16 {
            return col_b_exp_raw;
        }
        pub inline fn pBit(i: u16) u16 {
            return col_p_bits + i;
        }
        pub inline fn cBit(i: u16) u16 {
            return col_c_bits + i;
        }
        pub inline fn aExp() u16 {
            return aBit(mant);
        }
        pub inline fn aMant() u16 {
            return aBit(0);
        }
        pub inline fn bExp() u16 {
            return bBit(mant);
        }
        pub inline fn bMant() u16 {
            return bBit(0);
        }
        pub inline fn cExp() u16 {
            return cBit(mant);
        }
        pub inline fn cMant() u16 {
            return cBit(0);
        }
        pub inline fn aSign() u16 {
            return aBit(width - 1);
        }
        pub inline fn bSign() u16 {
            return bBit(width - 1);
        }
        pub inline fn cSign() u16 {
            return cBit(width - 1);
        }
        /// The output exponent's value, and the all-ones gap that says
        /// whether it reached emax.
        pub inline fn cExpVal() u16 {
            return col_c_exp_val;
        }
        /// The product bit that decides normalisation.
        pub inline fn topBit() u16 {
            return pBit(@intCast(f.normBit()));
        }
    };
}

/// The shared builder, moved out of this file so the other AIRs do not
/// reimplement the freeze-once discipline.
const bld = @import("./air_builder.zig");
const Builder = bld.Builder;
const LinTerm = bld.LinTerm;
pub const Trace = bld.Trace;
const FRange = bld.FRange;
const g = bld.g;
const kNegModPow2 = bld.kNegModPow2;
const kOne = bld.kOne;
const kNegOne = bld.kNegOne;

/// A constraint name built at comptime, so the numbers in it are the
/// format's numbers and not binary16's.
fn cname(comptime fmt: []const u8, comptime args: anytype) []const u8 {
    return std.fmt.comptimePrint(fmt, args);
}

/// Number of constraints per multiply — the spike's headline number.
pub const constraints_per_multiply: usize = 163;

/// The cost model, in constraints per multiply, derived from the format's
/// widths: 2·byteWidth bit-decompositions, the product's bit-decomposition
/// and reconstruction, the rounding witnesses, the sticky ORs, and the
/// per-format input guards. `buildSystem` prints if the built system ever
/// disagrees, and the test asserts it for every format.
pub fn expected_constraints(f: Format) usize {
    // Three bit-decompositions (a, b, c), the product's, and the kept
    // significand's are the only width-dependent parts; the other 27 are
    // the two significands, the two exponent reconstructions, the four
    // exponent guards, the product and its reconstruction, norm, round,
    // the two sticky ORs and their mux, the increment OR and its mux, the
    // carry booleanity, and the mantissa, exponent and sign equations.
    // The +10 over the pre-overflow 27. The old single mantissa equation is
    // replaced, not duplicated, so the additions are: the output exponent's
    // bit sum (+1) and its arithmetic column d0 (+1), the clamp's gap (+1)
    // and its product (+1), the flag's gap (+1) and inverse identity (+1),
    // the output mantissa's value (+1), the rounded-mantissa difference
    // (+1), the not-overflow complement (+1), the gated equation (+1), and
    // the zero mantissa on overflow (+1).
    // 4·width + productBits + sigBits + 66, MEASURED for the four formats
    // and pinned by the cost test. FOUR width-dependent parts, not three:
    // the two operand bit decompositions, the OUTPUT's, the product's, the
    // kept significand's, and the answer selector, which is one constraint
    // per output bit. The constant is the classifier (two mantissa values,
    // six zero-or-invertible pairs on the two exponent gaps, the two
    // mantissa-zero flags, eight class bits, four sanitised values), the
    // four answer rules with the exhaustiveness, the selector's gating
    // column, and the overflow switch.
    return 4 * @as(usize, f.byteWidth()) + f.productBits() + f.keptHigh() + 66;
}

/// Build the fp16 multiply AIR. `rows` multiplies, one per row.
pub const BuildError = bld.BuildError || error{BadWidth};

pub fn buildSystem(allocator: std.mem.Allocator, rows: usize, comptime f: Format) BuildError!bld.Owned {
    const L = Layout(f);
    // comptime so the constraint NAMES can carry the constants they
    // actually constrain: a name saying "1024" on a bf16 system is a lie.
    const nm_a_sig = comptime cname("a significand = {d} + a mantissa", .{f.mantImplicit()});
    const nm_b_sig = comptime cname("b significand = {d} + b mantissa", .{f.mantImplicit()});
    const nm_norm = comptime cname("norm = product bit {d}", .{f.normBit()});
    const nm_round = comptime cname("round = mux(norm, p{d}, p{d})", .{ f.keptHigh() - 1, f.keptHigh() - 2 });
    const nm_exp = comptime cname("ec = ea + eb + {d} + norm + carry − {d}", .{ f.keptLow(), @as(u16, f.bias) + @as(u16, f.mant_bits) - @as(u16, f.keptLow()) });
    std.debug.assert(rows > 0);
    var b = Builder{ .allocator = allocator };
    errdefer {
        b.factors.deinit(allocator);
        b.terms.deinit(allocator);
        b.constraints.deinit(allocator);
    }

    // Booleanity of the two input patterns.
    for (0..L.width) |i| {
        const col: u16 = @intCast(i);
        try b.lin("a bit boolean", .composed, &.{
            .{ .factors = try b.pair(col + 0, col + 0), .coefficient = kOne },
            .{ .factors = try b.pair(col + 0, col + 0), .coefficient = kNegOne },
        });
    }
    for (0..L.width) |i| {
        const col: u16 = @intCast(i);
        try b.lin("b bit boolean", .composed, &.{
            .{ .factors = try b.pair(col + 0, col + 0), .coefficient = kOne },
            .{ .factors = try b.pair(col + 0, col + 0), .coefficient = kNegOne },
        });
    }
    // Booleanity of the output pattern.
    for (0..L.width) |i| {
        const col: u16 = @intCast(i);
        try b.lin("c bit boolean", .composed, &.{
            .{ .factors = try b.pair(col + 0, col + 0), .coefficient = kOne },
            .{ .factors = try b.pair(col + 0, col + 0), .coefficient = kNegOne },
        });
    }

    // Input significands: sig = 1024 + mantissa.
    {
        var ts: [2 + L.mant]LinTerm = undefined;
        ts[0] = .{ .factors = try b.one(L.col_a_sig) };
        ts[1] = .{ .factors = try b.constant(g(f.mantImplicit())), .coefficient = kNegOne };
        for (0..L.mant) |i| {
            ts[2 + i] = .{
                .factors = try b.one(L.aMant() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
        }
        try b.lin(nm_a_sig, .composed, &ts);
    }
    {
        var ts: [2 + L.mant]LinTerm = undefined;
        ts[0] = .{ .factors = try b.one(L.col_b_sig) };
        ts[1] = .{ .factors = try b.constant(g(f.mantImplicit())), .coefficient = kNegOne };
        for (0..L.mant) |i| {
            ts[2 + i] = .{
                .factors = try b.one(L.bMant() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
        }
        try b.lin(nm_b_sig, .composed, &ts);
    }

    // The exponent as a VALUE, because a 5-bit sum cannot be multiplied
    // by anything without becoming degree 6. Two extra columns and two
    // linear constraints are what keep the inverse proofs at degree 2.
    {
        var ts: [L.exps + 1]LinTerm = undefined;
        ts[0] = .{ .factors = try b.one(L.aExpVal()) };
        for (0..L.exps) |i| {
            ts[1 + i] = .{
                .factors = try b.one(L.aExp() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
        }
        try b.lin("a exponent value", .composed, &ts);
    }
    {
        var ts: [L.exps + 1]LinTerm = undefined;
        ts[0] = .{ .factors = try b.one(L.bExpVal()) };
        for (0..L.exps) |i| {
            ts[1 + i] = .{
                .factors = try b.one(L.bExp() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
        }
        try b.lin("b exponent value", .composed, &ts);
    }

    // Input classification. The AIR used to ASSERT exp != 0 and exp != emax
    // with two inverses per operand; now it CLASSIFIES both operands, and
    // the selector below picks the answer. A subnormal input matches no
    // class at all, and the exhaustiveness constraint is what turns that
    // into "unprovable" instead of "silently wrong".
    for (0..2) |which| {
        const mant_val = if (which == 0) L.col_a_mant_val else L.col_b_mant_val;
        const exp_zero = if (which == 0) L.col_a_exp_zero else L.col_b_exp_zero;
        const mant_zero = if (which == 0) L.col_a_mant_zero else L.col_b_mant_zero;
        const exp_max = if (which == 0) L.col_a_exp_max else L.col_b_exp_max;
        const zero_inv = if (which == 0) L.col_a_zero_inv else L.col_b_zero_inv;
        const mant_inv = if (which == 0) L.col_a_mant_inv else L.col_b_mant_inv;
        const max_inv = if (which == 0) L.col_a_max_inv else L.col_b_max_inv;
        const gap = if (which == 0) L.col_a_gap else L.col_b_gap;

        var mv: [1 + f.mant_bits]LinTerm = undefined;
        mv[0] = .{ .factors = try b.one(mant_val) };
        for (0..f.mant_bits) |i| {
            mv[1 + @as(usize, @intCast(i))] = .{
                .factors = try b.one(if (which == 0) L.aBit(@intCast(i)) else L.bBit(@intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
        }
        try b.lin("input mantissa value is its bits", .composed, &mv);
        try b.lin("mantissa gap to emax", .composed, &.{
            .{ .factors = try b.one(gap) },
            .{ .factors = try b.one(if (which == 0) L.aExpVal() else L.bExpVal()) },
            .{ .factors = try b.constant(g(f.emax())), .coefficient = kNegOne },
        });
        const exp_val = if (which == 0) L.aExpVal() else L.bExpVal();
        // "[exp == 0]" and "[mantissa == 0]" are the zero-or-invertible
        // pattern: no range check, no comparison, one witness each.
        try b.zeroOrNonZero("exponent is zero", exp_val, zero_inv, exp_zero);
        try b.zeroOrNonZero("mantissa is zero", mant_val, mant_inv, mant_zero);
        try b.zeroOrNonZero("exponent is all ones", gap, max_inv, exp_max);
    }

    // The exact product. One degree-2 constraint is the entire "multiply".
    try b.lin("product = a_sig · b_sig", .composed, &.{
        .{ .factors = try b.one(L.col_product) },
        .{ .factors = try b.pair(L.col_a_sig_eff, L.col_b_sig_eff), .coefficient = kNegOne },
    });

    // The product's own bit decomposition: this is what makes the round
    // and sticky bits available as columns.
    {
        var ts: [1 + f.productBits()]LinTerm = undefined;
        ts[0] = .{ .factors = try b.one(L.col_product) };
        for (0..f.productBits()) |i| {
            ts[1 + i] = .{
                .factors = try b.one(L.pBit(@intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
        }
        try b.lin("product bits reconstruct", .composed, &ts);
    }
    for (0..f.productBits()) |i| {
        const col = L.pBit(@intCast(i));
        try b.lin("product bit boolean", .composed, &.{
            .{ .factors = try b.pair(col, col), .coefficient = kOne },
            .{ .factors = try b.pair(col, col), .coefficient = kNegOne },
        });
    }

    // norm = [product >= 2^normBit], which is exactly that bit.
    try b.lin(nm_norm, .composed, &.{
        .{ .factors = try b.one(L.col_norm) },
        .{ .factors = try b.one(L.topBit()), .coefficient = kNegOne },
    });

    // round = norm ? bit10 : bit9. A 2-way mux is degree 2, and a mux of
    // two booleans under a boolean selector is boolean for free.
    // y = norm ? a : b  is  y - norm·a - (1 - norm)·b = 0. Writing the
    // (1 - norm)·b term with a POSITIVE sign is correct for the STICKY mux
    // (where it is a sum of ORs) and wrong for every real mux.
    try b.lin(nm_round, .composed, &.{
        .{ .factors = try b.one(L.col_round) },
        .{ .factors = try b.pair(L.col_norm, L.pBit(@intCast(f.keptHigh() - 1))), .coefficient = kNegOne },
        .{ .factors = try b.one(L.pBit(@intCast(f.keptHigh() - 2))), .coefficient = kNegOne },
        .{ .factors = try b.pair(L.col_norm, L.pBit(@intCast(f.keptHigh() - 2))) },
    });

    // Sticky, as OR over the low product bits. An OR is an INEQUALITY
    // ("bit <= or"), and this IR has only equalities — the first attempt
    // wrote `p_i - or = 0`, which forces every bit to equal the OR and so
    // only holds when they all agree.
    //
    // The trick is the same field-inverse one used for the exponents: with
    // S = sum of the bits (all boolean, so S != 0 iff some bit is set),
    //   S · S⁻¹ = sticky   plus   sticky · (sticky - 1) = 0
    // forces sticky = 1 exactly when S != 0. Three constraints instead of
    // a dozen, and no inequality.
    try stickyOr(&b, L.col_sticky_hi_sum, L.col_sticky_hi_inv, L.col_sticky_hi, L.pBit(0), f.keptLow());
    try stickyOr(&b, L.col_sticky_lo_sum, L.col_sticky_lo_inv, L.col_sticky_lo, L.pBit(0), f.keptLow() - 1);
    try b.lin("sticky = mux(norm, sticky_hi, sticky_lo)", .composed, &.{
        .{ .factors = try b.one(L.col_sticky) },
        .{ .factors = try b.pair(L.col_norm, L.col_sticky_hi), .coefficient = kNegOne },
        .{ .factors = try b.one(L.col_sticky_lo), .coefficient = kNegOne },
        .{ .factors = try b.pair(L.col_norm, L.col_sticky_lo) },
    });

    // The kept field: kept_i = norm ? p_(i+11) : p_(i+10). 11 muxes, and
    // the result is boolean for free because it is a mux of two booleans.
    for (0..f.sigBits()) |i| {
        const idx: u16 = @intCast(i);
        try b.lin("kept bit mux", .composed, &.{
            .{ .factors = try b.one(L.col_kept + idx) },
            .{ .factors = try b.pair(L.col_norm, L.pBit(idx + @as(u16, f.keptLow()) + 1)), .coefficient = kNegOne },
            .{ .factors = try b.one(L.pBit(idx + @as(u16, f.keptLow()))), .coefficient = kNegOne },
            .{ .factors = try b.pair(L.col_norm, L.pBit(idx + @as(u16, f.keptLow()))) },
        });
    }

    // RNE: increment iff round AND (sticky OR the kept field's lsb). The
    // lsb has to be the PRE-increment one: on a tie with an odd kept field
    // the increment flips it to 0, so reading the output mantissa's bit 0
    // would say "no increment" exactly when RNE says "increment".
    // or_sl = sticky OR kept_lsb, by the same sum/inverse trick: two
    // booleans whose sum is non-zero exactly when the OR is 1.
    try b.lin("or_sl sum", .composed, &.{
        .{ .factors = try b.one(L.col_or_sl_sum) },
        .{ .factors = try b.one(L.col_sticky), .coefficient = kNegOne },
        .{ .factors = try b.one(L.col_kept), .coefficient = kNegOne },
    });
    try b.lin("or_sl sum · inv", .composed, &.{
        .{ .factors = try b.pair(L.col_or_sl_sum, L.col_or_sl_inv) },
        .{ .factors = try b.one(L.col_or_sl), .coefficient = kNegOne },
    });
    try b.lin("or_sl is boolean", .composed, &.{
        .{ .factors = try b.pair(L.col_or_sl, L.col_or_sl) },
        .{ .factors = try b.one(L.col_or_sl), .coefficient = kNegOne },
    });
    try b.lin("inc = round · or_sl", .composed, &.{
        .{ .factors = try b.one(L.col_inc) },
        .{ .factors = try b.pair(L.col_round, L.col_or_sl), .coefficient = kNegOne },
    });
    try b.lin("carry is boolean", .composed, &.{
        .{ .factors = try b.pair(L.col_carry, L.col_carry) },
        .{ .factors = try b.one(L.col_carry), .coefficient = kNegOne },
    });

    // The output exponent. The bit sum is routed through a value column
    // because that value is what the overflow flag reads.
    {
        var bits: [1 + f.exp_bits]LinTerm = undefined;
        bits[0] = .{ .factors = try b.one(L.cExpVal()) };
        for (0..L.exps) |i| {
            const d: u16 = @intCast(i);
            bits[1 + d] = .{
                .factors = try b.one(L.cExp() + d),
                .coefficient = g(kNegModPow2(d)),
            };
        }
        try b.lin("output exponent value is its bits", .composed, &bits);
    }
    {
        // ec = ea + eb + keep + carry − (bias + mant_bits), unchanged: an
        // overflow's exponent IS the all-ones field, so this equation is
        // what MAKES the flag correct rather than a separate assumption.
        // d0 = ea + eb + keep + carry − (bias + mant_bits), the arithmetic
        // exponent. The OUTPUT's exponent is the clamped copy, below.
        // GATED by s_normal: for a special row the arithmetic exponent is
        // meaningless, and the clamp below then has nothing sensible to
        // clamp. Gating costs one factor per term and stays degree 2.
        const gate: FRange = @as(FRange, try b.one(L.col_s_normal));
        const raw_terms = [_]LinTerm{.{ .factors = try b.one(L.col_d0) }};
        var ts: [2 * f.exp_bits + 4]LinTerm = undefined;
        var n: usize = 0;
        ts[n] = .{ .factors = try b.pairOf(gate, raw_terms[0].factors) };
        n += 1;
        for (0..L.exps) |i| {
            ts[n] = .{
                .factors = try b.pairOf(gate, @as(FRange, try b.one(L.aExp() + @as(u16, @intCast(i))))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
            n += 1;
        }
        for (0..L.exps) |i| {
            ts[n] = .{
                .factors = try b.pairOf(gate, @as(FRange, try b.one(L.bExp() + @as(u16, @intCast(i))))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
            n += 1;
        }
        ts[n] = .{ .factors = try b.pairOf(gate, @as(FRange, try b.one(L.col_norm))), .coefficient = g(kNegModPow2(0)) };
        n += 1;
        ts[n] = .{ .factors = try b.pairOf(gate, @as(FRange, try b.one(L.col_carry))), .coefficient = g(kNegModPow2(0)) };
        n += 1;
        // -(keep = keptLow + norm) and +(bias + mant_bits) (the exponent
        // unwind). The first was missing for one iteration, which showed up
        // as every row evaluating to exactly 10.
        const unwind: u16 = @as(u16, f.bias) + @as(u16, f.mant_bits) - @as(u16, f.keptLow());
        ts[n] = .{ .factors = try b.pairOf(gate, @as(FRange, try b.constant(g(unwind)))) };
        n += 1;
        try b.lin(nm_exp, .composed, ts[0..n]);
    }

    // The clamp. A big overflow lands the arithmetic exponent ABOVE emax —
    // max · max in binary16 gives 46, which does not fit in five bits — and
    // the output's exponent field is emax, because that field is what
    // infinity IS. "Clamp to emax" is an inequality, which this IR cannot
    // state, so it becomes a product with a quantity that is fully DETERMINED
    // and therefore leaves the prover no freedom at all:
    //
    //   gap0 = emax − d0
    //   ec   = d0 + overflow·gap0
    //
    // With overflow = 0 the output exponent is the arithmetic one; with
    // overflow = 1 the product is emax − d0 and the sum is emax. There is no
    // witness to choose badly, which is why this is the clamp to build
    // rather than a saturating one with a free overshoot column.
    try b.lin("arithmetic gap to emax", .composed, &.{
        .{ .factors = try b.one(L.col_d0_gap) },
        .{ .factors = try b.one(L.col_d0) },
        .{ .factors = try b.constant(g(f.emax())), .coefficient = kNegOne },
    });
    try b.lin("output exponent is the arithmetic one, clamped to emax", .composed, &.{
        .{ .factors = try b.one(L.cExpVal()) },
        .{ .factors = try b.one(L.col_d0), .coefficient = kNegOne },
        .{ .factors = try b.pair(L.col_overflow, L.col_d0_gap), .coefficient = kNegOne },
    });

    // The mantissa, and the overflow switch.
    {
        var bits: [1 + f.mant_bits]LinTerm = undefined;
        bits[0] = .{ .factors = try b.one(L.col_c_mant_val) };
        for (0..L.mant) |i| {
            const d: u16 = @intCast(i);
            bits[1 + d] = .{
                .factors = try b.one(L.cMant() + d),
                .coefficient = g(kNegModPow2(d)),
            };
        }
        try b.lin("output mantissa value is its bits", .composed, &bits);

        // diff = kept + inc − implicit·(1 + carry): the correctly rounded
        // mantissa of a NORMAL result, which an overflow does not have.
        var ts: [4 + f.sigBits()]LinTerm = undefined;
        var n: usize = 0;
        ts[n] = .{ .factors = try b.one(L.col_diff) };
        n += 1;
        for (0..f.sigBits()) |i| {
            ts[n] = .{
                .factors = try b.one(L.col_kept + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
            n += 1;
        }
        ts[n] = .{ .factors = try b.one(L.col_inc), .coefficient = kNegOne };
        n += 1;
        ts[n] = .{ .factors = try b.constant(g(f.mantImplicit())) };
        n += 1;
        // +mantImplicit·carry: a carry means the kept field reached
        // 2·mantImplicit, so the mantissa is exactly 1.0 and the equation
        // has to say so.
        ts[n] = .{ .factors = try b.one(L.col_carry), .coefficient = g(f.mantImplicit()) };
        n += 1;
        try b.lin("rounded mantissa before the overflow switch", .composed, ts[0..n]);

        try b.lin("not-overflow is 1 - overflow", .composed, &.{
            .{ .factors = try b.one(L.col_not_overflow) },
            .{ .factors = try b.one(L.col_overflow) },
            .{ .factors = try b.constant(kOne), .coefficient = kNegOne },
        });

        // The switch itself. Multiplying BOTH sides by (1 − overflow) is
        // degree 2, because `diff` and `c_mant_val` are columns and the
        // factor is a column too: gating the original equation instead
        // would have been degree 3 (implicit·carry·(1−overflow)).
        // "The arithmetic path is live" is normal AND not overflowing. Gating
        // the mantissa equation on not_overflow alone was not enough: a NaN
        // answer does not overflow, but its mantissa is the NaN pattern, which
        // the arithmetic path never produces.
        try b.lin("the arithmetic path is live", .composed, &.{
            .{ .factors = try b.one(L.col_path_on) },
            .{ .factors = try b.pair(L.col_s_normal, L.col_not_overflow), .coefficient = kNegOne },
        });
        try b.lin("mantissa equation, gated on the arithmetic path", .composed, &.{
            .{ .factors = try b.pair(L.col_diff, L.col_path_on), .coefficient = kNegOne },
            .{ .factors = try b.pair(L.col_c_mant_val, L.col_path_on) },
        });

        // And the output mantissa is ZERO when it overflows, which is what
        // makes the pattern infinity rather than a NaN. Without this the
        // prover could answer a NaN for a product whose truth is infinity,
        // and the gate above would not notice: it is switched off.
        try b.lin("an overflow has a zero mantissa", .composed, &.{
            .{ .factors = try b.pair(L.col_c_mant_val, L.col_overflow) },
        });
    }

    // The overflow flag: emax − ec == 0 exactly when the rounded exponent
    // The biconditional "overflow iff the exponent field is all ones" WAS
    // here, as a zeroOrNonZero over the gap emax − ec. Its flag meant "the
    // gap is zero", so it asserted exactly what NaN breaks: a NaN answer
    // has the all-ones exponent with a nonzero mantissa and overflow at
    // zero. Both directions are still forced without it. "overflow = 1
    // gives ec = emax" is the clamp below, reading gap = emax − d0. The
    // other way — overflow = 0 with ec = emax — is the d0 equation, whose
    // arithmetic exponent is above emax for a real overflow. The two
    // together leave no satisfying assignment, which was the point, for
    // three constraints cheaper.

    // The classes. Every one is a product of two booleans, so each is
    // boolean for free:
    //   zero   = expZero AND mantZero        inf = expMax AND mantZero
    //   nan    = expMax AND NOT mantZero    normal = NOT expZero AND NOT expMax
    // A subnormal is expZero AND NOT mantZero, which is none of them, and
    // the exhaustiveness constraint below is what rejects it.
    for (0..2) |which| {
        const exp_zero = if (which == 0) L.col_a_exp_zero else L.col_b_exp_zero;
        const mant_zero = if (which == 0) L.col_a_mant_zero else L.col_b_mant_zero;
        const exp_max = if (which == 0) L.col_a_exp_max else L.col_b_exp_max;
        const is_zero = if (which == 0) L.col_a_is_zero else L.col_b_is_zero;
        const is_inf = if (which == 0) L.col_a_is_inf else L.col_b_is_inf;
        const is_nan = if (which == 0) L.col_a_is_nan else L.col_b_is_nan;
        const is_normal = if (which == 0) L.col_a_is_normal else L.col_b_is_normal;

        try b.lin("input is zero", .composed, &.{
            .{ .factors = try b.one(is_zero) },
            .{ .factors = try b.pair(exp_zero, mant_zero), .coefficient = kNegOne },
        });
        try b.lin("input is infinity", .composed, &.{
            .{ .factors = try b.one(is_inf) },
            .{ .factors = try b.pair(exp_max, mant_zero), .coefficient = kNegOne },
        });
        // is_nan = exp_max − exp_max·mant_zero. The exponent is all ones
        // AND the mantissa is not zero; written as a sum so it stays
        // degree 2.
        try b.lin("input is NaN", .composed, &.{
            .{ .factors = try b.one(is_nan) },
            .{ .factors = try b.pair(exp_max, mant_zero), .coefficient = kOne },
            .{ .factors = try b.one(exp_max), .coefficient = kNegOne },
        });
        try b.lin("input is normal", .composed, &.{
            .{ .factors = try b.one(is_normal) },
            .{ .factors = try b.one(exp_zero) },
            .{ .factors = try b.one(exp_max) },
            .{ .factors = try b.constant(kOne), .coefficient = kNegOne },
            .{ .factors = try b.pair(exp_zero, exp_max), .coefficient = kNegOne },
        });
    }

    // The arithmetic path reads sanitised values, so it stays satisfiable
    // whatever the inputs are: a special operand becomes 1.0, and the
    // selector below discards whatever the path then computes. Without this
    // a zero operand would drive the exponent to a negative value that the
    // output's own exponent bits cannot hold, and the system would be
    // unsatisfiable for a case the answer exists.
    for (0..2) |which| {
        const is_normal = if (which == 0) L.col_a_is_normal else L.col_b_is_normal;
        const sig = if (which == 0) L.col_a_sig else L.col_b_sig;
        const sig_eff = if (which == 0) L.col_a_sig_eff else L.col_b_sig_eff;
        const exp_val = if (which == 0) L.aExpVal() else L.bExpVal();
        const exp_eff = if (which == 0) L.col_a_exp_eff else L.col_b_exp_eff;
        const bias: u16 = @intCast(f.bias);
        // eff = normal ? value : fallback, written as
        // eff − normal·value − fallback + normal·fallback = 0. The last
        // term is what makes it a mux instead of a blend: without it a
        // special operand would shift the result by the fallback.
        // The fallback rides on the same boolean as a second FACTOR, so the
        // product stays degree 2 instead of becoming a constant times a
        // product of three.
        const normal_col: FRange = @as(FRange, try b.one(is_normal));
        const sig_fb: FRange = @as(FRange, try b.constant(g(f.mantImplicit())));
        const exp_fb: FRange = @as(FRange, try b.constant(g(bias)));
        try b.lin("sanitised significand", .composed, &.{
            .{ .factors = try b.one(sig_eff) },
            .{ .factors = try b.pair(is_normal, sig), .coefficient = kNegOne },
            .{ .factors = sig_fb, .coefficient = kNegOne },
            .{ .factors = try b.pairOf(normal_col, sig_fb) },
        });
        try b.lin("sanitised exponent", .composed, &.{
            .{ .factors = try b.one(exp_eff) },
            .{ .factors = try b.pair(is_normal, exp_val), .coefficient = kNegOne },
            .{ .factors = exp_fb, .coefficient = kNegOne },
            .{ .factors = try b.pairOf(normal_col, exp_fb) },
        });
    }

    // Which answer. IEEE-754: NaN if either operand is NaN; NaN if one is
    // infinite and the other zero; infinity if one is infinite and the other
    // is finite and non-zero; zero if either is zero. `nan_a` and `nan_b`
    // are NOT mutually exclusive, so their combination is an OR, while the
    // two "inf times zero" cases cannot both hold (one operand cannot be
    // infinite and zero at once), so their sum is already the OR.
    try b.lin("either operand is NaN", .composed, &.{
        .{ .factors = try b.one(L.col_nan_any) },
        .{ .factors = try b.one(L.col_a_is_nan), .coefficient = kNegOne },
        .{ .factors = try b.one(L.col_b_is_nan), .coefficient = kNegOne },
        .{ .factors = try b.pair(L.col_a_is_nan, L.col_b_is_nan) },
    });
    try b.lin("infinity times zero is a bad pair", .composed, &.{
        .{ .factors = try b.one(L.col_bad_pair) },
        .{ .factors = try b.pair(L.col_a_is_inf, L.col_b_is_zero), .coefficient = kNegOne },
        .{ .factors = try b.pair(L.col_b_is_inf, L.col_a_is_zero), .coefficient = kNegOne },
    });
    try b.lin("the answer is NaN", .composed, &.{
        .{ .factors = try b.one(L.col_s_nan) },
        .{ .factors = try b.one(L.col_nan_any), .coefficient = kNegOne },
        .{ .factors = try b.one(L.col_bad_pair), .coefficient = kNegOne },
    });
    // infinity: some operand is infinite, the other is neither a NaN nor a
    // zero, and BOTH may be infinite (inf·inf = inf, so the OR needs its
    // a·b correction — the one case where two specials meet and the answer
    // is still infinity). Six degree-2 terms, no triple products.
    try b.lin("the answer is infinity", .composed, &.{
        .{ .factors = try b.one(L.col_s_inf) },
        .{ .factors = try b.pair(L.col_a_is_inf, L.col_b_is_inf) },
        .{ .factors = try b.one(L.col_a_is_inf), .coefficient = kNegOne },
        .{ .factors = try b.one(L.col_b_is_inf), .coefficient = kNegOne },
        .{ .factors = try b.pair(L.col_a_is_inf, L.col_b_is_nan) },
        .{ .factors = try b.pair(L.col_a_is_inf, L.col_b_is_zero) },
        .{ .factors = try b.pair(L.col_b_is_inf, L.col_a_is_nan) },
        .{ .factors = try b.pair(L.col_b_is_inf, L.col_a_is_zero) },
    });
    // zero: some operand is zero (both may be, hence the a·b correction),
    // minus the two bad pairs, which are exactly zero-with-infinity. A NaN
    // operand is never zero, so no NaN term is needed.
    try b.lin("the answer is zero", .composed, &.{
        .{ .factors = try b.one(L.col_s_zero) },
        .{ .factors = try b.pair(L.col_a_is_zero, L.col_b_is_zero) },
        .{ .factors = try b.one(L.col_a_is_zero), .coefficient = kNegOne },
        .{ .factors = try b.one(L.col_b_is_zero), .coefficient = kNegOne },
        .{ .factors = try b.pair(L.col_bad_pair, L.col_a_is_zero) },
        .{ .factors = try b.pair(L.col_bad_pair, L.col_b_is_zero) },
    });
    try b.lin("the answer is the normal path", .composed, &.{
        .{ .factors = try b.one(L.col_s_normal) },
        .{ .factors = try b.pair(L.col_a_is_normal, L.col_b_is_normal), .coefficient = kNegOne },
    });

    // EXHAUSTIVENESS, and this is the constraint that makes the whole thing
    // sound: a subnormal input is not zero, not infinite, not NaN and not
    // normal, so the four selectors are all zero and the sum below cannot
    // be one. Without it such an input would fall through to the normal
    // branch and the prover could claim 1.0 · 1.0 = 1.0 for it.
    try b.lin("exactly one answer class", .composed, &.{
        .{ .factors = try b.one(L.col_s_nan) },
        .{ .factors = try b.one(L.col_s_inf) },
        .{ .factors = try b.one(L.col_s_zero) },
        .{ .factors = try b.one(L.col_s_normal) },
        .{ .factors = try b.constant(kOne), .coefficient = kNegOne },
    });

    // The selected answer, one constraint per output bit. The three
    // special patterns are constants except for the sign, and the sign of a
    // NaN is 0 by the reference's canonical choice, so the NaN pattern adds
    // nothing to the sign bit. Each bit is a selection among booleans, so
    // the answer is boolean without a booleanity constraint.
    const nan_mantissa: u16 = @intCast((@as(u32, 1) << @intCast(f.mant_bits - 1)));
    for (0..L.width) |i| {
        const d: u16 = @intCast(i);
        const is_sign = d == L.width - 1;
        const is_exp = d >= f.mant_bits and d < f.mant_bits + f.exp_bits;
        const is_nan_mant = !is_sign and !is_exp and
            (d == f.mant_bits - 1) and nan_mantissa != 0;
        var terms: [5]LinTerm = undefined;
        var n: usize = 0;
        terms[n] = .{ .factors = try b.pair(L.col_s_normal, L.cBit(d)), .coefficient = kNegOne };
        n += 1;
        terms[n] = .{ .factors = try b.one(L.col_out + d) };
        n += 1;
        if (is_sign) {
            terms[n] = .{ .factors = try b.pair(L.col_s_zero, L.cSign()), .coefficient = kNegOne };
            n += 1;
            terms[n] = .{ .factors = try b.pair(L.col_s_inf, L.cSign()), .coefficient = kNegOne };
            n += 1;
        } else if (is_exp) {
            terms[n] = .{ .factors = try b.one(L.col_s_inf), .coefficient = kNegOne };
            n += 1;
            terms[n] = .{ .factors = try b.one(L.col_s_nan), .coefficient = kNegOne };
            n += 1;
        } else if (is_nan_mant) {
            terms[n] = .{ .factors = try b.one(L.col_s_nan), .coefficient = kNegOne };
            n += 1;
        }
        try b.lin("the answer's bit is selected", .composed, terms[0..n]);
    }

    // The output sign is the XOR of the input signs: a + b − 2ab.
    try b.lin("c_sign = a_sign XOR b_sign", .composed, &.{
        .{ .factors = try b.one(L.cSign()) },
        .{ .factors = try b.one(L.aSign()), .coefficient = kNegOne },
        .{ .factors = try b.one(L.bSign()), .coefficient = kNegOne },
        .{ .factors = try b.pair(L.aSign(), L.bSign()), .coefficient = g(2) },
    });

    // Freeze the three buffers, then RESOLVE the recorded indices into
    // slices. Only now do slices exist, and they point into buffers that
    // will not move again.
    return bld.freeze(allocator, &b, rows);
}

/// `out` = OR of the `count` boolean bits at `base`, via the
/// "the sum is non-zero" inverse trick. 3 constraints: the sum, the
/// inverse identity, and booleanity of the result.
fn stickyOr(b: *Builder, sum_col: u16, inv_col: u16, out_col: u16, base: u16, comptime count: u16) !void {
    var ts: [1 + count]LinTerm = undefined;
    ts[0] = .{ .factors = try b.one(sum_col) };
    for (0..count) |i| {
        ts[1 + i] = .{
            .factors = try b.one(base + @as(u16, @intCast(i))),
            .coefficient = kNegOne,
        };
    }
    try b.lin("sticky sum", .composed, ts[0 .. 1 + @as(usize, count)]);
    try b.lin("sum · inv = sticky", .composed, &.{
        .{ .factors = try b.pair(sum_col, inv_col) },
        .{ .factors = try b.one(out_col), .coefficient = kNegOne },
    });
    try b.lin("sticky is boolean", .composed, &.{
        .{ .factors = try b.pair(out_col, out_col) },
        .{ .factors = try b.one(out_col), .coefficient = kNegOne },
    });
}

// ---------------------------------------------------------------------------
// Trace construction
// ---------------------------------------------------------------------------

pub const BuildTraceError = error{
    OutOfMemory,
    /// The reference refused this pair (subnormal input or result, or an
    /// inf/NaN result). S1 does not cover those and says so instead of
    /// proving something plausible but wrong.
    UnsupportedCase,
};

fn set(cols: [][]Fp2, col: u16, r: usize, v: u64) void {
    cols[col][r] = Fp2.re(Goldilocks.fromU64(v));
}

fn a_sig_or_b_sig(p: Format.Parts, f: Format) u64 {
    return @as(u64, f.mantImplicit()) + @as(u64, p.mantissa);
}

/// How many of the low `n` bits are set. The AIR recomputes this as a
/// column and proves it is non-zero exactly when the OR must be 1.
fn popCount(v: u32, n: u8) u32 {
    const masked = v & ((@as(u32, 1) << @intCast(n)) - 1);
    var count: u32 = 0;
    var i: u32 = masked;
    while (i != 0) : (i &= i - 1) count += 1;
    return count;
}

/// One row per (a, b) pair of NORMAL binary16 values. The witness comes
/// from the reference's own decomposition, so the AIR and the reference
/// cannot silently disagree about the answer — a disagreement surfaces as
/// a failed proof, not a passing one.
/// The exponent the NORMAL path computes for a product of two normal
/// operands, before any rounding: the same arithmetic the AIR's `d0`
/// column carries, exposed so the scope guard and the test sweep can agree
/// on one definition instead of two copies of the formula.
///
/// It is signed and it is NOT the answer: a value of 0 or less means the
/// exact product lands in the subnormal range (or below it), which the
/// normal path cannot represent — and which still needs refusing when the
/// ROUNDED answer is the min normal, because the rounding that gets it
/// there is the subnormal path that is not built yet. 0x83FF's exact
/// product with 0x0400 is the case that proved the point: the answer is
/// 0x8400, a perfectly normal number, and the AIR still could not prove
/// it.
pub fn arithmeticExponent(comptime f: Format, a: u16, b: u16) i64 {
    const pa = f.parts(a);
    const pb = f.parts(b);
    const product: u64 = @as(u64, f.mantImplicit() + pa.mantissa) *
        @as(u64, f.mantImplicit() + pb.mantissa);
    const norm: i64 = if ((product >> @intCast(f.normBit())) & 1 == 1) 1 else 0;
    const keep: u8 = if ((product >> @intCast(f.normBit())) & 1 == 1) f.keptHigh() else f.keptLow();
    const carry: i64 = if ((product >> @intCast(@as(u8, keep) + f.sigBits())) & 1 == 1) 1 else 0;
    return @as(i64, pa.exponent) + @as(i64, pb.exponent) + norm + carry +
        @as(i64, f.keptLow()) - (@as(i64, f.bias) + @as(i64, f.mant_bits));
}

pub fn buildTrace(
    allocator: std.mem.Allocator,
    pairs: []const [2]u16,
    comptime f: Format,
) BuildTraceError!Trace {
    const L = Layout(f);
    const rows = pairs.len;
    std.debug.assert(rows > 0);

    var trace = try Trace.alloc(allocator, L.column_count, rows);
    errdefer trace.deinit(allocator);
    const cols = trace.columns;

    for (pairs, 0..) |pair, r| {
        const a = pair[0];
        const b = pair[1];
        const pa = f.parts(a);
        const pb = f.parts(b);
        // Subnormal inputs are out of scope, and the AIR proves it rather
        // than accepting them: a subnormal matches no class, so the
        // exhaustiveness constraint cannot be satisfied. Zero, infinity and
        // NaN are IN scope from here on.
        if (pa.exponent == 0 and pa.mantissa != 0) return BuildTraceError.UnsupportedCase;
        if (pb.exponent == 0 and pb.mantissa != 0) return BuildTraceError.UnsupportedCase;
        const expected = float_ref.multiply(f, a, b) catch return BuildTraceError.UnsupportedCase;
        const pc = f.parts(expected);
        // An infinite result is IN SCOPE now (the overflow flag is what
        // makes it one) and so is a ZERO one, which is what a zero operand
        // produces. What is still out of scope is any result with an
        // exponent field of zero that the normal path cannot have produced:
        // a subnormal, and the UNDERFLOW of two normal operands down to
        // zero. Both are refused rather than rounded, because the range
        // reduction that would prove them is not built yet — and note the
        // refusal is forced by the AIR too, not just by this guard: with
        // s_normal = 1 the exponent equation demands the arithmetic
        // exponent, which no representable result with a zero exponent
        // field has.
        const inputs_normal = pa.exponent != 0 and pa.exponent != f.emax() and
            pb.exponent != 0 and pb.exponent != f.emax();
        if (inputs_normal and arithmeticExponent(f, a, b) <= 0) {
            return BuildTraceError.UnsupportedCase;
        }
        if (pc.exponent == 0 and pc.mantissa != 0) return BuildTraceError.UnsupportedCase;
        // Overflow means INFINITY, not merely an all-ones exponent: a NaN
        // answer has one too, and the "overflow has a zero mantissa"
        // constraint says exactly the difference.
        const overflow: u32 = if (pc.exponent == f.emax() and pc.mantissa == 0) 1 else 0;

        trace.writeBits(L.col_a_bits, a, @intCast(f.byteWidth()), r);
        trace.writeBits(L.col_b_bits, b, @intCast(f.byteWidth()), r);
        trace.writeBits(L.col_c_bits, expected, @intCast(f.byteWidth()), r);

        const a_sig: u32 = @as(u32, f.mantImplicit()) + @as(u32, pa.mantissa);
        const b_sig: u32 = @as(u32, f.mantImplicit()) + @as(u32, pb.mantissa);
        cols[L.col_a_sig][r] = Fp2.re(Goldilocks.fromU64(a_sig));
        cols[L.col_b_sig][r] = Fp2.re(Goldilocks.fromU64(b_sig));

        // Field inverses witness "this exponent field is neither 0 nor 31".
        const a_exp: u64 = pa.exponent;
        const b_exp: u64 = pb.exponent;
        cols[L.aExpVal()][r] = Fp2.re(Goldilocks.fromU64(a_exp));
        cols[L.bExpVal()][r] = Fp2.re(Goldilocks.fromU64(b_exp));

        // The arithmetic runs on the SANITISED significands, so a special
        // operand multiplies the implicit one and the whole path is inert
        // until the selector turns it back off.
        const a_normal: bool = pa.exponent != 0 and pa.exponent != f.emax();
        const b_normal: bool = pb.exponent != 0 and pb.exponent != f.emax();
        const both_normal_row: u64 = if (a_normal and b_normal) 1 else 0;
        const a_exp_eff: u64 = if (a_normal) pa.exponent else @as(u16, f.bias);
        const b_exp_eff: u64 = if (b_normal) pb.exponent else @as(u16, f.bias);
        const a_sig_eff: u32 = if (a_normal) a_sig else f.mantImplicit();
        const b_sig_eff: u32 = if (b_normal) b_sig else f.mantImplicit();
        const product: u32 = a_sig_eff * b_sig_eff;
        // The classification, one operand at a time.
        for ([_]Format.Parts{ pa, pb }, 0..) |p, which| {
            const mant_val = if (which == 0) L.col_a_mant_val else L.col_b_mant_val;
            const exp_zero = if (which == 0) L.col_a_exp_zero else L.col_b_exp_zero;
            const mant_zero = if (which == 0) L.col_a_mant_zero else L.col_b_mant_zero;
            const exp_max = if (which == 0) L.col_a_exp_max else L.col_b_exp_max;
            const zero_inv = if (which == 0) L.col_a_zero_inv else L.col_b_zero_inv;
            const mant_inv = if (which == 0) L.col_a_mant_inv else L.col_b_mant_inv;
            const max_inv = if (which == 0) L.col_a_max_inv else L.col_b_max_inv;
            const gap = if (which == 0) L.col_a_gap else L.col_b_gap;
            const is_zero = if (which == 0) L.col_a_is_zero else L.col_b_is_zero;
            const is_inf = if (which == 0) L.col_a_is_inf else L.col_b_is_inf;
            const is_nan = if (which == 0) L.col_a_is_nan else L.col_b_is_nan;
            const is_normal = if (which == 0) L.col_a_is_normal else L.col_b_is_normal;
            const sig_eff = if (which == 0) L.col_a_sig_eff else L.col_b_sig_eff;
            const exp_eff = if (which == 0) L.col_a_exp_eff else L.col_b_exp_eff;

            const is_zero_v: u64 = if (p.exponent == 0 and p.mantissa == 0) 1 else 0;
            const is_inf_v: u64 = if (p.exponent == f.emax() and p.mantissa == 0) 1 else 0;
            const is_nan_v: u64 = if (p.exponent == f.emax() and p.mantissa != 0) 1 else 0;
            const is_normal_v: u64 = if (p.exponent != 0 and p.exponent != f.emax()) 1 else 0;
            const zero_v: u64 = if (p.exponent == 0) 1 else 0;
            const mant_zero_v: u64 = if (p.mantissa == 0) 1 else 0;
            const max_v: u64 = if (p.exponent == f.emax()) 1 else 0;
            const inv_or_zero = func: {
                const v = if (p.exponent == 0) 0 else p.exponent;
                if (v == 0) break :func Goldilocks.zero;
                break :func Goldilocks.fromU64(v).inv() catch unreachable;
            };
            const m_inv = if (p.mantissa == 0) Goldilocks.zero else Goldilocks.fromU64(p.mantissa).inv() catch unreachable;
            const g_inv = if (f.emax() == p.exponent) Goldilocks.zero else Goldilocks.fromU64(f.emax() - p.exponent).inv() catch unreachable;

            set(cols, mant_val, r, @intCast(p.mantissa));
            set(cols, exp_zero, r, zero_v);
            set(cols, mant_zero, r, mant_zero_v);
            set(cols, exp_max, r, max_v);
            set(cols, zero_inv, r, inv_or_zero.toU64());
            set(cols, mant_inv, r, m_inv.toU64());
            set(cols, max_inv, r, g_inv.toU64());
            set(cols, gap, r, @intCast(f.emax() - p.exponent));
            set(cols, is_zero, r, is_zero_v);
            set(cols, is_inf, r, is_inf_v);
            set(cols, is_nan, r, is_nan_v);
            set(cols, is_normal, r, is_normal_v);
            set(cols, sig_eff, r, if (is_normal_v == 1) a_sig_or_b_sig(p, f) else f.mantImplicit());
            set(cols, exp_eff, r, if (is_normal_v == 1) p.exponent else @as(u64, @intCast(f.bias)));
        }

        cols[L.col_product][r] = Fp2.re(Goldilocks.fromU64(product));
        trace.writeBits(L.col_p_bits, product, f.productBits(), r);

        const norm: u32 = if (product >= (@as(u32, 1) << @intCast(f.normBit()))) 1 else 0;
        cols[L.col_norm][r] = Fp2.re(Goldilocks.fromU64(norm));

        const keep: u8 = if (norm == 1) f.keptHigh() else f.keptLow();
        const round: u32 = @intCast((product >> @intCast(keep - 1)) & 1);
        const sum_hi: u32 = popCount(product, f.keptLow());
        const sum_lo: u32 = popCount(product, f.keptLow() - 1);
        cols[L.col_sticky_hi_sum][r] = Fp2.re(Goldilocks.fromU64(sum_hi));
        cols[L.col_sticky_lo_sum][r] = Fp2.re(Goldilocks.fromU64(sum_lo));
        // A zero sum gets a zero inverse: the identity sum·inv = sticky
        // then forces sticky = 0, which is the correct OR.
        const inv_hi = if (sum_hi == 0) Goldilocks.zero else Goldilocks.fromU64(sum_hi).inv() catch unreachable;
        const inv_lo = if (sum_lo == 0) Goldilocks.zero else Goldilocks.fromU64(sum_lo).inv() catch unreachable;
        cols[L.col_sticky_hi_inv][r] = Fp2.re(inv_hi);
        cols[L.col_sticky_lo_inv][r] = Fp2.re(inv_lo);
        const sticky_hi: u32 = if (sum_hi == 0) 0 else 1;
        const sticky_lo: u32 = if (sum_lo == 0) 0 else 1;
        const sticky: u32 = if (norm == 1) sticky_hi else sticky_lo;
        cols[L.col_round][r] = Fp2.re(Goldilocks.fromU64(round));
        cols[L.col_sticky_hi][r] = Fp2.re(Goldilocks.fromU64(sticky_hi));
        cols[L.col_sticky_lo][r] = Fp2.re(Goldilocks.fromU64(sticky_lo));
        cols[L.col_sticky][r] = Fp2.re(Goldilocks.fromU64(sticky));

        var kept_sum: u32 = 0;
        for (0..f.sigBits()) |i| {
            const bit = (product >> @intCast(i + keep)) & 1;
            cols[L.col_kept + @as(u16, @intCast(i))][r] = Fp2.re(Goldilocks.fromU64(bit));
            kept_sum |= bit << @intCast(i);
        }
        const lsb: u32 = (product >> @intCast(keep)) & 1;
        const or_sl_sum: u32 = sticky + lsb;
        const or_sl: u32 = if (or_sl_sum == 0) 0 else 1;
        cols[L.col_or_sl_sum][r] = Fp2.re(Goldilocks.fromU64(or_sl_sum));
        cols[L.col_or_sl_inv][r] = Fp2.re(
            if (or_sl_sum == 0) Goldilocks.zero else Goldilocks.fromU64(or_sl_sum).inv() catch unreachable,
        );
        const inc: u32 = if (round == 1 and or_sl == 1) 1 else 0;
        const carry: u32 = if (kept_sum + inc == (@as(u32, f.mantImplicit()) << 1)) 1 else 0;
        cols[L.col_or_sl][r] = Fp2.re(Goldilocks.fromU64(or_sl));
        cols[L.col_inc][r] = Fp2.re(Goldilocks.fromU64(inc));
        cols[L.col_carry][r] = Fp2.re(Goldilocks.fromU64(carry));

        // Overflow witness. `diff` is the correctly rounded mantissa a
        // NORMAL result would have, which an overflow does not use; the
        // output's own values come from the reference's answer, so the
        // exponent is the all-ones field exactly when the flag is set.
        const implicit: u64 = f.mantImplicit();
        const diff_signed: i64 = @as(i64, @intCast(kept_sum)) + @as(i64, @intCast(inc)) -
            @as(i64, @intCast(implicit)) - @as(i64, @intCast(implicit * carry));
        const diff: u64 = @intCast(@mod(diff_signed, @as(i64, @intCast(Goldilocks.p))));
        cols[L.col_diff][r] = Fp2.re(Goldilocks.fromU64(diff));
        cols[L.col_not_overflow][r] = Fp2.re(Goldilocks.fromU64(1 - overflow));
        set(cols, L.col_path_on, r, both_normal_row * (1 - overflow));
        // For a special row the gated equation above says nothing about
        // d0, so the witness points it at the ANSWER's exponent: that is
        // what makes the clamp below hold with the overflow flag at zero.
        const arithmetic_d0: u64 = @intCast(@mod(@as(i64, @intCast(a_exp_eff)) +
            @as(i64, @intCast(b_exp_eff)) +
            @as(i64, @intCast(f.keptLow())) +
            @as(i64, @intCast(norm)) + @as(i64, @intCast(carry)) -
            @as(i64, @intCast(@as(u16, f.bias) + @as(u16, f.mant_bits))), @as(i64, @intCast(Goldilocks.p))));
        const d0: u64 = both_normal_row * arithmetic_d0 + (1 - both_normal_row) * pc.exponent;
        cols[L.col_d0][r] = Fp2.re(Goldilocks.fromU64(d0));
        const d0_gap: u64 = @intCast(@mod(@as(i64, @intCast(f.emax())) - @as(i64, @intCast(d0)), @as(i64, @intCast(Goldilocks.p))));
        cols[L.col_d0_gap][r] = Fp2.re(Goldilocks.fromU64(d0_gap));
        const c_exp_val: u64 = pc.exponent;
        cols[L.col_c_exp_val][r] = Fp2.re(Goldilocks.fromU64(c_exp_val));
        cols[L.col_c_mant_val][r] = Fp2.re(Goldilocks.fromU64(pc.mantissa));
        cols[L.col_overflow][r] = Fp2.re(Goldilocks.fromU64(overflow));

        // Which answer, and the selected bits. The reference already knows
        // it, so the class is read off its decomposition rather than
        // recomputed: that keeps the AIR and the reference from silently
        // disagreeing about the class, which would show up as a failed
        // proof rather than a passing one.
        const da = float_ref.decompose(f, a);
        const db = float_ref.decompose(f, b);
        const a_zero: u64 = if (da.special == .zero) 1 else 0;
        const b_zero: u64 = if (db.special == .zero) 1 else 0;
        const a_inf: u64 = if (da.special == .inf) 1 else 0;
        const b_inf: u64 = if (db.special == .inf) 1 else 0;
        const nan_any: u64 = if (da.special == .nan or db.special == .nan) 1 else 0;
        const bad_pair: u64 = if ((a_inf == 1 and b_zero == 1) or (b_inf == 1 and a_zero == 1)) 1 else 0;
        const s_nan: u64 = if (nan_any == 1 or bad_pair == 1) 1 else 0;
        const s_inf: u64 = if (s_nan == 0 and (a_inf == 1 or b_inf == 1)) 1 else 0;
        const s_zero: u64 = if (s_nan == 0 and bad_pair == 0 and (a_zero == 1 or b_zero == 1)) 1 else 0;
        const both_normal: u64 = if (da.special == .finite and db.special == .finite) 1 else 0;
        set(cols, L.col_nan_any, r, nan_any);
        set(cols, L.col_bad_pair, r, bad_pair);
        set(cols, L.col_s_nan, r, s_nan);
        set(cols, L.col_s_inf, r, s_inf);
        set(cols, L.col_s_zero, r, s_zero);
        set(cols, L.col_s_normal, r, both_normal);

        // The answer's bits. The arithmetic path's bits are the normal
        // case's answer; the selector picks between them and the three
        // special patterns, whose sign is the XOR of the input signs and
        // whose NaN sign is zero.
        const sign_xor: u64 = @as(u64, pa.sign) ^ @as(u64, pb.sign);
        const nan_top: u16 = @intCast((@as(u32, 1) << @intCast(f.mant_bits - 1)));
        for (0..L.width) |i| {
            const d: u16 = @intCast(i);
            const is_sign = d == L.width - 1;
            const is_exp = d >= f.mant_bits and d < f.mant_bits + f.exp_bits;
            const is_nan_mant = !is_sign and !is_exp and d == f.mant_bits - 1;
            const normal_bit: u64 = if ((expected >> @intCast(d)) & 1 == 1) 1 else 0;
            const zero_bit: u64 = if (is_sign) sign_xor else 0;
            const inf_bit: u64 = if (is_sign) sign_xor else if (is_exp) 1 else 0;
            const nan_bit: u64 = if (is_exp) 1 else if (is_nan_mant and nan_top != 0) 1 else 0;
            set(cols, L.col_out + d, r, both_normal * normal_bit + s_zero * zero_bit + s_inf * inf_bit + s_nan * nan_bit);
        }
    }
    return trace;
}
