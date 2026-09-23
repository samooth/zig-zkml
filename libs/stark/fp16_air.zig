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
//! Normal x normal only, by the same explicit refusal the reference makes
//! (fp16_ref.zig). The spike does NOT implement subnormal results,
//! overflow to infinity, or NaN propagation: those are S2, and the trace
//! builder refuses to produce them rather than proving a wrong answer.

const std = @import("std");
const expr = @import("./expr.zig");
const range = @import("./range.zig");
const tensor = @import("../tensor/root.zig");
const fp16 = @import("./fp16_ref.zig");

pub const Fp2 = expr.Fp2;
pub const Goldilocks = tensor.Goldilocks;
pub const System = expr.System;
pub const Constraint = expr.Constraint;
pub const Term = expr.Term;
pub const Factor = expr.Factor;
pub const Scope = expr.Scope;

/// binary16 constants, from the reference so there is one source of truth.
pub const mant_bits: u8 = fp16.mant_bits;
pub const exp_bits: u8 = fp16.exp_bits;
pub const mant_implicit: u16 = fp16.mant_implicit;
pub const exp_bias: i32 = fp16.exp_bias;
pub const exp_max: u16 = fp16.exp_max;

/// A normal significand is 11 bits, so the exact product is 22 bits wide
/// and needs 22 booleanity constraints plus its reconstruction.
const product_bits: u8 = 22;
/// The kept field is the 11-bit output significand.
const kept_bits: u8 = 11;
/// Output pattern layout: sign at bit 15, exponent 14..10, mantissa 9..0.
pub const exp_base: u16 = 10;

// --- trace layout -----------------------------------------------------------

pub const col_a_bits: u16 = 0; // 16 columns, LSB first
pub const col_b_bits: u16 = col_a_bits + 16;
pub const col_a_sig: u16 = col_b_bits + 16;
pub const col_b_sig: u16 = col_a_sig + 1;
pub const col_a_exp_val: u16 = col_b_sig + 1;
pub const col_b_exp_val: u16 = col_a_exp_val + 1;
pub const col_a_exp_inv: u16 = col_b_exp_val + 1;
pub const col_a_nz_inv: u16 = col_a_exp_inv + 1;
pub const col_b_exp_inv: u16 = col_a_nz_inv + 1;
pub const col_b_nz_inv: u16 = col_b_exp_inv + 1;
pub const col_product: u16 = col_b_nz_inv + 1;
pub const col_p_bits: u16 = col_product + 1; // 22 columns
pub const col_norm: u16 = col_p_bits + product_bits;
pub const col_round: u16 = col_norm + 1;
pub const col_sticky_hi_sum: u16 = col_round + 1;
pub const col_sticky_hi_inv: u16 = col_sticky_hi_sum + 1;
pub const col_sticky_lo_sum: u16 = col_sticky_hi_inv + 1;
pub const col_sticky_lo_inv: u16 = col_sticky_lo_sum + 1;
pub const col_sticky_hi: u16 = col_sticky_lo_inv + 1;
pub const col_sticky_lo: u16 = col_sticky_hi + 1;
pub const col_sticky: u16 = col_sticky_lo + 1;
pub const col_kept: u16 = col_sticky + 1; // 11 columns
pub const col_or_sl_sum: u16 = col_kept + kept_bits;
pub const col_or_sl_inv: u16 = col_or_sl_sum + 1;
pub const col_or_sl: u16 = col_or_sl_inv + 1;
pub const col_inc: u16 = col_or_sl + 1;
pub const col_carry: u16 = col_inc + 1;
pub const col_c_bits: u16 = col_carry + 1; // 16 columns
pub const column_count: usize = col_c_bits + 16;

fn aBit(i: u16) u16 {
    return col_a_bits + i;
}
fn bBit(i: u16) u16 {
    return col_b_bits + i;
}
fn pBit(i: u16) u16 {
    return col_p_bits + i;
}
fn cBit(i: u16) u16 {
    return col_c_bits + i;
}
fn aExp() u16 {
    return aBit(exp_base);
}
fn aMant() u16 {
    return aBit(0);
}
fn bExp() u16 {
    return bBit(exp_base);
}
fn bMant() u16 {
    return bBit(0);
}
fn cExp() u16 {
    return cBit(exp_base);
}
fn cMant() u16 {
    return cBit(0);
}
fn aSign() u16 {
    return aBit(15);
}
fn bSign() u16 {
    return bBit(15);
}
fn cSign() u16 {
    return cBit(15);
}

const kOne = Fp2.one;
const kNegOne = Fp2.neg(Fp2.one);

fn g(v: u64) Fp2 {
    return Fp2.re(Goldilocks.fromU64(v));
}

/// A run of factors, recorded as INDICES while building.
///
/// Recording slices here would be a use-after-free waiting to happen: the
/// backing ArrayLists reallocate as they grow, so every Constraint written
/// before a growth points into a buffer that no longer exists. The first
/// version of this file did exactly that and segfaulted the verifier
/// mid-walk. Indices survive reallocation; slices are resolved once, at the
/// end, against the final buffers.
const FRange = struct {
    first: usize,
    len: usize,
};

const TSpec = struct {
    factors: FRange,
    coefficient: Fp2,
};

const LinTerm = struct {
    factors: FRange,
    coefficient: Fp2 = kOne,
};

const CSpec = struct {
    name: []const u8,
    scope: Scope,
    first_term: usize,
    n_terms: usize,
};

const Builder = struct {
    allocator: std.mem.Allocator,
    factors: std.ArrayList(Factor) = .empty,
    terms: std.ArrayList(TSpec) = .empty,
    constraints: std.ArrayList(CSpec) = .empty,

    fn one(self: *Builder, col: u16) !FRange {
        try self.factors.append(self.allocator, .{ .column = .{ .index = col } });
        return .{ .first = self.factors.items.len - 1, .len = 1 };
    }

    fn constant(self: *Builder, v: Fp2) !FRange {
        try self.factors.append(self.allocator, .{ .constant = v });
        return .{ .first = self.factors.items.len - 1, .len = 1 };
    }

    fn pair(self: *Builder, a: u16, b: u16) !FRange {
        try self.factors.append(self.allocator, .{ .column = .{ .index = a } });
        try self.factors.append(self.allocator, .{ .column = .{ .index = b } });
        return .{ .first = self.factors.items.len - 2, .len = 2 };
    }

    /// A constraint whose terms are the `terms` just described, in order.
    fn lin(self: *Builder, name: []const u8, scope: Scope, terms: []const LinTerm) !void {
        if (terms.len == 0) return;
        const first_term = self.terms.items.len;
        for (terms) |lt| {
            try self.terms.append(self.allocator, .{
                .factors = lt.factors,
                .coefficient = lt.coefficient,
            });
        }
        try self.constraints.append(self.allocator, .{
            .name = name,
            .scope = scope,
            .first_term = first_term,
            .n_terms = terms.len,
        });
    }
};

pub const BuildError = error{ OutOfMemory, BadWidth };

/// The full system, owning all storage its constraints point into.
pub const Owned = struct {
    allocator: std.mem.Allocator,
    factors: []Factor,
    terms: []Term,
    constraints: []Constraint,
    built: System,

    pub fn system(self: *const Owned) System {
        return self.built;
    }

    pub fn deinit(self: *Owned) void {
        self.allocator.free(self.factors);
        self.allocator.free(self.terms);
        self.allocator.free(self.constraints);
        self.* = undefined;
    }
};

/// Number of constraints per multiply — the spike's headline number.
pub const constraints_per_multiply: usize = 108;

/// Build the fp16 multiply AIR. `rows` multiplies, one per row.
pub fn buildSystem(allocator: std.mem.Allocator, rows: usize) BuildError!Owned {
    std.debug.assert(rows > 0);
    var b = Builder{ .allocator = allocator };
    errdefer {
        b.factors.deinit(allocator);
        b.terms.deinit(allocator);
        b.constraints.deinit(allocator);
    }

    // Booleanity of the two input patterns.
    for (0..16) |i| {
        const col: u16 = @intCast(i);
        try b.lin("a bit boolean", .composed, &.{
            .{ .factors = try b.pair(col + 0, col + 0), .coefficient = kOne },
            .{ .factors = try b.pair(col + 0, col + 0), .coefficient = kNegOne },
        });
    }
    for (0..16) |i| {
        const col: u16 = @intCast(i);
        try b.lin("b bit boolean", .composed, &.{
            .{ .factors = try b.pair(col + 0, col + 0), .coefficient = kOne },
            .{ .factors = try b.pair(col + 0, col + 0), .coefficient = kNegOne },
        });
    }
    // Booleanity of the output pattern.
    for (0..16) |i| {
        const col: u16 = @intCast(i);
        try b.lin("c bit boolean", .composed, &.{
            .{ .factors = try b.pair(col + 0, col + 0), .coefficient = kOne },
            .{ .factors = try b.pair(col + 0, col + 0), .coefficient = kNegOne },
        });
    }

    // Input significands: sig = 1024 + mantissa.
    {
        var ts: [12]LinTerm = undefined;
        ts[0] = .{ .factors = try b.one(col_a_sig) };
        ts[1] = .{ .factors = try b.constant(g(mant_implicit)), .coefficient = kNegOne };
        for (0..mant_bits) |i| {
            ts[2 + i] = .{
                .factors = try b.one(aMant() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
        }
        try b.lin("a_sig = 1024 + a_mant", .composed, &ts);
    }
    {
        var ts: [12]LinTerm = undefined;
        ts[0] = .{ .factors = try b.one(col_b_sig) };
        ts[1] = .{ .factors = try b.constant(g(mant_implicit)), .coefficient = kNegOne };
        for (0..mant_bits) |i| {
            ts[2 + i] = .{
                .factors = try b.one(bMant() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
        }
        try b.lin("b_sig = 1024 + b_mant", .composed, &ts);
    }

    // The exponent as a VALUE, because a 5-bit sum cannot be multiplied
    // by anything without becoming degree 6. Two extra columns and two
    // linear constraints are what keep the inverse proofs at degree 2.
    {
        var ts: [exp_bits + 1]LinTerm = undefined;
        ts[0] = .{ .factors = try b.one(col_a_exp_val) };
        for (0..exp_bits) |i| {
            ts[1 + i] = .{
                .factors = try b.one(aExp() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
        }
        try b.lin("a exponent value", .composed, &ts);
    }
    {
        var ts: [exp_bits + 1]LinTerm = undefined;
        ts[0] = .{ .factors = try b.one(col_b_exp_val) };
        for (0..exp_bits) |i| {
            ts[1 + i] = .{
                .factors = try b.one(bExp() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
        }
        try b.lin("b exponent value", .composed, &ts);
    }

    // The inputs are normal and finite: exp != 0 and exp != 31. Proved by
    // the field inverse trick — x·x⁻¹ = 1 has a solution iff x != 0 — so
    // no range check and no lookup is needed.
    try b.lin("a exponent is non-zero", .composed, &.{
        .{ .factors = try b.pair(col_a_exp_inv, col_a_exp_val), .coefficient = kOne },
        .{ .factors = try b.constant(kOne), .coefficient = kNegOne },
    });
    try b.lin("a exponent is not 31", .composed, &.{
        .{ .factors = try b.pair(col_a_nz_inv, col_a_exp_val), .coefficient = kNegOne },
        .{ .factors = try b.one(col_a_nz_inv), .coefficient = g(exp_max) },
        .{ .factors = try b.constant(kOne), .coefficient = kNegOne },
    });
    try b.lin("b exponent is non-zero", .composed, &.{
        .{ .factors = try b.pair(col_b_exp_inv, col_b_exp_val), .coefficient = kOne },
        .{ .factors = try b.constant(kOne), .coefficient = kNegOne },
    });
    try b.lin("b exponent is not 31", .composed, &.{
        .{ .factors = try b.pair(col_b_nz_inv, col_b_exp_val), .coefficient = kNegOne },
        .{ .factors = try b.one(col_b_nz_inv), .coefficient = g(exp_max) },
        .{ .factors = try b.constant(kOne), .coefficient = kNegOne },
    });

    // The exact product. One degree-2 constraint is the entire "multiply".
    try b.lin("product = a_sig · b_sig", .composed, &.{
        .{ .factors = try b.one(col_product) },
        .{ .factors = try b.pair(col_a_sig, col_b_sig), .coefficient = kNegOne },
    });

    // The product's own bit decomposition: this is what makes the round
    // and sticky bits available as columns.
    {
        var ts: [23]LinTerm = undefined;
        ts[0] = .{ .factors = try b.one(col_product) };
        for (0..product_bits) |i| {
            ts[1 + i] = .{
                .factors = try b.one(pBit(@intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
        }
        try b.lin("product bits reconstruct", .composed, &ts);
    }
    for (0..product_bits) |i| {
        const col = pBit(@intCast(i));
        try b.lin("product bit boolean", .composed, &.{
            .{ .factors = try b.pair(col, col), .coefficient = kOne },
            .{ .factors = try b.pair(col, col), .coefficient = kNegOne },
        });
    }

    // norm = [product >= 2^21], which is exactly bit 21.
    try b.lin("norm = product bit 21", .composed, &.{
        .{ .factors = try b.one(col_norm) },
        .{ .factors = try b.one(pBit(21)), .coefficient = kNegOne },
    });

    // round = norm ? bit10 : bit9. A 2-way mux is degree 2, and a mux of
    // two booleans under a boolean selector is boolean for free.
    // y = norm ? a : b  is  y - norm·a - (1 - norm)·b = 0. Writing the
    // (1 - norm)·b term with a POSITIVE sign is correct for the STICKY mux
    // (where it is a sum of ORs) and wrong for every real mux.
    try b.lin("round = mux(norm, p10, p9)", .composed, &.{
        .{ .factors = try b.one(col_round) },
        .{ .factors = try b.pair(col_norm, pBit(10)), .coefficient = kNegOne },
        .{ .factors = try b.one(pBit(9)), .coefficient = kNegOne },
        .{ .factors = try b.pair(col_norm, pBit(9)) },
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
    try stickyOr(&b, col_sticky_hi_sum, col_sticky_hi_inv, col_sticky_hi, pBit(0), 11);
    try stickyOr(&b, col_sticky_lo_sum, col_sticky_lo_inv, col_sticky_lo, pBit(0), 10);
    try b.lin("sticky = mux(norm, sticky_hi, sticky_lo)", .composed, &.{
        .{ .factors = try b.one(col_sticky) },
        .{ .factors = try b.pair(col_norm, col_sticky_hi), .coefficient = kNegOne },
        .{ .factors = try b.one(col_sticky_lo), .coefficient = kNegOne },
        .{ .factors = try b.pair(col_norm, col_sticky_lo) },
    });

    // The kept field: kept_i = norm ? p_(i+11) : p_(i+10). 11 muxes, and
    // the result is boolean for free because it is a mux of two booleans.
    for (0..kept_bits) |i| {
        const idx: u16 = @intCast(i);
        try b.lin("kept bit mux", .composed, &.{
            .{ .factors = try b.one(col_kept + idx) },
            .{ .factors = try b.pair(col_norm, pBit(idx + 11)), .coefficient = kNegOne },
            .{ .factors = try b.one(pBit(idx + 10)), .coefficient = kNegOne },
            .{ .factors = try b.pair(col_norm, pBit(idx + 10)) },
        });
    }

    // RNE: increment iff round AND (sticky OR the kept field's lsb, which
    // is the output mantissa's bit 0).
    // or_sl = sticky OR the kept field's lsb, by the same sum/inverse
    // trick: two booleans whose sum is non-zero exactly when the OR is 1.
    try b.lin("or_sl sum", .composed, &.{
        .{ .factors = try b.one(col_or_sl_sum) },
        .{ .factors = try b.one(col_sticky), .coefficient = kNegOne },
        .{ .factors = try b.one(cMant()), .coefficient = kNegOne },
    });
    try b.lin("or_sl sum · inv", .composed, &.{
        .{ .factors = try b.pair(col_or_sl_sum, col_or_sl_inv) },
        .{ .factors = try b.one(col_or_sl), .coefficient = kNegOne },
    });
    try b.lin("or_sl is boolean", .composed, &.{
        .{ .factors = try b.pair(col_or_sl, col_or_sl) },
        .{ .factors = try b.one(col_or_sl), .coefficient = kNegOne },
    });
    try b.lin("inc = round · or_sl", .composed, &.{
        .{ .factors = try b.one(col_inc) },
        .{ .factors = try b.pair(col_round, col_or_sl), .coefficient = kNegOne },
    });
    try b.lin("carry is boolean", .composed, &.{
        .{ .factors = try b.pair(col_carry, col_carry) },
        .{ .factors = try b.one(col_carry), .coefficient = kNegOne },
    });

    // The output significand: kept + inc = 1024 + c_mant + 1024·carry.
    // On a carry the kept field reached 2048, so the significand is
    // exactly 1.0 and the mantissa is zero — which is why the same linear
    // equation covers both cases.
    {
        // inc, the kept bits, 1024, the carry and the output mantissa.
        var ts: [3 + kept_bits + mant_bits]LinTerm = undefined;
        var n: usize = 0;
        ts[n] = .{ .factors = try b.one(col_inc) };
        n += 1;
        for (0..kept_bits) |i| {
            ts[n] = .{
                .factors = try b.one(col_kept + @as(u16, @intCast(i))),
                .coefficient = g(@as(u64, 1) << @intCast(i)),
            };
            n += 1;
        }
        ts[n] = .{ .factors = try b.constant(g(mant_implicit)), .coefficient = kNegOne };
        n += 1;
        ts[n] = .{ .factors = try b.one(col_carry), .coefficient = kNegOne };
        n += 1;
        for (0..mant_bits) |i| {
            ts[n] = .{
                .factors = try b.one(cMant() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
            n += 1;
        }
        try b.lin("kept + inc = 1024 + c_mant + 1024·carry", .composed, ts[0..n]);
    }

    // The output exponent: ec = ea + eb + keep + carry − 25, with
    // keep = 10 + norm. Rearranged as zero.
    {
        // 3 exponent fields of exp_bits terms each, plus norm, carry, 25.
        var ts: [3 * exp_bits + 4]LinTerm = undefined;
        var n: usize = 0;
        for (0..exp_bits) |i| {
            ts[n] = .{
                .factors = try b.one(cExp() + @as(u16, @intCast(i))),
                .coefficient = g(@as(u64, 1) << @intCast(i)),
            };
            n += 1;
        }
        for (0..exp_bits) |i| {
            ts[n] = .{
                .factors = try b.one(aExp() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
            n += 1;
        }
        for (0..exp_bits) |i| {
            ts[n] = .{
                .factors = try b.one(bExp() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
            n += 1;
        }
        ts[n] = .{ .factors = try b.one(col_norm), .coefficient = g(kNegModPow2(0)) };
        n += 1;
        ts[n] = .{ .factors = try b.one(col_carry), .coefficient = g(kNegModPow2(0)) };
        n += 1;
        // -10 (the base of keep = 10 + norm) and +25 (the exponent bias
        // unwind). The -10 was missing for one iteration, which showed up
        // as every row evaluating to exactly 10.
        ts[n] = .{ .factors = try b.constant(g(25 - 10)) };
        n += 1;
        try b.lin("ec = ea + eb + 10 + norm + carry − 25", .composed, ts[0..n]);
    }

    // The output sign is the XOR of the input signs: a + b − 2ab.
    try b.lin("c_sign = a_sign XOR b_sign", .composed, &.{
        .{ .factors = try b.one(cSign()) },
        .{ .factors = try b.one(aSign()), .coefficient = kNegOne },
        .{ .factors = try b.one(bSign()), .coefficient = kNegOne },
        .{ .factors = try b.pair(aSign(), bSign()), .coefficient = g(2) },
    });

    // Freeze the three buffers, then RESOLVE the recorded indices into
    // slices. Only now do slices exist, and they point into buffers that
    // will not move again.
    const factor_buf = try b.factors.toOwnedSlice(allocator);
    errdefer allocator.free(factor_buf);
    const term_specs = try b.terms.toOwnedSlice(allocator);
    defer allocator.free(term_specs);
    const c_specs = try b.constraints.toOwnedSlice(allocator);
    defer allocator.free(c_specs);

    const terms = try allocator.alloc(Term, term_specs.len);
    errdefer allocator.free(terms);
    for (term_specs, 0..) |spec, i| {
        terms[i] = .{
            .factors = factor_buf[spec.factors.first .. spec.factors.first + spec.factors.len],
            .coefficient = spec.coefficient,
        };
    }

    const per_row = c_specs.len;
    if (per_row != constraints_per_multiply) {
        std.debug.print("fp16 air: built {d} constraints per multiply, constant says {d}\n", .{ per_row, constraints_per_multiply });
    }
    const all = try allocator.alloc(Constraint, per_row * rows);
    for (0..rows) |r| {
        for (c_specs, 0..) |spec, i| {
            all[r * per_row + i] = .{
                .name = spec.name,
                .scope = spec.scope,
                .terms = terms[spec.first_term .. spec.first_term + spec.n_terms],
            };
        }
    }
    return .{
        .allocator = allocator,
        .factors = factor_buf,
        .terms = terms,
        .constraints = all,
        .built = .{ .constraints = all },
    };
}

/// `out` = OR of the `count` boolean bits at `base`, via the
/// "the sum is non-zero" inverse trick. 3 constraints: the sum, the
/// inverse identity, and booleanity of the result.
fn stickyOr(b: *Builder, sum_col: u16, inv_col: u16, out_col: u16, base: u16, count: u16) !void {
    var ts: [17]LinTerm = undefined;
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

/// -2^i as a field element, for the reconstruction coefficients.
fn kNegModPow2(i: u16) u64 {
    const p = Goldilocks.p;
    return p - ((@as(u64, 1) << @intCast(i)) % p);
}

// ---------------------------------------------------------------------------
// Trace construction
// ---------------------------------------------------------------------------

pub const Trace = struct {
    rows: usize,
    columns: [][]Fp2,

    pub fn deinit(self: *Trace, allocator: std.mem.Allocator) void {
        for (self.columns) |c| allocator.free(c);
        allocator.free(self.columns);
        self.* = undefined;
    }
};

pub const BuildTraceError = error{
    OutOfMemory,
    /// The reference refused this pair (subnormal input or result, or an
    /// inf/NaN result). S1 does not cover those and says so instead of
    /// proving something plausible but wrong.
    UnsupportedCase,
};

/// Bit i of `v` goes to COLUMN (base + i), row r — the trace is
/// column-major, so a run of bits is a stride over columns, not a
/// contiguous range.
fn writeBits(cols: [][]Fp2, base: u16, v: u32, n: u8, r: usize) void {
    for (0..n) |i| {
        cols[base + @as(u16, @intCast(i))][r] = Fp2.re(Goldilocks.fromU64((v >> @intCast(i)) & 1));
    }
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
pub fn buildTrace(
    allocator: std.mem.Allocator,
    pairs: []const [2]u16,
) BuildTraceError!Trace {
    const rows = pairs.len;
    std.debug.assert(rows > 0);

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

    for (pairs, 0..) |pair, r| {
        const a = pair[0];
        const b = pair[1];
        const pa = fp16.Parts.fromBits(a);
        const pb = fp16.Parts.fromBits(b);
        if (pa.exponent == 0 or pb.exponent == 0 or pa.exponent == exp_max or pb.exponent == exp_max) {
            return BuildTraceError.UnsupportedCase;
        }
        const expected = fp16.multiply(a, b) catch return BuildTraceError.UnsupportedCase;
        const pc = fp16.Parts.fromBits(expected);
        if (pc.exponent == 0 or pc.exponent == exp_max) return BuildTraceError.UnsupportedCase;

        writeBits(cols, col_a_bits, a, 16, r);
        writeBits(cols, col_b_bits, b, 16, r);
        writeBits(cols, col_c_bits, expected, 16, r);

        const a_sig: u32 = mant_implicit + @as(u32, pa.mantissa);
        const b_sig: u32 = mant_implicit + @as(u32, pb.mantissa);
        cols[col_a_sig][r] = Fp2.re(Goldilocks.fromU64(a_sig));
        cols[col_b_sig][r] = Fp2.re(Goldilocks.fromU64(b_sig));

        // Field inverses witness "this exponent field is neither 0 nor 31".
        const a_exp: u64 = pa.exponent;
        const b_exp: u64 = pb.exponent;
        cols[col_a_exp_val][r] = Fp2.re(Goldilocks.fromU64(a_exp));
        cols[col_b_exp_val][r] = Fp2.re(Goldilocks.fromU64(b_exp));
        cols[col_a_exp_inv][r] = Fp2.re(Goldilocks.fromU64(a_exp).inv() catch unreachable);
        cols[col_a_nz_inv][r] = Fp2.re(Goldilocks.fromU64(exp_max - a_exp).inv() catch unreachable);
        cols[col_b_exp_inv][r] = Fp2.re(Goldilocks.fromU64(b_exp).inv() catch unreachable);
        cols[col_b_nz_inv][r] = Fp2.re(Goldilocks.fromU64(exp_max - b_exp).inv() catch unreachable);

        const product: u32 = a_sig * b_sig;
        cols[col_product][r] = Fp2.re(Goldilocks.fromU64(product));
        writeBits(cols, col_p_bits, product, product_bits, r);

        const norm: u32 = if (product >= (1 << 21)) 1 else 0;
        cols[col_norm][r] = Fp2.re(Goldilocks.fromU64(norm));

        const keep: u8 = if (norm == 1) 11 else 10;
        const round: u32 = (product >> @intCast(keep - 1)) & 1;
        const sum_hi: u32 = popCount(product, 11);
        const sum_lo: u32 = popCount(product, 10);
        cols[col_sticky_hi_sum][r] = Fp2.re(Goldilocks.fromU64(sum_hi));
        cols[col_sticky_lo_sum][r] = Fp2.re(Goldilocks.fromU64(sum_lo));
        // A zero sum gets a zero inverse: the identity sum·inv = sticky
        // then forces sticky = 0, which is the correct OR.
        const inv_hi = if (sum_hi == 0) Goldilocks.zero else Goldilocks.fromU64(sum_hi).inv() catch unreachable;
        const inv_lo = if (sum_lo == 0) Goldilocks.zero else Goldilocks.fromU64(sum_lo).inv() catch unreachable;
        cols[col_sticky_hi_inv][r] = Fp2.re(inv_hi);
        cols[col_sticky_lo_inv][r] = Fp2.re(inv_lo);
        const sticky_hi: u32 = if (sum_hi == 0) 0 else 1;
        const sticky_lo: u32 = if (sum_lo == 0) 0 else 1;
        const sticky: u32 = if (norm == 1) sticky_hi else sticky_lo;
        cols[col_round][r] = Fp2.re(Goldilocks.fromU64(round));
        cols[col_sticky_hi][r] = Fp2.re(Goldilocks.fromU64(sticky_hi));
        cols[col_sticky_lo][r] = Fp2.re(Goldilocks.fromU64(sticky_lo));
        cols[col_sticky][r] = Fp2.re(Goldilocks.fromU64(sticky));

        var kept_sum: u32 = 0;
        for (0..kept_bits) |i| {
            const bit = (product >> @intCast(i + keep)) & 1;
            cols[col_kept + @as(u16, @intCast(i))][r] = Fp2.re(Goldilocks.fromU64(bit));
            kept_sum |= bit << @intCast(i);
        }
        const lsb: u32 = pc.mantissa & 1;
        const or_sl_sum: u32 = sticky + lsb;
        const or_sl: u32 = if (or_sl_sum == 0) 0 else 1;
        cols[col_or_sl_sum][r] = Fp2.re(Goldilocks.fromU64(or_sl_sum));
        cols[col_or_sl_inv][r] = Fp2.re(
            if (or_sl_sum == 0) Goldilocks.zero else Goldilocks.fromU64(or_sl_sum).inv() catch unreachable,
        );
        const inc: u32 = if (round == 1 and or_sl == 1) 1 else 0;
        const carry: u32 = if (kept_sum + inc == 2048) 1 else 0;
        cols[col_or_sl][r] = Fp2.re(Goldilocks.fromU64(or_sl));
        cols[col_inc][r] = Fp2.re(Goldilocks.fromU64(inc));
        cols[col_carry][r] = Fp2.re(Goldilocks.fromU64(carry));
    }
    return .{ .rows = rows, .columns = cols };
}
