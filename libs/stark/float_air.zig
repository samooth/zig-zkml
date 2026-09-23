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
        pub const col_a_exp_val: u16 = col_b_sig + 1;
        pub const col_b_exp_val: u16 = col_a_exp_val + 1;
        pub const col_a_exp_inv: u16 = col_b_exp_val + 1;
        pub const col_a_nz_inv: u16 = col_a_exp_inv + 1;
        pub const col_b_exp_inv: u16 = col_a_nz_inv + 1;
        pub const col_b_nz_inv: u16 = col_b_exp_inv + 1;
        pub const col_product: u16 = col_b_nz_inv + 1;
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
        pub const column_count: usize = col_c_bits + width;

        pub inline fn aBit(i: u16) u16 {
            return col_a_bits + i;
        }
        pub inline fn bBit(i: u16) u16 {
            return col_b_bits + i;
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
        /// The product bit that decides normalisation.
        pub inline fn topBit() u16 {
            return pBit(@intCast(f.normBit()));
        }
    };
}

pub const kOne = Fp2.one;
const kNegOne = Fp2.neg(Fp2.one);

/// A constraint name built at comptime, so the numbers in it are the
/// format's numbers and not binary16's.
fn cname(comptime fmt: []const u8, comptime args: anytype) []const u8 {
    return std.fmt.comptimePrint(fmt, args);
}

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
    return 3 * @as(usize, f.byteWidth()) + f.productBits() + f.keptHigh() + 27;
}

/// Build the fp16 multiply AIR. `rows` multiplies, one per row.
pub fn buildSystem(allocator: std.mem.Allocator, rows: usize, comptime f: Format) BuildError!Owned {
    const L = Layout(f);
    // comptime so the constraint NAMES can carry the constants they
    // actually constrain: a name saying "1024" on a bf16 system is a lie.
    const nm_a_sig = comptime cname("a significand = {d} + a mantissa", .{f.mantImplicit()});
    const nm_b_sig = comptime cname("b significand = {d} + b mantissa", .{f.mantImplicit()});
    const nm_norm = comptime cname("norm = product bit {d}", .{f.normBit()});
    const nm_round = comptime cname("round = mux(norm, p{d}, p{d})", .{ f.keptHigh() - 1, f.keptHigh() - 2 });
    const nm_kept = comptime cname("kept + inc = {d} + c mantissa + {d}·carry", .{ f.mantImplicit(), f.mantImplicit() });
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
        ts[0] = .{ .factors = try b.one(L.col_a_exp_val) };
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
        ts[0] = .{ .factors = try b.one(L.col_b_exp_val) };
        for (0..L.exps) |i| {
            ts[1 + i] = .{
                .factors = try b.one(L.bExp() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
        }
        try b.lin("b exponent value", .composed, &ts);
    }

    // The inputs are normal and finite: exp != 0 and exp != 31. Proved by
    // the field inverse trick — x·x⁻¹ = 1 has a solution iff x != 0 — so
    // no range check and no lookup is needed.
    try b.lin("a exponent is non-zero", .composed, &.{
        .{ .factors = try b.pair(L.col_a_exp_inv, L.col_a_exp_val), .coefficient = kOne },
        .{ .factors = try b.constant(kOne), .coefficient = kNegOne },
    });
    try b.lin("a exponent is below emax", .composed, &.{
        .{ .factors = try b.pair(L.col_a_nz_inv, L.col_a_exp_val), .coefficient = kNegOne },
        .{ .factors = try b.one(L.col_a_nz_inv), .coefficient = g(f.emax()) },
        .{ .factors = try b.constant(kOne), .coefficient = kNegOne },
    });
    try b.lin("b exponent is non-zero", .composed, &.{
        .{ .factors = try b.pair(L.col_b_exp_inv, L.col_b_exp_val), .coefficient = kOne },
        .{ .factors = try b.constant(kOne), .coefficient = kNegOne },
    });
    try b.lin("b exponent is below emax", .composed, &.{
        .{ .factors = try b.pair(L.col_b_nz_inv, L.col_b_exp_val), .coefficient = kNegOne },
        .{ .factors = try b.one(L.col_b_nz_inv), .coefficient = g(f.emax()) },
        .{ .factors = try b.constant(kOne), .coefficient = kNegOne },
    });

    // The exact product. One degree-2 constraint is the entire "multiply".
    try b.lin("product = a_sig · b_sig", .composed, &.{
        .{ .factors = try b.one(L.col_product) },
        .{ .factors = try b.pair(L.col_a_sig, L.col_b_sig), .coefficient = kNegOne },
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

    // The output significand: kept + inc = 1024 + c_mant + 1024·carry.
    // On a carry the kept field reached 2048, so the significand is
    // exactly 1.0 and the mantissa is zero — which is why the same linear
    // equation covers both cases.
    {
        // inc, the kept bits, the implicit one, the carry, the mantissa.
        var ts: [3 + f.sigBits() + f.mant_bits]LinTerm = undefined;
        var n: usize = 0;
        ts[n] = .{ .factors = try b.one(L.col_inc) };
        n += 1;
        for (0..f.sigBits()) |i| {
            ts[n] = .{
                .factors = try b.one(L.col_kept + @as(u16, @intCast(i))),
                .coefficient = g(@as(u64, 1) << @intCast(i)),
            };
            n += 1;
        }
        ts[n] = .{ .factors = try b.constant(g(f.mantImplicit())), .coefficient = kNegOne };
        n += 1;
        // -mantImplicit·carry, not -carry: a carry means the kept field
        // reached 2·mantImplicit, and the equation has to say so.
        ts[n] = .{ .factors = try b.one(L.col_carry), .coefficient = Fp2.neg(g(f.mantImplicit())) };
        n += 1;
        for (0..L.mant) |i| {
            ts[n] = .{
                .factors = try b.one(L.cMant() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
            n += 1;
        }
        try b.lin(nm_kept, .composed, ts[0..n]);
    }

    // The output exponent: ec = ea + eb + keep + carry − 25, with
    // keep = 10 + norm. Rearranged as zero.
    {
        // 3 exponent fields of exp_bits terms each, plus norm, carry, 25.
        var ts: [3 * f.exp_bits + 4]LinTerm = undefined;
        var n: usize = 0;
        for (0..L.exps) |i| {
            ts[n] = .{
                .factors = try b.one(L.cExp() + @as(u16, @intCast(i))),
                .coefficient = g(@as(u64, 1) << @intCast(i)),
            };
            n += 1;
        }
        for (0..L.exps) |i| {
            ts[n] = .{
                .factors = try b.one(L.aExp() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
            n += 1;
        }
        for (0..L.exps) |i| {
            ts[n] = .{
                .factors = try b.one(L.bExp() + @as(u16, @intCast(i))),
                .coefficient = g(kNegModPow2(@intCast(i))),
            };
            n += 1;
        }
        ts[n] = .{ .factors = try b.one(L.col_norm), .coefficient = g(kNegModPow2(0)) };
        n += 1;
        ts[n] = .{ .factors = try b.one(L.col_carry), .coefficient = g(kNegModPow2(0)) };
        n += 1;
        // -(keep = keptLow + norm) and +(bias + mant_bits) (the exponent
        // unwind). The first was missing for one iteration, which showed up
        // as every row evaluating to exactly 10.
        const unwind: u16 = @as(u16, f.bias) + @as(u16, f.mant_bits) - @as(u16, f.keptLow());
        ts[n] = .{ .factors = try b.constant(g(unwind)) };
        n += 1;
        try b.lin(nm_exp, .composed, ts[0..n]);
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
    const expected = expected_constraints(f);
    if (per_row != expected) {
        std.debug.print("float air: built {d} constraints per multiply, {s} should be {d}\n", .{ per_row, f.name, expected });
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
    comptime f: Format,
) BuildTraceError!Trace {
    const L = Layout(f);
    const rows = pairs.len;
    std.debug.assert(rows > 0);

    const cols = try allocator.alloc([]Fp2, L.column_count);
    errdefer allocator.free(cols);
    var made: usize = 0;
    errdefer for (cols[0..made]) |c| allocator.free(c);
    for (0..L.column_count) |i| {
        const buf = try allocator.alloc(Fp2, rows);
        @memset(buf, Fp2.zero);
        cols[i] = buf;
        made += 1;
    }

    for (pairs, 0..) |pair, r| {
        const a = pair[0];
        const b = pair[1];
        const pa = f.parts(a);
        const pb = f.parts(b);
        if (pa.exponent == 0 or pb.exponent == 0 or
            pa.exponent == f.emax() or pb.exponent == f.emax())
        {
            return BuildTraceError.UnsupportedCase;
        }
        const expected = float_ref.multiply(f, a, b) catch return BuildTraceError.UnsupportedCase;
        const pc = f.parts(expected);
        if (pc.exponent == 0 or pc.exponent == f.emax()) return BuildTraceError.UnsupportedCase;

        writeBits(cols, L.col_a_bits, a, @intCast(f.byteWidth()), r);
        writeBits(cols, L.col_b_bits, b, @intCast(f.byteWidth()), r);
        writeBits(cols, L.col_c_bits, expected, @intCast(f.byteWidth()), r);

        const a_sig: u32 = @as(u32, f.mantImplicit()) + @as(u32, pa.mantissa);
        const b_sig: u32 = @as(u32, f.mantImplicit()) + @as(u32, pb.mantissa);
        cols[L.col_a_sig][r] = Fp2.re(Goldilocks.fromU64(a_sig));
        cols[L.col_b_sig][r] = Fp2.re(Goldilocks.fromU64(b_sig));

        // Field inverses witness "this exponent field is neither 0 nor 31".
        const a_exp: u64 = pa.exponent;
        const b_exp: u64 = pb.exponent;
        cols[L.col_a_exp_val][r] = Fp2.re(Goldilocks.fromU64(a_exp));
        cols[L.col_b_exp_val][r] = Fp2.re(Goldilocks.fromU64(b_exp));
        cols[L.col_a_exp_inv][r] = Fp2.re(Goldilocks.fromU64(a_exp).inv() catch unreachable);
        cols[L.col_a_nz_inv][r] = Fp2.re(Goldilocks.fromU64(f.emax() - a_exp).inv() catch unreachable);
        cols[L.col_b_exp_inv][r] = Fp2.re(Goldilocks.fromU64(b_exp).inv() catch unreachable);
        cols[L.col_b_nz_inv][r] = Fp2.re(Goldilocks.fromU64(f.emax() - b_exp).inv() catch unreachable);

        const product: u32 = a_sig * b_sig;
        cols[L.col_product][r] = Fp2.re(Goldilocks.fromU64(product));
        writeBits(cols, L.col_p_bits, product, f.productBits(), r);

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
    }
    return .{ .rows = rows, .columns = cols };
}
