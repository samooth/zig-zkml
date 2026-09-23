//! Exact widening to binary32, as an AIR.
//!
//! Every engine this project has to match widens before it accumulates: a
//! bf16 or fp16 activation becomes an fp32 accumulator lane, an fp8
//! dequant becomes fp16 or bf16. Widening is the one float operation with
//! NO rounding — every bit of the source maps to a bit of the target and
//! nothing is lost — so its AIR is a set of copy constraints and needs no
//! arithmetic witnesses at all. That makes it the cheapest soundness
//! statement in this directory, and it is the first link of the chain the
//! real use case needs.
//!
//! bfloat16 → binary32 is total and needs no arithmetic: same bias, same
//! exponent width, so the exponent field is copied verbatim and the
//! mantissa moves up by 16 bits. The other three add 127 − bias to the
//! exponent field, which is a constant addition over bits and therefore a
//! carry chain of witnesses — still linear constraints, still degree 2
//! where the carries multiply.
//!
//! SCOPE, deliberately: those three cover NORMAL FINITE sources only. A
//! subnormal source needs a leading-zero count — a barrel shifter, the
//! same machinery the multiply's range reduction needs, and it lands with
//! it — and inf/NaN need the all-ones exponent mapped to all-ones rather
//! than shifted. `widen` refuses those cases instead of quietly rounding
//! them, the same discipline the multiply uses.

const std = @import("std");
const expr = @import("./expr.zig");
const float_ref = @import("./float_ref.zig");
const fmt_lib = @import("./float_format.zig");
const bld = @import("./air_builder.zig");

pub const Fp2 = expr.Fp2;
pub const System = expr.System;
pub const Constraint = expr.Constraint;
pub const Factor = expr.Factor;
pub const Scope = expr.Scope;
pub const Format = fmt_lib.Format;
pub const Builder = bld.Builder;
pub const Trace = bld.Trace;
pub const BuildError = bld.BuildError;

/// binary32's field layout. Not a `Format`: that one packs patterns into a
/// u16, and 32 bits do not fit. Only what a widening needs.
pub const binary32 = struct {
    pub const exp_bits: u8 = 8;
    pub const mant_bits: u8 = 23;
    pub const bias: i32 = 127;
    pub const width: u16 = 32;
    pub const sign: u16 = 31;
    pub const emax: u16 = 255;
};

/// The trace layout: the source pattern's bits, then the target's, then
/// the carry chain when the bias moves.
pub fn Layout(comptime f: Format) type {
    return struct {
        const src: u16 = f.byteWidth();
        const shift: u16 = binary32.mant_bits - f.mant_bits;
        const needs_carry = f.bias != binary32.bias;

        pub const col_src: u16 = 0;
        pub const col_dst: u16 = col_src + src;
        /// carry[j] is the carry INTO exponent bit j, for j >= 1. Bit 0's
        /// carry-in is the addend's own bit, which is a constant, so it
        /// needs no column. The chain runs over the TARGET's exponent
        /// width, because the bias difference is wider than a binary16
        /// exponent field: 112 needs seven bits and binary16 has five.
        pub const col_carry: u16 = col_dst + binary32.width;
        pub const carry_count: u16 = if (needs_carry) binary32.exp_bits - 1 else 0;
        pub const column_count: usize = col_carry + carry_count;

        pub inline fn srcBit(i: u16) u16 {
            return col_src + i;
        }
        pub inline fn dstBit(i: u16) u16 {
            return col_dst + i;
        }
        pub inline fn carry(j: u16) u16 {
            return col_carry + j - 1;
        }
        pub inline fn srcExpBit(i: u16) u16 {
            return srcBit(f.mant_bits + i);
        }
        pub inline fn dstExpBit(i: u16) u16 {
            return dstBit(binary32.mant_bits + i);
        }
    };
}

fn cname(comptime f: Format, comptime fmt: []const u8, comptime args: anytype) []const u8 {
    _ = f;
    return comptime std.fmt.comptimePrint(fmt, args);
}

/// How many constraints one widening costs: the source's booleanity, the
/// sign, the mantissa's rewiring including its zero padding, and the
/// exponent either copied (same bias) or summed with a carry chain.
pub fn expected_constraints(f: Format) usize {
    const src: usize = f.byteWidth();
    const shift: usize = binary32.mant_bits - f.mant_bits;
    const copied_exp: usize = f.exp_bits;
    const fixed: usize = src + 1 + shift + f.mant_bits;
    if (f.bias == binary32.bias) return fixed + copied_exp;
    // One equation per target exponent bit, plus a carry-out per bit but
    // the last, plus booleanity for each carry column.
    const target_exp: usize = binary32.exp_bits;
    return fixed + target_exp + (target_exp - 1) * 2;
}

/// Build the widening AIR: one row per source pattern.
pub fn buildSystem(allocator: std.mem.Allocator, rows: usize, comptime f: Format) BuildError!bld.Owned {
    const L = Layout(f);
    var b = Builder.init(allocator);
    const shift: u16 = binary32.mant_bits - f.mant_bits;

    inline for (0..L.src) |i| {
        const d: u16 = @intCast(i);
        try b.boolean(cname(f, "{s}: source bit {d} is boolean", .{ f.name, d }), L.srcBit(d));
    }

    try b.copy(cname(f, "{s}: sign copied", .{f.name}), L.dstBit(binary32.sign), L.srcBit(L.src - 1));

    inline for (0..shift) |i| {
        const d: u16 = @intCast(i);
        try b.lin(cname(f, "{s}: mantissa padding bit {d} is zero", .{ f.name, d }), .composed, &.{
            .{ .factors = try b.one(L.dstBit(d)) },
        });
    }
    inline for (0..f.mant_bits) |i| {
        const d: u16 = @intCast(i);
        try b.copy(cname(f, "{s}: mantissa bit {d} moved", .{ f.name, d }), L.dstBit(shift + d), L.srcBit(d));
    }

    if (f.bias == binary32.bias) {
        inline for (0..f.exp_bits) |i| {
            const d: u16 = @intCast(i);
            try b.copy(cname(f, "{s}: exponent bit {d} copied", .{ f.name, d }), L.dstExpBit(d), L.srcExpBit(d));
        }
    } else {
        // src_exp + 112 (or whatever 127 - bias is) is a three-input
        // addition per bit, and a ripple-carry adder is NOT linear: the
        // sum bit is an XOR and the carry is "at least two". Both become
        // linear plus ONE quadratic term because the addend's bit is a
        // comptime constant. With c = that bit, s the source bit and i the
        // carry in:
        //
        //   t = s XOR c = (1-2c)·s + c      and   s + c = t + 2·(s·c)
        //   dst = t XOR i = t + i - 2·t·i     (the 2·(s·c) is even)
        //   carry out = s·c + t·i
        //
        // which expand to the two equations below. Degree 2, no range
        // check, no comparison.
        const addend: u32 = @intCast(binary32.bias - @as(i32, f.bias));
        inline for (0..L.carry_count) |i| {
            const d: u16 = @intCast(i + 1);
            try b.boolean(cname(f, "{s}: carry into exponent bit {d} is boolean", .{ f.name, d }), L.carry(d));
        }
        inline for (0..binary32.exp_bits) |i| {
            const d: u16 = @intCast(i);
            const c: i32 = @intCast((addend >> @intCast(d)) & 1);
            const has_src = d < f.exp_bits;
            const has_in = d > 0;
            const sgn: Fp2 = if (c == 0) bld.kNegOne else Fp2.one;

            var eq: [5]bld.LinTerm = undefined;
            var n: usize = 0;
            eq[n] = .{ .factors = try b.one(L.dstExpBit(d)) };
            n += 1;
            if (has_src) {
                eq[n] = .{ .factors = try b.one(L.srcExpBit(d)), .coefficient = sgn };
                n += 1;
            }
            if (c == 1) {
                eq[n] = .{ .factors = try b.constant(Fp2.one), .coefficient = bld.kNegOne };
                n += 1;
            }
            if (has_in) {
                eq[n] = .{ .factors = try b.one(L.carry(d)), .coefficient = sgn };
                n += 1;
            }
            if (has_src and has_in) {
                eq[n] = .{ .factors = try b.pair(L.srcExpBit(d), L.carry(d)), .coefficient = Fp2.add(Fp2.neg(sgn), Fp2.neg(sgn)) };
                n += 1;
            }
            try b.lin(cname(f, "{s}: exponent bit {d} with carry", .{ f.name, d }), .composed, eq[0..n]);

            if (d + 1 >= binary32.exp_bits) continue;
            var cy: [4]bld.LinTerm = undefined;
            var m: usize = 0;
            cy[m] = .{ .factors = try b.one(L.carry(d + 1)) };
            m += 1;
            if (c == 1 and has_src) {
                cy[m] = .{ .factors = try b.one(L.srcExpBit(d)), .coefficient = bld.kNegOne };
                m += 1;
            }
            if (has_src and has_in) {
                cy[m] = .{ .factors = try b.pair(L.srcExpBit(d), L.carry(d)), .coefficient = sgn };
                m += 1;
            }
            if (c == 1 and has_in) {
                cy[m] = .{ .factors = try b.one(L.carry(d)), .coefficient = bld.kNegOne };
                m += 1;
            }
            try b.lin(cname(f, "{s}: carry out of exponent bit {d}", .{ f.name, d }), .composed, cy[0..m]);
        }
    }

    const expected = expected_constraints(f);
    if (b.count() != expected) {
        std.debug.print("widen air: built {d} constraints per widening, {s} should be {d}\n", .{ b.count(), f.name, expected });
    }
    return bld.freeze(allocator, &b, rows);
}

pub const BuildTraceError = error{ OutOfMemory, UnsupportedCase, OutOfRange };

/// One row per source pattern. The witness is `widen`'s own answer, so a
/// disagreement between the AIR and the reference shows up as a failed
/// proof rather than a passing one.
pub fn buildTrace(allocator: std.mem.Allocator, patterns: []const u16, comptime f: Format) BuildTraceError!Trace {
    const L = Layout(f);
    var trace = try Trace.alloc(allocator, L.column_count, patterns.len);
    errdefer trace.deinit(allocator);
    const addend: u32 = if (f.bias == binary32.bias) 0 else @intCast(binary32.bias - @as(i32, f.bias));
    for (patterns, 0..) |src, r| {
        const dst = try widen(f, src);
        trace.writeBits(L.col_src, src, L.src, r);
        trace.writeBits(L.col_dst, dst, binary32.width, r);
        // The carry chain is a witness like any other: recomputed here from
        // the source pattern, not read out of the target.
        var carry: u32 = 0;
        var d: u16 = 0;
        while (d < binary32.exp_bits) : (d += 1) {
            const src_bit: u32 = if (d < f.exp_bits) (src >> @intCast(f.mant_bits + d)) & 1 else 0;
            const sum = src_bit + ((addend >> @intCast(d)) & 1) + carry;
            if (d > 0 and d <= L.carry_count) {
                trace.columns[L.carry(d)][r] = Fp2.re(bld.Goldilocks.fromU64(carry));
            }
            carry = sum >> 1;
        }
    }
    return trace;
}

pub const WidenError = error{ UnsupportedCase, OutOfRange };

/// The reference. bfloat16 is total; the others refuse subnormal sources
/// and inf/NaN, which is exactly the AIR's scope.
pub fn widen(f: Format, src: u16) WidenError!u32 {
    if (src >= (@as(u32, 1) << @intCast(f.byteWidth()))) return WidenError.OutOfRange;
    if (f.bias == binary32.bias) return @as(u32, src) << 16;
    const p = f.parts(src);
    if (p.exponent == 0) return WidenError.UnsupportedCase;
    if (f.has_inf_nan and p.exponent == f.emax()) return WidenError.UnsupportedCase;
    const exponent: u16 = @intCast(@as(i32, p.exponent) + (binary32.bias - @as(i32, f.bias)));
    if (exponent >= binary32.emax) return WidenError.OutOfRange;
    const mantissa: u32 = @as(u32, p.mantissa) << @intCast(shiftOf(f));
    return (@as(u32, p.sign) << 31) | (@as(u32, exponent) << 23) | mantissa;
}

fn shiftOf(f: Format) u8 {
    return binary32.mant_bits - f.mant_bits;
}
