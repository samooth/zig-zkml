//! Finite field arithmetic — Goldilocks prime field p = 2^61 - 1.
//!
//! Own minimal L0 (docs/BLUE_PRINT.md §2: "200 líneas propias" fallback for the
//! zig-algebra path dependency, which cannot be fetched due to the
//! 0.16.0-dev semver lock). Only what zkML needs: add/sub/mul with lazy
//! reduction, pow/inverse, and serialization for hashing.
//!
//! Invariants: every arithmetic path stays < 2^61·k for at most one
//! unreduced step (lazy reduction allowed exactly once — see
//! docs/BLUE_PRINT.md §4.2 for the derivation of the accumulation budgets).

const std = @import("std");

pub const Goldilocks = struct {
    pub const p: u64 = (1 << 61) - 1;
    pub const fp_bits: u6 = 61;

    /// Fully-reduced representative in [0, p).
    rep: u64,

    pub const zero: Goldilocks = .{ .rep = 0 };
    pub const one: Goldilocks = .{ .rep = 1 };

    pub fn fromU64(v: u64) Goldilocks {
        return .{ .rep = v % p };
    }

    /// Canonical reduction for values < 2^64 that may exceed p once
    /// (products of two reduced elements are handled by `mul`).
    pub fn fromLazy(v: u64) Goldilocks {
        // v < 2^64 => v - p·(v >= p ? 1 : 0) can still be >= p if
        // v >= 2p (impossible for u64 < 2^64 when p ~ 2^61: v can hold up
        // to 15p). One conditional subtract per multiple is fine; loop.
        var x = v;
        while (x >= p) x -= p;
        return .{ .rep = x };
    }

    pub fn fromI64(v: i64) Goldilocks {
        if (v >= 0) return fromU64(@intCast(v));
        // -v mod p = p - (v mod p)
        const uv: u64 = @intCast(-v);
        return .{ .rep = p - (uv % p) };
    }

    pub fn add(a: Goldilocks, b: Goldilocks) Goldilocks {
        // a,b < p < 2^61 => sum < 2^62: single conditional subtract.
        const s = a.rep + b.rep;
        return .{ .rep = if (s >= p) s - p else s };
    }

    pub fn sub(a: Goldilocks, b: Goldilocks) Goldilocks {
        // a < p, b < p => a + p - b < 2^62: one conditional subtract.
        const d = a.rep + p - b.rep;
        return .{ .rep = if (d >= p) d - p else d };
    }

    /// Lazy add: returns unreduced sum (valid input to a following `mul`
    /// ONLY after reduction — mul() asserts inputs < p).
    pub fn addLazy(a: Goldilocks, b: Goldilocks) u64 {
        return a.rep + b.rep;
    }

    pub fn neg(a: Goldilocks) Goldilocks {
        return .{ .rep = if (a.rep == 0) 0 else p - a.rep };
    }

    pub fn mul(a: Goldilocks, b: Goldilocks) Goldilocks {
        std.debug.assert(a.rep < p and b.rep < p);
        const prod = @as(u128, a.rep) * @as(u128, b.rep);
        // prod < p^2 < 2^122. Reduce mod p = 2^61-1 using the identity
        // 2^61 ≡ 1 (mod p): fold the high 61 bits into the low part.
        const lo: u64 = @intCast(prod & p);
        const hi: u64 = @intCast(prod >> 61);
        var r = lo + hi;
        while (r >= p) r -= p;
        return .{ .rep = r };
    }

    pub fn eql(a: Goldilocks, b: Goldilocks) bool {
        return a.rep == b.rep;
    }

    pub fn isZero(a: Goldilocks) bool {
        return a.rep == 0;
    }

    pub fn pow(a: Goldilocks, e: u64) Goldilocks {
        var result = one;
        var base = a;
        var exp = e;
        while (exp > 0) {
            if (exp & 1 == 1) result = result.mul(base);
            base = base.mul(base);
            exp >>= 1;
        }
        return result;
    }

    /// Multiplicative inverse via Fermat: a^(p-2) mod p.
    /// Returns error on a == 0.
    pub fn inv(a: Goldilocks) error{ZeroInverse}!Goldilocks {
        if (a.isZero()) return error.ZeroInverse;
        return a.pow(p - 2);
    }

    pub fn toU64(a: Goldilocks) u64 {
        return a.rep;
    }

    /// 8-byte little-endian encoding for hashing/serialization.
    pub fn toBytes(a: Goldilocks, out: *[8]u8) void {
        std.mem.writeInt(u64, out, a.rep, .little);
    }

    pub fn fromBytes(bytes: *const [8]u8) Goldilocks {
        return fromU64(std.mem.readInt(u64, bytes, .little));
    }
};

test "goldilocks basic add/sub/mul" {
    const t = std.testing;
    const a = Goldilocks.fromU64(123456789);
    const b = Goldilocks.fromU64(987654321);
    try t.expectEqual(Goldilocks.fromU64(1111111110), a.add(b));
    try t.expectEqual(Goldilocks.zero, a.sub(a));
    // p-1 + 1 == 0
    try t.expectEqual(Goldilocks.zero, Goldilocks.fromU64(Goldilocks.p - 1).add(Goldilocks.one));
    // 2^61 ≡ 1
    try t.expectEqual(Goldilocks.one, Goldilocks.fromU64(1 << 61));
}

test "goldilocks mul reduction" {
    const t = std.testing;
    // (p-1)^2 mod p == 1
    const pm1 = Goldilocks.fromU64(Goldilocks.p - 1);
    try t.expectEqual(Goldilocks.one, pm1.mul(pm1));
    // big · big
    const big = Goldilocks.fromU64(Goldilocks.p - 2);
    const r = big.mul(big);
    // (p-2)^2 = p^2 -4p + 4 ≡ p - 4 + ... verify against u128 math
    const expected: u128 = (@as(u128, Goldilocks.p) - 2) * (@as(u128, Goldilocks.p) - 2);
    try t.expectEqual(@as(u64, @intCast(expected % Goldilocks.p)), r.toU64());
    // commutativity spot-check
    try t.expect(r.eql(big.mul(big)));
}

test "goldilocks inv and pow" {
    const t = std.testing;
    const a = Goldilocks.fromU64(0xDEADBEEF);
    const inv = try a.inv();
    try t.expectEqual(Goldilocks.one, a.mul(inv));
    try t.expectError(error.ZeroInverse, Goldilocks.zero.inv());
    try t.expectEqual(Goldilocks.fromU64(1), Goldilocks.one.pow(0));
    // a^p == a (Fermat)
    try t.expect(a.pow(Goldilocks.p).eql(a));
}

test "goldilocks negative and i64" {
    const t = std.testing;
    const neg1 = Goldilocks.fromI64(-1);
    try t.expectEqual(Goldilocks.fromU64(Goldilocks.p - 1), neg1);
    // a + (-a) == 0
    const a = Goldilocks.fromI64(-42);
    const b = Goldilocks.fromI64(42);
    try t.expectEqual(Goldilocks.zero, a.add(b));
    try t.expectEqual(neg1.mul(neg1), Goldilocks.one);
}

test "goldilocks bytes roundtrip" {
    const t = std.testing;
    var buf: [8]u8 = undefined;
    const a = Goldilocks.fromU64(0x0123456789ABCDEF % Goldilocks.p);
    a.toBytes(&buf);
    try t.expectEqual(a, Goldilocks.fromBytes(&buf));
}
