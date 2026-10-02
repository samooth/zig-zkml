//! F_{p^2} = F_p[i] with i^2 = -1, over Goldilocks p = 2^61 - 1.
//!
//! docs/BLUE_PRINT.md §4/§12 (post-spikes): the FRI domain lives in F_{p^2}:
//!   - p ≡ 3 (mod 4) => -1 is a quadratic non-residue => F_{p^2} = F_p[i]
//!   - the norm-1 torus has order p + 1 = 2^61 (a pure power of two):
//!     a 2-adic multiplicative subgroup closed under negation, with
//!     -1 = g^(2^60) the unique order-2 element.
//! Squaring x -> x^2 maps the order-2^k subgroup 2-to-1 onto the
//! order-2^(k-1) subgroup — exactly the FRI fold structure.
//!
//! Elements of norm 1 (a^2 + b^2 = 1 mod p) are closed under
//! multiplication and negation; conjugation is inversion on the torus.

const std = @import("std");
const field = @import("../field.zig");

pub const Goldilocks = field.Goldilocks;

// Comptime search (generator of the 2^61 torus in domain.zig) runs many
// field operations; raise the branch quota accordingly.
comptime {
    @setEvalBranchQuota(500_000);
}

/// F_p[i]: element = a + b·i, stored as two F_p values.
pub const Fp2 = struct {
    /// The 2-adicity the pin's FRI reads as "how big a 2-power subgroup can
    /// this field hold", defined upstream as `v2(p-1) + v2(p+1)`.
    ///
    /// For p = 2^61 - 1 that is 1 + 61 = 62, and the order-2^61 torus sits
    /// inside it. It is a property of the FIELD, computed from the prime, not a
    /// tuning knob: `tools/fri_conv_diff.sh` asserts it against the pin's own
    /// value, so a wrong constant here is caught rather than believed.
    ///
    /// It exists because the pin's `proveOn` guards `log_domain` with
    /// `F.two_adicity`, and without the member our own Fp2 is not a field it
    /// can be instantiated over. Adding the member does not make the two
    /// implementations the same code; `fri-diff` is what says they behave the
    /// same, and it says so by measurement.
    pub const two_adicity: usize = 62;

    a: Goldilocks,
    b: Goldilocks,

    pub const zero: Fp2 = .{ .a = Goldilocks.zero, .b = Goldilocks.zero };
    pub const one: Fp2 = .{ .a = Goldilocks.one, .b = Goldilocks.zero };

    pub fn re(x: Goldilocks) Fp2 {
        return .{ .a = x, .b = Goldilocks.zero };
    }

    pub fn fromRaw(a: u64, b: u64) Fp2 {
        return .{ .a = Goldilocks.fromU64(a), .b = Goldilocks.fromU64(b) };
    }

    /// The integer constructor the pin's field trait exposes: the value in the
    /// real part, imaginary part zero. Identical to `re`, and that is the point
    /// — the pin's is `fromBase(BaseField.fromInt(x))`, so both put nothing in
    /// the imaginary part.
    pub fn fromInt(x: anytype) Fp2 {
        return re(Goldilocks.fromU64(@intCast(x)));
    }

    /// Division, unchecked — matching the contract of the pin's field, which is
    /// `self.mul(other.inv())` and has a separate `divChecked` for the checked
    /// case. Our `inv` is the checked one, so a zero divisor would be an error
    /// here and the pin has no way to see it.
    ///
    /// It is left unchecked because the only caller is the pin's FRI computing
    /// `1/2`, and 2 is invertible in an odd characteristic: the branch is
    /// unreachable for that caller. It is NOT a general-purpose division and
    /// should not be used as one — `divChecked` below is that, and it exists so
    /// the unchecked name is a deliberate choice rather than the only one.
    pub fn div(x: Fp2, y: Fp2) Fp2 {
        return x.mul(y.inv() catch zero);
    }

    /// Division that says so when the divisor is zero.
    pub fn divChecked(x: Fp2, y: Fp2) error{ZeroInverse}!Fp2 {
        return x.mul(try y.inv());
    }

    pub fn eql(x: Fp2, y: Fp2) bool {
        return x.a.eql(y.a) and x.b.eql(y.b);
    }

    pub fn isZero(x: Fp2) bool {
        return x.a.isZero() and x.b.isZero();
    }

    pub fn add(x: Fp2, y: Fp2) Fp2 {
        return .{ .a = x.a.add(y.a), .b = x.b.add(y.b) };
    }

    pub fn sub(x: Fp2, y: Fp2) Fp2 {
        return .{ .a = x.a.sub(y.a), .b = x.b.sub(y.b) };
    }

    pub fn neg(x: Fp2) Fp2 {
        return .{ .a = x.a.neg(), .b = x.b.neg() };
    }

    /// (a + b·i)(c + d·i) = (ac - bd) + (ad + bc)·i.
    pub fn mul(x: Fp2, y: Fp2) Fp2 {
        const ac = x.a.mul(y.a);
        const bd = x.b.mul(y.b);
        const ad = x.a.mul(y.b);
        const bc = x.b.mul(y.a);
        return .{ .a = ac.sub(bd), .b = ad.add(bc) };
    }

    pub fn sqr(x: Fp2) Fp2 {
        return x.mul(x);
    }

    /// Scalar mul by F_p element.
    pub fn mulReal(x: Fp2, s: Goldilocks) Fp2 {
        return .{ .a = x.a.mul(s), .b = x.b.mul(s) };
    }

    /// Complex conjugate: a + b·i -> a - b·i.
    pub fn conj(x: Fp2) Fp2 {
        return .{ .a = x.a, .b = x.b.neg() };
    }

    /// Norm N(x) = x·conj(x) = a^2 + b^2 ∈ F_p.
    pub fn norm(x: Fp2) Goldilocks {
        return x.a.mul(x.a).add(x.b.mul(x.b));
    }

    /// Trace Tr(x) = x + conj(x) = 2a ∈ F_p.
    pub fn trace(x: Fp2) Goldilocks {
        return x.a.add(x.a);
    }

    pub fn pow(x: Fp2, e: u64) Fp2 {
        var result = one;
        var base = x;
        var exp = e;
        while (exp > 0) {
            if (exp & 1 == 1) result = result.mul(base);
            base = base.sqr();
            exp >>= 1;
        }
        return result;
    }

    /// Inverse via norm: x^-1 = conj(x) / N(x).
    pub fn inv(x: Fp2) error{ZeroInverse}!Fp2 {
        const n = x.norm();
        const n_inv = try n.inv();
        return x.conj().mulReal(n_inv);
    }

    pub fn random(rng: std.Random) Fp2 {
        return .{
            .a = Goldilocks.fromU64(rng.int(u64)),
            .b = Goldilocks.fromU64(rng.int(u64)),
        };
    }

    /// 16-byte little-endian encoding (a || b).
    pub fn toBytes(x: Fp2) [16]u8 {
        var out: [16]u8 = undefined;
        x.a.toBytes(out[0..8]);
        x.b.toBytes(out[8..16]);
        return out;
    }

    pub const NUM_BYTES: usize = 16;

    /// Parse 16 LE bytes; rejects non-canonical encodings (a or b >= p) so
    /// transcript rejection sampling works.
    pub fn fromBytes(bytes: []const u8) error{ InvalidLength, OutOfField }!Fp2 {
        if (bytes.len != NUM_BYTES) return error.InvalidLength;
        const a = Goldilocks.fromU64(std.mem.readInt(u64, bytes[0..8], .little));
        const b = Goldilocks.fromU64(std.mem.readInt(u64, bytes[8..16], .little));
        // fromU64 reduces mod p, hiding non-canonical encodings — check
        // the raw values instead so rejection sampling is uniform.
        const ra = std.mem.readInt(u64, bytes[0..8], .little);
        const rb = std.mem.readInt(u64, bytes[8..16], .little);
        if (ra >= Goldilocks.p or rb >= Goldilocks.p) return error.OutOfField;
        return .{ .a = a, .b = b };
    }
};

/// i (i^2 = -1): -1 is a non-residue for p ≡ 3 (mod 4), so x^2 + 1 is
/// irreducible — verified by a comptime assertion.
pub const imaginary_unit: Fp2 = .{ .a = Goldilocks.zero, .b = Goldilocks.one };

comptime {
    // p mod 4 must be 3 for F_p[i] to be a field.
    std.debug.assert((Goldilocks.p & 3) == 3);
    // (0 + 1i)^2 = -1 + 0i.
    const ii = imaginary_unit.sqr();
    std.debug.assert(ii.b.isZero());
    std.debug.assert(ii.a.eql(Goldilocks.zero.sub(Goldilocks.one)));
}

test "fp2 field axioms" {
    const t = std.testing;
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    for (0..32) |_| {
        const x = Fp2.random(rng);
        const y = Fp2.random(rng);
        const z = Fp2.random(rng);
        if (x.isZero() or y.isZero() or z.isZero()) continue;

        // commutativity / associativity / distributivity
        try t.expect(x.add(y).eql(y.add(x)));
        try t.expect(x.mul(y).eql(y.mul(x)));
        try t.expect(x.add(y).add(z).eql(x.add(y.add(z))));
        try t.expect(x.mul(y).mul(z).eql(x.mul(y.mul(z))));
        try t.expect(x.mul(y.add(z)).eql(x.mul(y).add(x.mul(z))));

        // identity / inverse
        try t.expect(x.add(Fp2.zero).eql(x));
        try t.expect(x.mul(Fp2.one).eql(x));
        try t.expect(x.mul((try x.inv())).eql(Fp2.one));
        try t.expect(x.sub(x).isZero());

        // conj / norm: N(x) in F_p, N(x·y) = N(x)·N(y)
        const n = x.norm();
        try t.expect(n.inv() != error.ZeroInverse or !n.isZero());
        try t.expect(x.norm().mul(y.norm()).eql(x.mul(y).norm()));
    }
}

test "fp2 i^2 = -1" {
    const t = std.testing;
    const ii = imaginary_unit.sqr();
    try t.expect(ii.eql(Fp2.re(Goldilocks.zero.sub(Goldilocks.one))));
}

test "fp2 norm of torus elements" {
    // Elements of norm 1 form a group under multiplication (the torus):
    // x^(p-1) has norm 1 for any x != 0.
    const t = std.testing;
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const one_g = Goldilocks.one;

    for (0..16) |_| {
        const x = Fp2.random(rng);
        if (x.isZero()) continue;
        const u = x.pow(Goldilocks.p - 1);
        try t.expect(u.norm().eql(one_g));
        // the torus is closed under multiplication
        const v = Fp2.random(rng);
        if (v.isZero()) continue;
        const w = v.pow(Goldilocks.p - 1);
        try t.expect(u.mul(w).norm().eql(one_g));
    }
}
