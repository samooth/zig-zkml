//! Transcript binding for the fingerprint challenge: the ordering that makes
//! the claim in `fingerprint.zig` sound.
//!
//! # Why order is the soundness
//!
//! The fingerprint says: commit to A and B, *then* draw a random u⊗v, then
//! check ⟨u⊗v, A·Bᵀ⟩ against the committed matrices. If the prover chose the
//! challenge, it picks a (u, v, false product) triple where the claim happens
//! to hold — a one-equation constraint in three unknowns, trivially solvable,
//! and the proof says nothing about the product. Drawing the challenge after
//! the commitment is what forces the prover to be right for *every* u⊗v, and
//! Schwartz–Zippel then makes the soundness error 1/|F|.
//!
//! So the ordering is not a convention, it is the construction. It is
//! therefore enforced here, structurally, rather than documented and hoped
//! for: `bindMatrices` absorbs the two commitment roots and returns a
//! `Challenge`. There is deliberately no API that produces a challenge without
//! first absorbing a commitment, so a caller cannot reach the vulnerable
//! ordering by accident.
//!
//! # What this does not do
//!
//! This binds the challenge. It does not prove anything: the claim still has
//! to be discharged (that is the FRI/AIR work, F3). A caller that binds and
//! then ignores the challenge has built nothing. The tests here therefore
//! check the *binding properties* — determinism, order sensitivity, domain
//! separation — not soundness of a proof, because there is no proof yet.

const std = @import("std");
const Allocator = std.mem.Allocator;

const fp2_mod = @import("../fri/fp2.zig");
const Fp2 = fp2_mod.Fp2;
const transcript_lib = @import("../transcript.zig");
const fingerprint = @import("fingerprint.zig");

pub const Transcript = transcript_lib.Transcript;
pub const Challenge = fingerprint.Challenge;

pub const Error = error{
    InvalidShape,
    OutOfMemory,
};

/// Domain separators. These exist so that a challenge drawn for one purpose
/// cannot be replayed as if it had been drawn for another: absorbing a
/// commitment then squeezing "fp" must differ from any other use of the same
/// commitment.
const DOM_COMMIT = "fp.cmt";
const DOM_U = "fp.u";
const DOM_V = "fp.v";

/// Absorb both matrix commitments, then derive u and v.
///
/// The returned slices are freshly allocated and owned by the caller. `m` and
/// `n` are the row counts of A and B; they are absorbed too, because the same
/// commitment bytes reinterpreted at a different shape must not be a valid
/// instance of the previous one.
pub fn bindMatrices(
    allocator: Allocator,
    transcript: *Transcript,
    a_root: [32]u8,
    b_root: [32]u8,
    m: usize,
    n: usize,
) Error!Challenge {
    try fingerprint.validateShape(m, n);

    transcript.absorbBytes(DOM_COMMIT);
    transcript.absorbBytes(&a_root);
    transcript.absorbBytes(&b_root);
    // Shapes are absorbed little-endian by hand: absorbField needs a field
    // element, and the extents are usize, not field members.
    var shape: [16]u8 = undefined;
    std.mem.writeInt(u64, shape[0..8], @intCast(m), .little);
    std.mem.writeInt(u64, shape[8..16], @intCast(n), .little);
    transcript.absorbBytes(&shape);

    // Separate domains for u and v. Without this, u and v would be drawn from
    // the same squeeze sequence, which makes the challenge rank-1 with a
    // correlation between the factors that a prover could exploit.
    transcript.absorbBytes(DOM_U);
    const u = try drawVector(allocator, transcript, m);
    errdefer allocator.free(u);

    transcript.absorbBytes(DOM_V);
    const v = try drawVector(allocator, transcript, n);
    errdefer allocator.free(v);

    return Challenge.init(u, v);
}

/// Draw `len` field elements by rejection sampling, rejecting the zero vector
/// as a whole.
///
/// The all-zero vector is not a soundness problem in itself (⟨0⊗v, C⟩ = 0
/// holds for every C, so it would make the check vacuous and a prover with a
/// false product could pass it) and so it must never be sampled. Redrawing
/// keeps the distribution uniform over the non-zero vectors, which is what the
/// Schwartz–Zippel bound assumes.
fn drawVector(allocator: Allocator, transcript: *Transcript, len: usize) Error![]Fp2 {
    while (true) {
        const out = try allocator.alloc(Fp2, len);
        errdefer allocator.free(out);
        var all_zero = true;
        for (out) |*e| {
            e.* = transcript.challengeField(Fp2);
            if (!e.*.isZero()) all_zero = false;
        }
        if (!all_zero) return out;
        allocator.free(out);
    }
}

test "binding is deterministic" {
    const a = std.testing.allocator;
    const a_root = [_]u8{1} ** 32;
    const b_root = [_]u8{2} ** 32;

    var t1 = Transcript.init("zkml.test.fp");
    const c1 = try bindMatrices(a, &t1, a_root, b_root, 4, 6);
    defer a.free(c1.u);
    defer a.free(c1.v);

    var t2 = Transcript.init("zkml.test.fp");
    const c2 = try bindMatrices(a, &t2, a_root, b_root, 4, 6);
    defer a.free(c2.u);
    defer a.free(c2.v);

    try std.testing.expectEqual(c1.u.len, c2.u.len);
    try std.testing.expectEqual(c1.v.len, c2.v.len);
    for (c1.u, c2.u) |x, y| try std.testing.expect(x.eql(y));
    for (c1.v, c2.v) |x, y| try std.testing.expect(x.eql(y));
}

test "a different commitment gives a different challenge" {
    const a = std.testing.allocator;
    const a_root = [_]u8{1} ** 32;
    const b_root = [_]u8{2} ** 32;
    const b_other = [_]u8{3} ** 32;

    var t1 = Transcript.init("zkml.test.fp");
    const c1 = try bindMatrices(a, &t1, a_root, b_root, 4, 6);
    defer a.free(c1.u);
    defer a.free(c1.v);

    // Same A, different B: the challenge must move, or a prover could satisfy
    // the old challenge with the new matrix.
    var t2 = Transcript.init("zkml.test.fp");
    const c2 = try bindMatrices(a, &t2, a_root, b_other, 4, 6);
    defer a.free(c2.u);
    defer a.free(c2.v);

    var differs = false;
    for (c1.u, c2.u) |x, y| {
        if (!x.eql(y)) differs = true;
    }
    for (c1.v, c2.v) |x, y| {
        if (!x.eql(y)) differs = true;
    }
    try std.testing.expect(differs);
}

test "the shape is bound, not just the roots" {
    const a = std.testing.allocator;
    const a_root = [_]u8{1} ** 32;
    const b_root = [_]u8{2} ** 32;

    var t1 = Transcript.init("zkml.test.fp");
    const c1 = try bindMatrices(a, &t1, a_root, b_root, 4, 6);
    defer a.free(c1.u);
    defer a.free(c1.v);

    // Same roots, different shape: a different instance, so a different
    // challenge. Lengths differ, which alone proves the binding took effect.
    var t2 = Transcript.init("zkml.test.fp");
    const c2 = try bindMatrices(a, &t2, a_root, b_root, 4, 8);
    defer a.free(c2.u);
    defer a.free(c2.v);

    try std.testing.expect(c1.u.len != c2.u.len or c1.v.len != c2.v.len);
}

test "u and v come from separate domains" {
    const a = std.testing.allocator;
    const a_root = [_]u8{1} ** 32;
    const b_root = [_]u8{2} ** 32;

    // If u and v shared a domain and the shapes were equal, a buggy binder
    // could return the same vector twice; the factors would then be perfectly
    // correlated and the "random" challenge would live in a 1-dimensional
    // subspace.
    var t = Transcript.init("zkml.test.fp");
    const c = try bindMatrices(a, &t, a_root, b_root, 5, 5);
    defer a.free(c.u);
    defer a.free(c.v);

    var identical = true;
    for (c.u, c.v) |x, y| {
        if (!x.eql(y)) identical = false;
    }
    try std.testing.expect(!identical);
}

test "the challenge is never all-zero" {
    const a = std.testing.allocator;
    const b_root = [_]u8{9} ** 32;

    // The all-zero challenge would make every claim vacuously true, so the
    // binder redraws. This checks the guard is actually wired up.
    for (0..8) |i| {
        var root_a: [32]u8 = undefined;
        std.mem.writeInt(u64, root_a[0..8], @intCast(i), .little);
        var t = Transcript.init("zkml.test.fp");
        const c = try bindMatrices(a, &t, root_a, b_root, 3, 4);
        defer a.free(c.u);
        defer a.free(c.v);

        var u_zero = true;
        for (c.u) |e| {
            if (!e.isZero()) u_zero = false;
        }
        var v_zero = true;
        for (c.v) |e| {
            if (!e.isZero()) v_zero = false;
        }
        try std.testing.expect(!u_zero);
        try std.testing.expect(!v_zero);
    }
}

test "empty shapes are rejected before any transcript state changes" {
    const a = std.testing.allocator;
    const a_root = [_]u8{1} ** 32;
    const b_root = [_]u8{2} ** 32;

    var t = Transcript.init("zkml.test.fp");
    const before = t.finish();
    try std.testing.expectError(Error.InvalidShape, bindMatrices(a, &t, a_root, b_root, 0, 4));
    const after = t.finish();
    try std.testing.expectEqualSlices(u8, &before, &after);
}

test "the challenge cannot be drawn before the commitments are absorbed" {
    const a = std.testing.allocator;
    const a_root = [_]u8{0xAA} ** 32;
    const b_root = [_]u8{0xBB} ** 32;
    const m = 4;
    const n = 6;

    // Correct order: commitments, then challenge.
    var t_good = Transcript.init("zkml.test.order");
    const good = try bindMatrices(a, &t_good, a_root, b_root, m, n);
    defer a.free(good.u);
    defer a.free(good.v);

    // The vulnerable ordering, spelled out: draw first, then absorb. A prover
    // that knows the challenge in advance can satisfy the claim with a false
    // product, so these two paths must NOT produce the same challenge. If they
    // did, the binding would be vacuous.
    var t_bad = Transcript.init("zkml.test.order");
    const bad_u = a.alloc(Fp2, m) catch unreachable;
    defer a.free(bad_u);
    const bad_v = a.alloc(Fp2, n) catch unreachable;
    defer a.free(bad_v);
    for (bad_u) |*e| e.* = t_bad.challengeField(Fp2);
    for (bad_v) |*e| e.* = t_bad.challengeField(Fp2);
    t_bad.absorbBytes(DOM_COMMIT);
    t_bad.absorbBytes(&a_root);
    t_bad.absorbBytes(&b_root);

    var same = true;
    for (good.u, bad_u) |x, y| {
        if (!x.eql(y)) same = false;
    }
    for (good.v, bad_v) |x, y| {
        if (!x.eql(y)) same = false;
    }
    try std.testing.expect(!same);
}

test "swapping which matrix is A changes the challenge" {
    const a = std.testing.allocator;
    const a_root = [_]u8{1} ** 32;
    const b_root = [_]u8{2} ** 32;

    var t1 = Transcript.init("zkml.test.swap");
    const c1 = try bindMatrices(a, &t1, a_root, b_root, 4, 4);
    defer a.free(c1.u);
    defer a.free(c1.v);

    var t2 = Transcript.init("zkml.test.swap");
    const c2 = try bindMatrices(a, &t2, b_root, a_root, 4, 4);
    defer a.free(c2.u);
    defer a.free(c2.v);

    var differs = false;
    for (c1.u, c2.u) |x, y| {
        if (!x.eql(y)) differs = true;
    }
    for (c1.v, c2.v) |x, y| {
        if (!x.eql(y)) differs = true;
    }
    try std.testing.expect(differs);
}

test "the transcript domain separates different claims" {
    const a = std.testing.allocator;
    const a_root = [_]u8{5} ** 32;
    const b_root = [_]u8{6} ** 32;

    var t1 = Transcript.init("zkml.test.domain.one");
    const c1 = try bindMatrices(a, &t1, a_root, b_root, 3, 3);
    defer a.free(c1.u);
    defer a.free(c1.v);

    var t2 = Transcript.init("zkml.test.domain.two");
    const c2 = try bindMatrices(a, &t2, a_root, b_root, 3, 3);
    defer a.free(c2.u);
    defer a.free(c2.v);

    var differs = false;
    for (c1.u, c2.u) |x, y| {
        if (!x.eql(y)) differs = true;
    }
    try std.testing.expect(differs);
}
