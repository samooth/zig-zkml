//! Every guard that was a pre-condition assertion and became a returned error.
//!
//! The conversion is the point; this file is what makes it a tested claim
//! instead of a comment. Each test names the exact error rather than
//! asserting "it fails", because `EmptyRows` and `BadKeep` are two different
//! bugs and a test that only checks that something went wrong passes just as
//! happily while being wrong.
//!
//! It also pins the reason all six moved together. A pre-condition assertion
//! is removed by ReleaseFast, so each of these was reachable in a release
//! build with no diagnostic at all — not a wrong answer, no answer. That is
//! the same defect `zig_algebra` v0.5.1 fixed in its own `Domain.init`, and
//! the reason its author wrote down "the old assert is compiled out in
//! ReleaseFast" next to the fix.

const std = @import("std");
const testing = std.testing;

const air_builder = @import("air_builder.zig");
const float_ref = @import("float_ref.zig");
const float_air = @import("float_air.zig");
const domain = @import("../fri/domain.zig");
const float_format = @import("float_format.zig");

test "air_builder.freeze: zero rows is EmptyRows, not an empty system" {
    const a = testing.allocator;
    var b = air_builder.Builder.init(a);
    defer b.deinit();
    try testing.expectError(
        air_builder.BuildError.EmptyRows,
        air_builder.freeze(a, &b, 0),
    );
}

test "air_builder.Trace.alloc: zero rows is EmptyRows" {
    try testing.expectError(
        air_builder.BuildError.EmptyRows,
        air_builder.Trace.alloc(testing.allocator, 3, 0),
    );
}

test "float_air.buildSystem: zero rows is EmptyRows" {
    try testing.expectError(
        float_air.BuildError.EmptyRows,
        float_air.buildSystem(testing.allocator, 0, float_format.bfloat16),
    );
}

test "float_ref.roundToNearestEven: keep out of range is BadKeep, not a shifted word" {
    // keep == 0 shifted by a full word; keep == 64 is an undefined shift. In
    // ReleaseFast both used to return a Rounded with no diagnostic.
    try testing.expectError(float_ref.Error.BadKeep, float_ref.roundToNearestEven(1, 0));
    try testing.expectError(float_ref.Error.BadKeep, float_ref.roundToNearestEven(1, 64));
    // The boundaries that must keep working, so the guard is not over-tight.
    _ = try float_ref.roundToNearestEven(1, 1);
    _ = try float_ref.roundToNearestEven(1, 63);
}

test "fri.Domain.init: log_n above the torus order is DomainTooLarge" {
    // torus_log_order is 61, and log_n is a u6, so 62 is reachable from a
    // prover config. The old assert disappeared in ReleaseFast and
    // `torus_log_order - log_n` wrapped, yielding a generator of the wrong
    // order — a proof that cannot verify, for a reason nobody could see.
    try testing.expectError(error.DomainTooLarge, domain.Domain.init(62));
    try testing.expectError(error.DomainTooLarge, domain.Domain.init(63));
    // 0 is the trivial subgroup and is explicitly valid, as is the boundary.
    _ = try domain.Domain.init(0);
    _ = try domain.Domain.init(61);
}
