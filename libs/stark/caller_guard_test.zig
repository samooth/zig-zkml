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
const widen_air = @import("widen_air.zig");
const chunk_binding = @import("chunk_binding.zig");
const quant_binding = @import("quant_binding.zig");
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

test "float_air.buildTraceShifted: no pairs is EmptyRows" {
    // Same shape as buildSystem, and a separate entry point — the two are
    // closed independently because a guard is only a claim about the site it
    // stands on.
    const no_pairs: [0][2]u16 = .{};
    try testing.expectError(
        float_air.BuildTraceError.EmptyRows,
        float_air.buildTraceShifted(testing.allocator, &no_pairs, float_format.bfloat16, 1),
    );
    // One pair is the boundary that must still build, so the guard is not
    // over-tight.
    const one_pair = [_][2]u16{.{ 0x3F80, 0x4000 }};
    var t = try float_air.buildTraceShifted(testing.allocator, &one_pair, float_format.bfloat16, 1);
    defer t.deinit(testing.allocator);
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

// --- the composed error sets -------------------------------------------------
//
// Composing `air_builder.BuildError` into three bindings was a signature
// change, and a signature change nothing can tumble has the same shape as
// everything the other six commits closed. So: two of these are runtime tests
// of a path that can actually be taken, and the third is honest about the
// fact that its case is currently unreachable.

test "widen_air.buildSystem: EmptyRows crosses the composed error set" {
    // `widen_air` forwards the caller's `rows` to `air_builder.freeze`, so the
    // zero-row case is reachable here and must arrive under its own name.
    try testing.expectError(
        air_builder.BuildError.EmptyRows,
        widen_air.buildSystem(testing.allocator, 0, float_format.bfloat16),
    );
}

test "widen_air.buildTrace: EmptyRows crosses the composed error set" {
    // Same for the trace path, which composes into `BuildTraceError`.
    const no_patterns: [0]u16 = .{};
    try testing.expectError(
        air_builder.BuildError.EmptyRows,
        widen_air.buildTrace(testing.allocator, &no_patterns, float_format.bfloat16),
    );
}

test "chunk_binding and quant_binding admit the composed case, though it is unreachable" {
    // Both call `air_builder.freeze(allocator, &b, 1)` with a literal 1, so
    // `EmptyRows` cannot be raised on those paths today. The composition is
    // still correct — it is what will let a future caller-supplied row count
    // through without a second round of signature surgery — and this test pins
    // the part of that claim which is checkable: the sets admit the value.
    //
    // What it deliberately does NOT do is pretend a runtime test happened.
    // There is no input that produces one, and a test that asserted one would
    // be asserting nothing.
    const from_builder: air_builder.BuildError = error.EmptyRows;
    comptime {
        // Fails to compile if either binding ever stops composing the
        // builder's set, which is the regression this composition exists to
        // prevent.
        const as_chunk: chunk_binding.BuildError = from_builder;
        const as_quant: quant_binding.BuildError = from_builder;
        const as_widen: widen_air.BuildError = from_builder;
        std.mem.doNotOptimizeAway(&as_chunk);
        std.mem.doNotOptimizeAway(&as_quant);
        std.mem.doNotOptimizeAway(&as_widen);
    }
}

// --- the class: every guard, asked whether it leaks ---------------------------
//
// Converting an assertion into a returned error converts an abort into an
// unwind path that did not exist before. An aborting process leaks nothing;
// an unwinding one leaks everything acquired ahead of the guard unless an
// `errdefer` says otherwise. So each conversion is a leak site, and the
// question is binary per guard: does it fire after an allocation?
//
// A grep cannot answer that — it counts `errdefer`s next to guards, which is
// a ratio, and `widen_air` had two `errdefer`s and still leaked. What answers
// it is running every guard through `testing.allocator`, which fails the
// test on a leak. That is the whole test: if a guard fires after an
// allocation and nothing releases it, the allocator reports it here.

test "every converted guard unwinds without leaking" {
    // One block, one allocator, every guard in turn. Kept together on
    // purpose: the claim is a property of the *set* of conversions, and a
    // test that only ran the first of them would pass just as happily.
    const a = testing.allocator;

    // Before any allocation: pure, so it cannot leak, and the test says so by
    // running it rather than by asserting that it cannot.
    try testing.expectError(float_ref.Error.BadKeep, float_ref.roundToNearestEven(1, 0));
    try testing.expectError(error.DomainTooLarge, domain.Domain.init(63));

    // Guards that fire before their first allocation.
    try testing.expectError(
        float_air.BuildError.EmptyRows,
        float_air.buildSystem(a, 0, float_format.bfloat16),
    );
    {
        const no_pairs: [0][2]u16 = .{};
        try testing.expectError(
            float_air.BuildTraceError.EmptyRows,
            float_air.buildTraceShifted(a, &no_pairs, float_format.bfloat16, 1),
        );
    }
    try testing.expectError(
        air_builder.BuildError.EmptyRows,
        widen_air.buildSystem(a, 0, float_format.bfloat16),
    );
    {
        const no_patterns: [0]u16 = .{};
        try testing.expectError(
            air_builder.BuildError.EmptyRows,
            widen_air.buildTrace(a, &no_patterns, float_format.bfloat16),
        );
    }

    // The two that reach an `errdefer`-protected allocation: the guard *is*
    // the allocating call, so a failure inside it must leave nothing behind.
    {
        var b = air_builder.Builder.init(a);
        defer b.deinit();
        try testing.expectError(air_builder.BuildError.EmptyRows, air_builder.freeze(a, &b, 0));
    }
    try testing.expectError(
        air_builder.BuildError.EmptyRows,
        air_builder.Trace.alloc(a, 3, 0),
    );
}
