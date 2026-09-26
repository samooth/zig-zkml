//! Normalization gadgets — RMSNorm and LayerNorm.
//!
//! docs/BLUE_PRINT.md §4.3: sums are proven by LogUp of ranges (byte lookups).
//!
//! # NOT IMPLEMENTED — this module is a compile-time trap, on purpose
//!
//! What was here before was worse than nothing: two gadget structs that
//! declared columns and then emitted a single `degree = 1` constraint with no
//! expression behind it. That compiles, satisfies the `&gadgets.norm...
//! .airFragment` reference in zkml.zig, and would let a caller believe a
//! normalization was verified when nothing was. The SiLU table had the same
//! shape of problem and it survived because nothing ever called it.
//!
//! So the entry points now fail to compile until the work is real. Turning
//! them back into functions is the signal that there is an implementation to
//! test; a stub that returns a Fragment is not it.
//!
//! # What building this actually requires
//!
//! 1. A float reference for RMSNorm, to test against. `libs/stark/float_ref.zig`
//!    covers the multiply formats; there is no norm reference yet, and an AIR
//!    with nothing to check against is exactly the stub this replaces.
//! 2. The sum of squares by LogUp over the per-element magnitude bounds
//!    (`nonlin.rsqrt_q8_8` is already bound in the statement; the sum is not).
//! 3. The rsqrt lookup wired into the same fragment, with the zero-sum case
//!    rejected rather than reading `rsqrt_q8_8[0]`, which is 0 by construction.
//! 4. Negative tests: a wrong epsilon, a wrong weight scale, and a tampered
//!    sum must all be rejected.
//!
//! LayerNorm additionally needs the mean, which is a signed LogUp and a
//! subtler range proof than the sum of squares. It is not attempted before
//! RMSNorm lands.

const std = @import("std");
const tensor = @import("../../tensor/root.zig");
const air = @import("../../air/root.zig");
const nonlin = @import("../nonlin/root.zig");

pub const Goldilocks = tensor.Goldilocks;

/// The canonical rsqrt is already committed to the statement via
/// `gadgets.nonlin.digest()`, so the table this module will use is bound. It
/// is re-exported here so callers have one obvious import.
pub const rsqrt_q8_8 = &nonlin.SiLULookup.rsqrt_q8_8;

pub const NotImplemented = error{NotImplemented};

pub const RmsNormGadget = struct {
    rows: usize,
    cols: usize,
    weight: []const i8,

    /// Always fails. See the module header: there is no implementation, and a
    /// Fragment with one `degree = 1` constraint and no expression is a
    /// forgery, not a gadget.
    pub fn airFragment(self: @This(), gpa: std.mem.Allocator) NotImplemented!air.Fragment {
        _ = self;
        _ = gpa;
        return error.NotImplemented;
    }
};

pub const LayerNormGadget = struct {
    rows: usize,
    cols: usize,
    weight: []const i8,
    bias: []const i8,

    /// Always fails. See the module header.
    pub fn airFragment(self: @This(), gpa: std.mem.Allocator) NotImplemented!air.Fragment {
        _ = self;
        _ = gpa;
        return error.NotImplemented;
    }
};

test "the norm gadgets refuse to build a fragment" {
    // The previous version of this test asserted `frag.rows == 4` against a
    // fragment that contained no expressions at all. It passed, and it was
    // asserting that a stub had the right row count. The contract now is that
    // there is no fragment, and this is what pins that.
    const t = std.testing;
    const ln = LayerNormGadget{ .rows = 4, .cols = 128, .weight = &[_]i8{1} ** 128, .bias = &[_]i8{0} ** 128 };
    try t.expectError(error.NotImplemented, ln.airFragment(t.allocator));

    const rms = RmsNormGadget{ .rows = 4, .cols = 128, .weight = &[_]i8{1} ** 128 };
    try t.expectError(error.NotImplemented, rms.airFragment(t.allocator));
}

test "the rsqrt table the norm will use is already bound in the statement" {
    // RMSNorm is not built, but the table it will need is not floating: it is
    // covered by the nonlinearity digest the statement absorbs.
    const t = std.testing;
    try t.expectEqual(@as(usize, 256), rsqrt_q8_8.len);
    try t.expect(rsqrt_q8_8[1] > 0);
    // rsqrt(0) has no finite value and is 0 by construction; callers must
    // reject a zero sum of squares before indexing.
    try t.expectEqual(@as(i16, 0), rsqrt_q8_8[0]);
}
