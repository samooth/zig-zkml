//! Single boundary between the adapter and the raw C ABI pointer types.
//!
//! Both `gguf_attestation` and `witness_hooks` create handles through here so
//! the const-correctness rules around the allocator argument live in exactly
//! one place.

const std = @import("std");
const zkml = @import("zig_zkml");

pub const api = zkml.api;

/// The ABI entry points take `*anyopaque`; the handle dereferences it once and
/// stores a *copy* of the allocator value (`allocator: std.mem.Allocator`),
/// never the pointer. `gpa` must therefore be valid for the duration of the
/// ABI call — pass the address of a local, not of a temporary.
pub fn allocatorHandle(gpa: *const std.mem.Allocator) *anyopaque {
    return @ptrCast(@constCast(gpa));
}

pub fn ok(rc: i32) bool {
    return rc == 0;
}
