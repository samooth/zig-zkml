//! zig-ai-facing glue for the zig-zkml adapter (Stage 2).
//!
//! This file is NOT compiled inside zig-zkml: it imports zig-ai's `gguf`
//! module, which only exists inside the engine's build graph. It is
//! compile-ready glue to be compiled by zig-ai once the build patch in
//! `README.md` is applied (that patch creates this module with `gguf` and
//! `zig_zkml` mapped). The adapter internals it re-exports are compiled and
//! tested in this repo through `test_adapter.zig`.
//!
//! Wire-up at the call site (src/main.zig, after `GgufModel.load`):
//!
//!     var snap = try adapter.GgufSnapshot.init(gpa, &model.file);
//!     defer snap.deinit();
//!     const att = try attestation.attestSource(gpa, snap.source());
//!     // publish att.root (hex) next to the model / in the serving response

const std = @import("std");
const gguf = @import("gguf");
const zkml = @import("zig_zkml");

const attestation = @import("gguf_attestation.zig");
const witness_hooks = @import("witness_hooks.zig");

pub const GgufSnapshot = struct {
    gpa: std.mem.Allocator,
    file: *const gguf.GgufFile,
    /// Tensor names in one stable order. The keys borrow the mmap-backed
    /// `GgufFile.data`, so only the slice of slices is owned here.
    names: []const []const u8,

    pub fn init(gpa: std.mem.Allocator, file: *const gguf.GgufFile) !GgufSnapshot {
        var list = std.ArrayList([]const u8).empty;
        errdefer list.deinit(gpa);

        var it = file.tensors.iterator();
        while (it.next()) |entry| {
            try list.append(gpa, entry.key_ptr.*);
        }
        return .{
            .gpa = gpa,
            .file = file,
            .names = try list.toOwnedSlice(gpa),
        };
    }

    pub fn deinit(self: *GgufSnapshot) void {
        self.gpa.free(self.names);
        self.* = undefined;
    }

    pub fn source(self: *GgufSnapshot) attestation.TensorSource {
        return .{
            .ctx = @ptrCast(self),
            .countFn = struct {
                fn f(ctx: *anyopaque) usize {
                    const s: *GgufSnapshot = @ptrCast(@alignCast(ctx));
                    return s.names.len;
                }
            }.f,
            .nameFn = struct {
                fn f(ctx: *anyopaque, i: usize) []const u8 {
                    const s: *GgufSnapshot = @ptrCast(@alignCast(ctx));
                    return s.names[i];
                }
            }.f,
            .dataFn = struct {
                fn f(ctx: *anyopaque, i: usize) []const u8 {
                    const s: *GgufSnapshot = @ptrCast(@alignCast(ctx));
                    const info = s.file.getTensor(s.names[i]) orelse return &[_]u8{};
                    return s.file.tensorData(info);
                }
            }.f,
        };
    }
};

/// Compile-time proof that the engine's metrics struct still matches the
/// adapter's mirror; call it once from the engine glue.
pub fn assertLayerMetricsLayout(comptime Metrics: type) void {
    witness_hooks.layoutMatches(Metrics);
}

/// `MetricHooks` for the engine contract. `on_layer` receives a
/// `*const LayerMetrics`, so the pointer is cast to the layout-identical
/// mirror; `layoutMatches` (above) makes an upstream field reorder a
/// compile error rather than silent corruption.
pub fn metricHooks() struct {
    on_layer: *const fn (*const witness_hooks.LayerMetricsView) void,
} {
    return .{ .on_layer = &witness_hooks.onLayer };
}
