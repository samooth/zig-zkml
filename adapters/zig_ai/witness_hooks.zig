//! Witness recording bridge for the zig-ai engine adapter (Stage 2).
//!
//! zig-ai's engine contract (`src/engine_api/contract.zig`) exposes
//! `MetricHooks { on_layer: ?*const fn (*const LayerMetrics) void, ... }`.
//! Two properties of that contract shape this bridge:
//!
//!   1. The callbacks carry **no userdata** — a hook cannot be a bound method.
//!      The bridge therefore keeps the active session in a `threadlocal`
//!      slot. Inference runs its layer loop on one thread, so thread-local is
//!      both sufficient and race-free for begin/end/finalize; `record_op`
//!      itself is already thread-safe inside the core (TraceRecorder).
//!   2. `on_layer` fires **after** the layer forward completes. With that
//!      alone the bridge still produces a valid trace (one metrics op per
//!      layer, self-contained begin/end). An engine that also records real op
//!      payloads opens the layer itself with `beginLayer`/`endLayer` around
//!      the forward and the hook's entry lands inside it.
//!
//! `LayerMetricsView` mirrors zig-ai's `LayerMetrics` field-for-field
//! (`extern struct`, same order); `layoutMatches` asserts it at comptime in
//! the consuming build so a future upstream reorder fails loudly.

const std = @import("std");
const zkml = @import("zig_zkml");
const abi = @import("abi.zig");

const api = abi.api;
const trace = zkml.trace;

pub const SlotKey = struct {
    layer: u32,
    op: trace.Op,
    expert: u16 = 0,
    rank: u8 = 0,
};

pub const WitnessError = error{
    OutOfMemory,
    StateRejected,
    PayloadRejected,
    FinalizeRejected,
};

/// Owns a `ZKML_Witness` handle and mirrors its layer state machine so the
/// engine cannot desynchronize begin/end.
pub const Recorder = struct {
    gpa: std.mem.Allocator,
    handle: *api.ZKML_Witness,
    open_layer: ?u32 = null,

    pub fn init(gpa: std.mem.Allocator) WitnessError!Recorder {
        const handle = api.zkml_witness_session_create(abi.allocatorHandle(&gpa)) orelse
            return error.OutOfMemory;
        return .{ .gpa = gpa, .handle = handle };
    }

    pub fn deinit(self: *Recorder) void {
        api.zkml_witness_session_destroy(self.handle);
        self.* = undefined;
    }

    pub fn beginLayer(self: *Recorder, layer: u32) WitnessError!void {
        if (api.zkml_witness_begin_layer(self.handle, layer) != 0) return error.StateRejected;
        self.open_layer = layer;
    }

    pub fn endLayer(self: *Recorder) WitnessError!void {
        if (api.zkml_witness_end_layer(self.handle) != 0) return error.StateRejected;
        self.open_layer = null;
    }

    pub fn record(self: *Recorder, key: SlotKey, payload: []const u8) WitnessError!void {
        const c_key = api.ZKML_SlotKey{
            .layer = key.layer,
            .expert = key.expert,
            .op = @intCast(@intFromEnum(key.op)),
            .rank = key.rank,
        };
        if (api.zkml_witness_record_op(self.handle, &c_key, payload.ptr, payload.len) != 0) {
            return error.PayloadRejected;
        }
    }

    /// Record into the currently open layer; caller must have begun it.
    pub fn recordOpen(self: *Recorder, op: trace.Op, payload: []const u8) WitnessError!void {
        const layer = self.open_layer orelse return error.StateRejected;
        return self.record(.{ .layer = layer, .op = op }, payload);
    }

    pub fn finalize(self: *Recorder, stmt_hash: [32]u8) WitnessError![32]u8 {
        if (self.open_layer != null) return error.StateRejected;
        var out: [32]u8 = undefined;
        if (api.zkml_witness_finalize(self.handle, &stmt_hash, &out) != 0) {
            return error.FinalizeRejected;
        }
        return out;
    }
};

/// Field-for-field mirror of zig-ai `engine_api.LayerMetrics`.
pub const LayerMetricsView = extern struct {
    layer_idx: usize,
    forward_ms: f64,
    attention_ms: f64,
    ffn_ms: f64,
    memory_kb: f64,
    is_attention: bool,
};

/// Comptime guard for the engine-side glue: fails if zig-ai's `LayerMetrics`
/// stops being layout-compatible with this mirror.
pub fn layoutMatches(comptime Metrics: type) void {
    if (@sizeOf(Metrics) != @sizeOf(LayerMetricsView)) @compileError("LayerMetrics size mismatch");
    if (@offsetOf(Metrics, "layer_idx") != @offsetOf(LayerMetricsView, "layer_idx")) @compileError("layer_idx offset mismatch");
    if (@offsetOf(Metrics, "forward_ms") != @offsetOf(LayerMetricsView, "forward_ms")) @compileError("forward_ms offset mismatch");
    if (@offsetOf(Metrics, "attention_ms") != @offsetOf(LayerMetricsView, "attention_ms")) @compileError("attention_ms offset mismatch");
    if (@offsetOf(Metrics, "ffn_ms") != @offsetOf(LayerMetricsView, "ffn_ms")) @compileError("ffn_ms offset mismatch");
    if (@offsetOf(Metrics, "memory_kb") != @offsetOf(LayerMetricsView, "memory_kb")) @compileError("memory_kb offset mismatch");
    if (@offsetOf(Metrics, "is_attention") != @offsetOf(LayerMetricsView, "is_attention")) @compileError("is_attention offset mismatch");
}

threadlocal var active: ?*Recorder = null;

/// Bind `rec` to the calling thread's hook callbacks. The engine installs it
/// for the lifetime of one inference session and calls `uninstall` before
/// the session's recorder goes out of scope.
pub fn install(rec: *Recorder) void {
    active = rec;
}

pub fn uninstall() void {
    active = null;
}

pub fn activeRecorder() ?*Recorder {
    return active;
}

/// Plugs into `MetricHooks.on_layer` via a pointer cast (the engine's
/// `LayerMetrics*` is layout-identical; see `layoutMatches`).
///
/// Records the layer's metrics as a `ZKML_OP_OTHER` entry:
///   - no layer open  -> self-contained begin/record/end for that layer;
///   - layer open     -> recorded into it, left open for the engine.
pub fn onLayer(m: *const LayerMetricsView) void {
    const rec = active orelse return;
    const layer: u32 = @intCast(m.layer_idx);

    // A layer left open by a previous, unfinished forward is closed first.
    if (rec.open_layer) |open| {
        if (open != layer) rec.endLayer() catch return;
    }

    // If the engine did not open this layer itself, the entry is
    // self-contained: open, record, close.
    const self_opened = rec.open_layer == null;
    if (self_opened) rec.beginLayer(layer) catch return;

    rec.record(.{ .layer = layer, .op = .other }, std.mem.asBytes(m)) catch return;

    if (self_opened) rec.endLayer() catch return;
}
