//! Gate tests for the zig-ai adapter (Stage 2) — run by `zig build test`.
//!
//! The engine itself is not imported: `GgufSnapshot` (integration.zig) binds
//! the same `TensorSource` interface to zig-ai's `GgufFile`, so these tests
//! exercise the exact code path the engine takes, with a synthetic loader
//! table. What is covered:
//!
//!   - root is order-independent (hash-map iteration order is neither the
//!     GGUF directory order nor stable across runs);
//!   - one flipped tensor byte changes the root (negative, PLAN rule);
//!   - duplicate tensor names are rejected, not silently folded;
//!   - the adapter root equals the core attestor's root for the same tensors;
//!   - an inclusion proof round-trips through the standalone verifier;
//!   - GGUF tensor-name parsing matches zig-ai's documented conventions;
//!   - the `MetricHooks` bridge records a deterministic trace and respects
//!     an engine-opened layer.

const std = @import("std");
const zkml = @import("zig_zkml");

const testing = std.testing;
const abi = @import("abi.zig");
const attestation = @import("gguf_attestation.zig");
const hooks = @import("witness_hooks.zig");
const api = abi.api;
const trace = zkml.trace;

const Tensor = struct { name: []const u8, data: []const u8 };

/// Stand-in for `GgufFile.tensors` + `tensorData()`.
const Table = struct {
    items: []const Tensor,

    fn count(ctx: *anyopaque) usize {
        const t: *const Table = @ptrCast(@alignCast(ctx));
        return t.items.len;
    }

    fn name(ctx: *anyopaque, i: usize) []const u8 {
        const t: *const Table = @ptrCast(@alignCast(ctx));
        return t.items[i].name;
    }

    fn data(ctx: *anyopaque, i: usize) []const u8 {
        const t: *const Table = @ptrCast(@alignCast(ctx));
        return t.items[i].data;
    }

    fn source(self: *const Table) attestation.TensorSource {
        return .{
            .ctx = @constCast(self),
            .countFn = &count,
            .nameFn = &name,
            .dataFn = &data,
        };
    }
};

const fixture = [_]Tensor{
    .{ .name = "token_embd.weight", .data = "EMB-16-BYTES-OK!" },
    .{ .name = "blk.0.attn_q.weight", .data = "Q0" },
    .{ .name = "blk.0.ffn_down.weight", .data = "DOWN0-PADDED-XX" },
    .{ .name = "blk.7.mlp.up_proj.weight", .data = "UP7" },
    .{ .name = "output_norm.weight", .data = "NORM" },
};

test "adapter: root is independent of tensor iteration order" {
    var reversed = fixture;
    std.mem.reverse(Tensor, &reversed);

    const forward = Table{ .items = &fixture };
    const backward = Table{ .items = &reversed };

    const root_a = try attestation.attestSource(testing.allocator, forward.source());
    const root_b = try attestation.attestSource(testing.allocator, backward.source());

    try testing.expectEqual(@as(usize, fixture.len), root_a.tensor_count);
    try testing.expectEqualSlices(u8, &root_a.root, &root_b.root);
}

test "adapter: one flipped tensor byte changes the root" {
    const baseline_table = Table{ .items = &fixture };
    const baseline = try attestation.attestSource(testing.allocator, baseline_table.source());

    var corrupted: [fixture.len]Tensor = undefined;
    @memcpy(&corrupted, &fixture);
    const original = corrupted[1].data;
    var buf: [fixture[1].data.len]u8 = undefined;
    @memcpy(&buf, original);
    buf[0] ^= 0xFF;
    corrupted[1].data = &buf;

    const corrupted_table = Table{ .items = &corrupted };
    const after = try attestation.attestSource(testing.allocator, corrupted_table.source());
    try testing.expect(!std.mem.eql(u8, &baseline.root, &after.root));
}

test "adapter: duplicate tensor names are rejected" {
    const dup = [_]Tensor{
        .{ .name = "blk.0.attn_q.weight", .data = "A" },
        .{ .name = "blk.0.attn_q.weight", .data = "B" },
    };
    const table = Table{ .items = &dup };
    try testing.expectError(error.DuplicateName, attestation.attestSource(testing.allocator, table.source()));
}

test "adapter: root matches the core attestor over the same tensors" {
    const table = Table{ .items = &fixture };
    const via_adapter = try attestation.attestSource(testing.allocator, table.source());

    const handle = api.zkml_attestor_create(abi.allocatorHandle(&testing.allocator)) orelse
        return error.TestUnexpectedResult;
    defer api.zkml_attestor_destroy(handle);
    for (fixture) |item| {
        _ = api.zkml_attestor_add(handle, item.name.ptr, item.name.len, item.data.ptr, item.data.len);
    }
    _ = api.zkml_attestor_finish(handle);
    var root: [32]u8 = undefined;
    _ = api.zkml_attestor_root(handle, &root);

    try testing.expectEqualSlices(u8, &root, &via_adapter.root);
}

test "adapter: inclusion proof verifies through the standalone verifier" {
    const table = Table{ .items = &fixture };
    var session = try attestation.ProofSession.build(testing.allocator, table.source());
    defer session.deinit();

    const wire = try session.proof("blk.0.attn_q.weight");
    defer session.freeProof(wire);

    const root = try session.root();
    const rc = api.zkml_proof_verify(abi.allocatorHandle(&testing.allocator), wire.ptr, wire.len, &root);
    try testing.expectEqual(@as(i32, 0), rc);

    var wrong = root;
    wrong[0] ^= 0xFF;
    const bad_rc = api.zkml_proof_verify(abi.allocatorHandle(&testing.allocator), wire.ptr, wire.len, &wrong);
    try testing.expectEqual(@as(i32, -3), bad_rc);
}

test "adapter: gguf tensor-name parsing" {
    const cases = [_]struct { name: []const u8, layer: ?u32, role: attestation.Role }{
        .{ .name = "token_embd.weight", .layer = null, .role = .token_embd },
        .{ .name = "output.weight", .layer = null, .role = .output },
        .{ .name = "output_norm.weight", .layer = null, .role = .output_norm },
        .{ .name = "blk.0.attn_q.weight", .layer = 0, .role = .attn_q },
        .{ .name = "blk.12.ffn_down.bias", .layer = 12, .role = .ffn_down },
        .{ .name = "blk.3.ffn_norm.weight", .layer = 3, .role = .ffn_norm },
        .{ .name = "blk.7.mlp.gate_proj.weight", .layer = 7, .role = .ffn_gate },
        .{ .name = "blk.1.feed_forward.w2.weight", .layer = 1, .role = .ffn_down },
        .{ .name = "blk.4.rope_freqs.weight", .layer = 4, .role = .other },
        .{ .name = "some.unknown.tensor", .layer = null, .role = .other },
    };
    for (cases) |c| {
        const parsed = attestation.parseTensorName(c.name);
        try testing.expectEqual(c.layer, parsed.layer);
        try testing.expectEqual(c.role, parsed.role);
    }

    try testing.expectEqual(attestation.Category.attention, attestation.roleCategory(.attn_k));
    try testing.expectEqual(attestation.Category.ffn, attestation.roleCategory(.ffn_up));
    try testing.expectEqual(attestation.Category.norm, attestation.roleCategory(.attn_norm));
}

/// Two sessions fed the same per-layer metrics in the same order must agree.
fn hookTraceHash(stmt: [32]u8, layers: usize) ![32]u8 {
    var rec = try hooks.Recorder.init(testing.allocator);
    defer rec.deinit();
    hooks.install(&rec);
    defer hooks.uninstall();

    for (0..layers) |layer| {
        const m = hooks.LayerMetricsView{
            .layer_idx = layer,
            .forward_ms = 1.5,
            .attention_ms = 0.5,
            .ffn_ms = 1.0,
            .memory_kb = 128,
            .is_attention = (layer % 2) == 0,
        };
        hooks.onLayer(&m);
    }
    return rec.finalize(stmt);
}

test "adapter: witness hook records a deterministic per-layer trace" {
    const stmt = [_]u8{0x5A} ** 32;
    const first = try hookTraceHash(stmt, 4);
    const second = try hookTraceHash(stmt, 4);
    try testing.expectEqualSlices(u8, &first, &second);

    // A different layer count is a different trace.
    const longer = try hookTraceHash(stmt, 5);
    try testing.expect(!std.mem.eql(u8, &first, &longer));
}

test "adapter: hook records into an engine-opened layer without closing it" {
    var rec = try hooks.Recorder.init(testing.allocator);
    defer rec.deinit();
    hooks.install(&rec);
    defer hooks.uninstall();

    try rec.beginLayer(5);
    const m = hooks.LayerMetricsView{
        .layer_idx = 5,
        .forward_ms = 2.0,
        .attention_ms = 1.0,
        .ffn_ms = 1.0,
        .memory_kb = 256,
        .is_attention = true,
    };
    hooks.onLayer(&m);
    // The engine owns the boundary: still open.
    try testing.expectEqual(@as(?u32, 5), rec.open_layer);

    // A real op payload joins the same layer.
    try rec.recordOpen(.gemm_c, "GEMM-C-BYTES");
    try rec.endLayer();

    const stmt = [_]u8{0x11} ** 32;
    const hash = try rec.finalize(stmt);
    try testing.expect(!std.mem.eql(u8, &[_]u8{0} ** 32, &hash));

    // Frozen after finalize.
    try testing.expectError(error.StateRejected, rec.beginLayer(6));
    try testing.expectError(error.FinalizeRejected, rec.finalize(stmt));
}

test "adapter: hook is inert with no installed recorder" {
    // Must not crash or trap when the engine fires a hook outside a session.
    hooks.uninstall();
    const m = hooks.LayerMetricsView{
        .layer_idx = 0,
        .forward_ms = 0,
        .attention_ms = 0,
        .ffn_ms = 0,
        .memory_kb = 0,
        .is_attention = false,
    };
    hooks.onLayer(&m);
    try testing.expectEqual(@as(?*hooks.Recorder, null), hooks.activeRecorder());
}

test "adapter: recorder rejects state-machine violations" {
    var rec = try hooks.Recorder.init(testing.allocator);
    defer rec.deinit();

    const stmt = [_]u8{1} ** 32;
    // finalize with a layer open.
    try rec.beginLayer(0);
    try testing.expectError(error.StateRejected, rec.finalize(stmt));
    // record with no layer open.
    try rec.endLayer();
    try testing.expectError(error.StateRejected, rec.recordOpen(.gemm_a, "X"));
    // double begin.
    try rec.beginLayer(0);
    try testing.expectError(error.StateRejected, rec.beginLayer(1));
    try rec.endLayer();
    _ = try rec.finalize(stmt);
}

test "adapter: witness ops map onto trace ordinals" {
    // The C ABI ordinals and trace.Op must stay in lockstep; a mismatch
    // would silently mislabel every recorded slot.
    try testing.expectEqual(@as(u8, 0), @intFromEnum(trace.Op.gemm_a));
    try testing.expectEqual(@as(u8, 3), @intFromEnum(trace.Op.dequant));
    try testing.expectEqual(@as(u8, 8), @intFromEnum(trace.Op.routing_topk));
    try testing.expectEqual(@as(u8, 10), @intFromEnum(trace.Op.other));
}
