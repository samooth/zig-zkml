//! C ABI — F0/F1/F5 surface for host inference engines (BLUE_PRINT §8).
//!
//! Conventions (mirroring the engine-adapter pattern):
//!   - Opaque handle (`ZKML_Attestor` / `ZKML_Witness`); allocator passed
//!     at creation (B1: everything derived from the handle is freed by
//!     its destroy / free functions — no orphaned memory).
//!   - Status codes: 0 = OK, negative = error (`Status`).
//!   - Every entry point is `export fn` (C convention by default) and
//!     force-emitted via the comptime trap at the bottom; `zig build abi`
//!     asserts the symbols land in the binary.
//!   - Caller pointers are read within the call only; names are duped
//!     where retained, tensor data / payloads are copied (never retained).
//!
//! ABI versioning: exported names are additive; changing a signature
//! bumps ZKML_ABI_VERSION (checked by the Python audit tool).
//!   v1: attestor + proof_verify + transcript_seed
//!   v2: witness session (create/begin_layer/record_op/end_layer/
//!       finalize/destroy) — additive, v1 functions unchanged.

const std = @import("std");
const merkle = @import("merkle.zig");
const transcript = @import("transcript.zig");
const trace = @import("trace/root.zig");

pub const ZKML_ABI_VERSION: u32 = 2;

/// Status codes returned by the C API. Stable across versions.
pub const Status = enum(i32) {
    ok = 0,
    /// Allocation failure (from any layer).
    out_of_memory = -1,
    /// Null/misaligned handle or pointer argument.
    invalid_argument = -2,
    /// Proof buffer failed to parse (bad magic/version/length) or
    /// failed verification against the expected root.
    bad_proof = -3,
    /// Tensor name not present in the attestation.
    leaf_not_found = -4,
    /// Duplicate tensor name at attestor build time.
    duplicate_name = -5,

    fn of(err: anyerror) Status {
        return switch (err) {
            error.OutOfMemory => .out_of_memory,
            error.BadProofFormat => .bad_proof,
            error.LeafNotFound => .leaf_not_found,
            error.DuplicateName => .duplicate_name,
            else => .invalid_argument,
        };
    }
};

/// Weights attestation handle. Opaque from C (`void*`); the internal
/// state machine is: building (add*) → finished (root/proof*).
pub const ZKML_Attestor = struct {
    allocator: std.mem.Allocator,
    builder: merkle.Builder,
    finished: ?merkle.MerkleTree = null,

    fn freeNames(self: *ZKML_Attestor, names: [][]const u8) void {
        for (names) |nm| self.allocator.free(nm);
    }
};

// --- Process allocator (C-boundary entry for pure-C consumers) ---

/// Stable process-wide allocator returned to C callers who have no Zig
/// allocator of their own (llama.cpp wrapper, vLLM ctypes, ...). Pass the
/// result to any `*_create` entry (B1: captured; never free it).
var process_allocator: std.mem.Allocator = std.heap.smp_allocator;

pub export fn zkml_allocator_process() *anyopaque {
    return &process_allocator;
}

// --- Attestor lifecycle (F0) ---

/// Create an attestor. `allocator` must remain valid for the handle's
/// lifetime (it is captured — B1).
pub export fn zkml_attestor_create(allocator: *anyopaque) ?*ZKML_Attestor {
    const a: *std.mem.Allocator = @ptrCast(@alignCast(allocator));
    const self = a.create(ZKML_Attestor) catch return null;
    self.* = .{
        .allocator = a.*,
        .builder = merkle.Builder.init(a.*),
    };
    return self;
}

/// Hash one tensor into the attestation as it streams past (Blake3
/// incremental — `data` is read once, never retained; `name` is duped).
/// Tensors may be added in ANY order — the root is order-independent.
pub export fn zkml_attestor_add(
    self: *ZKML_Attestor,
    name: ?[*]const u8,
    name_len: usize,
    data: ?[*]const u8,
    data_len: usize,
) i32 {
    const n = name orelse return @intFromEnum(Status.invalid_argument);
    const d = data orelse return @intFromEnum(Status.invalid_argument);
    if (name_len == 0) return @intFromEnum(Status.invalid_argument);
    // Tree is frozen once finished: no adds past finish().
    if (self.finished != null) return @intFromEnum(Status.invalid_argument);
    self.builder.add(n[0..name_len], d[0..data_len]) catch |err| {
        return @intFromEnum(Status.of(err));
    };
    return @intFromEnum(Status.ok);
}

/// Finish the attestation: computes and caches the Merkle root; leaves
/// stay available for inclusion proofs. Idempotent on a finished
/// attestor; adding after finish fails (frozen tree).
pub export fn zkml_attestor_finish(self: *ZKML_Attestor) i32 {
    if (self.finished != null) return @intFromEnum(Status.ok);
    if (self.builder.leaves.items.len == 0) {
        return @intFromEnum(Status.invalid_argument);
    }
    const tree = self.builder.finish() catch |err| {
        return @intFromEnum(Status.of(err));
    };
    // Builder handed its arrays to the tree; builder lists are now empty.
    self.finished = tree;
    return @intFromEnum(Status.ok);
}

/// Write the 32-byte Merkle root (order-independent, cached).
pub export fn zkml_attestor_root(self: *ZKML_Attestor, root_out: ?[*]u8) i32 {
    const out = root_out orelse return @intFromEnum(Status.invalid_argument);
    const tree = self.finished orelse return @intFromEnum(Status.invalid_argument);
    @memcpy(out[0..32], &tree.root());
    return @intFromEnum(Status.ok);
}

/// Serialize the inclusion proof for tensor `name`. `proof_out`/`len_out`
/// receive a buffer allocated from the captured allocator; free it with
/// `zkml_attestor_free_proof`. Wire format: merkle.Proof.serialize
/// ("ZKMP" v1, 48-byte header + siblings).
pub export fn zkml_attestor_proof(
    self: *ZKML_Attestor,
    name: ?[*]const u8,
    name_len: usize,
    proof_out: ?*[*]u8,
    proof_len_out: ?*usize,
) i32 {
    const n = name orelse return @intFromEnum(Status.invalid_argument);
    const out = proof_out orelse return @intFromEnum(Status.invalid_argument);
    const len_out = proof_len_out orelse return @intFromEnum(Status.invalid_argument);
    if (name_len == 0) return @intFromEnum(Status.invalid_argument);
    const tree = self.finished orelse return @intFromEnum(Status.invalid_argument);

    var p = tree.proof(self.allocator, n[0..name_len]) catch |err| {
        return @intFromEnum(Status.of(err));
    };
    defer p.deinit(self.allocator);
    const wire = p.serialize(self.allocator) catch |err| {
        return @intFromEnum(Status.of(err));
    };
    out.* = wire.ptr;
    len_out.* = wire.len;
    return @intFromEnum(Status.ok);
}

/// Free a proof buffer obtained from `zkml_attestor_proof` (B1).
pub export fn zkml_attestor_free_proof(
    self: *ZKML_Attestor,
    proof: ?[*]u8,
    proof_len: usize,
) void {
    const p = proof orelse return;
    if (proof_len == 0) return;
    self.allocator.free(p[0..proof_len]);
}

/// Destroy the attestor: frees the tree (leaf hashes + duped names) and
/// the handle. Safe on unfinished attestors (builder state included).
pub export fn zkml_attestor_destroy(self: ?*ZKML_Attestor) void {
    const s = self orelse return;
    if (s.finished) |*tree| {
        // Tree owns duped names: free strings, then arrays (MerkleTree
        // .deinit frees the same shape, but in-place here to keep the
        // optional-move semantics simple).
        for (tree.names) |nm| s.allocator.free(nm);
        s.allocator.free(tree.names);
        s.allocator.free(tree.leaves);
        s.finished = null;
    }
    // Unfinished builder state (e.g. destroy mid-load): builder lists
    // still own the duped names. After finish() the lists are empty.
    s.builder.deinit();
    s.allocator.destroy(s);
}

// --- Standalone verification (no tree, no model needed) ---

/// Verify a serialized proof against `expected_root` (32 bytes) — the
/// auditor's entry point: only the published root and the proof bytes.
/// Returns 0 on verification success, negative otherwise.
pub export fn zkml_proof_verify(
    allocator: *anyopaque,
    proof: ?[*]const u8,
    proof_len: usize,
    expected_root: ?[*]const u8,
) i32 {
    const p = proof orelse return @intFromEnum(Status.invalid_argument);
    const r = expected_root orelse return @intFromEnum(Status.invalid_argument);
    if (proof_len == 0) return @intFromEnum(Status.invalid_argument);
    const a: *std.mem.Allocator = @ptrCast(@alignCast(allocator));
    const ok = merkle.Proof.verifySerialized(a.*, p[0..proof_len], r[0..32].*) catch |err| {
        return @intFromEnum(Status.of(err));
    };
    return if (ok) @intFromEnum(Status.ok) else @intFromEnum(Status.bad_proof);
}

// --- Deterministic transcript seed (F1) ---

/// Derive a deterministic 32-byte sampling seed from a context
/// (BLUE_PRINT §8: reproducible sampling with no hidden state: same
/// context → same seed, always).
pub export fn zkml_transcript_seed(
    context: ?[*]const u8,
    context_len: usize,
    seed_out: ?[*]u8,
) i32 {
    const c = context orelse return @intFromEnum(Status.invalid_argument);
    const out = seed_out orelse return @intFromEnum(Status.invalid_argument);
    if (context_len == 0) return @intFromEnum(Status.invalid_argument);
    var tr = transcript.Transcript.init("zkml.seed.v1");
    tr.absorb(c[0..context_len]);
    const seed = tr.finish();
    @memcpy(out[0..32], &seed);
    return @intFromEnum(Status.ok);
}

// --- Witness session (ABI v2 — recorded inference feed) ---

/// Slot key as seen from C: flat mirror of `trace.SlotKey`.
/// `op` is the `trace.Op` ordinal (see ZKML_OP_* in zkml_c.h).
/// Field order matches the C struct exactly: 8 bytes, no padding.
pub const ZKML_SlotKey = extern struct {
    layer: u32,
    expert: u16,
    op: u8,
    rank: u8,
};

/// Witness session handle. Wraps `TraceRecorder` + layer state machine:
/// open layer (begin) → record* → close layer (end) → finalize (frozen).
pub const ZKML_Witness = struct {
    allocator: std.mem.Allocator,
    recorder: trace.TraceRecorder,
    /// Layer opened by begin_layer and not yet closed (record target).
    open_layer: ?u32 = null,
    /// Set by finalize: no further begin/record/end accepted.
    frozen: bool = false,
};

/// Create a witness session. `allocator` is captured (B1), same rule as
/// the attestor. Returns NULL on allocation failure.
pub export fn zkml_witness_session_create(allocator: *anyopaque) ?*ZKML_Witness {
    const a: *std.mem.Allocator = @ptrCast(@alignCast(allocator));
    const self = a.create(ZKML_Witness) catch return null;
    self.* = .{
        .allocator = a.*,
        .recorder = trace.TraceRecorder.init(a.*),
    };
    return self;
}

/// Open layer `layer_idx` for recording. Only one layer may be open at a
/// time; call end_layer before opening the next. Rejected after finalize.
pub export fn zkml_witness_begin_layer(self: *ZKML_Witness, layer_idx: u32) i32 {
    if (self.frozen) return @intFromEnum(Status.invalid_argument);
    if (self.open_layer != null) return @intFromEnum(Status.invalid_argument);
    self.open_layer = layer_idx;
    return @intFromEnum(Status.ok);
}

/// Record one op payload into slot `key`. May be called concurrently from
/// multiple threads while a layer is open (TraceRecorder is thread-safe);
/// the engine serializes begin/end/finalize itself. `key.layer` must
/// match the open layer. `payload` is copied, never retained.
pub export fn zkml_witness_record_op(
    self: *ZKML_Witness,
    key: ?*const ZKML_SlotKey,
    payload: ?[*]const u8,
    payload_len: usize,
) i32 {
    const k = key orelse return @intFromEnum(Status.invalid_argument);
    const p = payload orelse return @intFromEnum(Status.invalid_argument);
    if (self.frozen) return @intFromEnum(Status.invalid_argument);
    const open = self.open_layer orelse return @intFromEnum(Status.invalid_argument);
    if (k.layer != open) return @intFromEnum(Status.invalid_argument);
    if (k.op >= @intFromEnum(trace.Op.other) + 1) return @intFromEnum(Status.invalid_argument);
    const slot_key = trace.SlotKey{
        .layer = k.layer,
        .op = @enumFromInt(k.op),
        .expert = k.expert,
        .rank = k.rank,
    };
    self.recorder.record(slot_key, p[0..payload_len]) catch |err| {
        return @intFromEnum(Status.of(err));
    };
    return @intFromEnum(Status.ok);
}

/// Close the currently open layer. Must pair with begin_layer.
pub export fn zkml_witness_end_layer(self: *ZKML_Witness) i32 {
    if (self.frozen) return @intFromEnum(Status.invalid_argument);
    if (self.open_layer == null) return @intFromEnum(Status.invalid_argument);
    self.open_layer = null;
    return @intFromEnum(Status.ok);
}

/// Finalize: absorb all recorded slots in CANONICAL order (BLUE_PRINT
/// §6.2) bound to `stmt_hash`, writing the 32-byte trace hash to
/// `trace_hash_out`. Freezes the session (no further recording).
pub export fn zkml_witness_finalize(
    self: *ZKML_Witness,
    stmt_hash: ?[*]const u8,
    trace_hash_out: ?[*]u8,
) i32 {
    const s = stmt_hash orelse return @intFromEnum(Status.invalid_argument);
    const out = trace_hash_out orelse return @intFromEnum(Status.invalid_argument);
    if (self.frozen) return @intFromEnum(Status.invalid_argument);
    if (self.open_layer != null) return @intFromEnum(Status.invalid_argument);
    const h = self.recorder.finalize(s[0..32]) catch |err| {
        return @intFromEnum(Status.of(err));
    };
    @memcpy(out[0..32], &h);
    self.frozen = true;
    return @intFromEnum(Status.ok);
}

/// Destroy the session: frees all slot buffers and the handle (B1).
/// Safe on partially-used sessions; destroy(NULL) is a no-op.
pub export fn zkml_witness_session_destroy(self: ?*ZKML_Witness) void {
    const w = self orelse return;
    w.recorder.deinit();
    w.allocator.destroy(w);
}

comptime {
    // Force emission (BLUE_PRINT §8: comptime { _ = &fn } verified with nm).
    _ = &zkml_allocator_process;
    _ = &zkml_attestor_create;
    _ = &zkml_attestor_add;
    _ = &zkml_attestor_finish;
    _ = &zkml_attestor_root;
    _ = &zkml_attestor_proof;
    _ = &zkml_attestor_free_proof;
    _ = &zkml_attestor_destroy;
    _ = &zkml_proof_verify;
    _ = &zkml_transcript_seed;
    _ = &zkml_witness_session_create;
    _ = &zkml_witness_begin_layer;
    _ = &zkml_witness_record_op;
    _ = &zkml_witness_end_layer;
    _ = &zkml_witness_finalize;
    _ = &zkml_witness_session_destroy;
}

// --- Tests: exercise the ABI exactly as C would (F0 go/no-go §13) ---

const testing = std.testing;

test "abi: process allocator usable for attestor" {
    // Pure-C consumers (llama.cpp wrapper) pass zkml_allocator_process()
    // instead of constructing a Zig allocator themselves.
    const h = zkml_attestor_create(zkml_allocator_process()) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(i32, 0), zkml_attestor_add(h, "t".ptr, 1, "x".ptr, 1));
    try testing.expectEqual(@as(i32, 0), zkml_attestor_finish(h));
    var root: [32]u8 = undefined;
    try testing.expectEqual(@as(i32, 0), zkml_attestor_root(h, &root));
    zkml_attestor_destroy(h);

    const w = zkml_witness_session_create(zkml_allocator_process()) orelse return error.TestUnexpectedResult;
    zkml_witness_session_destroy(w);
}

test "abi: attestor lifecycle, proof, standalone verify" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var a = arena.allocator();

    const h = zkml_attestor_create(@ptrCast(&a)) orelse return error.TestUnexpectedResult;

    const names = [_][]const u8{ "down.0.w", "gate.0.w", "up.0.w" };
    const datas = [_][]const u8{ "CCCC", "AAAA", "BBBB" };
    for (names, datas) |nm, dt| {
        const rc = zkml_attestor_add(h, nm.ptr, nm.len, dt.ptr, dt.len);
        try testing.expectEqual(@as(i32, 0), rc);
    }

    // Finish (idempotent).
    try testing.expectEqual(@as(i32, 0), zkml_attestor_finish(h));
    try testing.expectEqual(@as(i32, 0), zkml_attestor_finish(h));

    // Adding after finish is rejected (frozen tree).
    try testing.expectEqual(@as(i32, -2), zkml_attestor_add(h, names[0].ptr, names[0].len, datas[0].ptr, datas[0].len));

    // Root.
    var root: [32]u8 = undefined;
    try testing.expectEqual(@as(i32, 0), zkml_attestor_root(h, &root));

    // Root equals a tree built the library way (order-independence).
    var entries_buf: [3]merkle.Entry = undefined;
    for (names, datas, 0..) |nm, dt, i| {
        entries_buf[i] = .{ .name = nm, .data = dt };
    }
    var m = try merkle.MerkleTree.init(a, &entries_buf);
    defer m.deinit(a);
    try testing.expectEqualSlices(u8, &root, &m.root());

    // Proof for one tensor + roundtrip via the wire format.
    var wire: [*]u8 = undefined;
    var wire_len: usize = 0;
    try testing.expectEqual(@as(i32, 0), zkml_attestor_proof(h, "gate.0.w".ptr, 8, &wire, &wire_len));
    defer _ = zkml_attestor_free_proof(h, wire, wire_len);
    try testing.expect(wire_len > 48);

    // Standalone verification (fresh call — the auditor's path).
    try testing.expectEqual(@as(i32, 0), zkml_proof_verify(@ptrCast(&a), wire, wire_len, &root));
    // Negative: wrong root.
    var bad_root = root;
    bad_root[0] ^= 0xFF;
    try testing.expectEqual(@as(i32, -3), zkml_proof_verify(@ptrCast(&a), wire, wire_len, &bad_root));
    // Negative: unknown tensor.
    try testing.expectEqual(@as(i32, -4), zkml_attestor_proof(h, "nope".ptr, 4, &wire, &wire_len));

    zkml_attestor_destroy(h);
}

test "abi: null and invalid arguments" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var a = arena.allocator();

    const h = zkml_attestor_create(@ptrCast(&a)) orelse return error.TestUnexpectedResult;

    // finish with zero tensors: invalid.
    try testing.expectEqual(@as(i32, -2), zkml_attestor_finish(h));
    // root before finish: invalid.
    var root: [32]u8 = undefined;
    try testing.expectEqual(@as(i32, -2), zkml_attestor_root(h, &root));
    // null name/data.
    try testing.expectEqual(@as(i32, -2), zkml_attestor_add(h, null, 4, "x".ptr, 1));
    try testing.expectEqual(@as(i32, -2), zkml_attestor_add(h, "x".ptr, 1, null, 4));
    // empty name.
    try testing.expectEqual(@as(i32, -2), zkml_attestor_add(h, "x".ptr, 0, "y".ptr, 1));

    zkml_attestor_destroy(h);
    // destroy(null) is a safe no-op. (Double destroy on the same pointer
    // is use-after-free, as with any C handle — not tested by design.)
    zkml_attestor_destroy(null);
}

test "abi: transcript seed deterministic" {
    const ctx = "sampling context v1";
    var s1: [32]u8 = undefined;
    var s2: [32]u8 = undefined;
    try testing.expectEqual(@as(i32, 0), zkml_transcript_seed(ctx.ptr, ctx.len, &s1));
    try testing.expectEqual(@as(i32, 0), zkml_transcript_seed(ctx.ptr, ctx.len, &s2));
    try testing.expectEqualSlices(u8, &s1, &s2);

    // Different context → different seed.
    const ctx2 = "sampling context v2";
    var s3: [32]u8 = undefined;
    try testing.expectEqual(@as(i32, 0), zkml_transcript_seed(ctx2.ptr, ctx2.len, &s3));
    try testing.expect(!std.mem.eql(u8, &s1, &s3));

    // Null/zero-len rejected.
    try testing.expectEqual(@as(i32, -2), zkml_transcript_seed(null, 1, &s1));
    try testing.expectEqual(@as(i32, -2), zkml_transcript_seed(ctx.ptr, 0, &s1));
    try testing.expectEqual(@as(i32, -2), zkml_transcript_seed(ctx.ptr, ctx.len, null));
}

test "abi: duplicate names rejected at finish" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var a = arena.allocator();

    const h = zkml_attestor_create(@ptrCast(&a)) orelse return error.TestUnexpectedResult;
    _ = zkml_attestor_add(h, "w".ptr, 1, "1".ptr, 1);
    _ = zkml_attestor_add(h, "w".ptr, 1, "2".ptr, 1);
    // -5 = duplicate_name; builder state stays valid for destroy.
    try testing.expectEqual(@as(i32, -5), zkml_attestor_finish(h));
    zkml_attestor_destroy(h);
}

test "abi: witness session lifecycle and determinism" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var a = arena.allocator();

    const w = zkml_witness_session_create(@ptrCast(&a)) orelse return error.TestUnexpectedResult;

    const stmt = [_]u8{0x77} ** 32;
    const k0 = ZKML_SlotKey{ .layer = 0, .expert = 3, .op = @intFromEnum(trace.Op.gemm_a), .rank = 0 };
    const k1 = ZKML_SlotKey{ .layer = 1, .expert = 0, .op = @intFromEnum(trace.Op.gemm_c), .rank = 1 };

    try testing.expectEqual(@as(i32, 0), zkml_witness_begin_layer(w, 0));
    try testing.expectEqual(@as(i32, 0), zkml_witness_record_op(w, &k0, "AAAA", 4));
    try testing.expectEqual(@as(i32, 0), zkml_witness_record_op(w, &k0, "BBBB", 4));
    try testing.expectEqual(@as(i32, 0), zkml_witness_end_layer(w));
    try testing.expectEqual(@as(i32, 0), zkml_witness_begin_layer(w, 1));
    try testing.expectEqual(@as(i32, 0), zkml_witness_record_op(w, &k1, "CCCC", 4));
    try testing.expectEqual(@as(i32, 0), zkml_witness_end_layer(w));

    var h1: [32]u8 = undefined;
    try testing.expectEqual(@as(i32, 0), zkml_witness_finalize(w, &stmt, &h1));

    // Frozen after finalize.
    try testing.expectEqual(@as(i32, -2), zkml_witness_begin_layer(w, 2));
    try testing.expectEqual(@as(i32, -2), zkml_witness_record_op(w, &k0, "X", 1));
    try testing.expectEqual(@as(i32, -2), zkml_witness_end_layer(w));
    try testing.expectEqual(@as(i32, -2), zkml_witness_finalize(w, &stmt, &h1));
    zkml_witness_session_destroy(w);

    // Same data in a second session (different insertion order) → same hash.
    const w2 = zkml_witness_session_create(@ptrCast(&a)) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(i32, 0), zkml_witness_begin_layer(w2, 1));
    try testing.expectEqual(@as(i32, 0), zkml_witness_record_op(w2, &k1, "CCCC", 4));
    try testing.expectEqual(@as(i32, 0), zkml_witness_end_layer(w2));
    try testing.expectEqual(@as(i32, 0), zkml_witness_begin_layer(w2, 0));
    try testing.expectEqual(@as(i32, 0), zkml_witness_record_op(w2, &k0, "AAAA", 4));
    try testing.expectEqual(@as(i32, 0), zkml_witness_record_op(w2, &k0, "BBBB", 4));
    try testing.expectEqual(@as(i32, 0), zkml_witness_end_layer(w2));
    var h2: [32]u8 = undefined;
    try testing.expectEqual(@as(i32, 0), zkml_witness_finalize(w2, &stmt, &h2));
    zkml_witness_session_destroy(w2);
    try testing.expectEqualSlices(u8, &h1, &h2);

    // Different payload → different hash.
    const w3 = zkml_witness_session_create(@ptrCast(&a)) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(i32, 0), zkml_witness_begin_layer(w3, 0));
    try testing.expectEqual(@as(i32, 0), zkml_witness_record_op(w3, &k0, "AAAB", 4));
    try testing.expectEqual(@as(i32, 0), zkml_witness_end_layer(w3));
    var h3: [32]u8 = undefined;
    try testing.expectEqual(@as(i32, 0), zkml_witness_finalize(w3, &stmt, &h3));
    zkml_witness_session_destroy(w3);
    try testing.expect(!std.mem.eql(u8, &h1, &h3));
}

test "abi: witness state machine negatives" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var a = arena.allocator();

    // C-layout sanity: ZKML_SlotKey must match zkml_c.h (8 bytes).
    try testing.expectEqual(@as(usize, 8), @sizeOf(ZKML_SlotKey));
    try testing.expectEqual(@as(usize, 0), @offsetOf(ZKML_SlotKey, "layer"));
    try testing.expectEqual(@as(usize, 4), @offsetOf(ZKML_SlotKey, "expert"));
    try testing.expectEqual(@as(usize, 6), @offsetOf(ZKML_SlotKey, "op"));
    try testing.expectEqual(@as(usize, 7), @offsetOf(ZKML_SlotKey, "rank"));

    const w = zkml_witness_session_create(@ptrCast(&a)) orelse return error.TestUnexpectedResult;
    defer zkml_witness_session_destroy(w);

    const stmt = [_]u8{1} ** 32;
    var out: [32]u8 = undefined;
    const k = ZKML_SlotKey{ .layer = 0, .expert = 0, .op = 0, .rank = 0 };

    // record with no open layer.
    try testing.expectEqual(@as(i32, -2), zkml_witness_record_op(w, &k, "x", 1));
    // end with no open layer.
    try testing.expectEqual(@as(i32, -2), zkml_witness_end_layer(w));
    // double begin.
    try testing.expectEqual(@as(i32, 0), zkml_witness_begin_layer(w, 0));
    try testing.expectEqual(@as(i32, -2), zkml_witness_begin_layer(w, 1));
    // record for a different layer than open.
    const k_wrong = ZKML_SlotKey{ .layer = 5, .expert = 0, .op = 0, .rank = 0 };
    try testing.expectEqual(@as(i32, -2), zkml_witness_record_op(w, &k_wrong, "x", 1));
    // invalid op ordinal.
    const k_bop = ZKML_SlotKey{ .layer = 0, .expert = 0, .op = 200, .rank = 0 };
    try testing.expectEqual(@as(i32, -2), zkml_witness_record_op(w, &k_bop, "x", 1));
    // null args.
    try testing.expectEqual(@as(i32, -2), zkml_witness_record_op(w, null, "x", 1));
    try testing.expectEqual(@as(i32, -2), zkml_witness_record_op(w, &k, null, 1));
    // finalize with layer still open.
    try testing.expectEqual(@as(i32, -2), zkml_witness_finalize(w, &stmt, &out));
    // close the layer so finalize is unblocked.
    try testing.expectEqual(@as(i32, 0), zkml_witness_end_layer(w));
    // finalize with null args.
    try testing.expectEqual(@as(i32, -2), zkml_witness_finalize(w, null, &out));
    try testing.expectEqual(@as(i32, -2), zkml_witness_finalize(w, &stmt, null));
    // happy path now works.
    try testing.expectEqual(@as(i32, 0), zkml_witness_finalize(w, &stmt, &out));
}
