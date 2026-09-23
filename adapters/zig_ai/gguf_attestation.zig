//! GGUF weights attestation for the zig-ai engine adapter (Stage 2).
//!
//! zig-ai loads models through `gguf.GgufModel` / `gguf.GgufFile`
//! (`src/loader/gguf.zig`), whose public surface is
//! `GgufFile.tensors: std.StringHashMap(TensorInfo)` plus
//! `GgufFile.tensorData(info) []const u8` (mmap-backed, zero-copy).
//!
//! That map is a hash map, so iteration order is neither the GGUF directory
//! order nor stable across runs. The adapter therefore consumes a structural
//! `TensorSource` (count / name / data) rather than importing zig-ai's
//! module: `integration.zig` in this directory holds the 20-line shim that
//! binds it to `GgufFile`, and the engine build adds it as one module.
//!
//! The attestation root is order-independent (leaves are name-sorted before
//! folding), so any iteration order yields the same root.

const std = @import("std");
const abi = @import("abi.zig");

const api = abi.api;

/// Structural view of a loader's tensor table. `data` slices are borrowed and
/// read once inside `attestor_add` (never retained), which is what makes the
/// mmap fast-path free of extra copies.
pub const TensorSource = struct {
    ctx: *anyopaque,
    countFn: *const fn (ctx: *anyopaque) usize,
    nameFn: *const fn (ctx: *anyopaque, i: usize) []const u8,
    dataFn: *const fn (ctx: *anyopaque, i: usize) []const u8,

    pub fn count(self: TensorSource) usize {
        return self.countFn(self.ctx);
    }

    pub fn name(self: TensorSource, i: usize) []const u8 {
        return self.nameFn(self.ctx, i);
    }

    pub fn data(self: TensorSource, i: usize) []const u8 {
        return self.dataFn(self.ctx, i);
    }
};

/// Result of streaming a whole tensor table through the attestor.
pub const Attestation = struct {
    root: [32]u8,
    tensor_count: usize,
};

pub const AttestError = error{
    OutOfMemory,
    DuplicateName,
    AddFailed,
    FinishFailed,
    RootFailed,
};

/// Stream every tensor of `source` into a fresh attestor and return the
/// Merkle root. The handle is destroyed before returning; the caller keeps
/// only the root. `gpa` is captured by the handle for the call's duration.
pub fn attestSource(gpa: std.mem.Allocator, source: TensorSource) AttestError!Attestation {
    const handle = api.zkml_attestor_create(abi.allocatorHandle(&gpa)) orelse return error.OutOfMemory;
    defer api.zkml_attestor_destroy(handle);

    const n = source.count();
    for (0..n) |i| {
        const name = source.name(i);
        const data = source.data(i);
        const rc = api.zkml_attestor_add(handle, name.ptr, name.len, data.ptr, data.len);
        switch (rc) {
            0 => {},
            @intFromEnum(api.Status.duplicate_name) => return error.DuplicateName,
            else => return error.AddFailed,
        }
    }
    // Duplicates are only detectable once the leaves are sorted, so the
    // core reports them here rather than at add time.
    switch (api.zkml_attestor_finish(handle)) {
        0 => {},
        @intFromEnum(api.Status.duplicate_name) => return error.DuplicateName,
        else => return error.FinishFailed,
    }

    var root: [32]u8 = undefined;
    if (api.zkml_attestor_root(handle, &root) != 0) return error.RootFailed;
    return .{ .root = root, .tensor_count = n };
}

/// A finished attestation kept alive so inclusion proofs can still be
/// materialized. `zkml_attestor_proof` allocates from the handle's captured
/// allocator, so the handle must outlive the buffer: free the proof through
/// this session, then `deinit`.
pub const ProofSession = struct {
    handle: *api.ZKML_Attestor,

    pub fn build(gpa: std.mem.Allocator, source: TensorSource) AttestError!ProofSession {
        const handle = api.zkml_attestor_create(abi.allocatorHandle(&gpa)) orelse return error.OutOfMemory;
        errdefer api.zkml_attestor_destroy(handle);

        const n = source.count();
        for (0..n) |i| {
            const name = source.name(i);
            const data = source.data(i);
            if (api.zkml_attestor_add(handle, name.ptr, name.len, data.ptr, data.len) != 0) {
                return error.AddFailed;
            }
        }
        // Duplicates surface at finish, not at add (leaves are sorted
        // there); a caller inspecting proofs wants the precise reason.
        switch (api.zkml_attestor_finish(handle)) {
            0 => {},
            @intFromEnum(api.Status.duplicate_name) => return error.DuplicateName,
            else => return error.FinishFailed,
        }

        return .{ .handle = handle };
    }

    pub fn deinit(self: *ProofSession) void {
        api.zkml_attestor_destroy(self.handle);
        self.* = undefined;
    }

    pub fn root(self: *const ProofSession) AttestError![32]u8 {
        var out: [32]u8 = undefined;
        if (api.zkml_attestor_root(self.handle, &out) != 0) return error.RootFailed;
        return out;
    }

    /// Wire-format inclusion proof; release it with `freeProof`.
    pub fn proof(self: *ProofSession, name: []const u8) AttestError![]u8 {
        var wire: [*]u8 = undefined;
        var wire_len: usize = 0;
        const rc = api.zkml_attestor_proof(self.handle, name.ptr, name.len, &wire, &wire_len);
        if (rc == @intFromEnum(api.Status.leaf_not_found)) return error.AddFailed;
        if (rc != 0) return error.AddFailed;
        return wire[0..wire_len];
    }

    pub fn freeProof(self: *ProofSession, wire: []u8) void {
        api.zkml_attestor_free_proof(self.handle, wire.ptr, wire.len);
    }
};

/// Coarse classification of a GGUF tensor name, mirroring the conventions
/// documented in zig-ai's `src/loader/gguf.zig:parseTensorName`
/// (`blk.{i}.attn_q.weight`, `blk.{i}.ffn_down.weight`,
/// `blk.{i}.mlp.gate_proj.weight`, `token_embd.weight`, ...).
///
/// The engine may extend `Role`; unknown roles classify as `.other` while
/// still yielding the parsed layer index.
pub const Role = enum {
    token_embd,
    output_norm,
    output,
    attn_q,
    attn_k,
    attn_v,
    attn_o,
    attn_norm,
    ffn_norm,
    ffn_gate,
    ffn_up,
    ffn_down,
    other,
};

pub const Category = enum {
    embedding,
    attention,
    ffn,
    norm,
    output,
    other,
};

pub fn roleCategory(role: Role) Category {
    return switch (role) {
        .token_embd => .embedding,
        .output => .output,
        .output_norm, .attn_norm, .ffn_norm => .norm,
        .attn_q, .attn_k, .attn_v, .attn_o => .attention,
        .ffn_gate, .ffn_up, .ffn_down => .ffn,
        .other => .other,
    };
}

pub const ParsedName = struct {
    layer: ?u32,
    role: Role,
};

/// Parse `<prefix>blk.<layer>.<role>.<weight|bias>` plus the handful of
/// unlayered tensors. Returns `.other` for anything unrecognized.
pub fn parseTensorName(name: []const u8) ParsedName {
    if (std.mem.startsWith(u8, name, "blk.")) return parseBlk(name);

    if (std.mem.eql(u8, name, "token_embd.weight")) return .{ .layer = null, .role = .token_embd };
    if (std.mem.eql(u8, name, "output_norm.weight")) return .{ .layer = null, .role = .output_norm };
    if (std.mem.eql(u8, name, "output.weight")) return .{ .layer = null, .role = .output };

    return .{ .layer = null, .role = .other };
}

fn parseBlk(name: []const u8) ParsedName {
    const rest = name["blk.".len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return .{ .layer = null, .role = .other };
    const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return .{ .layer = null, .role = .other };

    const tail = rest[dot + 1 ..];
    // mlp.* / feed_forward.* / proj variants map onto the ffn roles.
    if (roleFromName(tail)) |role| return .{ .layer = layer, .role = role };
    if (std.mem.startsWith(u8, tail, "mlp.")) {
        if (roleFromName(tail["mlp.".len..])) |role| return .{ .layer = layer, .role = role };
    }
    if (std.mem.startsWith(u8, tail, "feed_forward.")) {
        const ff = tail["feed_forward.".len..];
        if (std.mem.startsWith(u8, ff, "w1.")) return .{ .layer = layer, .role = .ffn_gate };
        if (std.mem.startsWith(u8, ff, "w2.")) return .{ .layer = layer, .role = .ffn_down };
        if (std.mem.startsWith(u8, ff, "w3.")) return .{ .layer = layer, .role = .ffn_up };
    }
    return .{ .layer = layer, .role = .other };
}

fn roleFromName(tail: []const u8) ?Role {
    const table = .{
        .{ "attn_q.", Role.attn_q },
        .{ "attn_k.", Role.attn_k },
        .{ "attn_v.", Role.attn_v },
        .{ "attn_o.", Role.attn_o },
        .{ "attn_norm.", Role.attn_norm },
        .{ "ffn_norm.", Role.ffn_norm },
        .{ "ffn_gate.", Role.ffn_gate },
        .{ "ffn_up.", Role.ffn_up },
        .{ "ffn_down.", Role.ffn_down },
        .{ "gate_proj.", Role.ffn_gate },
        .{ "up_proj.", Role.ffn_up },
        .{ "down_proj.", Role.ffn_down },
    };
    inline for (table) |entry| {
        if (std.mem.startsWith(u8, tail, entry[0])) return entry[1];
    }
    return null;
}
