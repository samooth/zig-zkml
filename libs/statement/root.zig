//! Statement — public inputs for the zkML proof.
//!
//! BLUE_PRINT §6.1: without H(X)/H(Y) a malicious prover chooses X',Y'.
//! Without weights_leaf he chooses W'. Without scheme_ids he chooses
//! cheaper arithmetic. The statement makes all of this binding.

const std = @import("std");
const tensor = @import("../tensor/root.zig");

pub const Hash = [32]u8;

/// Statement format version — bump on ANY serialization change.
pub const statement_version: u8 = 1;

pub const StatementLayer = struct {
    version: u8 = statement_version,
    scheme_ids: []const tensor.Scheme,
    weights_root: Hash,
    weights_leaf: Hash,
    layer_idx: u32,
    /// Selected experts (DeepSeek-V3 routes among 256 — u32 length, not u8).
    expert_ids: []const u16,
    h_input: Hash,
    h_output: Hash,
    dims: Dims,
    security_params_id: u8 = 0,

    pub const Dims = struct {
        m: u32,
        k: u32,
        n: u32,
    };

    /// Serialize the statement into a byte buffer for Fiat-Shamir binding.
    /// The layout is deterministic and canonical. Scheme enum ordinals are
    /// pinned below — never reorder `tensor.Scheme` (serialization
    /// compatibility across versions relies on it).
    pub fn serialize(self: StatementLayer, allocator: std.mem.Allocator) error{OutOfMemory}![]u8 {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);

        // version
        try buf.append(allocator, self.version);

        // scheme count (u32) + schemes (1 byte each — pinned ordinals)
        var tmp4: [4]u8 = undefined;
        std.mem.writeInt(u32, &tmp4, @intCast(self.scheme_ids.len), .little);
        try buf.appendSlice(allocator, &tmp4);
        for (self.scheme_ids) |s| {
            try buf.append(allocator, @intFromEnum(s));
        }

        // weights_root, weights_leaf
        try buf.appendSlice(allocator, &self.weights_root);
        try buf.appendSlice(allocator, &self.weights_leaf);

        // layer_idx
        std.mem.writeInt(u32, &tmp4, self.layer_idx, .little);
        try buf.appendSlice(allocator, &tmp4);

        // expert count (u32) + experts (u16 LE each)
        std.mem.writeInt(u32, &tmp4, @intCast(self.expert_ids.len), .little);
        try buf.appendSlice(allocator, &tmp4);
        var tmp2: [2]u8 = undefined;
        for (self.expert_ids) |e| {
            std.mem.writeInt(u16, &tmp2, e, .little);
            try buf.appendSlice(allocator, &tmp2);
        }

        // h_input, h_output
        try buf.appendSlice(allocator, &self.h_input);
        try buf.appendSlice(allocator, &self.h_output);

        // dims
        std.mem.writeInt(u32, &tmp4, self.dims.m, .little);
        try buf.appendSlice(allocator, &tmp4);
        std.mem.writeInt(u32, &tmp4, self.dims.k, .little);
        try buf.appendSlice(allocator, &tmp4);
        std.mem.writeInt(u32, &tmp4, self.dims.n, .little);
        try buf.appendSlice(allocator, &tmp4);

        // security_params_id
        try buf.append(allocator, self.security_params_id);

        return try buf.toOwnedSlice(allocator);
    }

    /// Compute a Blake3 hash of the serialized statement.
    pub fn hash(self: StatementLayer, allocator: std.mem.Allocator) error{OutOfMemory}!Hash {
        const serialized = try self.serialize(allocator);
        defer allocator.free(serialized);
        var h: Hash = undefined;
        std.crypto.hash.Blake3.hash(serialized, &h, .{});
        return h;
    }
};

test "statement serialize deterministic" {
    const t = std.testing;
    const a = t.allocator;

    const schemes = [_]tensor.Scheme{ .int8_symmetric, .int4_gguf_q4_k };
    const experts = [_]u16{ 0, 1, 2 };

    const stmt = StatementLayer{
        .scheme_ids = &schemes,
        .weights_root = [_]u8{0x11} ** 32,
        .weights_leaf = [_]u8{0x22} ** 32,
        .layer_idx = 7,
        .expert_ids = &experts,
        .h_input = [_]u8{0x33} ** 32,
        .h_output = [_]u8{0x44} ** 32,
        .dims = .{ .m = 2048, .k = 1408, .n = 2048 },
    };

    const s1 = try stmt.serialize(a);
    defer a.free(s1);
    const s2 = try stmt.serialize(a);
    defer a.free(s2);
    try t.expectEqualSlices(u8, s1, s2);
    try t.expect(s1.len > 0);
}

test "statement hash deterministic and binding" {
    const t = std.testing;
    const a = t.allocator;

    const schemes = [_]tensor.Scheme{.fixed_q16_16};
    const stmt = StatementLayer{
        .scheme_ids = &schemes,
        .weights_root = [_]u8{0xaa} ** 32,
        .weights_leaf = [_]u8{0xbb} ** 32,
        .layer_idx = 0,
        .expert_ids = &.{},
        .h_input = [_]u8{0xcc} ** 32,
        .h_output = [_]u8{0xdd} ** 32,
        .dims = .{ .m = 1024, .k = 512, .n = 1024 },
    };

    const h1 = try stmt.hash(a);
    const h2 = try stmt.hash(a);
    try t.expectEqual(h1, h2);

    // Binding: ±1 in layer_idx must change the hash.
    var stmt2 = stmt;
    stmt2.layer_idx = 1;
    try t.expect(!std.mem.eql(u8, &h1, &(try stmt2.hash(a))));

    // Binding: different output hash must change the statement hash.
    var stmt3 = stmt;
    stmt3.h_output[0] ^= 1;
    try t.expect(!std.mem.eql(u8, &h1, &(try stmt3.hash(a))));
}

test "statement handles 256-expert layers (DeepSeek-V3)" {
    const t = std.testing;
    const a = t.allocator;

    // DeepSeek-V3: 256 routed experts per layer — a u8 count would
    // have trapped on @intCast. u32 lengths handle any realistic model.
    var experts: [256]u16 = undefined;
    for (&experts, 0..) |*e, i| e.* = @intCast(i);

    const stmt = StatementLayer{
        .scheme_ids = &.{},
        .weights_root = [_]u8{0} ** 32,
        .weights_leaf = [_]u8{0} ** 32,
        .layer_idx = 0,
        .expert_ids = &experts,
        .h_input = [_]u8{0} ** 32,
        .h_output = [_]u8{0} ** 32,
        .dims = .{ .m = 1, .k = 1, .n = 1 },
    };

    const s = try stmt.serialize(a);
    defer a.free(s);
    try t.expect(s.len > 256 * 2);
    _ = try stmt.hash(a);
}
