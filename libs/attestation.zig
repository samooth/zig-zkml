//! Weights attestation — F0 Merkle root over model tensors.
//!
//! BLUE_PRINT §5.3: `loadWeights` hashes every tensor into a leaf and the
//! root is exposed as `kt_weights_merkle_root`. This module wraps the
//! Blake3 Merkle tree from `libs/merkle.zig` with an API oriented toward
//! model weight loading.

const std = @import("std");
const merkle = @import("merkle.zig");

pub const Hash = merkle.Hash;
pub const Entry = merkle.Entry;

pub const WeightsAttestor = struct {
    tree: merkle.MerkleTree,
    allocator: std.mem.Allocator,

    /// Build the attestation tree from named weight tensors.
    pub fn init(allocator: std.mem.Allocator, entries: []const Entry) !WeightsAttestor {
        return .{
            .tree = try merkle.MerkleTree.init(allocator, entries),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *WeightsAttestor) void {
        self.tree.deinit(self.allocator);
    }

    /// The 32-byte Merkle root binding all weights.
    pub fn root(self: *const WeightsAttestor) Hash {
        return self.tree.root();
    }

    /// Leaf hash for a specific tensor (by name, binary search).
    pub fn leafHash(self: *const WeightsAttestor, name: []const u8) ?Hash {
        // Names are sorted — mirror the tree's binary lookup.
        const S = struct {
            fn order(key: []const u8, item: []const u8) std.math.Order {
                return std.mem.order(u8, key, item);
            }
        };
        const idx = std.sort.binarySearch([]const u8, self.tree.names, name, S.order) orelse return null;
        return self.tree.leaves[idx];
    }

    /// Inclusion proof for a specific tensor. Free with `freeProof` —
    /// the allocator is captured here (BLUE_PRINT B1: no orphaned memory).
    pub fn proof(self: *const WeightsAttestor, name: []const u8) !merkle.Proof {
        return self.tree.proof(self.allocator, name);
    }

    /// Free a proof obtained from `proof` (same captured allocator).
    pub fn freeProof(self: *const WeightsAttestor, p: *merkle.Proof) void {
        p.deinit(self.allocator);
    }
};

test "attestation root from entries" {
    const t = std.testing;
    const a = t.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();

    const entries = [_]Entry{
        .{ .name = "layer0.gate_up", .data = "AAAA" },
        .{ .name = "layer0.down", .data = "BBBB" },
        .{ .name = "layer1.gate_up", .data = "CCCC" },
    };

    var att = try WeightsAttestor.init(alloc, &entries);
    defer att.deinit();

    const r = att.root();
    // Root is non-zero.
    try t.expect(!std.mem.eql(u8, &r, &([_]u8{0} ** 32)));

    // Leaf hash for known tensor.
    const leaf = att.leafHash("layer0.gate_up");
    try t.expect(leaf != null);

    // Unknown tensor returns null.
    try t.expect(att.leafHash("nope") == null);
}

test "attestation proof verifies" {
    const t = std.testing;
    const a = t.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();

    const entries = [_]Entry{
        .{ .name = "w1", .data = "data1" },
        .{ .name = "w2", .data = "data2" },
        .{ .name = "w3", .data = "data3" },
    };

    var att = try WeightsAttestor.init(alloc, &entries);
    defer att.deinit();
    const r = att.root();

    var p = try att.proof("w2");
    defer att.freeProof(&p);
    try t.expect(p.verify(r));

    // Regression: root() must not corrupt leaves — proofs still verify
    // after any number of root() calls (cached root since the fix).
    _ = att.root();
    _ = att.root();
    var p2 = try att.proof("w2");
    defer att.freeProof(&p2);
    try t.expect(p2.verify(r));

    // Negative: a leaf hash from a DIFFERENT tree must not verify.
    var p3 = try att.proof("w1");
    defer att.freeProof(&p3);
    p3.leaf[0] ^= 0x01;
    try t.expect(!p3.verify(r));
}
