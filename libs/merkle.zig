//! Merkle tree over tensor leaves — Blake3 domain-separated.
//!
//! F0 weights attestation (docs/BLUE_PRINT.md §5.3): `loadWeights` hashes every
//! tensor into a leaf and the root is exposed as `zkml_attestor_root`
//! (engine adapters may re-wrap it under engine-specific names).
//! Design points:
//!   - Leaf = Blake3("zkml.wleaf" || tag || len_le64 || data...) where
//!     `tag` names the tensor (stable across load-order permutations — the
//!     tree is keyed by name, not by insertion order).
//!   - Node = Blake3("zkml.wnode" || left || right).
//!   - Odd levels self-pair the orphan node: left == right == orphan. The
//!     classic duplicate-leaf second-preimage attacks don't apply because
//!     leaf and node hashes live in separate keyed domains, so a node hash
//!     can never be reinterpreted as a leaf. Self-pairing keeps root, proof
//!     and verify consistent for every tree shape (an orphan leaf proves
//!     with itself as sibling).
//!   - The root is computed once at init and cached: the tree is immutable
//!     after construction, so `root()` is O(1) and never mutates state.

const std = @import("std");

pub const Hash = [32]u8;

const leaf_domain = "zkml.wleaf";
const node_domain = "zkml.wnode";

pub const Entry = struct {
    name: []const u8,
    data: []const u8,
};

fn hashLeafInner(name: []const u8, data: []const u8) Hash {
    var out: Hash = undefined;
    var h = std.crypto.hash.Blake3.init(.{});
    h.update(leaf_domain);
    h.update(name);
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, data.len, .little);
    h.update(&len_buf);
    h.update(data);
    h.final(&out);
    return out;
}

fn hashNode(l: *const Hash, r: *const Hash) Hash {
    var out: Hash = undefined;
    var h = std.crypto.hash.Blake3.init(.{});
    h.update(node_domain);
    h.update(l);
    h.update(r);
    h.final(&out);
    return out;
}

/// Empty-tree root: Blake3 over the leaf domain only.
fn emptyRoot() Hash {
    var out: Hash = undefined;
    std.crypto.hash.Blake3.hash(leaf_domain, &out, .{});
    return out;
}

/// Fold a full level in place. `scratch` is consumed (destroyed) — callers
/// pass a private copy. Odd orphans self-pair: node = H(orphan, orphan).
fn foldLevels(level: []Hash) Hash {
    var cur = level;
    while (cur.len > 1) {
        const pairs = cur.len / 2;
        for (0..pairs) |i| {
            const h = hashNode(&cur[2 * i], &cur[2 * i + 1]);
            cur[i] = h;
        }
        if (cur.len % 2 == 1) {
            // Self-pair the orphan (see module doc): consistent with proofs.
            cur[pairs] = hashNode(&cur[cur.len - 1], &cur[cur.len - 1]);
            cur = cur[0 .. pairs + 1];
        } else {
            cur = cur[0..pairs];
        }
    }
    return cur[0];
}

pub const MerkleTree = struct {
    /// Leaves sorted by name (root is load-order independent).
    leaves: []Hash,
    names: [][]const u8,
    /// Cached root — computed once at init; the tree is immutable.
    root_hash: Hash,
    /// Build from named tensors. The root is stable under any permutation
    /// of `entries` (F0 go/no-go: "root estable ante reorden de lectura").
    pub fn init(allocator: std.mem.Allocator, entries: []const Entry) !MerkleTree {
        const n = entries.len;
        var leaves = try allocator.alloc(Hash, n);
        errdefer allocator.free(leaves);
        var names = try allocator.alloc([]const u8, n);
        errdefer allocator.free(names);

        var duped: usize = 0;
        errdefer for (names[0..duped]) |nm| allocator.free(nm);

        for (entries, 0..) |e, i| {
            leaves[i] = hashLeafInner(e.name, e.data);
            names[i] = try allocator.dupe(u8, e.name);
            duped += 1;
        }

        // Insertion sort by name: attestation trees have at most hundreds
        // of tensors; allocation-free and obviously correct beats clever.
        var i: usize = 1;
        while (i < n) : (i += 1) {
            var j = i;
            while (j > 0 and std.mem.order(u8, names[j - 1], names[j]) == .gt) : (j -= 1) {
                const tmp_h = leaves[j];
                leaves[j] = leaves[j - 1];
                leaves[j - 1] = tmp_h;
                const tmp_n = names[j];
                names[j] = names[j - 1];
                names[j - 1] = tmp_n;
            }
        }

        // Compute and cache the root on a scratch copy — the stored leaves
        // must stay pristine for inclusion proofs.
        const root_hash = if (n == 0) emptyRoot() else blk: {
            const scratch = try allocator.dupe(Hash, leaves);
            defer allocator.free(scratch);
            break :blk foldLevels(scratch);
        };

        return .{ .leaves = leaves, .names = names, .root_hash = root_hash };
    }

    pub fn deinit(self: *MerkleTree, allocator: std.mem.Allocator) void {
        for (self.names) |nm| allocator.free(nm);
        allocator.free(self.names);
        allocator.free(self.leaves);
        self.* = undefined;
    }

    pub fn hashLeaf(name: []const u8, data: []const u8) Hash {
        return hashLeafInner(name, data);
    }

    /// The cached root over the sorted leaves. O(1), non-mutating,
    /// idempotent (was previously a destructive in-place fold — see git
    /// history for the bug this replaced).
    pub fn root(self: *const MerkleTree) Hash {
        return self.root_hash;
    }

    /// Inclusion proof for the leaf named `name`: one sibling hash per
    /// level (orphan positions carry the node itself as sibling — it was
    /// self-paired at root-construction time).
    pub fn proof(self: *const MerkleTree, allocator: std.mem.Allocator, name: []const u8) !Proof {
        const idx = self.indexOf(name) orelse return error.LeafNotFound;
        var siblings: std.ArrayList(Hash) = .empty;
        errdefer siblings.deinit(allocator);

        const scratch = try allocator.alloc(Hash, self.leaves.len);
        defer allocator.free(scratch);
        @memcpy(scratch, self.leaves);

        var level = scratch;
        var idx_run = idx;
        while (level.len > 1) {
            const sibling = if (idx_run % 2 == 0)
                (if (idx_run + 1 < level.len) level[idx_run + 1] else level[idx_run])
            else
                level[idx_run - 1];
            try siblings.append(allocator, sibling);
            const pairs = level.len / 2;
            for (0..pairs) |i| {
                const h = hashNode(&level[2 * i], &level[2 * i + 1]);
                level[i] = h;
            }
            if (level.len % 2 == 1) {
                const orphan = level[level.len - 1];
                level[pairs] = hashNode(&orphan, &orphan);
                level = level[0 .. pairs + 1];
            } else {
                level = level[0..pairs];
            }
            idx_run /= 2;
        }
        return .{
            .leaf = self.leaves[idx], // sorted position == proof position
            .siblings = try siblings.toOwnedSlice(allocator),
            .leaf_index = idx,
        };
    }

    /// Names are sorted at init — binary search.
    fn indexOf(self: *const MerkleTree, name: []const u8) ?usize {
        // compareFn(context=key, item): order of the KEY relative to item.
        const S = struct {
            fn order(key: []const u8, item: []const u8) std.math.Order {
                return std.mem.order(u8, key, item);
            }
        };
        return std.sort.binarySearch([]const u8, self.names, name, S.order);
    }
};

/// Streaming tree builder for production weight loading.
///
/// `MerkleTree.init` requires all entries up front; a 671B model cannot
/// materialize every tensor at once. The builder hashes each tensor as
/// it is added (Blake3 incremental — the tensor bytes are read once and
/// never retained), keeps only the 32-byte leaf hashes plus duped name
/// copies, and finishes into a `MerkleTree`.
///
/// Ownership matches `MerkleTree.init`: names are duped on `add` (the
/// caller's buffers may go away immediately), the finished tree owns
/// everything, and `MerkleTree.deinit` frees it all. The root is
/// order-independent (names are sorted at `finish`).
pub const Builder = struct {
    allocator: std.mem.Allocator,
    leaves: std.ArrayList(Hash) = .empty,
    names: std.ArrayList([]const u8) = .empty,

    pub fn init(allocator: std.mem.Allocator) Builder {
        return .{ .allocator = allocator };
    }

    /// Free the unfinished builder state (duped names included).
    /// NOT valid after a successful `finish` — the tree owns everything
    /// from that point (this deinit is then a no-op on empty lists).
    /// Resets the lists to a valid empty state, so a stray second
    /// deinit cannot double-free (it sees empty slices).
    pub fn deinit(self: *Builder) void {
        for (self.names.items) |nm| self.allocator.free(nm);
        // ArrayList.deinit frees the backing buffer and resets to .empty,
        // which also keeps a stray second deinit harmless. Assigning
        // `.empty` directly would silently drop the capacity and leak.
        self.names.deinit(self.allocator);
        self.leaves.deinit(self.allocator);
    }

    /// Hash one tensor into the tree as it streams past. `name` is duped
    /// (caller's buffer may be transient); `data` is read once, hashed
    /// incrementally and never retained.
    pub fn add(self: *Builder, name: []const u8, data: []const u8) error{OutOfMemory}!void {
        const name_copy = self.allocator.dupe(u8, name) catch |err| return err;
        self.leaves.append(self.allocator, hashLeafInner(name, data)) catch |err| {
            self.allocator.free(name_copy);
            return err;
        };
        self.names.append(self.allocator, name_copy) catch |err| {
            _ = self.leaves.pop();
            self.allocator.free(name_copy);
            return err;
        };
    }

    /// Finish into an immutable tree (root computed and cached).
    /// Duplicate names are rejected — two tensors with the same name
    /// would make inclusion proofs ambiguous.
    ///
    /// On ANY error the builder stays fully valid (lists intact, caller
    /// may retry or `deinit`): validation runs in place and the arrays
    /// are only extracted once every failure mode is excluded.
    pub fn finish(self: *Builder) error{ OutOfMemory, DuplicateName }!MerkleTree {
        const n = self.leaves.items.len;
        std.debug.assert(self.names.items.len == n);

        // Sort both arrays together by name, in place (hundreds of
        // tensors at most, same rationale as MerkleTree.init).
        var i: usize = 1;
        while (i < n) : (i += 1) {
            var j = i;
            while (j > 0 and std.mem.order(u8, self.names.items[j - 1], self.names.items[j]) == .gt) : (j -= 1) {
                const tmp_h = self.leaves.items[j];
                self.leaves.items[j] = self.leaves.items[j - 1];
                self.leaves.items[j - 1] = tmp_h;
                const tmp_n = self.names.items[j];
                self.names.items[j] = self.names.items[j - 1];
                self.names.items[j - 1] = tmp_n;
            }
        }

        // Duplicate detection: cheap on the sorted array, and it protects
        // proof unambiguity (indexOf would otherwise pick one of them).
        // (n >= 2 required: 1..0 would overflow on an empty tree.)
        if (n >= 2) {
            for (1..n) |k| {
                if (std.mem.eql(u8, self.names.items[k - 1], self.names.items[k])) {
                    return error.DuplicateName; // builder state still valid
                }
            }
        }

        // Validation done — extract. Remaining failure modes are OOM;
        // each hand-back keeps the builder deinit-able and retryable.
        const leaves = try self.leaves.toOwnedSlice(self.allocator);
        const names = self.names.toOwnedSlice(self.allocator) catch |err| {
            self.allocator.free(leaves);
            return err;
        };

        const root_hash = if (n == 0) emptyRoot() else blk: {
            const scratch = self.allocator.dupe(Hash, leaves) catch |err| {
                self.leaves = .{ .items = leaves[0..0], .capacity = leaves.len };
                self.names = .{ .items = names[0..0], .capacity = names.len };
                return err;
            };
            defer self.allocator.free(scratch);
            break :blk foldLevels(scratch);
        };

        return .{ .leaves = leaves, .names = names, .root_hash = root_hash };
    }
};

pub const Proof = struct {
    leaf: Hash,
    siblings: []Hash,
    leaf_index: usize,

    pub fn deinit(self: *Proof, allocator: std.mem.Allocator) void {
        allocator.free(self.siblings);
        self.* = undefined;
    }

    /// Verify against an expected root.
    pub fn verify(self: *const Proof, expected_root: Hash) bool {
        var h = self.leaf;
        var idx = self.leaf_index;
        for (self.siblings) |sib| {
            h = if (idx % 2 == 0) hashNode(&h, &sib) else hashNode(&sib, &h);
            idx /= 2;
        }
        return std.mem.eql(u8, &h, &expected_root);
    }

    // --- Wire format (C ABI boundary + Python audit tool) ---
    //
    // Canonical little-endian layout, versioned by leading magic:
    //   [4]  magic "ZKMP"          (zkml proof)
    //   [1]  version = 1
    //   [1]  reserved = 0
    //   [2]  reserved = 0
    //   [32] leaf hash
    //   [4]  leaf index (u32 LE)
    //   [4]  sibling count (u32 LE)
    //   [n*32] siblings, bottom-up in tree order
    // Total: 48 + 32*n bytes. Any parser deviation → error (never a
    // reinterpreted proof).

    pub const wire_magic = [4]u8{ 'Z', 'K', 'M', 'P' };
    pub const wire_version: u8 = 1;
    pub const wire_header_len: usize = 48;

    /// Serialize into a freshly allocated buffer (caller frees).
    pub fn serialize(self: *const Proof, allocator: std.mem.Allocator) error{OutOfMemory}![]u8 {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);
        try buf.appendSlice(allocator, &wire_magic);
        try buf.append(allocator, wire_version);
        try buf.append(allocator, 0); // reserved
        try buf.appendSlice(allocator, &[_]u8{ 0, 0 }); // reserved
        try buf.appendSlice(allocator, &self.leaf);
        var tmp4: [4]u8 = undefined;
        std.mem.writeInt(u32, &tmp4, @intCast(self.leaf_index), .little);
        try buf.appendSlice(allocator, &tmp4);
        std.mem.writeInt(u32, &tmp4, @intCast(self.siblings.len), .little);
        try buf.appendSlice(allocator, &tmp4);
        for (self.siblings) |sib| {
            try buf.appendSlice(allocator, &sib);
        }
        return buf.toOwnedSlice(allocator);
    }

    /// Parse the wire format. Rejects trailing bytes, bad magic/version.
    pub fn deserialize(allocator: std.mem.Allocator, wire: []const u8) error{ OutOfMemory, BadProofFormat }!Proof {
        if (wire.len < wire_header_len) return error.BadProofFormat;
        if (!std.mem.eql(u8, wire[0..4], &wire_magic)) return error.BadProofFormat;
        if (wire[4] != wire_version) return error.BadProofFormat;

        var leaf: Hash = undefined;
        @memcpy(&leaf, wire[8..40]);
        const leaf_index: usize = std.mem.readInt(u32, wire[40..44], .little);
        const n_sib: usize = std.mem.readInt(u32, wire[44..48], .little);

        if (wire.len != wire_header_len + n_sib * @sizeOf(Hash)) {
            return error.BadProofFormat;
        }
        const siblings = try allocator.alloc(Hash, n_sib);
        errdefer allocator.free(siblings);
        for (siblings, 0..) |*s, i| {
            @memcpy(s, wire[wire_header_len + i * @sizeOf(Hash) ..][0..@sizeOf(Hash)]);
        }
        return .{ .leaf = leaf, .siblings = siblings, .leaf_index = leaf_index };
    }

    /// One-shot: parse a serialized proof and verify it against
    /// `expected_root`. Frees the intermediate on both paths.
    pub fn verifySerialized(allocator: std.mem.Allocator, wire: []const u8, expected_root: Hash) error{ OutOfMemory, BadProofFormat }!bool {
        var p = try deserialize(allocator, wire);
        defer p.deinit(allocator);
        return p.verify(expected_root);
    }
};

test "merkle root stable under leaf permutation" {
    const t = std.testing;
    const a = t.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();

    const e1 = [_]Entry{
        .{ .name = "gate.0.w", .data = "AAAA" },
        .{ .name = "up.0.w", .data = "BBBB" },
        .{ .name = "down.0.w", .data = "CCCC" },
    };
    const e2 = [_]Entry{
        .{ .name = "down.0.w", .data = "CCCC" },
        .{ .name = "gate.0.w", .data = "AAAA" },
        .{ .name = "up.0.w", .data = "BBBB" },
    };
    var m1 = try MerkleTree.init(arena.allocator(), &e1);
    defer m1.deinit(arena.allocator());
    var m2 = try MerkleTree.init(arena.allocator(), &e2);
    defer m2.deinit(arena.allocator());
    try t.expectEqualSlices(u8, &m1.root(), &m2.root());
}

test "merkle root changes when data changes" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();

    const e1 = [_]Entry{.{ .name = "w", .data = "AAAA" }};
    const e2 = [_]Entry{.{ .name = "w", .data = "AAAB" }}; // 1 byte differs
    var m1 = try MerkleTree.init(arena.allocator(), &e1);
    defer m1.deinit(arena.allocator());
    var m2 = try MerkleTree.init(arena.allocator(), &e2);
    defer m2.deinit(arena.allocator());
    try std.testing.expect(!std.mem.eql(u8, &m1.root(), &m2.root()));
}

test "merkle root changes when name changes" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();

    const e1 = [_]Entry{.{ .name = "w1", .data = "AAAA" }};
    const e2 = [_]Entry{.{ .name = "w2", .data = "AAAA" }};
    var m1 = try MerkleTree.init(arena.allocator(), &e1);
    defer m1.deinit(arena.allocator());
    var m2 = try MerkleTree.init(arena.allocator(), &e2);
    defer m2.deinit(arena.allocator());
    try std.testing.expect(!std.mem.eql(u8, &m1.root(), &m2.root()));
}

test "merkle empty and single leaf" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();

    var m0 = try MerkleTree.init(alloc, &.{});
    defer m0.deinit(alloc);
    const e = [_]Entry{.{ .name = "solo", .data = "xyz" }};
    var m1 = try MerkleTree.init(alloc, &e);
    defer m1.deinit(alloc);
    // Single-leaf root == its own leaf hash (loop never runs).
    try std.testing.expectEqualSlices(u8, &hashLeafInner("solo", "xyz"), &m1.root());
    // Empty root differs from single-leaf root.
    try std.testing.expect(!std.mem.eql(u8, &m0.root(), &m1.root()));
}

test "merkle proof verifies and rejects" {
    const t = std.testing;
    const a = t.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();

    // 5 leaves (odd count exercises orphan self-pairing at multiple levels)
    const entries = [_]Entry{
        .{ .name = "a", .data = "1" },
        .{ .name = "b", .data = "2" },
        .{ .name = "c", .data = "3" },
        .{ .name = "d", .data = "4" },
        .{ .name = "e", .data = "5" },
    };
    var m = try MerkleTree.init(alloc, &entries);
    defer m.deinit(alloc);
    const r = m.root();

    // Positive: every leaf proves against the root — including the orphan.
    for (entries) |e| {
        var p = try m.proof(alloc, e.name);
        defer p.deinit(alloc);
        try t.expect(p.verify(r));
    }

    // Negative: tampered leaf hash (flip a bit) must fail.
    var p_bad = try m.proof(alloc, "c");
    defer p_bad.deinit(alloc);
    p_bad.leaf[0] ^= 0x01;
    try t.expect(!p_bad.verify(r));

    // Negative: tampered sibling must fail.
    var p_bad2 = try m.proof(alloc, "b");
    defer p_bad2.deinit(alloc);
    if (p_bad2.siblings.len > 0) p_bad2.siblings[0][3] ^= 0xFF;
    try t.expect(!p_bad2.verify(r));

    // Unknown leaf name.
    try t.expectError(error.LeafNotFound, m.proof(alloc, "nope"));
}

// --- Regression tests for the bugs found in review ---

test "merkle root is idempotent (root() twice)" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();

    const entries = [_]Entry{
        .{ .name = "a", .data = "1" },
        .{ .name = "b", .data = "2" },
        .{ .name = "c", .data = "3" },
        .{ .name = "d", .data = "4" },
    };
    var m = try MerkleTree.init(alloc, &entries);
    defer m.deinit(alloc);

    const r1 = m.root();
    const r2 = m.root();
    try std.testing.expectEqualSlices(u8, &r1, &r2);
}

test "merkle proofs work after root() was called (no leaf corruption)" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();

    const entries = [_]Entry{
        .{ .name = "a", .data = "1" },
        .{ .name = "b", .data = "2" },
        .{ .name = "c", .data = "3" },
    };
    var m = try MerkleTree.init(alloc, &entries);
    defer m.deinit(alloc);

    // root() first, then proofs — previously corrupted the leaves.
    _ = m.root();
    _ = m.root(); // and twice, previously returned a different value
    for (entries) |e| {
        var p = try m.proof(alloc, e.name);
        defer p.deinit(alloc);
        try std.testing.expect(p.verify(m.root()));
    }
}

test "merkle orphan leaf proves in every odd tree size" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();

    // Trees of 1, 3, 5, 7, 9 leaves: every leaf must prove, especially
    // the promoted/self-paired orphan at each level.
    for ([_]usize{ 1, 3, 5, 7, 9 }) |n| {
        var entries_buf: [9]Entry = undefined;
        var names_buf: [9][2]u8 = undefined;
        var data_buf: [9][2]u8 = undefined;
        for (0..n) |i| {
            names_buf[i] = .{ 'a' + @as(u8, @intCast(i)), 'w' };
            data_buf[i] = .{ 'A' + @as(u8, @intCast(i)), 'x' };
            entries_buf[i] = .{ .name = &names_buf[i], .data = &data_buf[i] };
        }
        var m = try MerkleTree.init(alloc, entries_buf[0..n]);
        defer m.deinit(alloc);
        const r = m.root();
        for (entries_buf[0..n]) |e| {
            var p = try m.proof(alloc, e.name);
            defer p.deinit(alloc);
            try std.testing.expect(p.verify(r));
        }
    }
}

test "builder: streaming add matches MerkleTree.init" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const entries = [_]Entry{
        .{ .name = "down.0.w", .data = "CCCC" },
        .{ .name = "gate.0.w", .data = "AAAA" },
        .{ .name = "up.0.w", .data = "BBBB" },
    };

    var b = Builder.init(alloc);
    defer b.deinit();
    for (entries) |e| try b.add(e.name, e.data);
    var m = try b.finish();
    defer m.deinit(alloc);

    var m2 = try MerkleTree.init(alloc, &entries);
    defer m2.deinit(alloc);
    try t.expectEqualSlices(u8, &m.root(), &m2.root());

    // Proofs work on the finished tree (borrowed names stay valid).
    for (entries) |e| {
        var p = try m.proof(alloc, e.name);
        defer p.deinit(alloc);
        try t.expect(p.verify(m.root()));
    }
}

test "builder: order-independent root" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var b1 = Builder.init(alloc);
    defer b1.deinit();
    try b1.add("a", "1");
    try b1.add("b", "2");
    try b1.add("c", "3");
    var m1 = try b1.finish();
    defer m1.deinit(alloc);

    var b2 = Builder.init(alloc);
    defer b2.deinit();
    try b2.add("c", "3");
    try b2.add("a", "1");
    try b2.add("b", "2");
    var m2 = try b2.finish();
    defer m2.deinit(alloc);

    try t.expectEqualSlices(u8, &m1.root(), &m2.root());
}

test "builder: duplicate names rejected" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var b = Builder.init(alloc);
    defer b.deinit();
    try b.add("w", "1");
    try b.add("w", "2"); // same name twice → ambiguous proofs
    try t.expectError(error.DuplicateName, b.finish());
}

test "builder: empty tree" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var b = Builder.init(alloc);
    defer b.deinit();
    var m = try b.finish();
    defer m.deinit(alloc);

    var m0 = try MerkleTree.init(alloc, &.{});
    defer m0.deinit(alloc);
    try t.expectEqualSlices(u8, &m.root(), &m0.root());
}

test "proof wire format roundtrip" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const entries = [_]Entry{
        .{ .name = "a", .data = "1" },
        .{ .name = "b", .data = "2" },
        .{ .name = "c", .data = "3" },
        .{ .name = "d", .data = "4" },
        .{ .name = "e", .data = "5" },
    };
    var m = try MerkleTree.init(alloc, &entries);
    defer m.deinit(alloc);

    var p = try m.proof(alloc, "c");
    defer p.deinit(alloc);
    const wire = try p.serialize(alloc);
    defer alloc.free(wire);

    // Exact layout: 48-byte header + 5 leaves → ceil levels = 3 siblings.
    try t.expectEqual(Proof.wire_header_len + p.siblings.len * 32, wire.len);

    var p2 = try Proof.deserialize(alloc, wire);
    defer p2.deinit(alloc);
    try t.expectEqualSlices(u8, &p.leaf, &p2.leaf);
    try t.expectEqual(p.leaf_index, p2.leaf_index);
    try t.expectEqual(p.siblings.len, p2.siblings.len);

    // One-shot verify against the real root — positive.
    try t.expect(try Proof.verifySerialized(alloc, wire, m.root()));
    // Negative: wrong root.
    var bad_root = m.root();
    bad_root[0] ^= 0xFF;
    try t.expect(!(try Proof.verifySerialized(alloc, wire, bad_root)));
}

test "proof wire format rejects malformed input" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const entries = [_]Entry{
        .{ .name = "a", .data = "1" },
        .{ .name = "b", .data = "2" },
    };
    var m = try MerkleTree.init(alloc, &entries);
    defer m.deinit(alloc);
    var p = try m.proof(alloc, "a");
    defer p.deinit(alloc);
    const wire = try p.serialize(alloc);
    defer alloc.free(wire);

    // Truncated header.
    try t.expectError(error.BadProofFormat, Proof.deserialize(alloc, wire[0..10]));
    // Bad magic.
    var bad = try alloc.dupe(u8, wire);
    defer alloc.free(bad);
    bad[0] = 'X';
    try t.expectError(error.BadProofFormat, Proof.deserialize(alloc, bad));
    // Bad version.
    var bad2 = try alloc.dupe(u8, wire);
    defer alloc.free(bad2);
    bad2[4] = 99;
    try t.expectError(error.BadProofFormat, Proof.deserialize(alloc, bad2));
    // Trailing bytes.
    var long = try alloc.alloc(u8, wire.len + 1);
    defer alloc.free(long);
    @memcpy(long[0..wire.len], wire);
    long[wire.len] = 0;
    try t.expectError(error.BadProofFormat, Proof.deserialize(alloc, long));
    // Lying sibling count.
    var bad3 = try alloc.dupe(u8, wire);
    defer alloc.free(bad3);
    bad3[44] += 1;
    try t.expectError(error.BadProofFormat, Proof.deserialize(alloc, bad3));
}
