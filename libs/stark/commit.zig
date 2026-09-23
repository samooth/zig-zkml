//! Column commitment for the STARK backend: a Merkle tree whose leaf at
//! index i covers every column's LDE values at rows i AND i+1.
//!
//! Two-point leaves are what make row-shifted constraints (the running-sum
//! `s' = s + a*b`) checkable with a single opening: the verifier needs the
//! columns at the query point and at its successor, and one path
//! authenticates both. The successor wraps cyclically (row N-1's successor
//! is row 0), so the trace domain is treated as a cycle — a running sum
//! that does not wrap consistently is a trace the prover cannot attest.
//!
//! Hashing follows the FRI leaf convention (domain-separated Blake3 over
//! the 16-byte Fp2 encodings) and reuses zig-merkle's tree, so both
//! commitments in a proof share one hashing discipline.

const std = @import("std");
const fp2 = @import("../fri/fp2.zig");
const merkle_pkg = @import("zig-merkle");

pub const Fp2 = fp2.Fp2;
pub const HASH_LEN = 32;

const NodeHash = struct {
    pub fn hashBytes(input: []const u8) [HASH_LEN]u8 {
        var out: [HASH_LEN]u8 = undefined;
        std.crypto.hash.Blake3.hash(input, &out, .{});
        return out;
    }
};
pub const MerkleTree = merkle_pkg.MerkleTree(NodeHash);
pub const MerkleProof = merkle_pkg.MerkleProof;

/// The one hashing convention for a column window: every column's values
/// at the previous, current and next rows. Both the prover (which has the
/// full LDE) and the verifier (which has only the opening) must agree on
/// it byte for byte, so it lives here alone.
pub fn hashWindow(
    ncols: usize,
    prev: []const Fp2,
    current: []const Fp2,
    next: []const Fp2,
) [HASH_LEN]u8 {
    var h = std.crypto.hash.Blake3.init(.{});
    h.update("zkml.stark.leaf");
    for (0..ncols) |k| {
        h.update(&prev[k].toBytes());
        h.update(&current[k].toBytes());
        h.update(&next[k].toBytes());
    }
    var out: [HASH_LEN]u8 = undefined;
    h.final(&out);
    return out;
}

/// Leaf for LDE row `i`, gathered from the full column vectors.
pub fn leafHash(columns: []const []const Fp2, i: usize, stride: usize) [HASH_LEN]u8 {
    const n = columns[0].len;
    const prev_i = (i + n - stride) % n;
    const next_i = (i + stride) % n;
    const ncols = columns.len;
    var h = std.crypto.hash.Blake3.init(.{});
    h.update("zkml.stark.leaf");
    for (0..ncols) |k| {
        const col = columns[k];
        h.update(&col[prev_i].toBytes());
        h.update(&col[i].toBytes());
        h.update(&col[next_i].toBytes());
    }
    var out: [HASH_LEN]u8 = undefined;
    h.final(&out);
    return out;
}

/// Build the leaf vector for a commitment (one leaf per LDE position).
pub fn buildLeaves(allocator: std.mem.Allocator, columns: []const []const Fp2, stride: usize) ![][HASH_LEN]u8 {
    const n = columns[0].len;
    const leaves = try allocator.alloc([HASH_LEN]u8, n);
    for (0..n) |i| leaves[i] = leafHash(columns, i, stride);
    return leaves;
}

pub const Commitment = struct {
    root: [HASH_LEN]u8,
    log_size: u6,
    num_columns: u16,
};

/// Commit to the LDE columns and return the root.
pub fn commit(allocator: std.mem.Allocator, columns: []const []const Fp2, stride: usize) !Commitment {
    const leaves = try buildLeaves(allocator, columns, stride);
    defer allocator.free(leaves);
    var tree = MerkleTree.initFromHashes(allocator, leaves) catch return error.OutOfMemory;
    defer tree.deinit();
    const n = columns[0].len;
    return .{
        .root = tree.root(),
        .log_size = @intCast(std.math.log2_int(usize, n)),
        .num_columns = @intCast(columns.len),
    };
}

pub const Error = error{ InvalidLeaf, OutOfMemory };

/// Verify a leaf against the commitment root.
pub fn verifyLeaf(
    commitment: Commitment,
    index: usize,
    leaf: [HASH_LEN]u8,
    path: MerkleProof,
) Error!void {
    if (!MerkleTree.verifyHashed(commitment.root, index, leaf, path)) {
        return Error.InvalidLeaf;
    }
}

const testing = std.testing;

fn f(v: u64) Fp2 {
    return Fp2.re(fp2.Goldilocks.fromU64(v));
}

test "commit: leaf binds both the row and its cyclic successor" {
    const a = testing.allocator;
    const n = 8;
    const col_a = [_]Fp2{ f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8) };
    var col_b = col_a;
    col_b[0] = f(99);
    const columns = [_][]const Fp2{ &col_a, &col_b };

    const c = try commit(a, &columns, 1);
    try testing.expectEqual(@as(u16, 2), c.num_columns);
    try testing.expectEqual(@as(u6, 3), c.log_size);

    const leaves = try buildLeaves(a, &columns, 1);
    defer a.free(leaves);
    var tree = MerkleTree.initFromHashes(a, leaves) catch return error.OutOfMemory;
    defer tree.deinit();

    // A genuine leaf verifies.
    {
        const path = tree.prove(3, a) catch return error.OutOfMemory;
        defer path.deinit(a);
        try verifyLeaf(c, 3, leaves[3], path);
    }

    // Flipping the successor row (i+1) breaks leaf i even though the value
    // at row i itself is untouched — this is the row-shift guarantee.
    {
        const tampered = a.dupe(Fp2, &col_b) catch return error.OutOfMemory;
        defer a.free(tampered);
        tampered[4] = f(1234);
        const bad_columns = [_][]const Fp2{ &col_a, tampered };
        const bad_leaf = leafHash(&bad_columns, 3, 1);
        try testing.expect(!std.mem.eql(u8, &bad_leaf, &leaves[3]));

        const path = tree.prove(3, a) catch return error.OutOfMemory;
        defer path.deinit(a);
        try testing.expectError(Error.InvalidLeaf, verifyLeaf(c, 3, bad_leaf, path));
    }

    // The previous row is bound too (offset -1).
    {
        const tampered = a.dupe(Fp2, &col_a) catch return error.OutOfMemory;
        defer a.free(tampered);
        tampered[2] = f(4321);
        const bad_columns = [_][]const Fp2{ tampered, &col_b };
        try testing.expect(!std.mem.eql(u8, &leafHash(&bad_columns, 3, 1), &leaves[3]));
    }

    // The wrap-around leaf (N-1) commits row 0 as its successor.
    const last = leafHash(&columns, n - 1, 1);
    const first = leafHash(&columns, 0, 1);
    try testing.expect(!std.mem.eql(u8, &last, &first));
}
