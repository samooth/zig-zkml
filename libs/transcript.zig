//! Fiat-Shamir transcript — Blake3-based, domain-separated.
//!
//! F1 deterministic sampling (docs/BLUE_PRINT.md §6.2): absorbs serialized data in
//! canonical order and squeezes challenges. The transcript is initialized
//! with a domain separator derived from the statement so that different
//! statements produce independent challenge streams.

const std = @import("std");

/// Field interface required by absorbField/challengeField: NUM_BYTES,
/// toBytes(), fromBytes([]const u8) error{...}!Self, and a modulus bound
/// (rejection sampling needs fromBytes to fail on non-canonical bytes).
pub const Transcript = struct {
    state: std.crypto.hash.Blake3,

    /// Create a new transcript. `domain` is absorbed first to separate
    /// different protocols/statements.
    pub fn init(domain: []const u8) Transcript {
        var t = Transcript{
            .state = std.crypto.hash.Blake3.init(.{}),
        };
        t.state.update(domain);
        return t;
    }

    /// Absorb arbitrary data into the transcript.
    pub fn absorb(self: *Transcript, data: []const u8) void {
        // Length-prefix to prevent length-extension ambiguity.
        var len_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &len_buf, data.len, .little);
        self.state.update(&len_buf);
        self.state.update(data);
    }

    /// Absorb raw bytes WITHOUT length prefix (Merkle roots, fixed-size
    /// digests): domain-tagged so it cannot be confused with absorb().
    pub fn absorbBytes(self: *Transcript, bytes: []const u8) void {
        self.state.update("raw");
        self.state.update(bytes);
    }

    /// Squeeze a 64-byte challenge (two independent 32-byte chunks).
    pub fn challenge64(self: *Transcript) [64]u8 {
        var out: [64]u8 = undefined;
        // Blake3 supports extended output natively.
        self.state.final(&out);
        // Re-initialize from the challenge to chain (squeeze-and-rekey).
        self.state = std.crypto.hash.Blake3.init(.{});
        self.state.update(&out);
        return out;
    }

    /// Squeeze a u64 challenge (lower 8 bytes of a 32-byte challenge).
    pub fn challengeU64(self: *Transcript) u64 {
        const full = self.challenge32();
        return std.mem.readInt(u64, full[0..8], .little);
    }

    /// Squeeze a 32-byte challenge.
    pub fn challenge32(self: *Transcript) [32]u8 {
        var out: [32]u8 = undefined;
        self.state.final(&out);
        self.state = std.crypto.hash.Blake3.init(.{});
        self.state.update(&out);
        return out;
    }

    /// Finalize and return the 32-byte running state hash.
    /// Absorb a field element (needs NUM_BYTES + toBytes + a small
    /// prefix domain-separating field absorptions from raw bytes).
    pub fn absorbField(self: *Transcript, comptime F: type, e: F) void {
        self.state.update("fld");
        self.state.update(&e.toBytes());
    }

    /// Squeeze a field element via rejection sampling: F.fromBytes must
    /// reject non-canonical encodings (>= modulus), which resqueezes.
    pub fn challengeField(self: *Transcript, comptime F: type) F {
        while (true) {
            const c = self.challenge32();
            if (F.fromBytes(c[0..F.NUM_BYTES])) |v| return v else |_| {}
        }
    }

    pub fn finish(self: *Transcript) [32]u8 {
        var out: [32]u8 = undefined;
        self.state.final(&out);
        return out;
    }
};

test "transcript determinism" {
    const t = std.testing;

    // Same absorb order → same challenge.
    var tr1 = Transcript.init("test-domain");
    tr1.absorb("hello");
    tr1.absorb("world");
    const c1 = tr1.challengeU64();

    var tr2 = Transcript.init("test-domain");
    tr2.absorb("hello");
    tr2.absorb("world");
    const c2 = tr2.challengeU64();

    try t.expectEqual(c1, c2);
}

test "transcript domain separation" {
    const t = std.testing;

    var tr1 = Transcript.init("domain-A");
    tr1.absorb("data");
    const c1 = tr1.challengeU64();

    var tr2 = Transcript.init("domain-B");
    tr2.absorb("data");
    const c2 = tr2.challengeU64();

    try t.expect(c1 != c2);
}

test "transcript challenge32 roundtrip" {
    const t = std.testing;
    var tr = Transcript.init("roundtrip");
    tr.absorb("input");
    const c = tr.challenge32();
    // Verify we can chain: absorb challenge, squeeze again.
    tr.absorb(&c);
    const c2 = tr.challenge32();
    try t.expect(!std.mem.eql(u8, &c, &c2));
}
