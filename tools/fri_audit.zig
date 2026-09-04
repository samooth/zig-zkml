//! Spike: zig-algebra FRI audit — is it a real low-degree test?
//!
//! zig-algebra's FRI is "index-pairing" over arbitrary array positions:
//! the fold is layer[j] = layer[2j] + alpha*layer[2j+1] with NO evaluation
//! domain (no coset of roots of unity, no f(x)/f(-x) mirroring) and NO
//! final degree check. Canonical FRI derives soundness from the RS-code
//! structure: evaluations lie on a 2-adic coset, the fold pairs x with -x,
//! and the degree halves. Index-pairing adjacent slots with one linear
//! challenge encodes no polynomial structure at all.
//!
//! THE GO/NO-GO TEST: random data, honestly folded and committed by the
//! prover, must REJECT for a real low-degree test (random data is
//! maximally far from any RS code). If it verifies, this "FRI" is only a
//! Merkle-consistency game and cannot anchor a STARK's degree claim.
//!
//! Self-contained by design: the field and transcript are defined inline
//! (FRI takes them via comptime F / anytype), avoiding the upstream
//! zig-transcript duplicate-module bug ("file exists in modules
//! 'zig-transcript' and 'zig-transcript0'" — its build.zig registers the
//! same source as two distinct modules for fri's inner import and for
//! consumers; any consumer importing both fri and the transcript breaks).
//!
//! Field: Goldilocks p = 2^61-1 (BLUE_PRINT §4). Run: zig build spike

const std = @import("std");
const fri = @import("zig-fri");

/// Minimal Goldilocks p = 2^61-1 satisfying exactly the interface the FRI
/// consumes: NUM_BYTES / toBytes / fromBytes / add / mul / eql (+ test
/// helpers). Arithmetic mirrors libs/field.zig (lazy-reduction fold).
const M61 = struct {
    pub const NUM_BYTES: usize = 8;
    pub const MODULUS: u64 = (1 << 61) - 1;

    rep: u64,

    pub fn fromInt(x: u64) M61 {
        return .{ .rep = x % MODULUS };
    }
    pub fn zero() M61 {
        return .{ .rep = 0 };
    }
    pub fn one() M61 {
        return .{ .rep = 1 };
    }
    pub fn eql(a: M61, b: M61) bool {
        return a.rep == b.rep;
    }
    pub fn add(a: M61, b: M61) M61 {
        const s = a.rep + b.rep; // both < p < 2^61: single conditional sub
        return .{ .rep = if (s >= MODULUS) s - MODULUS else s };
    }
    pub fn mul(a: M61, b: M61) M61 {
        // 2^61 ≡ 1 (mod p): fold high bits into low, then subtract p at most twice.
        const prod = @as(u128, a.rep) * @as(u128, b.rep);
        const lo: u64 = @intCast(prod & MODULUS);
        const hi: u64 = @intCast(prod >> 61);
        var r = lo + hi;
        while (r >= MODULUS) r -= MODULUS;
        return .{ .rep = r };
    }
    pub fn sqr(a: M61) M61 {
        return a.mul(a);
    }
    pub fn toBytes(self: M61) [NUM_BYTES]u8 {
        var b: [NUM_BYTES]u8 = undefined;
        std.mem.writeInt(u64, &b, self.rep, .little);
        return b;
    }
    pub fn fromBytes(bytes: []const u8) !M61 {
        if (bytes.len != NUM_BYTES) return error.InvalidLength;
        const v = std.mem.readInt(u64, bytes[0..NUM_BYTES], .little);
        if (v >= MODULUS) return error.OutOfField;
        return .{ .rep = v };
    }
};

/// Fiat-Shamir transcript matching the interface FRI drives (absorbBytes /
/// absorbField / challengeField / challengeU64). Blake3 with
/// rekey-on-squeeze; field challenges reduce by rejection sampling.
const FriTranscript = struct {
    state: std.crypto.hash.Blake3,

    pub fn init(domain: []const u8) FriTranscript {
        var t = FriTranscript{ .state = std.crypto.hash.Blake3.init(.{}) };
        t.state.update("zkml.fri-audit/");
        t.state.update(domain);
        return t;
    }

    pub fn absorbBytes(self: *FriTranscript, bytes: []const u8) void {
        self.state.update(bytes);
    }

    pub fn absorbField(self: *FriTranscript, comptime F: type, e: F) void {
        self.state.update(&e.toBytes());
    }

    pub fn challenge32(self: *FriTranscript) [32]u8 {
        var out: [32]u8 = undefined;
        self.state.final(&out);
        // Squeeze-and-rekey: every challenge advances the state.
        self.state = std.crypto.hash.Blake3.init(.{});
        self.state.update(&out);
        return out;
    }

    pub fn challengeU64(self: *FriTranscript) u64 {
        const c = self.challenge32();
        return std.mem.readInt(u64, c[0..8], .little);
    }

    pub fn challengeField(self: *FriTranscript, comptime F: type) F {
        // Rejection sampling: resqueeze until the bytes are < p.
        while (true) {
            const c = self.challenge32();
            if (F.fromBytes(c[0..F.NUM_BYTES])) |v| return v else |_| {}
        }
    }
};

pub fn main() !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer _ = debug_allocator.deinit();
    const allocator = debug_allocator.allocator();

    const n = 256;
    const config = fri.Config{
        .domain_size = n,
        .final_length = 8,
        .num_queries = 20,
    };

    var pass_count: usize = 0;
    var total: usize = 0;

    // --- Test 1: sanity — a degree-2 polynomial (degree < final_length) ---
    {
        var evals: [n]M61 = undefined;
        const c3 = M61.fromInt(3);
        const c7 = M61.fromInt(7);
        for (0..n) |i| {
            const x = M61.fromInt(i);
            evals[i] = x.sqr().add(x.mul(c3)).add(c7);
        }
        const ok = try proveAndVerify(allocator, &evals, config);
        total += 1;
        if (ok) {
            pass_count += 1;
            std.debug.print("test 1  degree-2 poly (deg < final 8): VERIFIES (expected)\n", .{});
        } else {
            std.debug.print("test 1  degree-2 poly: REJECTS (BROKEN — even honest low-degree data fails)\n", .{});
        }
    }

    // --- Test 2: GO/NO-GO — arbitrary random data, honestly proven ---
    {
        var prng = std.Random.DefaultPrng.init(0xC0FFEE);
        const rnd = prng.random();
        var accepted: usize = 0;
        const trials = 16;
        for (0..trials) |t| {
            var evals: [n]M61 = undefined;
            for (0..n) |i| {
                const r = rnd.int(u32) ^ (@as(u64, t) << 32) ^ i;
                evals[i] = M61.fromInt(r);
            }
            // Honest prover: layers derived from the data, real Merkle
            // commitments, consistent everything. A REAL low-degree test
            // must reject this with overwhelming probability (random data
            // is maximally far from any Reed-Solomon code).
            const ok = try proveAndVerify(allocator, &evals, config);
            if (ok) accepted += 1;
        }
        total += 1;
        std.debug.print("test 2  ARBITRARY random data, honest prover: {d}/{d} verified\n", .{ accepted, trials });
        if (accepted == 0) {
            pass_count += 1;
            std.debug.print("         -> rejected as expected: FRI behaves as a low-degree test\n", .{});
        } else {
            std.debug.print("         -> ACCEPTED non-low-degree data: the FRI certifies NOTHING about degree\n", .{});
        }
    }

    // --- Test 3: medium degree (128 >> final_length 8), honestly proven ---
    {
        var accepted: usize = 0;
        const trials = 16;
        for (0..trials) |t| {
            var evals: [n]M61 = undefined;
            const tt = M61.fromInt(t);
            for (0..n) |i| {
                const x = M61.fromInt(i);
                var acc = M61.zero();
                var xp = M61.one();
                for (0..129) |k| {
                    const c = M61.fromInt(k *% 2654435761).add(tt);
                    acc = acc.add(c.mul(xp));
                    xp = xp.mul(x);
                }
                evals[i] = acc;
            }
            const ok = try proveAndVerify(allocator, &evals, config);
            if (ok) accepted += 1;
        }
        total += 1;
        std.debug.print("test 3  degree-128 data (>> final 8): {d}/{d} verified\n", .{ accepted, trials });
        if (accepted <= 1) pass_count += 1; // allowance for a fluke
    }

    std.debug.print("\n=== FRI AUDIT: {d}/{d} gates passed ===\n", .{ pass_count, total });
    if (pass_count != total) {
        std.debug.print("VERDICT: zig-algebra FRI is NOT a sound low-degree test — do NOT build F2 on it.\n", .{});
    } else {
        std.debug.print("VERDICT: FRI rejects non-low-degree data — viable to evaluate further.\n", .{});
    }
}

fn proveAndVerify(allocator: std.mem.Allocator, evals: []const M61, config: fri.Config) !bool {
    var pt = FriTranscript.init("v1");
    var proof = try fri.prove(M61, allocator, &pt, evals, config);
    defer proof.deinit(allocator);
    var vt = FriTranscript.init("v1");
    return fri.verify(M61, &vt, &proof, config);
}
