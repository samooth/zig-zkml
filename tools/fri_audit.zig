//! Spike: zig-algebra FRI v2 audit — is it a real low-degree test?
//!
//! FRI v2 uses the canonical 2-adic subgroup, antipodal x/-x folds,
//! Merkle layer commitments and a final degree anchor. The audit keeps
//! the same three gates: a low-degree polynomial must verify, arbitrary
//! data must be rejected, and a polynomial above the configured bound
//! must be rejected.
//!
//! The transcript is kept local so this audit exercises only the FRI
//! interface. The field comes from zig-algebra's zig-field module.
//!
//! Field: zig-algebra Goldilocks. Run: zig build spike

const std = @import("std");
const field = @import("zig-field");
const fri = @import("zig-fri");
const Field = field.Goldilocks;

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
    const domain = fri.Domain(Field).init(Field, 8);
    const config = fri.Config{
        .log_domain = 8,
        .log_initial_degree = 7,
        .log_final = 3,
        .log_residual_degree = 2,
        .num_queries = 20,
    };

    var pass_count: usize = 0;
    var total: usize = 0;

    // --- Test 1: sanity — a degree-2 polynomial below the configured bound ---
    {
        var evals: [n]Field = undefined;
        const c3 = Field.fromInt(3);
        const c7 = Field.fromInt(7);
        for (0..n) |i| {
            const x = domain.at(i);
            evals[i] = x.sqr().add(x.mul(c3)).add(c7);
        }
        const ok = try proveAndVerify(allocator, &evals, config);
        total += 1;
        if (ok) {
            pass_count += 1;
            std.debug.print("test 1  degree-2 poly (below configured bound): VERIFIES (expected)\n", .{});
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
            var evals: [n]Field = undefined;
            for (0..n) |i| {
                const r = rnd.int(u32) ^ (@as(u64, t) << 32) ^ i;
                evals[i] = Field.fromInt(r);
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

    // --- Test 3: medium degree above the configured bound, honestly proven ---
    {
        var accepted: usize = 0;
        const trials = 16;
        for (0..trials) |t| {
            var evals: [n]Field = undefined;
            const tt = Field.fromInt(t);
            for (0..n) |i| {
                const x = domain.at(i);
                var acc = Field.zero();
                var xp = Field.one();
                for (0..129) |k| {
                    const c = Field.fromInt(k *% 2654435761).add(tt);
                    acc = acc.add(c.mul(xp));
                    xp = xp.mul(x);
                }
                evals[i] = acc;
            }
            const ok = try proveAndVerify(allocator, &evals, config);
            if (ok) accepted += 1;
        }
        total += 1;
        std.debug.print("test 3  degree-128 data (above configured bound): {d}/{d} verified\n", .{ accepted, trials });
        if (accepted <= 1) pass_count += 1; // allowance for a fluke
    }

    std.debug.print("\n=== FRI AUDIT: {d}/{d} gates passed ===\n", .{ pass_count, total });
    if (pass_count != total) {
        std.debug.print("VERDICT: zig-algebra FRI is NOT a sound low-degree test — do NOT build F2 on it.\n", .{});
        return error.FriAuditFailed;
    } else {
        std.debug.print("VERDICT: FRI rejects non-low-degree data — viable to evaluate further.\n", .{});
    }
}

fn proveAndVerify(allocator: std.mem.Allocator, evals: []const Field, config: fri.Config) !bool {
    var pt = FriTranscript.init("v2");
    var proof = try fri.prove(Field, allocator, &pt, evals, config);
    defer proof.deinit(allocator);
    var vt = FriTranscript.init("v2");
    return fri.verify(Field, &vt, &proof, config);
}
