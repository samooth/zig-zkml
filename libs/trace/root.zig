//! Trace recorder — captures witness data from native inference kernels.
//!
//! BLUE_PRINT §6.2: the TraceRecorder is multi-threaded (ktransformers-zig
//! runs experts in parallel). Recording is by slot (layer, op, expert,
//! tp_rank) with per-slot buffers. `finalize()` absorbs in CANONICAL order
//! so the Fiat-Shamir transcript is deterministic regardless of thread
//! scheduling.

const std = @import("std");
const field = @import("../field.zig");
const transcript = @import("../transcript.zig");

pub const Op = enum {
    gemm_a,
    gemm_b,
    gemm_c,
    dequant,
    requant,
    swiglu,
    layernorm,
    rmsnorm,
    routing_topk,
    routing_gate,
    other,

    pub fn order(a: Op, b: Op) std.math.Order {
        return std.math.order(@intFromEnum(a), @intFromEnum(b));
    }
};

pub const SlotKey = struct {
    layer: u32,
    op: Op,
    expert: u16,
    rank: u8,

    /// Canonical ordering for deterministic absorption.
    pub fn order(a: SlotKey, b: SlotKey) std.math.Order {
        if (a.layer != b.layer) return std.math.order(a.layer, b.layer);
        if (@intFromEnum(a.op) != @intFromEnum(b.op))
            return std.math.order(@intFromEnum(a.op), @intFromEnum(b.op));
        if (a.expert != b.expert) return std.math.order(a.expert, b.expert);
        return std.math.order(a.rank, b.rank);
    }
};

pub const Slot = struct {
    /// Raw witness data (Goldilocks elements serialized as 8-byte LE).
    data: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Slot, allocator: std.mem.Allocator) void {
        self.data.deinit(allocator);
    }
};

pub const TraceRecorder = struct {
    gpa: std.mem.Allocator,
    mutex: std.atomic.Mutex = .unlocked,
    slots: std.AutoHashMap(SlotKey, Slot),

    pub fn init(gpa: std.mem.Allocator) TraceRecorder {
        return .{
            .gpa = gpa,
            .slots = std.AutoHashMap(SlotKey, Slot).init(gpa),
        };
    }

    pub fn deinit(self: *TraceRecorder) void {
        var it = self.slots.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit(self.gpa);
        }
        self.slots.deinit();
    }

    /// Record witness data for a slot. Thread-safe (spinlock-protected;
    /// backoff hint keeps contention cheap).
    /// Data is copied into the slot's buffer; no absorption happens here.
    pub fn record(self: *TraceRecorder, key: SlotKey, data: []const u8) error{OutOfMemory}!void {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();
        const gop = try self.slots.getOrPut(key);
        if (!gop.found_existing) gop.value_ptr.* = .{};
        try gop.value_ptr.data.appendSlice(self.gpa, data);
    }

    /// Record a single Goldilocks element for a slot.
    pub fn recordElement(self: *TraceRecorder, key: SlotKey, elem: field.Goldilocks) error{OutOfMemory}!void {
        var buf: [8]u8 = undefined;
        elem.toBytes(&buf);
        try self.record(key, &buf);
    }

    /// Finalize: absorb all recorded slots into a transcript in CANONICAL
    /// order (§6.2). Deterministic hash independent of thread scheduling.
    /// Allocation failures PROPAGATE — a transcript must never silently
    /// drop witness data (previously `catch continue`d, a soundness hole).
    pub fn finalize(self: *TraceRecorder, stmt_hash: *const [32]u8) error{OutOfMemory}![32]u8 {
        while (!self.mutex.tryLock()) {
            std.atomic.spinLoopHint();
        }
        defer self.mutex.unlock();

        var tr = transcript.Transcript.init("zkml.trace");
        tr.absorb(stmt_hash);

        // Collect keys and sort canonically.
        var keys = std.ArrayList(SlotKey).empty;
        defer keys.deinit(self.gpa);
        try keys.ensureTotalCapacity(self.gpa, self.slots.count());
        var it = self.slots.keyIterator();
        while (it.next()) |key| {
            keys.appendAssumeCapacity(key.*);
        }

        std.sort.insertion(SlotKey, keys.items, {}, struct {
            fn lessThan(_: void, a: SlotKey, b: SlotKey) bool {
                return a.order(b) == .lt;
            }
        }.lessThan);

        // Absorb each slot: tag (layer|op|expert|rank) || len || data.
        for (keys.items) |key| {
            var tag: [12]u8 = undefined;
            std.mem.writeInt(u32, tag[0..4], key.layer, .little);
            std.mem.writeInt(u16, tag[4..6], @intFromEnum(key.op), .little);
            std.mem.writeInt(u16, tag[6..8], key.expert, .little);
            tag[8] = key.rank;
            tag[9] = 0; // padding
            tag[10] = 0;
            tag[11] = 0;
            tr.absorb(&tag);

            if (self.slots.get(key)) |slot| {
                var len_buf: [8]u8 = undefined;
                std.mem.writeInt(u64, &len_buf, slot.data.items.len, .little);
                tr.absorb(&len_buf);
                tr.absorb(slot.data.items);
            }
        }

        return tr.finish();
    }
};

test "trace recorder determinism" {
    const t = std.testing;
    const a = t.allocator;

    // Two recorders with same data in different insertion order.
    var rec1 = TraceRecorder.init(a);
    defer rec1.deinit();
    var rec2 = TraceRecorder.init(a);
    defer rec2.deinit();

    const k1 = SlotKey{ .layer = 0, .op = .gemm_a, .expert = 5, .rank = 0 };
    const k2 = SlotKey{ .layer = 0, .op = .gemm_b, .expert = 5, .rank = 0 };

    // Rec1: k1 then k2
    try rec1.record(k1, "AAAA");
    try rec1.record(k2, "BBBB");

    // Rec2: k2 then k1 (different order)
    try rec2.record(k2, "BBBB");
    try rec2.record(k1, "AAAA");

    const stmt = [_]u8{0xaa} ** 32;
    const h1 = try rec1.finalize(&stmt);
    const h2 = try rec2.finalize(&stmt);

    try t.expectEqualSlices(u8, &h1, &h2);
}

test "trace recorder different data differs" {
    const t = std.testing;
    const a = t.allocator;

    var rec1 = TraceRecorder.init(a);
    defer rec1.deinit();
    var rec2 = TraceRecorder.init(a);
    defer rec2.deinit();

    const k = SlotKey{ .layer = 1, .op = .gemm_a, .expert = 0, .rank = 0 };
    try rec1.record(k, "AAAA");
    try rec2.record(k, "AAAB"); // 1 bit differs

    const stmt = [_]u8{0xaa} ** 32;
    const h1 = try rec1.finalize(&stmt);
    const h2 = try rec2.finalize(&stmt);
    try t.expect(!std.mem.eql(u8, &h1, &h2));
}

test "trace recorder slot key ordering" {
    const t = std.testing;
    const k1 = SlotKey{ .layer = 0, .op = .gemm_a, .expert = 0, .rank = 0 };
    const k2 = SlotKey{ .layer = 1, .op = .gemm_a, .expert = 0, .rank = 0 };
    const k3 = SlotKey{ .layer = 0, .op = .gemm_b, .expert = 0, .rank = 0 };

    try t.expect(k1.order(k2) == .lt);
    try t.expect(k2.order(k1) == .gt);
    try t.expect(k1.order(k3) == .lt);
    try t.expect(k3.order(k1) == .gt);
}

test "trace multi-thread determinism (§13: mismo witness multi-thread)" {
    const t = std.testing;
    const a = t.allocator;

    const n_threads = 4;
    const n_slots_per_thread = 16;

    // Multi-threaded recorder: each thread writes to ITS OWN set of slots
    // (the realistic case — intra-slot order is semantically meaningful).
    var rec_mt = TraceRecorder.init(a);
    defer rec_mt.deinit();

    const Worker = struct {
        fn run(rec: *TraceRecorder, rank: u8) void {
            for (0..n_slots_per_thread) |s| {
                const key = SlotKey{
                    .layer = @intCast(s / 4),
                    .op = .gemm_a,
                    .expert = @intCast(s % 4),
                    .rank = rank,
                };
                var buf: [8]u8 = undefined;
                std.mem.writeInt(u64, &buf, @as(u64, rank) *% 100 +% s, .little);
                rec.record(key, &buf) catch @panic("record OOM in test");
            }
        }
    };

    var threads: [n_threads]std.Thread = undefined;
    for (0..n_threads) |i| {
        threads[i] = try std.Thread.spawn(.{}, Worker.run, .{ &rec_mt, @as(u8, @intCast(i)) });
    }
    for (&threads) |*th| th.join();

    // Single-threaded recorder: same data, written sequentially in a
    // scrambled (non-canonical) order.
    var rec_st = TraceRecorder.init(a);
    defer rec_st.deinit();
    for (0..n_threads) |r| {
        for (0..n_slots_per_thread) |s| {
            const key = SlotKey{
                .layer = @intCast(s / 4),
                .op = .gemm_a,
                .expert = @intCast(s % 4),
                .rank = @intCast(r),
            };
            var buf: [8]u8 = undefined;
            std.mem.writeInt(u64, &buf, @as(u64, r) *% 100 +% s, .little);
            try rec_st.record(key, &buf);
        }
    }

    const stmt = [_]u8{0x5a} ** 32;
    const h_mt = try rec_mt.finalize(&stmt);
    const h_st = try rec_st.finalize(&stmt);

    // Same witness from multi-threaded execution → same transcript hash
    // (BLUE_PRINT §13 determinism requirement).
    try t.expectEqualSlices(u8, &h_mt, &h_st);
}
