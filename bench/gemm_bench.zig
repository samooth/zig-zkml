//! F2 bench: what the STARK prover and verifier actually cost per GEMM
//! output element, and how the two trace layouts compare.
//!
//! Run it through the build so it is always ReleaseFast — measuring Debug
//! code would be measuring the wrong thing:
//!
//!   zig build bench
//!   zig build bench -- --k 1024 --repeat 3
//!
//! Both layouts use the same operand binding (raw nibble + block scale), but
//! the exact no-padding shape is different: 1 MAC/row accepts k = 2^m - 1,
//! while 16 MACs/row accepts k = 16·(2^m - 1). Those sets are disjoint, so
//! the bench reports the nearest valid shape for each layout and prints both k
//! values instead of claiming a same-reduction comparison:
//!
//!   1 MAC/row    gemm_air    + quant_binding   (4 + 16 columns)
//!   16 MACs/row  gemm_chunk  + chunk_binding   (34 + 226 columns)
//!
//! Chunking cuts the TRACE 16x, but it also multiplies the COLUMNS by 14, and
//! columns dominate. The lever is therefore not the row count, it is the
//! 192 binding columns: 12 per operand. A LogUp membership check against a
//! 16-entry nibble table would replace 5 columns with 1.

const std = @import("std");
const zkml = @import("zkml");

const tensor = zkml.tensor;
const stark = zkml.stark;
const gemm_air = zkml.gemm_air;
const gemm_chunk = zkml.gemm_chunk;
const quant = zkml.quant_binding;
const cbind = zkml.chunk_binding;

const Goldilocks = tensor.Goldilocks;

const Fp2 = stark.Fp2;

const Stats = struct {
    label: []const u8,
    k: usize,
    rows: usize,
    columns: usize,
    composed: usize,
    prove_ns: u64,
    verify_ns: u64,
    proof_bytes: usize,
    verified: bool,
};

/// Bytes the proof actually carries (field elements, hashes and the
/// small scalars), not the in-memory bookkeeping around them.
fn proofBytes(proof: stark.Proof) usize {
    const fp2_bytes = 2 * @sizeOf(u64);
    const hash_bytes = 32;
    var total: usize = hash_bytes; // commitment root
    total += @sizeOf(u16) + @sizeOf(u6); // num_columns, log_size
    for (proof.fri_proof.layers) |_| total += hash_bytes;
    total += proof.fri_proof.residual.len * fp2_bytes;
    total += proof.fri_proof.queries.len * @sizeOf(usize); // pair index
    for (proof.fri_proof.queries) |q| {
        total += q.values.len * fp2_bytes;
        total += q.paths.len * hash_bytes;
        for (q.paths) |p| total += p.siblings.len * hash_bytes;
    }
    for (proof.openings) |o| total += openingBytes(o, fp2_bytes, hash_bytes);
    for (proof.boundary_openings) |o| total += openingBytes(o, fp2_bytes, hash_bytes);
    return total;
}

fn openingBytes(o: stark.Opening, fp2_bytes: usize, hash_bytes: usize) usize {
    var total: usize = 3 * o.current.len * fp2_bytes;
    total += hash_bytes; // leaf
    total += o.path.siblings.len * hash_bytes;
    total += @sizeOf(usize); // index
    return total;
}

/// Config derived from the trace height. Both layouts hold degree-2
/// constraints, so the blowup and the FRI shape are the same.
fn configFor(rows: usize) stark.Config {
    const log_trace: u6 = @intCast(std.math.log2_int(usize, rows));
    const log_blowup: u6 = 2;
    return .{
        .log_trace = log_trace,
        .log_blowup = log_blowup,
        .fri = .{
            .log_domain = log_trace + log_blowup,
            .log_final = log_trace + 1,
            .log_residual_degree = log_trace,
            .num_queries = 4,
        },
    };
}

/// A Q4_K block whose bytes encode known raw nibbles: byte j is (j%16) in
/// both halves, so element 2j and 2j+1 both have raw nibble j%16.
fn blockWithRamp() [128]u8 {
    var out: [128]u8 = undefined;
    for (0..128) |j| {
        const raw: u8 = @intCast(j % 16);
        out[j] = (raw << 4) | raw;
    }
    return out;
}

fn rawNibble(i: usize) u8 {
    return @intCast((i / 2) % 16);
}

/// Two alternating fp16 block scales (1.0 and 0.5), one per 256-element
/// Q4_K block, so multi-block runs exercise the per-MAC scale array.
fn blockScale(i: usize) u16 {
    return if ((i / 256) % 2 == 0) 0x3C00 else 0x3800;
}

const Data = struct {
    a: []Goldilocks,
    b: []Goldilocks,
    nib_a: []u8,
    nib_b: []u8,
    scale_a: []u16,
    scale_b: []u16,
    c: Goldilocks,

    fn deinit(self: *Data, allocator: std.mem.Allocator) void {
        allocator.free(self.a);
        allocator.free(self.b);
        allocator.free(self.nib_a);
        allocator.free(self.nib_b);
        allocator.free(self.scale_a);
        allocator.free(self.scale_b);
        self.* = undefined;
    }
};

/// Real dequantized operands: the same path the engine's loader takes.
fn buildData(allocator: std.mem.Allocator, k: usize) !Data {
    const block = blockWithRamp();
    var d = Data{
        .a = try allocator.alloc(Goldilocks, k),
        .b = try allocator.alloc(Goldilocks, k),
        .nib_a = try allocator.alloc(u8, k),
        .nib_b = try allocator.alloc(u8, k),
        .scale_a = try allocator.alloc(u16, k),
        .scale_b = try allocator.alloc(u16, k),
        .c = Goldilocks.zero,
    };
    errdefer d.deinit(allocator);

    var acc = Goldilocks.zero;
    for (0..k) |i| {
        const deq_a = tensor.dequantQ4K(&block, blockScale(i)) catch unreachable;
        const deq_b = tensor.dequantQ4K(&block, blockScale(i)) catch unreachable;
        d.a[i] = deq_a[i % 256];
        d.b[i] = deq_b[i % 256];
        d.nib_a[i] = rawNibble(i % 256);
        d.nib_b[i] = rawNibble(i % 256);
        d.scale_a[i] = blockScale(i);
        d.scale_b[i] = d.scale_a[i];
        acc = acc.add(d.a[i].mul(d.b[i]));
    }
    d.c = acc;
    return d;
}

fn benchPerMac(allocator: std.mem.Allocator, io: std.Io, data: *const Data) !Stats {
    var sys = try quant.buildSystem(allocator, data.a.len);
    defer sys.deinit();
    const system = sys.system();

    var gemm = try gemm_air.buildTrace(allocator, data.a, data.b, data.c);
    defer gemm.deinit(allocator);
    var bound = try quant.bindOperands(
        allocator,
        &gemm,
        data.nib_a,
        data.scale_a,
        data.nib_b,
        data.scale_b,
    );
    defer bound.deinit(allocator);

    const cfg = configFor(bound.rows);
    var transcript = stark.Transcript.init("zkml.bench.permac");
    const t0 = std.Io.Timestamp.now(io, .awake);
    var proof = try stark.prove(allocator, &transcript, .{
        .rows = bound.rows,
        .columns = bound.columns,
    }, system, cfg);
    const t1 = std.Io.Timestamp.now(io, .awake);
    defer proof.deinit(allocator);

    var vt = stark.Transcript.init("zkml.bench.permac");
    const ok = try stark.verify(&vt, &proof, system, cfg);
    const t2 = std.Io.Timestamp.now(io, .awake);

    return .{
        .label = "1 MAC/row",
        .k = data.a.len,
        .rows = bound.rows,
        .columns = bound.columns.len,
        .composed = system.composedCount(),
        .prove_ns = @intCast(t1.nanoseconds - t0.nanoseconds),
        .verify_ns = @intCast(t2.nanoseconds - t1.nanoseconds),
        .proof_bytes = proofBytes(proof),
        .verified = ok,
    };
}

fn benchChunked(allocator: std.mem.Allocator, io: std.Io, data: *const Data) !Stats {
    var sys = try cbind.buildSystem(allocator, data.a.len);
    defer sys.deinit();
    const system = sys.system();

    var gemm = try gemm_chunk.buildTrace(allocator, data.a, data.b, data.c);
    defer gemm.deinit(allocator);
    var bound = try cbind.bindOperands(
        allocator,
        &gemm,
        data.nib_a,
        data.scale_a,
        data.nib_b,
        data.scale_b,
    );
    defer bound.deinit(allocator);

    const cfg = configFor(bound.rows);
    var transcript = stark.Transcript.init("zkml.bench.chunked");
    const t0 = std.Io.Timestamp.now(io, .awake);
    var proof = try stark.prove(allocator, &transcript, .{
        .rows = bound.rows,
        .columns = bound.columns,
    }, system, cfg);
    const t1 = std.Io.Timestamp.now(io, .awake);
    defer proof.deinit(allocator);

    var vt = stark.Transcript.init("zkml.bench.chunked");
    const ok = try stark.verify(&vt, &proof, system, cfg);
    const t2 = std.Io.Timestamp.now(io, .awake);

    return .{
        .label = "16 MACs/row",
        .k = data.a.len,
        .rows = bound.rows,
        .columns = bound.columns.len,
        .composed = system.composedCount(),
        .prove_ns = @intCast(t1.nanoseconds - t0.nanoseconds),
        .verify_ns = @intCast(t2.nanoseconds - t1.nanoseconds),
        .proof_bytes = proofBytes(proof),
        .verified = ok,
    };
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn kib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1024.0;
}

fn largestPowerOfTwoAtMost(n: usize) usize {
    return @as(usize, 1) << @intCast(std.math.log2_int(usize, n));
}

fn perMacK(requested: usize) !usize {
    if (requested == 0) return error.BadK;
    const p = largestPowerOfTwoAtMost(requested + 1);
    if (p < 2) return error.BadK;
    return p - 1;
}

fn chunkedK(requested: usize) !usize {
    if (requested < gemm_chunk.slots) return error.BadK;
    const max_chunks = requested / gemm_chunk.slots;
    const p = largestPowerOfTwoAtMost(max_chunks + 1);
    if (p < 2) return error.BadK;
    return (p - 1) * gemm_chunk.slots;
}

fn run(allocator: std.mem.Allocator, io: std.Io, requested: usize, repeat: usize) !void {
    const per_mac_k = try perMacK(requested);
    const chunked_k = try chunkedK(requested);
    std.debug.print("\n=== F2 GEMM bench (ReleaseFast) ===\n", .{});
    std.debug.print("requested k = {d}; valid shapes: per-mac {d}, chunked {d}; best of {d}\n\n", .{
        requested,
        per_mac_k,
        chunked_k,
        repeat,
    });

    var per_data = try buildData(allocator, per_mac_k);
    defer per_data.deinit(allocator);
    var chunk_data = try buildData(allocator, chunked_k);
    defer chunk_data.deinit(allocator);
    std.debug.print("outputs: per-mac C = {d}, chunked C = {d}\n\n", .{
        per_data.c.toU64(),
        chunk_data.c.toU64(),
    });

    var per_mac = Stats{
        .label = "1 MAC/row",
        .k = per_mac_k,
        .rows = 0,
        .columns = 0,
        .composed = 0,
        .prove_ns = std.math.maxInt(u64),
        .verify_ns = std.math.maxInt(u64),
        .proof_bytes = 0,
        .verified = false,
    };
    var chunked = per_mac;
    chunked.label = "16 MACs/row";
    chunked.k = chunked_k;

    for (0..repeat) |i| {
        const a = try benchPerMac(allocator, io, &per_data);
        const c = try benchChunked(allocator, io, &chunk_data);
        per_mac.prove_ns = @min(per_mac.prove_ns, a.prove_ns);
        per_mac.verify_ns = @min(per_mac.verify_ns, a.verify_ns);
        per_mac.proof_bytes = a.proof_bytes;
        per_mac.rows = a.rows;
        per_mac.columns = a.columns;
        per_mac.composed = a.composed;
        per_mac.verified = a.verified;
        chunked.prove_ns = @min(chunked.prove_ns, c.prove_ns);
        chunked.verify_ns = @min(chunked.verify_ns, c.verify_ns);
        chunked.proof_bytes = c.proof_bytes;
        chunked.rows = c.rows;
        chunked.columns = c.columns;
        chunked.composed = c.composed;
        chunked.verified = c.verified;
        std.debug.print("  run {d}: per-mac {d:.1} ms prove, chunked {d:.1} ms prove\n", .{
            i + 1,
            ms(a.prove_ns),
            ms(c.prove_ns),
        });
    }

    std.debug.print("\n{s: <14} {s: >7} {s: >8} {s: >9} {s: >10} {s: >10} {s: >10} {s: >9} {s: >5}\n", .{
        "layout", "k", "rows", "columns", "composed", "prove ms", "verify ms", "proof KiB", "ok",
    });
    for ([_]Stats{ per_mac, chunked }) |s| {
        std.debug.print("{s: <14} {d: >7} {d: >8} {d: >9} {d: >10} {d: >10.2} {d: >10.2} {d: >9.1} {s: >5}\n", .{
            s.label, s.k, s.rows, s.columns, s.composed, ms(s.prove_ns), ms(s.verify_ns), kib(s.proof_bytes), if (s.verified) "yes" else "NO",
        });
    }

    std.debug.print("\nchunked vs per-mac: prove {d:.2}x, verify {d:.2}x, proof {d:.2}x, rows {d:.2}x\n", .{
        ratio(chunked.prove_ns, per_mac.prove_ns),
        ratio(chunked.verify_ns, per_mac.verify_ns),
        ratio(chunked.proof_bytes, per_mac.proof_bytes),
        ratio(chunked.rows, per_mac.rows),
    });

    const per_macs: f64 = @floatFromInt(per_mac.k);
    const chunked_macs: f64 = @floatFromInt(chunked.k);
    std.debug.print("per MAC: per-mac layout {d:.3} us prove, chunked {d:.3} us prove\n", .{
        ms(per_mac.prove_ns) * 1000.0 / per_macs,
        ms(chunked.prove_ns) * 1000.0 / chunked_macs,
    });
}

fn ratio(num: usize, den: usize) f64 {
    if (den == 0) return 0;
    return @as(f64, @floatFromInt(num)) / @as(f64, @floatFromInt(den));
}

fn ratioU(num: u64, den: u64) f64 {
    if (den == 0) return 0;
    return @as(f64, @floatFromInt(num)) / @as(f64, @floatFromInt(den));
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());

    var k: usize = 256;
    var repeat: usize = 3;
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--k") and i + 1 < argv.len) {
            i += 1;
            k = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--repeat") and i + 1 < argv.len) {
            i += 1;
            repeat = try std.fmt.parseInt(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--help")) {
            std.debug.print("usage: gemm_bench [--k N] [--repeat N]\n", .{});
            return;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            return error.BadArgument;
        }
    }
    if (k == 0) return error.BadK;
    if (repeat == 0) repeat = 1;

    try run(allocator, init.io, k, repeat);
}
