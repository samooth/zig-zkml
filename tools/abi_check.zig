//! abi_check.zig — end-to-end ABI check: builds an attestation through
//! the exported C API, self-verifies via zkml_proof_verify, and dumps the
//! root + serialized proof + manifest for cross-verification with
//! tools/verify_weights.py (independent implementation — the real audit).
//!
//! Usage: zkml_abi_check <outdir> [proof_name]
//! Writes: <outdir>/{root.hex, name.txt, proof.bin, manifest.json}

const std = @import("std");
const api = @import("api");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena;
    var allocator = init.gpa;

    const args = try init.minimal.args.toSlice(arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: zkml_abi_check <outdir> [proof_name]\n", .{});
        return error.Usage;
    }
    const outdir = args[1];
    const proof_name = if (args.len > 2) args[2] else "gate.0.w";

    // --- Build the attestation through the C ABI exactly as kt would ---
    const handle = api.api.zkml_attestor_create(@ptrCast(&allocator)) orelse {
        std.debug.print("attestor_create failed\n", .{});
        return error.AbiCreate;
    };
    defer api.api.zkml_attestor_destroy(handle);

    const tensors = [_]struct { name: []const u8, data: []const u8 }{
        .{ .name = "gate.0.w", .data = "GATE-TENSOR-BYTES-0" },
        .{ .name = "up.0.w", .data = "UP-TENSOR-BYTES-1" },
        .{ .name = "down.0.w", .data = "DOWN-TENSOR-BYTES-2" },
        .{ .name = "norm.0.w", .data = "NORM-TENSOR-BYTES-3" },
        .{ .name = "routing.0.w", .data = "ROUTING-TENSOR-4" },
    };
    for (tensors) |t| {
        const rc = api.api.zkml_attestor_add(handle, t.name.ptr, t.name.len, t.data.ptr, t.data.len);
        if (rc != 0) return error.AbiAdd;
    }
    if (api.api.zkml_attestor_finish(handle) != 0) return error.AbiFinish;

    var root: [32]u8 = undefined;
    if (api.api.zkml_attestor_root(handle, &root) != 0) return error.AbiRoot;

    var wire: [*]u8 = undefined;
    var wire_len: usize = 0;
    const rc = api.api.zkml_attestor_proof(
        handle,
        proof_name.ptr,
        proof_name.len,
        &wire,
        &wire_len,
    );
    if (rc != 0) {
        std.debug.print("attestor_proof failed: {d}\n", .{rc});
        return error.AbiProof;
    }
    defer _ = api.api.zkml_attestor_free_proof(handle, wire, wire_len);

    // --- Self-verify through the standalone verifier ---
    if (api.api.zkml_proof_verify(@ptrCast(&allocator), wire, wire_len, &root) != 0) {
        return error.SelfVerify;
    }

    // --- Dump artifacts for the Python cross-check ---
    try std.Io.Dir.cwd().createDirPath(io, outdir);
    const out = try std.Io.Dir.cwd().openDir(io, outdir, .{});

    var fbuf: [4096]u8 = undefined;
    {
        var root_buf: [64]u8 = undefined;
        const root_hex = try std.fmt.bufPrint(&root_buf, "{x}", .{&root});
        const f = try out.createFile(io, "root.hex", .{});
        var fw = f.writer(io, &fbuf);
        try fw.interface.writeAll(root_hex);
        try fw.interface.writeAll("\n");
        try fw.interface.flush();
        f.close(io);
    }
    {
        const f = try out.createFile(io, "name.txt", .{});
        var fw = f.writer(io, &fbuf);
        try fw.interface.writeAll(proof_name);
        try fw.interface.flush();
        f.close(io);
    }
    {
        const f = try out.createFile(io, "proof.bin", .{});
        var fw = f.writer(io, &fbuf);
        try fw.interface.writeAll(wire[0..wire_len]);
        try fw.interface.flush();
        f.close(io);
    }
    {
        var manifest_buf = std.ArrayList(u8).empty;
        defer manifest_buf.deinit(arena.allocator());
        const ma = arena.allocator();
        try manifest_buf.appendSlice(ma, "[");
        for (tensors, 0..) |t, i| {
            if (i > 0) try manifest_buf.appendSlice(ma, ",");
            try manifest_buf.appendSlice(ma, "{\"name\": \"");
            try manifest_buf.appendSlice(ma, t.name);
            try manifest_buf.appendSlice(ma, "\", \"data_b64\": \"");
            const enc = std.base64.standard.Encoder;
            const b64 = try ma.alloc(u8, enc.calcSize(t.data.len));
            _ = enc.encode(b64, t.data);
            try manifest_buf.appendSlice(ma, b64);
            try manifest_buf.appendSlice(ma, "\"}");
        }
        try manifest_buf.appendSlice(ma, "]");
        const f = try out.createFile(io, "manifest.json", .{});
        var fw = f.writer(io, &fbuf);
        try fw.interface.writeAll(manifest_buf.items);
        try fw.interface.flush();
        f.close(io);
    }

    var root_buf: [64]u8 = undefined;
    const root_hex = try std.fmt.bufPrint(&root_buf, "{x}", .{root});
    {
        var obuf: [512]u8 = undefined;
        var w = std.Io.File.stdout().writer(io, &obuf);
        try w.interface.print(
            "abi check OK\n  root:  {s}\n  proof: {d} bytes for '{s}'\n  self-verify: passed\n  artifacts in {s}/ for tools/verify_weights.py\n",
            .{ root_hex, wire_len, proof_name, outdir },
        );
        try w.interface.flush();
    }
}
