const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Single module rooted at the project root — libs/** are file imports
    // inside the module path, so `zig build test` collects ALL unit tests
    // (tests are only gathered from the root module's file set).
    const lib_mod = b.createModule(.{
        .root_source_file = b.path("zkml.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addLibrary(.{
        .name = "zkml",
        .root_module = lib_mod,
    });
    b.installArtifact(lib);

    // Unit tests for the library (all files, via the single module root).
    const tests = b.addTest(.{
        .root_module = lib_mod,
    });
    const run_tests = b.addRunArtifact(tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // ABI check: exercise the exported C API end-to-end (attestor
    // lifecycle, proof, standalone verify, artifact dump for the Python
    // cross-check). Also assert the symbols really landed in the lib.
    const abi_mod = b.createModule(.{
        .root_source_file = b.path("tools/abi_check.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "api", .module = lib_mod },
        },
    });
    const abi_exe = b.addExecutable(.{
        .name = "zkml_abi_check",
        .root_module = abi_mod,
    });
    const run_abi = b.addRunArtifact(abi_exe);
    run_abi.addArg(b.pathFromRoot(".zig-cache/abi-check"));
    const abi_step = b.step("abi", "End-to-end C-ABI check + symbol emission");
    abi_step.dependOn(&run_abi.step);
    // Symbol assertion: every zkml_* export must be present in the lib.
    const nm = b.addSystemCommand(&.{ "sh", "-c",
        \\nm zig-out/lib/libzkml.a | grep -q zkml_attestor_create && \
        \\nm zig-out/lib/libzkml.a | grep -q zkml_attestor_destroy && \
        \\nm zig-out/lib/libzkml.a | grep -q zkml_proof_verify && \
        \\nm zig-out/lib/libzkml.a | grep -q zkml_transcript_seed && \
        \\echo 'symbols: ok'
    });
    nm.step.dependOn(b.getInstallStep());
    abi_step.dependOn(&nm.step);

    // Cross-check: artifacts from the ABI run verified by the INDEPENDENT
    // Python implementation (no shared code — a genuine audit, F0 §11).
    const py_proof = b.addSystemCommand(&.{ "sh", "-c",
        \\python3 tools/verify_weights.py proof --root "$(tr -d '\n' < .zig-cache/abi-check/root.hex)" \
        \\  --name "$(cat .zig-cache/abi-check/name.txt)" < .zig-cache/abi-check/proof.bin
    });
    py_proof.step.dependOn(&run_abi.step);
    const py_manifest = b.addSystemCommand(&.{ "sh", "-c",
        \\python3 tools/verify_weights.py manifest --root "$(tr -d '\n' < .zig-cache/abi-check/root.hex)" \
        \\  .zig-cache/abi-check/manifest.json
    });
    py_manifest.step.dependOn(&run_abi.step);
    const py_selftest = b.addSystemCommand(&.{ "python3", "tools/verify_weights.py", "selftest" });

    const verify_step = b.step("verify", "End-to-end: tests + ABI + independent Python audit");
    verify_step.dependOn(&run_tests.step);
    verify_step.dependOn(&run_abi.step);
    verify_step.dependOn(&py_proof.step);
    verify_step.dependOn(&py_manifest.step);
    verify_step.dependOn(&py_selftest.step);

    // Fmt check.
    const fmt = b.addFmt(.{
        .paths = &.{ "zkml.zig", "libs", "tools" },
        .check = true,
    });
    const fmt_step = b.step("fmt", "Check formatting");
    fmt_step.dependOn(&fmt.step);
}
