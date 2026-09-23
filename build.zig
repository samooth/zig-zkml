const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Single module rooted at the project root — libs/** are file imports
    // inside the module path, so `zig build test` collects ALL unit tests
    // (tests are only gathered from the root module's file set).
    const algebra_dep = b.dependency("zig_algebra", .{
        .target = target,
        .optimize = optimize,
    });
    const zmerkle_mod = algebra_dep.module("zig-merkle");

    const lib_mod = b.addModule("zig_zkml", .{
        .root_source_file = b.path("zkml.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zig-merkle", .module = zmerkle_mod },
        },
    });

    // Re-export zig-merkle so consumers don't need zig_algebra in their own deps
    lib_mod.addImport("zig-merkle", zmerkle_mod);

    const lib = b.addLibrary(.{
        .name = "zkml",
        .root_module = lib_mod,
    });
    b.installArtifact(lib);

    // Shared library for out-of-language consumers (llama.cpp CMake
    // adapter, vLLM ctypes). Same exports as the static lib.
    const dylib = b.addLibrary(.{
        .name = "zkml",
        .root_module = lib_mod,
        .linkage = .dynamic,
    });
    b.installArtifact(dylib);

    // Unit tests for the library (all files, via the single module root).
    const tests = b.addTest(.{
        .root_module = lib_mod,
    });
    const run_tests = b.addRunArtifact(tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // Engine adapter tests: separate root module so the adapter sources can
    // use `@import("zig_zkml")` — exactly how an engine build consumes them
    // (a path dependency + module import), while still being gated here.
    const zig_ai_adapter_mod = b.createModule(.{
        .root_source_file = b.path("adapters/zig_ai/test_adapter.zig"),
        .target = target,
        .optimize = optimize,
    });
    zig_ai_adapter_mod.addImport("zig_zkml", lib_mod);
    const adapter_tests = b.addTest(.{ .root_module = zig_ai_adapter_mod });
    const run_adapter_tests = b.addRunArtifact(adapter_tests);
    test_step.dependOn(&run_adapter_tests.step);

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
    const nm = b.addSystemCommand(&.{
        "sh", "-c",
        \\nm zig-out/lib/libzkml.a | grep -q zkml_attestor_create && \
        \\nm zig-out/lib/libzkml.a | grep -q zkml_attestor_destroy && \
        \\nm zig-out/lib/libzkml.a | grep -q zkml_proof_verify && \
        \\nm zig-out/lib/libzkml.a | grep -q zkml_transcript_seed && \
        \\nm zig-out/lib/libzkml.a | grep -q zkml_allocator_process && \
        \\nm zig-out/lib/libzkml.a | grep -q zkml_witness_session_create && \
        \\nm zig-out/lib/libzkml.a | grep -q zkml_witness_record_op && \
        \\nm zig-out/lib/libzkml.a | grep -q zkml_witness_finalize && \
        \\nm zig-out/lib/libzkml.a | grep -q zkml_witness_session_destroy && \
        \\echo 'symbols: ok'
    });
    nm.step.dependOn(b.getInstallStep());
    abi_step.dependOn(&nm.step);

    // Cross-check: artifacts from the ABI run verified by the INDEPENDENT
    // Python implementation (no shared code — a genuine audit, F0 §11).
    const py_proof = b.addSystemCommand(&.{
        "sh", "-c",
        \\python3 tools/verify_weights.py proof --root "$(tr -d '\n' < .zig-cache/abi-check/root.hex)" \
        \\  --name "$(cat .zig-cache/abi-check/name.txt)" < .zig-cache/abi-check/proof.bin
    });
    py_proof.step.dependOn(&run_abi.step);
    const py_manifest = b.addSystemCommand(&.{
        "sh", "-c",
        \\python3 tools/verify_weights.py manifest --root "$(tr -d '\n' < .zig-cache/abi-check/root.hex)" \
        \\  .zig-cache/abi-check/manifest.json
    });
    py_manifest.step.dependOn(&run_abi.step);
    const py_selftest = b.addSystemCommand(&.{ "python3", "tools/verify_weights.py", "selftest" });

    // --- Stage 3: vLLM adapter (ctypes) ----------------------------------
    // Zero-dependency Python test over the shared library: attestation
    // algebra, order independence, flipped-byte negative, proof round-trip,
    // witness session, and a cross-check against the independent auditor.
    const vllm_test = b.addSystemCommand(&.{ "python3", "adapters/vllm/test_zkml.py" });
    vllm_test.step.dependOn(b.getInstallStep());
    const vllm_step = b.step(
        "vllm-adapter",
        "Run the vLLM ctypes adapter tests (Stage 3; needs python3, no vLLM)",
    );
    vllm_step.dependOn(&vllm_test.step);

    const verify_step = b.step("verify", "End-to-end: tests + ABI + independent Python audit");
    verify_step.dependOn(&run_tests.step);
    verify_step.dependOn(&run_abi.step);
    verify_step.dependOn(&py_proof.step);
    verify_step.dependOn(&py_manifest.step);
    verify_step.dependOn(&py_selftest.step);
    // vLLM ctypes adapter rides the default gate: it is hermetic, needs no
    // vLLM/torch install, and it is the only test that exercises libzkml.so
    // through a foreign function boundary.
    verify_step.dependOn(&vllm_test.step);

    // --- F2 spikes -------------------------------------------------------
    // zig-algebra exposes its libs as named modules (zig-fri, ...). The FRI
    // audit (tools/fri_audit.zig) empirically answers whether its
    // field-generic FRI is a sound low-degree test over OUR Goldilocks
    // p = 2^61-1. The tool is self-contained (own field + transcript
    // passed through FRI's comptime-F / anytype interfaces) — importing
    // the package's zig-transcript alongside zig-fri breaks the module
    // graph (upstream registers the same source as two modules:
    // 'zig-transcript' and 'zig-transcript0').
    const fri_mod = algebra_dep.module("zig-fri");

    const fri_audit_mod = b.createModule(.{
        .root_source_file = b.path("tools/fri_audit.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zig-fri", .module = fri_mod },
        },
    });
    const fri_audit_exe = b.addExecutable(.{
        .name = "zkml_fri_audit",
        .root_module = fri_audit_mod,
    });
    const run_fri_audit = b.addRunArtifact(fri_audit_exe);
    const spike_step = b.step("spike", "F2 spikes: FRI soundness audit over Goldilocks");
    spike_step.dependOn(&run_fri_audit.step);

    // --- Stage 4: ktransformers-zig reference glue (optional) ------------
    // Same shape as the llama adapter: CMake configure/build, gated test,
    // independent Python audit of the emitted root. Needs a built
    // ktransformers-zig sibling checkout.
    const kt_dir = b.option(
        []const u8,
        "kt-dir",
        "Path to a built ktransformers-zig checkout (default: sibling ../ktransformers-zig)",
    ) orelse b.fmt("{s}/../ktransformers-zig", .{b.build_root.path orelse "."});
    const kt_variant = b.option(
        []const u8,
        "kt-variant",
        "Which prebuilt ktransformers-zig kernel variant to link (avx2, amx, ...)",
    ) orelse "avx2";

    const kt_cfg = b.addSystemCommand(&.{
        "cmake", "-S", "adapters/ktransformers", "-B", ".zig-cache/kt_adapter",
    });
    kt_cfg.addArg(b.fmt("-DKT_DIR={s}", .{kt_dir}));
    kt_cfg.addArg(b.fmt("-DKT_VARIANT={s}", .{kt_variant}));
    kt_cfg.step.dependOn(b.getInstallStep());

    const kt_build = b.addSystemCommand(&.{
        "cmake", "--build", ".zig-cache/kt_adapter", "--parallel",
    });
    kt_build.step.dependOn(&kt_cfg.step);

    const kt_artifacts = ".zig-cache/kt_adapter/artifacts";
    const kt_test = b.addSystemCommand(&.{
        "sh", "-c",
        "mkdir -p " ++ kt_artifacts ++ " && .zig-cache/kt_adapter/kt_glue_test " ++ kt_artifacts,
    });
    kt_test.step.dependOn(&kt_build.step);

    const kt_py = b.addSystemCommand(&.{
        "python3", "tools/integration/verify_adapter_root.py", kt_artifacts,
    });
    kt_py.step.dependOn(&kt_test.step);

    const kt_step = b.step(
        "kt-adapter",
        "Build + test the ktransformers-zig reference glue (Stage 4; needs a built sibling checkout)",
    );
    kt_step.dependOn(&kt_py.step);

    // --- Stage 1: llama.cpp adapter (optional — needs CMake + built llama.cpp) ---
    // Build the zero-fork GGUF attestation wrapper and run its gated test
    // (positive + negative + Python cross-check). Not part of the default
    // build; invoke with `zig build llama-adapter`.
    const llama_dir = b.option(
        []const u8,
        "llama-dir",
        "Path to a built llama.cpp checkout (default: sibling ../llama.cpp)",
    ) orelse b.fmt("{s}/../llama.cpp", .{b.build_root.path orelse "."});

    const llama_cfg = b.addSystemCommand(&.{
        "cmake", "-S", "adapters/llama_cpp", "-B", ".zig-cache/llama_adapter",
    });
    llama_cfg.addArg(b.fmt("-DLLAMA_DIR={s}", .{llama_dir}));
    // Ensure libzkml.so exists before CMake configure checks for it.
    llama_cfg.step.dependOn(b.getInstallStep());

    const llama_build = b.addSystemCommand(&.{
        "cmake", "--build", ".zig-cache/llama_adapter", "--parallel",
    });
    llama_build.step.dependOn(&llama_cfg.step);

    const llama_artifacts = ".zig-cache/llama_adapter/artifacts";
    const llama_test = b.addSystemCommand(&.{
        "sh", "-c",
        "mkdir -p " ++ llama_artifacts ++ " && " ++
            ".zig-cache/llama_adapter/zkml_llama_test " ++ llama_artifacts,
    });
    llama_test.step.dependOn(&llama_build.step);

    const llama_nm = b.addSystemCommand(&.{
        "sh", "-c",
        "nm -D .zig-cache/llama_adapter/libzkml_llama.so | grep -q zkml_llama_attest_gguf && " ++
            "nm -D .zig-cache/llama_adapter/libzkml_llama.so | grep -q zkml_attestor_create && " ++
            "echo 'llama adapter symbols: ok'",
    });
    llama_nm.step.dependOn(&llama_build.step);

    const llama_py = b.addSystemCommand(&.{
        "python3", "tools/integration/verify_adapter_root.py", llama_artifacts,
    });
    llama_py.step.dependOn(&llama_test.step);

    const llama_step = b.step(
        "llama-adapter",
        "Build + test llama.cpp attestation adapter (Stage 1; needs CMake + built llama.cpp)",
    );
    llama_step.dependOn(&llama_nm.step);
    llama_step.dependOn(&llama_py.step);

    // Fmt check.
    const fmt = b.addFmt(.{
        .paths = &.{ "zkml.zig", "libs", "tools", "adapters" },
        .check = true,
    });
    const fmt_step = b.step("fmt", "Check formatting");
    fmt_step.dependOn(&fmt.step);
}
