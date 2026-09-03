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

    // Fmt check.
    const fmt = b.addFmt(.{
        .paths = &.{ "zkml.zig", "libs" },
        .check = true,
    });
    const fmt_step = b.step("fmt", "Check formatting");
    fmt_step.dependOn(&fmt.step);
}
