//! zig-zkml — verifiable-inference layer for ktransformers-zig.
//!
//! Single module root: all libraries are file imports so unit tests in
//! every file are collected by `zig build test` (test blocks are only
//! gathered from the root module's file set — see BLUE_PRINT §13).
//!
//! Phase: F0/F1 (weights attestation + deterministic transcript) plus the
//! L0/L2 foundations (field, merkle, transcript, tensor, trace, statement).
//! The STARK prover (F2+) will build on this; see BLUE_PRINT.md.

const std = @import("std");

pub const field = @import("libs/field.zig");
pub const merkle = @import("libs/merkle.zig");
pub const transcript = @import("libs/transcript.zig");
pub const tensor = @import("libs/tensor/root.zig");
pub const trace = @import("libs/trace/root.zig");
pub const statement = @import("libs/statement/root.zig");
pub const attestation = @import("libs/attestation.zig");
pub const api = @import("libs/api.zig");

comptime {
    // Lazy-analysis trap: re-exported decls are not analyzed until referenced.
    // Force emission of the module API so unit tests and consumers see it.
    _ = &field.Goldilocks.add;
    _ = &merkle.MerkleTree.hashLeaf;
    _ = &merkle.MerkleTree.proof;
    _ = &merkle.Builder.finish;
    _ = &merkle.Proof.serialize;
    _ = &transcript.Transcript.init;
    _ = &transcript.Transcript.challengeU64;
    _ = &tensor.Scheme.magnitudeBound;
    _ = &tensor.dequantQ4K;
    _ = &trace.TraceRecorder.init;
    _ = &trace.TraceRecorder.finalize;
    _ = &statement.StatementLayer.serialize;
    _ = &attestation.WeightsAttestor.init;
    _ = &api.zkml_attestor_create;
}

test {
    // Pull in every library's tests (same-module file imports are collected).
    std.testing.refAllDecls(@This());
    _ = @import("libs/field.zig");
    _ = @import("libs/merkle.zig");
    _ = @import("libs/transcript.zig");
    _ = @import("libs/tensor/root.zig");
    _ = @import("libs/trace/root.zig");
    _ = @import("libs/statement/root.zig");
    _ = @import("libs/attestation.zig");
    _ = @import("libs/api.zig");
}
