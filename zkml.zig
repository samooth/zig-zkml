//! zig-zkml — verifiable-inference layer for host inference engines
//! (llama.cpp, vLLM, zig-ai, ktransformers-zig — via adapters/<engine>/).
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
pub const fri = @import("libs/fri/root.zig");
pub const air = @import("libs/air/root.zig");
pub const gadgets = @import("libs/gadgets/root.zig");
pub const compile = @import("libs/compile/root.zig");
pub const prove = @import("libs/prove/root.zig");

// F2 STARK backend. Exported so tools (bench/gemm_bench.zig) and the F3
// C API can reach the real prover without reaching into libs/ by path.
pub const stark = @import("libs/stark/root.zig");
pub const gemm_air = @import("libs/stark/gemm_air.zig");
pub const gemm_chunk = @import("libs/stark/gemm_chunk.zig");
pub const quant_binding = @import("libs/stark/quant_binding.zig");
pub const chunk_binding = @import("libs/stark/chunk_binding.zig");
pub const logup = @import("libs/stark/logup.zig");

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
    _ = &api.zkml_allocator_process;
    _ = &api.zkml_attestor_create;
    _ = &api.zkml_witness_session_create;
    _ = &api.zkml_witness_record_op;
    _ = &api.zkml_witness_finalize;
    _ = &gadgets.gemm.GemmGadget.airFragment;
    _ = &gadgets.nonlin.SiLULookup.airFragment;
    _ = &gadgets.norm.RmsNormGadget.airFragment;
    _ = &gadgets.routing.GroupTop2Gadget.airFragment;
    _ = &compile.CircuitGraph.init;
    _ = &prove.prove;
    _ = &prove.verify;
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
    _ = @import("libs/fri/fp2.zig");
    _ = @import("libs/stark/expr.zig");
    _ = @import("libs/stark/commit.zig");
    _ = @import("libs/stark/root.zig");
    _ = @import("libs/stark/range.zig");
    _ = @import("libs/stark/logup.zig");
    _ = @import("libs/stark/logup_test.zig");
    _ = @import("libs/stark/gemm_air.zig");
    _ = @import("libs/stark/gemm_chunk.zig");
    _ = @import("libs/stark/chunk_binding.zig");
    _ = @import("libs/stark/chunk_binding_test.zig");
    _ = @import("libs/stark/gemm_chunk_test.zig");
    _ = @import("libs/stark/quant_test.zig");
    _ = @import("libs/stark/gemm_test.zig");
    _ = @import("libs/fri/domain.zig");
    _ = @import("libs/fri/root.zig");
    _ = @import("libs/air/root.zig");
    _ = @import("libs/gadgets/gemm/root.zig");
    _ = @import("libs/gadgets/quant/root.zig");
    _ = @import("libs/gadgets/nonlin/root.zig");
    _ = @import("libs/gadgets/norm/root.zig");
    _ = @import("libs/gadgets/routing/root.zig");
    _ = @import("libs/compile/root.zig");
    _ = @import("libs/prove/root.zig");
}
