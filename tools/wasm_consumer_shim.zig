//! The question `wasm-portability` should have been asking.
//!
//! `tools/wasm_expected.txt` swept `zig build test -Dtarget=wasm32-freestanding`
//! and pinned which files failed. That inventory mixed three unrelated things:
//! stdlib internals (`posix.zig`, `Thread.zig`), repository code, and the
//! *test harness itself* — which takes `std.process.Init` and spawns threads
//! and can never compile freestanding, on any platform, for any reason
//! connected to this library. A file that is not supposed to compile should
//! not appear in a list of portability defects, and its presence meant every
//! future fix would move the list for a reason unrelated to portability.
//!
//! So the shim is a *consumer*: it imports the public API, touches the
//! entry points a real embedder calls, and nothing else. That is the literal
//! question "can a consumer compile freestanding?", and the answer is a yes or
//! a no rather than a list.
//!
//! It is deliberately not a build of `zkml.zig` — that root re-exports the
//! benchmark and test-adjacent modules, which is how the harness leaked in.
//! Importing `libs/api.zig` is what a C-ABI embedder actually pulls in, since
//! every exported `zkml_*` symbol lives there.
//!
//! No `test` block: a test needs a test runner, and a test runner is exactly
//! the harness this file exists to exclude. The gate compiles it; that is the
//! assertion.

const builtin = @import("builtin");
const api = @import("api");

// This list is a DELIBERATE FAILURE and the reason this gate is trustworthy.
//
// The first version of this shim held `&api.zkml_witness_session_create` and
// friends — pointers to the exported functions. It compiled cleanly against
// wasm32-freestanding, and it was wrong: taking a function's address forces
// its SIGNATURE to be analysed, not its BODY. The body is where
// `std.heap.smp_allocator` lives, so the one defect this gate exists to catch
// was invisible to it, and the gate passed.
//
// That is the `&f` shape from AGENTS.md, committed in the gate built to
// prevent exactly that class. The instrument has to be shown to fail on a
// known defect before its green is worth anything, and it was not.
//
// So the entry points are CALLED. An `export fn` body is always analysed,
// which makes everything it calls reachable, which is what pulls in the
// allocator. The calls are never executed — this file is compiled, not run —
// so their arguments only have to typecheck.
extern fn zkml_probe_allocator() void;

/// The `*anyopaque` allocator parameter every `*_create` entry takes. These
/// calls are never executed — the file is compiled, not run — so the value only
/// has to typecheck. It cannot be `null`: `*anyopaque` is non-optional, and
/// address zero is rejected, so a real one-element buffer stands in. What is
/// being asked here is whether the entry point COMPILES, not what it returns.
var allocator_slot: [1]u8 = undefined;
const some_allocator: *anyopaque = @ptrCast(&allocator_slot);

/// Optional out-pointers are typed, not bare `0`: `0` is a comptime_int and
/// does not infer to a pointer. These stand in for the caller's buffers; the
/// file is compiled, never run, so only the types matter.
const no_name: ?[*]const u8 = null;
const no_const_data: ?[*]const u8 = null;
const no_bytes_out: ?[*]u8 = null;
const no_proof_out: ?*[*]u8 = null;
const no_len_out: ?*usize = null;
const no_proof: ?[*]u8 = null;

export fn zkml_wasm_consumer_probe() void {
    // Every one of these takes the allocator out of the caller in api.zig —
    // the same `process_allocator` the web demo tripped over. Calling them is
    // what forces the bodies to be analysed; see the note above.
    const att = api.zkml_attestor_create(some_allocator) orelse unreachable;
    _ = api.zkml_attestor_add(att, no_name, 0, no_const_data, 0);
    _ = api.zkml_attestor_root(att, no_bytes_out);
    _ = api.zkml_attestor_finish(att);
    _ = api.zkml_attestor_proof(att, no_name, 0, no_proof_out, no_len_out);
    _ = api.zkml_attestor_free_proof(att, no_proof, 0);
    api.zkml_attestor_destroy(att);

    const wit = api.zkml_witness_session_create(some_allocator) orelse unreachable;
    _ = api.zkml_witness_begin_layer(wit, 0);
    _ = api.zkml_witness_end_layer(wit);
    _ = api.zkml_witness_finalize(wit, no_const_data, no_bytes_out);
    api.zkml_witness_session_destroy(wit);

    var seed: [32]u8 = undefined;
    _ = api.zkml_transcript_seed(no_const_data, 0, &seed);
    _ = &seed;

    zkml_probe_allocator();
}

// A freestanding build has no stdout, so printing would reintroduce exactly
// the dependency the shim is isolating.
pub const referenced_entry_points: usize = 15;

/// A consumer can also read the ABI version without linking a runtime.
pub const abi_version = api.ZKML_ABI_VERSION;

/// `smp_allocator` is unavailable without threads. This is not a hypothetical:
/// the whole reason the web demo could not build freestanding was one
/// `smp_allocator` in `libs/api.zig`, untouched since the 27th, behind twelve
/// commits of tests and documentation. The allocator choice is a compile-time
/// property of the target, so a consumer-side check is a real check — but the
/// gate compiles the shim, and that is what decides.
pub const is_single_threaded_target = builtin.single_threaded;
