/*
 * zkml_engine.h — engine adapter contract for zig-zkml.
 *
 * zig-zkml core is engine-agnostic. Each host inference engine (llama.cpp,
 * vLLM, zig-ai, ktransformers-zig, ...) integrates via an adapter under
 * adapters/<engine>/ that implements this contract. The core never iterates
 * model tensors and never knows the engine runtime: the engine's loader
 * streams weights in, the core hashes/proves/verifies.
 *
 * This header is the shared vocabulary. The actual zkml_* entry points live
 * in zkml_c.h (ABI is additive; witness functions arrive with ABI v2 —
 * Stage 5 of PLAN_MULTI_ENGINE.md).
 *
 * See PLAN_MULTI_ENGINE.md for the adapter matrix, staging and gates.
 */
#ifndef ZKML_ENGINE_H
#define ZKML_ENGINE_H

#include "zkml_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ---------------------------------------------------------------------------
 * Weight stream (F0 attestation)
 * ---------------------------------------------------------------------------
 * Lifecycle (load path):
 *
 *   ZKML_Attestor* a = zkml_attestor_create(allocator);
 *   for each tensor T in the engine's loader:
 *       zkml_attestor_add(a, T.name, T.name_len, T.data, T.data_len);
 *   zkml_attestor_finish(a);
 *   zkml_attestor_root(a, root_out);          // publish / compare
 *
 * Contract:
 *   - `name` is the engine's canonical tensor name (stable across loads;
 *     the root is keyed by name, order-independent).
 *   - `data` is read once and never retained — safe to pass mmap pointers
 *     or transient decode buffers.
 *   - >= 1 tensor or finish() fails: an empty root is not an attestation.
 *   - The engine owns iteration; the core owns hashing. Adapters that
 *     cannot reach tensor bytes directly (e.g. a wrapper over a loader
 *     callback) call add() from inside the loader's per-tensor callback.
 */

/*
 * Optional helper: a single-callback view of the weight stream, useful for
 * engines whose loader already fans out to a per-tensor callback
 * (llama.cpp progress/set_tensor_data, vLLM safetensors iteration, ...).
 * The adapter's callback forwards to zkml_attestor_add.
 */
typedef int (*zkml_weight_stream_fn)(
    void *user,
    const char *name, size_t name_len,
    const void *data, size_t data_len);

/*
 * ---------------------------------------------------------------------------
 * Witness stream (recorded inference — feeds TraceRecorder)
 * ---------------------------------------------------------------------------
 * Lifecycle (forward path, ABI v2 — declared in zkml_c.h in Stage 5):
 *
 *   int s = zkml_witness_session_create(allocator);
 *   for each layer L executed by the engine:
 *       zkml_witness_begin_layer(s, layer_idx, ...);
 *       for each recorded op (gemm_a/b/c, dequant, routing, ...):
 *           zkml_witness_record_op(s, &slot_key, payload, payload_len);
 *       zkml_witness_end_layer(s, trace_hash_out);
 *   zkml_witness_finalize(s, witness_id_out);
 *
 * SlotKey (flat, C-layout): layer u32, op enum, expert u16 (0 for dense),
 * tp_rank u8 — mirrors TraceRecorder's canonical ordering. Payload is the
 * raw op I/O bytes the engine's recorded-mode kernel produced.
 *
 * Contract:
 *   - Intra-slot order is semantically significant (same thread/expert);
 *     inter-slot order is canonicalized at finalize (BLUE_PRINT §6.2).
 *   - Engines with no MoE concept pass expert = 0, rank = 0.
 *   - Witness hooks are opt-in: F0 (attestation-only) adapters skip this
 *     whole section.
 */

/*
 * ---------------------------------------------------------------------------
 * Prove / verify (F2+, after the STARK backend lands)
 * ---------------------------------------------------------------------------
 *   zkml_prover_new(...)            // binds weights root + statement
 *   zkml_prove_layer(prover, ...)   // L3 compile -> L2 constraints -> L1 FRI
 *   zkml_verify_layer(...)          // standalone: no runtime, no model
 */

/*
 * ---------------------------------------------------------------------------
 * Adapter stage checklist (PLAN_MULTI_ENGINE.md)
 * ---------------------------------------------------------------------------
 * F0 (attestation):
 *   [ ] weight stream: every loader tensor -> zkml_attestor_add
 *   [ ] root published through the engine's normal surface
 *   [ ] negative test: corrupt one tensor byte -> root changes
 *   [ ] root cross-checks against tools/verify_weights.py
 * F1 (seed):
 *   [ ] zkml_transcript_seed exposed if the engine samples
 * Witness (ABI v2):
 *   [ ] begin_layer/record_op/end_layer per recorded op
 *   [ ] determinism: same trace -> same finalize hash (multi-thread)
 * F2+ (proof):
 *   [ ] prove_layer on the engine's exact-path witness
 *   [ ] negative: +-1 ulp / bad routing / bad scale -> verify rejects
 */

#ifdef __cplusplus
}
#endif

#endif /* ZKML_ENGINE_H */
