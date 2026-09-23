/*
 * kt_glue.h — reference glue between ktransformers-zig and zig-zkml.
 *
 * This is the REFERENCE adapter: the simplest possible mapping, used to
 * calibrate the checklist every other adapter (llama.cpp, zig-ai, vLLM)
 * is measured against. The engine's conventions (`kt_*`) are mapped onto
 * the engine-agnostic core (`zkml_*`); see README.md for the full table.
 *
 * Naming: the functions in ktransformers-zig's design doc (`zkML.md`) were
 * sketched as `kt_weights_merkle_root` / `kt_mla_weights_merkle_root` /
 * `kt_transcript_seed`. This glue deliberately prefixes them with
 * `kt_zkml_` instead of claiming those names, so that if the engine later
 * implements them, two shared objects cannot export the same symbol and
 * leave the loader's choice ambiguous.
 *
 * The engine repo is never modified from here: this header is compiled
 * against ktransformers-zig's own `include/kt_kernel.h` and links its
 * prebuilt shared library.
 */
#ifndef ZKML_KT_GLUE_H
#define ZKML_KT_GLUE_H

#include <stddef.h>
#include <stdint.h>

#include "kt_kernel.h"
#include "zkml_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ---------------------------------------------------------------------------
 * Generic weight stream (the reference primitive)
 * ---------------------------------------------------------------------------
 * A tensor view is the engine's own (name, bytes) pair. `name` must be
 * canonical and stable; `bytes` is the exact on-disk/on-wire extent. Data
 * is read once and never retained, so views may point into mmap'd weights.
 */
typedef struct kt_zkml_tensor_view {
    const char* name;
    const void* data;
    size_t bytes;
} kt_zkml_tensor_view;

/*
 * Attest an array of tensor views. Order-independent. Returns a ZKML_*
 * status code (0 = ZKML_OK).
 */
int kt_zkml_attest_tensors(const kt_zkml_tensor_view* tensors, size_t count,
                           uint8_t root_out[32]);

/*
 * ---------------------------------------------------------------------------
 * F0 — engine-shaped helpers
 * ---------------------------------------------------------------------------
 * MoE weights are BF16 with shapes pinned by kt_kernel.h:
 *   gate_proj / up_proj : [expert_num, intermediate_size, hidden_size]
 *   down_proj           : [expert_num, hidden_size, intermediate_size]
 * so the extents are derivable from the config alone. Quantization scales
 * and zero points are deliberately NOT attested: their layout depends on
 * `quant_config` (group_size / per_channel / zero_point), which the C
 * header does not pin, and a root that guessed would be worse than no
 * root. A deployment that needs them must pass explicit views.
 *
 * `cfg->hidden_type` must be KT_TYPE_BF16; anything else is rejected rather
 * than mis-sized.
 */
int kt_zkml_moe_weights_merkle_root(const kt_moe_config_t* cfg,
                                    uint8_t root_out[32]);

/*
 * GGUF-quantized LlamaMoe weights (any of the 16 GGML block formats, one
 * per projection). Extents come from the engine's own
 * `kt_type_row_bytes(n, type)` helper, so no block-size table is duplicated
 * here: if the engine cannot size a row it returns 0 and this fails with
 * ZKML_INVALID_ARGUMENT instead of hashing the wrong number of bytes.
 */
int kt_zkml_llama_moe_weights_merkle_root(const kt_llama_moe_config_t* cfg,
                                           uint8_t root_out[32]);

/*
 * MLA and DSV3 layer projections are NOT attested by a typed helper:
 * kt_kernel.h gives the dimensions (hidden_size, q_lora_rank, num_heads,
 * nope_size, rope_size, kv_lora_rank) but not the per-projection shapes,
 * and `v_head_dim` in particular is absent, so every shape is a guess.
 * Use kt_zkml_attest_tensors() with views the caller knows are right.
 */

/*
 * ---------------------------------------------------------------------------
 * F1 — deterministic sampling seed
 * ---------------------------------------------------------------------------
 * Direct pass-through to the core: context bytes -> 32-byte seed, no clock,
 * no hidden state.
 */
int kt_zkml_transcript_seed(const uint8_t* context, size_t context_len,
                            uint8_t seed_out[32]);

/*
 * ---------------------------------------------------------------------------
 * Witness (ABI v2)
 * ---------------------------------------------------------------------------
 * ktransformers-zig has NO recorded mode today: kt_kernel.h exposes no
 * begin/end-recording lifecycle, no op-record type, and the only internal
 * cache (ForwardCache in the SFT path) is for backpropagation, is not
 * reachable from C, and is not replayable. These wrappers are therefore the
 * hooks the engine should call once a recorded mode exists; they are
 * exercised by the adapter's own tests today.
 *
 * Slot keys and op ordinals are the core's ZKML_SlotKey / ZKML_OP_* — the
 * mapping is a pass-through by design (this is the reference adapter).
 */
typedef ZKML_SlotKey kt_zkml_slot_key;

typedef struct kt_zkml_witness kt_zkml_witness;

kt_zkml_witness* kt_zkml_witness_new(void);
int kt_zkml_witness_begin_layer(kt_zkml_witness* w, uint32_t layer);
int kt_zkml_witness_record(kt_zkml_witness* w, const kt_zkml_slot_key* key,
                           const void* payload, size_t payload_len);
int kt_zkml_witness_end_layer(kt_zkml_witness* w);
int kt_zkml_witness_finalize(kt_zkml_witness* w, const uint8_t stmt_hash[32],
                             uint8_t trace_hash_out[32]);
void kt_zkml_witness_free(kt_zkml_witness* w);

/*
 * Engine variant string this glue was compiled against, e.g.
 * "avx2". Pass NULL to query. Useful in logs next to a published root:
 * a root is only meaningful together with the kernel that will consume it.
 */
const char* kt_zkml_cpu_variant(void);

#ifdef __cplusplus
}
#endif

#endif /* ZKML_KT_GLUE_H */
