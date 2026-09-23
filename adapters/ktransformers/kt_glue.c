/*
 * kt_glue.c — reference glue between ktransformers-zig and zig-zkml.
 * See kt_glue.h for the contract and README.md for the kt_* -> zkml_*
 * mapping table.
 */
#include "kt_glue.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- Generic weight stream ------------------------------------------------- */

int kt_zkml_attest_tensors(const kt_zkml_tensor_view* tensors, size_t count,
                           uint8_t root_out[32]) {
    if (tensors == NULL || root_out == NULL || count == 0) {
        return ZKML_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < count; ++i) {
        if (tensors[i].name == NULL || tensors[i].name[0] == '\0') {
            return ZKML_INVALID_ARGUMENT;
        }
        if (tensors[i].data == NULL) {
            return ZKML_INVALID_ARGUMENT;
        }
    }

    ZKML_Attestor* att = zkml_attestor_create(zkml_allocator_process());
    if (att == NULL) {
        return ZKML_OUT_OF_MEMORY;
    }

    int rc = ZKML_OK;
    for (size_t i = 0; i < count; ++i) {
        rc = zkml_attestor_add(att, tensors[i].name, strlen(tensors[i].name),
                               tensors[i].data, tensors[i].bytes);
        if (rc != ZKML_OK) {
            break;
        }
    }
    if (rc == ZKML_OK) {
        rc = zkml_attestor_finish(att);
    }
    if (rc == ZKML_OK) {
        rc = zkml_attestor_root(att, root_out);
    }
    zkml_attestor_destroy(att);
    return rc;
}

/* --- MoE (BF16, shapes pinned by kt_kernel.h) ------------------------------ */

int kt_zkml_moe_weights_merkle_root(const kt_moe_config_t* cfg,
                                    uint8_t root_out[32]) {
    if (cfg == NULL || root_out == NULL) {
        return ZKML_INVALID_ARGUMENT;
    }
    if (cfg->hidden_type != KT_TYPE_BF16) {
        /* The extents below assume 2-byte elements. Refusing beats hashing
         * a prefix of a differently-typed buffer. */
        return ZKML_INVALID_ARGUMENT;
    }
    if (cfg->expert_num <= 0 || cfg->intermediate_size <= 0 || cfg->hidden_size <= 0) {
        return ZKML_INVALID_ARGUMENT;
    }
    if (cfg->gate_proj == NULL || cfg->up_proj == NULL || cfg->down_proj == NULL) {
        return ZKML_INVALID_ARGUMENT;
    }

    const size_t experts = (size_t)cfg->expert_num;
    const size_t inter = (size_t)cfg->intermediate_size;
    const size_t hidden = (size_t)cfg->hidden_size;
    const size_t bytes_2d = experts * inter * hidden * 2u; /* [E, I, H] */
    const size_t bytes_d = experts * hidden * inter * 2u;  /* [E, H, I] */

    char name_gate[64];
    char name_up[64];
    char name_down[64];
    snprintf(name_gate, sizeof(name_gate), "layers.%d.moe.gate_proj", cfg->layer_idx);
    snprintf(name_up, sizeof(name_up), "layers.%d.moe.up_proj", cfg->layer_idx);
    snprintf(name_down, sizeof(name_down), "layers.%d.moe.down_proj", cfg->layer_idx);

    const kt_zkml_tensor_view views[3] = {
        { name_gate, cfg->gate_proj, bytes_2d },
        { name_up, cfg->up_proj, bytes_2d },
        { name_down, cfg->down_proj, bytes_d },
    };
    return kt_zkml_attest_tensors(views, 3, root_out);
}

/* --- LlamaMoe (GGUF block formats, sized by the engine) ------------------- */

/* rows * row_bytes, with overflow checked. Returns 0 on overflow. */
static size_t mul_checked(size_t rows, size_t row_bytes) {
    if (row_bytes == 0) {
        return 0;
    }
    if (rows != 0 && row_bytes > (size_t)-1 / rows) {
        return 0;
    }
    return rows * row_bytes;
}

int kt_zkml_llama_moe_weights_merkle_root(const kt_llama_moe_config_t* cfg,
                                           uint8_t root_out[32]) {
    if (cfg == NULL || root_out == NULL) {
        return ZKML_INVALID_ARGUMENT;
    }
    if (cfg->gate_proj == NULL || cfg->up_proj == NULL || cfg->down_proj == NULL) {
        return ZKML_INVALID_ARGUMENT;
    }
    if (cfg->expert_num == 0 || cfg->intermediate_size == 0 || cfg->hidden_size == 0) {
        return ZKML_INVALID_ARGUMENT;
    }

    const int experts = (int)cfg->expert_num;
    const size_t inter = cfg->intermediate_size;
    const size_t hidden = cfg->hidden_size;

    /* gate/up: [E, I, H] in `type` blocks; down: [E, H, I]. */
    const int row_gate = kt_type_row_bytes(hidden, (int)cfg->gate_type);
    const int row_up = kt_type_row_bytes(hidden, (int)cfg->up_type);
    const int row_down = kt_type_row_bytes(inter, (int)cfg->down_type);
    if (row_gate <= 0 || row_up <= 0 || row_down <= 0) {
        /* The engine cannot size one of these rows (unsupported type or a
         * row length that is not a whole number of blocks). */
        return ZKML_INVALID_ARGUMENT;
    }

    const size_t bytes_gate = mul_checked((size_t)experts * inter, (size_t)row_gate);
    const size_t bytes_up = mul_checked((size_t)experts * inter, (size_t)row_up);
    const size_t bytes_down = mul_checked((size_t)experts * hidden, (size_t)row_down);
    if (bytes_gate == 0 || bytes_up == 0 || bytes_down == 0) {
        return ZKML_INVALID_ARGUMENT;
    }

    char name_gate[64];
    char name_up[64];
    char name_down[64];
    snprintf(name_gate, sizeof(name_gate), "layers.%zu.moe.gate_proj", cfg->layer_idx);
    snprintf(name_up, sizeof(name_up), "layers.%zu.moe.up_proj", cfg->layer_idx);
    snprintf(name_down, sizeof(name_down), "layers.%zu.moe.down_proj", cfg->layer_idx);

    const kt_zkml_tensor_view views[3] = {
        { name_gate, cfg->gate_proj, bytes_gate },
        { name_up, cfg->up_proj, bytes_up },
        { name_down, cfg->down_proj, bytes_down },
    };
    return kt_zkml_attest_tensors(views, 3, root_out);
}

/* --- F1: transcript seed --------------------------------------------------- */

int kt_zkml_transcript_seed(const uint8_t* context, size_t context_len,
                            uint8_t seed_out[32]) {
    return zkml_transcript_seed(context, context_len, seed_out);
}

/* --- Witness (ABI v2) ------------------------------------------------------ */

struct kt_zkml_witness {
    ZKML_Witness* handle;
};

kt_zkml_witness* kt_zkml_witness_new(void) {
    if (zkml_allocator_process() == NULL) {
        return NULL;
    }
    ZKML_Witness* w = zkml_witness_session_create(zkml_allocator_process());
    if (w == NULL) {
        return NULL;
    }
    kt_zkml_witness* self = (kt_zkml_witness*)malloc(sizeof(kt_zkml_witness));
    if (self == NULL) {
        zkml_witness_session_destroy(w);
        return NULL;
    }
    self->handle = w;
    return self;
}

int kt_zkml_witness_begin_layer(kt_zkml_witness* w, uint32_t layer) {
    if (w == NULL) {
        return ZKML_INVALID_ARGUMENT;
    }
    return zkml_witness_begin_layer(w->handle, layer);
}

int kt_zkml_witness_record(kt_zkml_witness* w, const kt_zkml_slot_key* key,
                           const void* payload, size_t payload_len) {
    if (w == NULL || key == NULL) {
        return ZKML_INVALID_ARGUMENT;
    }
    return zkml_witness_record_op(w->handle, key, (const uint8_t*)payload, payload_len);
}

int kt_zkml_witness_end_layer(kt_zkml_witness* w) {
    if (w == NULL) {
        return ZKML_INVALID_ARGUMENT;
    }
    return zkml_witness_end_layer(w->handle);
}

int kt_zkml_witness_finalize(kt_zkml_witness* w, const uint8_t stmt_hash[32],
                             uint8_t trace_hash_out[32]) {
    if (w == NULL || stmt_hash == NULL || trace_hash_out == NULL) {
        return ZKML_INVALID_ARGUMENT;
    }
    return zkml_witness_finalize(w->handle, stmt_hash, trace_hash_out);
}

void kt_zkml_witness_free(kt_zkml_witness* w) {
    if (w == NULL) {
        return;
    }
    zkml_witness_session_destroy(w->handle);
    free(w);
}

const char* kt_zkml_cpu_variant(void) {
    return kt_get_cpu_variant();
}
