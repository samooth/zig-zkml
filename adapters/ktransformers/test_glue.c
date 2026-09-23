/*
 * test_glue.c — gate for the ktransformers-zig reference adapter (Stage 4).
 *
 * Same shape as the llama.cpp adapter's gate: positive, determinism,
 * negative (one flipped byte), plus the engine-specific paths (BF16 MoE
 * config, GGUF-block LlamaMoe config sized by the engine's own
 * kt_type_row_bytes), the F1 seed, the witness shim, and artifacts for the
 * independent Python auditor.
 *
 * Usage: kt_glue_test <workdir>
 * Writes: <workdir>/{root.hex,manifest.json}
 */
#include "kt_glue.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void ok(const char* what) {
    printf("ok   %s\n", what);
}

static void fail(const char* what) {
    printf("FAIL %s\n", what);
    failures++;
}

static void hexdump(const uint8_t* b, size_t n, char* out) {
    static const char* d = "0123456789abcdef";
    for (size_t i = 0; i < n; ++i) {
        out[2 * i] = d[b[i] >> 4];
        out[2 * i + 1] = d[b[i] & 15];
    }
    out[2 * n] = '\0';
}

/* --- fixtures ------------------------------------------------------------- */

#define N_TENSORS 4

static const char* const kNames[N_TENSORS] = {
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "layers.0.input_layernorm.weight",
};

static uint8_t g_data[N_TENSORS][16];

static void fill_fixture(uint8_t seed) {
    for (int t = 0; t < N_TENSORS; ++t) {
        for (size_t i = 0; i < sizeof(g_data[t]); ++i) {
            g_data[t][i] = (uint8_t)(seed + t * 7 + i * 3);
        }
    }
}

static size_t build_views(kt_zkml_tensor_view* views) {
    for (int t = 0; t < N_TENSORS; ++t) {
        views[t].name = kNames[t];
        views[t].data = g_data[t];
        views[t].bytes = sizeof(g_data[t]);
    }
    return N_TENSORS;
}

/* --- tests ---------------------------------------------------------------- */

static void test_generic_positive_and_determinism(void) {
    kt_zkml_tensor_view views[N_TENSORS];
    fill_fixture(1);
    build_views(views);

    uint8_t r1[32];
    uint8_t r2[32];
    if (kt_zkml_attest_tensors(views, N_TENSORS, r1) != ZKML_OK) {
        fail("generic: first pass");
        return;
    }
    if (kt_zkml_attest_tensors(views, N_TENSORS, r2) != ZKML_OK) {
        fail("generic: second pass");
        return;
    }
    if (memcmp(r1, r2, 32) != 0) {
        fail("generic: not deterministic");
        return;
    }
    ok("generic: deterministic root across two passes");
}

static void test_generic_order_independence(void) {
    kt_zkml_tensor_view forward[N_TENSORS];
    kt_zkml_tensor_view backward[N_TENSORS];
    fill_fixture(1);
    build_views(forward);
    for (int t = 0; t < N_TENSORS; ++t) {
        backward[t] = forward[N_TENSORS - 1 - t];
    }

    uint8_t a[32];
    uint8_t b[32];
    if (kt_zkml_attest_tensors(forward, N_TENSORS, a) != ZKML_OK ||
        kt_zkml_attest_tensors(backward, N_TENSORS, b) != ZKML_OK) {
        fail("order: attest failed");
        return;
    }
    if (memcmp(a, b, 32) != 0) {
        fail("order: root depends on iteration order");
        return;
    }
    ok("generic: root is order-independent");
}

static void test_generic_negative_flipped_byte(void) {
    kt_zkml_tensor_view views[N_TENSORS];
    fill_fixture(1);
    build_views(views);

    uint8_t before[32];
    if (kt_zkml_attest_tensors(views, N_TENSORS, before) != ZKML_OK) {
        fail("negative: baseline attest failed");
        return;
    }
    g_data[1][3] ^= 0xFF;
    uint8_t after[32];
    if (kt_zkml_attest_tensors(views, N_TENSORS, after) != ZKML_OK) {
        fail("negative: corrupt attest failed");
        return;
    }
    if (memcmp(before, after, 32) == 0) {
        fail("negative: flipped byte produced the same root");
        return;
    }
    g_data[1][3] ^= 0xFF; /* restore */
    ok("negative: one flipped tensor byte changes the root");
}

static void test_argument_validation(void) {
    kt_zkml_tensor_view views[N_TENSORS];
    uint8_t root[32];
    fill_fixture(1);
    build_views(views);

    if (kt_zkml_attest_tensors(NULL, N_TENSORS, root) != ZKML_INVALID_ARGUMENT) {
        fail("validation: null views accepted");
        return;
    }
    if (kt_zkml_attest_tensors(views, 0, root) != ZKML_INVALID_ARGUMENT) {
        fail("validation: zero count accepted");
        return;
    }
    kt_zkml_tensor_view bad = views[0];
    bad.name = "";
    if (kt_zkml_attest_tensors(&bad, 1, root) != ZKML_INVALID_ARGUMENT) {
        fail("validation: empty name accepted");
        return;
    }
    bad = views[0];
    bad.data = NULL;
    if (kt_zkml_attest_tensors(&bad, 1, root) != ZKML_INVALID_ARGUMENT) {
        fail("validation: null data accepted");
        return;
    }

    kt_zkml_tensor_view dup[2] = { views[0], views[0] };
    int rc = kt_zkml_attest_tensors(dup, 2, root);
    if (rc == ZKML_OK) {
        fail("validation: duplicate names accepted");
        return;
    }
    ok("argument validation: nulls, empty name, duplicates rejected");
}

static void test_moe_bf16_config(void) {
    /* 2 experts x 4 intermediate x 8 hidden, BF16. */
    const int experts = 2;
    const int inter = 4;
    const int hidden = 8;
    const size_t n2d = (size_t)experts * inter * hidden;
    uint16_t* gate = calloc(n2d, sizeof(uint16_t));
    uint16_t* up = calloc(n2d, sizeof(uint16_t));
    uint16_t* down = calloc(n2d, sizeof(uint16_t));
    if (gate == NULL || up == NULL || down == NULL) {
        free(gate);
        free(up);
        free(down);
        fail("moe: allocation failed");
        return;
    }
    for (size_t i = 0; i < n2d; ++i) {
        gate[i] = (uint16_t)(0x3f80 + i); /* BF16 1.0 + small delta */
        up[i] = (uint16_t)(0x4000 + i);
        down[i] = (uint16_t)(0x3f00 + i);
    }

    kt_moe_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.expert_num = experts;
    cfg.intermediate_size = inter;
    cfg.hidden_size = hidden;
    cfg.layer_idx = 3;
    cfg.gate_proj = gate;
    cfg.up_proj = up;
    cfg.down_proj = down;
    cfg.hidden_type = KT_TYPE_BF16;

    uint8_t r1[32];
    uint8_t r2[32];
    if (kt_zkml_moe_weights_merkle_root(&cfg, r1) != ZKML_OK) {
        free(gate);
        free(up);
        free(down);
        fail("moe: attest failed");
        return;
    }
    if (kt_zkml_moe_weights_merkle_root(&cfg, r2) != ZKML_OK || memcmp(r1, r2, 32) != 0) {
        free(gate);
        free(up);
        free(down);
        fail("moe: not deterministic");
        return;
    }

    /* Negative: one flipped weight byte changes the root. */
    down[n2d / 2] ^= 0x0001;
    uint8_t r3[32];
    int rc = kt_zkml_moe_weights_merkle_root(&cfg, r3);
    down[n2d / 2] ^= 0x0001;
    free(gate);
    free(up);
    free(down);
    if (rc != ZKML_OK || memcmp(r1, r3, 32) == 0) {
        fail("moe: flipped weight did not change the root");
        return;
    }

    /* A non-BF16 hidden type must be refused, not mis-sized. */
    memset(&cfg, 0, sizeof(cfg));
    cfg.expert_num = experts;
    cfg.intermediate_size = inter;
    cfg.hidden_size = hidden;
    cfg.layer_idx = 3;
    cfg.gate_proj = gate; /* dangling, but rejected before use */
    cfg.up_proj = up;
    cfg.down_proj = down;
    cfg.hidden_type = KT_TYPE_F32;
    if (kt_zkml_moe_weights_merkle_root(&cfg, r1) != ZKML_INVALID_ARGUMENT) {
        fail("moe: non-BF16 hidden type accepted");
        return;
    }
    ok("moe: BF16 config attested; flip detected; non-BF16 refused");
}

static void test_llama_moe_gguf_config(void) {
    /* Q8_0: 32 elements per 34-byte block. Sizes come from the engine. */
    const int experts = 2;
    const int inter = 64; /* multiple of 32 */
    const int hidden = 64;
    const size_t n_gate = (size_t)experts * inter * (hidden / 32) * 34;
    const size_t n_down = (size_t)experts * hidden * (inter / 32) * 34;
    uint8_t* gate = calloc(1, n_gate);
    uint8_t* up = calloc(1, n_gate);
    uint8_t* down = calloc(1, n_down);
    if (gate == NULL || up == NULL || down == NULL) {
        free(gate);
        free(up);
        free(down);
        fail("llama_moe: allocation failed");
        return;
    }
    for (size_t i = 0; i < n_gate; ++i) {
        gate[i] = (uint8_t)(i * 5);
        up[i] = (uint8_t)(i * 7);
    }
    for (size_t i = 0; i < n_down; ++i) {
        down[i] = (uint8_t)(i * 11);
    }

    kt_llama_moe_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.expert_num = experts;
    cfg.intermediate_size = inter;
    cfg.hidden_size = hidden;
    cfg.layer_idx = 5;
    cfg.gate_proj = gate;
    cfg.up_proj = up;
    cfg.down_proj = down;
    cfg.gate_type = KT_TYPE_Q8_0;
    cfg.up_type = KT_TYPE_Q8_0;
    cfg.down_type = KT_TYPE_Q8_0;

    uint8_t r1[32];
    uint8_t r2[32];
    if (kt_zkml_llama_moe_weights_merkle_root(&cfg, r1) != ZKML_OK) {
        free(gate);
        free(up);
        free(down);
        fail("llama_moe: attest failed");
        return;
    }
    if (kt_zkml_llama_moe_weights_merkle_root(&cfg, r2) != ZKML_OK || memcmp(r1, r2, 32) != 0) {
        free(gate);
        free(up);
        free(down);
        fail("llama_moe: not deterministic");
        return;
    }
    down[n_down / 2] ^= 0xFF;
    int rc = kt_zkml_llama_moe_weights_merkle_root(&cfg, r2);
    free(gate);
    free(up);
    free(down);
    if (rc != ZKML_OK || memcmp(r1, r2, 32) == 0) {
        fail("llama_moe: flipped weight did not change the root");
        return;
    }
    ok("llama_moe: Q8_0 config attested via kt_type_row_bytes; flip detected");
}

static void test_transcript_seed(void) {
    uint8_t s1[32];
    uint8_t s2[32];
    uint8_t s3[32];
    const char* ctx = "ktransformers sampling context";
    if (kt_zkml_transcript_seed((const uint8_t*)ctx, strlen(ctx), s1) != ZKML_OK) {
        fail("seed: first call failed");
        return;
    }
    if (kt_zkml_transcript_seed((const uint8_t*)ctx, strlen(ctx), s2) != ZKML_OK) {
        fail("seed: second call failed");
        return;
    }
    if (memcmp(s1, s2, 32) != 0) {
        fail("seed: not deterministic");
        return;
    }
    const char* other = "different context";
    if (kt_zkml_transcript_seed((const uint8_t*)other, strlen(other), s3) != ZKML_OK) {
        fail("seed: other call failed");
        return;
    }
    if (memcmp(s1, s3, 32) == 0) {
        fail("seed: different contexts collided");
        return;
    }
    ok("transcript seed: deterministic and context-sensitive");
}

static void test_witness_shim(void) {
    kt_zkml_witness* w = kt_zkml_witness_new();
    if (w == NULL) {
        fail("witness: allocation failed");
        return;
    }
    uint8_t stmt[32];
    memset(stmt, 0x5A, sizeof(stmt));

    if (kt_zkml_witness_begin_layer(w, 0) != ZKML_OK) {
        fail("witness: begin failed");
        kt_zkml_witness_free(w);
        return;
    }
    kt_zkml_slot_key key = { .layer = 0, .expert = 3, .op = ZKML_OP_GEMM_C, .rank = 1 };
    if (kt_zkml_witness_record(w, &key, "GEMM-C-BYTES", 11) != ZKML_OK) {
        fail("witness: record failed");
        kt_zkml_witness_free(w);
        return;
    }
    if (kt_zkml_witness_end_layer(w) != ZKML_OK) {
        fail("witness: end failed");
        kt_zkml_witness_free(w);
        return;
    }
    uint8_t trace1[32];
    uint8_t trace2[32];
    if (kt_zkml_witness_finalize(w, stmt, trace1) != ZKML_OK) {
        fail("witness: finalize failed");
        kt_zkml_witness_free(w);
        return;
    }
    /* Frozen after finalize. */
    if (kt_zkml_witness_begin_layer(w, 1) == ZKML_OK) {
        fail("witness: session not frozen after finalize");
        kt_zkml_witness_free(w);
        return;
    }
    kt_zkml_witness_free(w);

    /* Same inputs, second session -> same hash. */
    w = kt_zkml_witness_new();
    if (w == NULL) {
        fail("witness: second allocation failed");
        return;
    }
    kt_zkml_witness_begin_layer(w, 0);
    kt_zkml_witness_record(w, &key, "GEMM-C-BYTES", 11);
    kt_zkml_witness_end_layer(w);
    kt_zkml_witness_finalize(w, stmt, trace2);
    kt_zkml_witness_free(w);

    if (memcmp(trace1, trace2, 32) != 0) {
        fail("witness: trace not deterministic");
        return;
    }
    ok("witness: shim lifecycle, frozen state, deterministic trace");
}

static void test_engine_variant(void) {
    const char* v = kt_zkml_cpu_variant();
    if (v == NULL || v[0] == '\0') {
        fail("engine: no CPU variant reported");
        return;
    }
    printf("     engine variant: %s\n", v);
    ok("engine: kt_get_cpu_variant reachable through the glue");
}

/* --- artifacts for the independent auditor -------------------------------- */

static void write_artifacts(const char* dir, const uint8_t* root,
                            const kt_zkml_tensor_view* views, size_t count) {
    char path[512];
    snprintf(path, sizeof(path), "%s/root.hex", dir);
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        return;
    }
    char hex[65];
    hexdump(root, 32, hex);
    fprintf(f, "%s\n", hex);
    fclose(f);

    static const char* b64 =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    snprintf(path, sizeof(path), "%s/manifest.json", dir);
    f = fopen(path, "wb");
    if (f == NULL) {
        return;
    }
    fputc('[', f);
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* d = (const uint8_t*)views[i].data;
        const size_t n = views[i].bytes;
        if (i > 0) {
            fputc(',', f);
        }
        fprintf(f, "{\"name\": \"%s\", \"data_b64\": \"", views[i].name);
        for (size_t j = 0; j < n; ++j) {
            fputc(b64[d[j] >> 2], f);
            fputc(b64[((d[j] & 3) << 4) | (j + 1 < n ? d[j + 1] >> 4 : 0)], f);
            if (j + 1 < n) {
                fputc(b64[((d[j + 1] & 15) << 2) | (j + 2 < n ? d[j + 2] >> 6 : 0)], f);
                fputc(b64[d[j + 2] & 63], f);
                j += 2;
            }
        }
        const size_t rem = n % 3;
        if (rem == 1) {
            fputs("==", f);
        } else if (rem == 2) {
            fputc('=', f);
        }
        fputs("\"}", f);
    }
    fputs("]\n", f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <workdir>\n", argv[0]);
        return 2;
    }

    test_generic_positive_and_determinism();
    test_generic_order_independence();
    test_generic_negative_flipped_byte();
    test_argument_validation();
    test_moe_bf16_config();
    test_llama_moe_gguf_config();
    test_transcript_seed();
    test_witness_shim();
    test_engine_variant();

    /* Artifacts from the generic path (the same tensors the auditor sees). */
    kt_zkml_tensor_view views[N_TENSORS];
    uint8_t root[32];
    fill_fixture(1);
    build_views(views);
    if (kt_zkml_attest_tensors(views, N_TENSORS, root) == ZKML_OK) {
        write_artifacts(argv[1], root, views, N_TENSORS);
    }

    char hex[65];
    hexdump(root, 32, hex);
    printf("root: %s\n", hex);
    if (failures == 0) {
        printf("PASS\n");
        return 0;
    }
    printf("FAILED: %d check(s)\n", failures);
    return 1;
}
