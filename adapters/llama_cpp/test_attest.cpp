/*
 * test_attest.cpp — Stage 1 gate for the llama.cpp adapter.
 *
 * Builds a tiny 3-tensor GGUF, attests it through zkml_llama_attest_gguf
 * (positive + determinism), then corrupts one tensor byte and asserts the
 * root changes (negative). Emits root.hex + manifest.json for the
 * independent Python auditor (tools/verify_weights.py).
 *
 * Usage: zkml_llama_test <workdir>
 *   writes <workdir>/{model.gguf,corrupt.gguf,root.hex,manifest.json}
 * Exit 0 = pass.
 */
#include "zkml_llama.h"
#include "zkml_c.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool Fail(const char* msg) {
    fprintf(stderr, "FAIL: %s\n", msg);
    return false;
}

void HexDump(const uint8_t* b, size_t n, char* out /* 2*n+1 */) {
    static const char* hexd = "0123456789abcdef";
    for (size_t i = 0; i < n; ++i) {
        out[2 * i] = hexd[b[i] >> 4];
        out[2 * i + 1] = hexd[b[i] & 15];
    }
    out[2 * n] = '\0';
}

// Write a minimal GGUF with three named F32 tensors.
bool WriteTinyGguf(const char* path) {
    struct ggml_init_params mp;
    mp.mem_size = 16u * 1024u * 1024u;
    mp.mem_buffer = nullptr;
    mp.no_alloc = false;
    struct ggml_context* ml = ggml_init(mp);
    if (ml == nullptr) {
        return false;
    }

    struct gguf_context* guf = gguf_init_empty();

    // tensor A: 4x4 f32, tensor B: 8x1 f32, tensor C: 1x1 f32
    struct ggml_tensor* a = ggml_new_tensor_2d(ml, GGML_TYPE_F32, 4, 4);
    ggml_set_name(a, "layer.0.weight");
    auto* ad = (float*)a->data;
    for (int i = 0; i < 16; ++i) {
        ad[i] = float(i) * 0.25f;
    }

    struct ggml_tensor* b = ggml_new_tensor_2d(ml, GGML_TYPE_F32, 8, 1);
    ggml_set_name(b, "layer.1.weight");
    auto* bd = (float*)b->data;
    for (int i = 0; i < 8; ++i) {
        bd[i] = float(i + 1) * -1.5f;
    }

    struct ggml_tensor* c = ggml_new_tensor_2d(ml, GGML_TYPE_F32, 1, 1);
    ggml_set_name(c, "bias");
    auto* cd = (float*)c->data;
    cd[0] = 3.14159f;

    gguf_add_tensor(guf, a);
    gguf_add_tensor(guf, b);
    gguf_add_tensor(guf, c);
    gguf_set_val_str(guf, "general.architecture", "zkml_tiny");

    const bool ok = gguf_write_to_file(guf, path, /*only_meta=*/false);
    gguf_free(guf);
    ggml_free(ml);
    return ok;
}

// Flip one byte inside the data section of `src` and write to `dst`.
bool CorruptOneByte(const char* src, const char* dst) {
    struct gguf_init_params params;
    params.no_alloc = true;
    params.ctx = nullptr;
    struct gguf_context* guf = gguf_init_from_file(src, params);
    if (guf == nullptr) {
        return false;
    }
    FILE* f = fopen(src, "rb");
    if (f == nullptr) {
        gguf_free(guf);
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long file_size = ftell(f);
    const size_t data_offset = gguf_get_data_offset(guf);
    // Target the first byte of the first tensor's payload.
    const size_t target = data_offset + gguf_get_tensor_offset(guf, 0);
    if (static_cast<long>(target) >= file_size) {
        fclose(f);
        gguf_free(guf);
        return false;
    }

    std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
    fseek(f, 0, SEEK_SET);
    const bool read_ok = fread(bytes.data(), 1, bytes.size(), f) == bytes.size();
    fclose(f);
    gguf_free(guf);
    if (!read_ok) {
        return false;
    }
    bytes[target] ^= 0xFF;

    FILE* out = fopen(dst, "wb");
    if (out == nullptr) {
        return false;
    }
    const bool write_ok =
        fwrite(bytes.data(), 1, bytes.size(), out) == bytes.size();
    fclose(out);
    return write_ok;
}

bool WriteHexFile(const char* path, const uint8_t* root) {
    char hex[65];
    HexDump(root, 32, hex);
    FILE* f = fopen(path, "wb");
    if (f == nullptr) {
        return false;
    }
    fputs(hex, f);
    fputs("\n", f);
    fclose(f);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <workdir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const std::string model = dir + "/model.gguf";
    const std::string corrupt = dir + "/corrupt.gguf";
    const std::string root_path = dir + "/root.hex";
    const std::string manifest_path = dir + "/manifest.json";

    if (!WriteTinyGguf(model.c_str())) {
        return Fail("write tiny gguf") ? 0 : 1;
    }

    // --- Positive + determinism ---
    uint8_t root1[32] = {0};
    uint8_t root2[32] = {0};
    if (zkml_llama_attest_gguf(model.c_str(), root1) != ZKML_OK) {
        return Fail("attest_gguf (first pass)") ? 0 : 1;
    }
    if (zkml_llama_attest_gguf(model.c_str(), root2) != ZKML_OK) {
        return Fail("attest_gguf (second pass)") ? 0 : 1;
    }
    if (memcmp(root1, root2, 32) != 0) {
        return Fail("attestation not deterministic") ? 0 : 1;
    }
    printf("ok  positive: root is stable across two passes\n");

    // --- Manifest path (same root) ---
    uint8_t root_m[32] = {0};
    if (zkml_llama_attest_gguf_manifest(model.c_str(), root_m,
                                        manifest_path.c_str()) != ZKML_OK) {
        return Fail("attest_gguf_manifest") ? 0 : 1;
    }
    if (memcmp(root1, root_m, 32) != 0) {
        return Fail("manifest path root mismatch") ? 0 : 1;
    }
    printf("ok  manifest emitted (%s)\n", manifest_path.c_str());

    // --- Negative: corrupt one tensor byte -> root must change ---
    if (!CorruptOneByte(model.c_str(), corrupt.c_str())) {
        return Fail("corrupt copy") ? 0 : 1;
    }
    uint8_t root_bad[32] = {0};
    if (zkml_llama_attest_gguf(corrupt.c_str(), root_bad) != ZKML_OK) {
        return Fail("attest corrupt gguf") ? 0 : 1;
    }
    if (memcmp(root1, root_bad, 32) == 0) {
        return Fail("corrupted GGUF produced the SAME root") ? 0 : 1;
    }
    printf("ok  negative: one flipped byte changes the root\n");

    // --- Publish artifacts for tools/verify_weights.py ---
    if (!WriteHexFile(root_path.c_str(), root1)) {
        return Fail("write root.hex") ? 0 : 1;
    }
    char hex[65];
    HexDump(root1, 32, hex);
    printf("root: %s\n", hex);
    printf("artifacts: %s/{model.gguf,corrupt.gguf,root.hex,manifest.json}\n",
           dir.c_str());
    printf("PASS\n");
    return 0;
}
