/*
 * wrapper.cpp — llama.cpp attestation wrapper (Stage 1, zero-fork).
 *
 * Path (a) attestation: open GGUF with the public gguf reader
 * (no_alloc metadata + streamed tensor reads), feed every tensor to
 * zkml_attestor_*, finish, publish the root. Same container walk as
 * examples/gguf-hash; no llama_model is created for pure attestation.
 *
 * Path (b) witness (cb_eval injection) lands with F2 engine integration;
 * the witness ABI itself is already exported from libzkml (ABI v2).
 */
#include "zkml_llama.h"
#include "zkml_c.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Minimal base64 (standard alphabet, with padding) for manifest output.
static const char kB64Alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string Base64Encode(const uint8_t* data, size_t len) {
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    size_t i = 0;
    while (i + 2 < len) {
        uint32_t n = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | data[i + 2];
        out.push_back(kB64Alphabet[(n >> 18) & 63]);
        out.push_back(kB64Alphabet[(n >> 12) & 63]);
        out.push_back(kB64Alphabet[(n >> 6) & 63]);
        out.push_back(kB64Alphabet[n & 63]);
        i += 3;
    }
    if (i + 1 == len) {
        uint32_t n = uint32_t(data[i]) << 16;
        out.push_back(kB64Alphabet[(n >> 18) & 63]);
        out.push_back(kB64Alphabet[(n >> 12) & 63]);
        out.push_back('=');
        out.push_back('=');
    } else if (i + 2 == len) {
        uint32_t n = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8);
        out.push_back(kB64Alphabet[(n >> 18) & 63]);
        out.push_back(kB64Alphabet[(n >> 12) & 63]);
        out.push_back(kB64Alphabet[(n >> 6) & 63]);
        out.push_back('=');
    }
    return out;
}

struct TensorRecord {
    std::string name;
    std::vector<uint8_t> data;
};

// Read one tensor's bytes from the GGUF data section at `abs_offset`.
bool ReadAt(FILE* f, size_t abs_offset, size_t size, std::vector<uint8_t>& out) {
    out.resize(size);
    if (size == 0) {
        return true;
    }
    if (fseek(f, static_cast<long>(abs_offset), SEEK_SET) != 0) {
        return false;
    }
    return fread(out.data(), 1, size, f) == size;
}

// Shared walk: metadata-only GGUF open + streamed per-tensor reads.
// If `records` is non-null, tensor bytes are also retained for the manifest.
int WalkGguf(const char* path,
             uint8_t* root_out,
             std::vector<TensorRecord>* records) {
    if (path == nullptr || root_out == nullptr) {
        return ZKML_INVALID_ARGUMENT;
    }

    struct gguf_init_params params;
    params.no_alloc = true;
    params.ctx = nullptr;
    struct gguf_context* gguf = gguf_init_from_file(path, params);
    if (gguf == nullptr) {
        return ZKML_INVALID_ARGUMENT;
    }

    FILE* f = fopen(path, "rb");
    if (f == nullptr) {
        gguf_free(gguf);
        return ZKML_INVALID_ARGUMENT;
    }

    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    if (n_tensors <= 0) {
        fclose(f);
        gguf_free(gguf);
        return ZKML_INVALID_ARGUMENT;
    }

    ZKML_Attestor* attestor =
        zkml_attestor_create(zkml_allocator_process());
    if (attestor == nullptr) {
        fclose(f);
        gguf_free(gguf);
        return ZKML_OUT_OF_MEMORY;
    }

    const size_t data_offset = gguf_get_data_offset(gguf);
    int rc = ZKML_OK;
    std::vector<uint8_t> buf;

    if (records != nullptr) {
        records->clear();
        records->reserve(static_cast<size_t>(n_tensors));
    }

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf, i);
        const size_t size = gguf_get_tensor_size(gguf, i);
        const size_t offset = data_offset + gguf_get_tensor_offset(gguf, i);

        if (!ReadAt(f, offset, size, buf)) {
            rc = ZKML_INVALID_ARGUMENT;
            break;
        }
        const int add_rc = zkml_attestor_add(
            attestor, name, strlen(name),
            buf.empty() ? (const void*)"" : buf.data(), size);
        if (add_rc != ZKML_OK) {
            rc = add_rc;
            break;
        }
        if (records != nullptr) {
            TensorRecord rec;
            rec.name = name;
            rec.data = buf;
            records->push_back(std::move(rec));
        }
    }

    fclose(f);
    gguf_free(gguf);

    if (rc == ZKML_OK) {
        rc = zkml_attestor_finish(attestor);
    }
    if (rc == ZKML_OK) {
        rc = zkml_attestor_root(attestor, root_out);
    }
    zkml_attestor_destroy(attestor);
    return rc;
}

}  // namespace

extern "C" int zkml_llama_attest_gguf(const char* path, uint8_t root_out[32]) {
    return WalkGguf(path, root_out, nullptr);
}

extern "C" int zkml_llama_attest_gguf_manifest(const char* path,
                                               uint8_t root_out[32],
                                               const char* manifest_path) {
    std::vector<TensorRecord> records;
    const int rc = WalkGguf(path, root_out,
                            manifest_path != nullptr ? &records : nullptr);
    if (rc != ZKML_OK || manifest_path == nullptr) {
        return rc;
    }

    FILE* mf = fopen(manifest_path, "wb");
    if (mf == nullptr) {
        return ZKML_INVALID_ARGUMENT;
    }
    fputs("[", mf);
    for (size_t i = 0; i < records.size(); ++i) {
        const TensorRecord& rec = records[i];
        if (i > 0) {
            fputs(",", mf);
        }
        fputs("{\"name\": \"", mf);
        fwrite(rec.name.data(), 1, rec.name.size(), mf);
        fputs("\", \"data_b64\": \"", mf);
        const std::string b64 = Base64Encode(
            rec.data.empty() ? (const uint8_t*)"" : rec.data.data(),
            rec.data.size());
        fwrite(b64.data(), 1, b64.size(), mf);
        fputs("\"}", mf);
    }
    fputs("]", mf);
    fclose(mf);
    return ZKML_OK;
}
