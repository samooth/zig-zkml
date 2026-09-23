/*
 * zkml_llama.h — zero-fork attestation wrapper for llama.cpp (Stage 1).
 *
 * Streams every tensor of a GGUF file into the zig-zkml weights
 * attestation (zkml_attestor_*) and exposes the Merkle root. Uses only
 * the public gguf/ggml reader API (same pattern as examples/gguf-hash) —
 * no llama.cpp fork, no model-load side effects for pure attestation.
 *
 * Contract: see include/zkml_engine.h. The 9 zkml_* v1 functions and
 * the v2 witness entry points are re-exported through libzkml; this
 * header only adds the llama.cpp-specific glue.
 */
#ifndef ZKML_LLAMA_H
#define ZKML_LLAMA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Attest a GGUF file: open the container, stream each tensor's bytes
 * into zkml_attestor_add, finish, and write the 32-byte Merkle root to
 * root_out. Returns 0 on success, or a negative zkml status code
 * (see zkml_c.h). root_out must point to 32 writable bytes.
 *
 * Memory: one tensor-sized buffer is allocated at a time (streamed —
 * the whole model is never held). The attestor uses
 * zkml_allocator_process(); nothing the caller must free.
 */
int zkml_llama_attest_gguf(const char* path, uint8_t root_out[32]);

/*
 * Same as zkml_llama_attest_gguf, but also emits a JSON manifest of
 * {"name", "data_b64"} entries (one per tensor) to manifest_path, for
 * cross-checking the root with tools/verify_weights.py. Pass
 * manifest_path = NULL to skip the manifest. Returns 0 / negative.
 */
int zkml_llama_attest_gguf_manifest(const char* path,
                                    uint8_t root_out[32],
                                    const char* manifest_path);

#ifdef __cplusplus
}
#endif

#endif /* ZKML_LLAMA_H */
