/*
 * zkml_c.h — C ABI for zig-zkml (F0/F1 surface, BLUE_PRINT §8).
 *
 * Verifiable-inference foundations for ktransformers-zig:
 *   - F0: weights attestation (Merkle root over model tensors)
 *   - F1: deterministic transcript seed (reproducible sampling)
 *
 * Conventions:
 *   - Opaque handle ZKML_ATTESTOR; the allocator passed to
 *     zkml_attestor_create is captured (B1): everything derived from the
 *     handle is freed by its destroy/free functions.
 *   - Status codes: 0 = ZKML_OK, negative = error.
 *   - All functions are thread-safe ONLY per distinct handles; a single
 *     attestor is single-threaded (it hashes at load time, before
 *     inference threads exist).
 *   - ABI is additive; signature changes bump ZKML_ABI_VERSION.
 */
#ifndef ZKML_C_H
#define ZKML_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZKML_ABI_VERSION 1

/* Status codes (stable). */
enum {
    ZKML_OK = 0,
    ZKML_OUT_OF_MEMORY = -1,
    ZKML_INVALID_ARGUMENT = -2,
    ZKML_BAD_PROOF = -3,     /* parse failure or verification failure */
    ZKML_LEAF_NOT_FOUND = -4,
    ZKML_DUPLICATE_NAME = -5,
};

/* Opaque weights-attestation handle. */
typedef struct ZKML_Attestor ZKML_Attestor;

/*
 * Create an attestor. `allocator` is a pointer to a stable
 * std.mem.Allocator (from the host runtime); it is captured and must
 * outlive the handle. Returns NULL on allocation failure.
 *
 * The typical kt_kernel.h glue passes its own captured allocator here.
 */
ZKML_Attestor* zkml_attestor_create(void* allocator);

/*
 * Hash one tensor into the attestation as it streams past. Blake3
 * incremental: `data` is read once and NEVER retained; `name` is copied.
 * Tensors may arrive in ANY order — the root is order-independent.
 *
 * Call this from loadWeights as each tensor is read. Returns ZKML_OK.
 */
int zkml_attestor_add(ZKML_Attestor* attestor,
                      const char* name, size_t name_len,
                      const void* data, size_t data_len);

/*
 * Finish: compute and cache the Merkle root. After this call the tree is
 * frozen (add returns ZKML_INVALID_ARGUMENT). Requires >= 1 tensor.
 * Idempotent on a finished attestor.
 */
int zkml_attestor_finish(ZKML_Attestor* attestor);

/*
 * Write the 32-byte Merkle root (order-independent, cached).
 * `root_out` must point to 32 writable bytes.
 */
int zkml_attestor_root(ZKML_Attestor* attestor, uint8_t* root_out);

/*
 * Serialize the inclusion proof for tensor `name`. `proof_out`/`len_out`
 * receive a buffer from the captured allocator — free it with
 * zkml_attestor_free_proof. The wire format is versioned ("ZKMP" v1):
 * 48-byte header + 32 bytes per tree level. An auditor only needs the
 * published root, the tensor bytes and this proof to verify a tensor.
 */
int zkml_attestor_proof(ZKML_Attestor* attestor,
                        const char* name, size_t name_len,
                        uint8_t** proof_out, size_t* len_out);

/* Free a proof buffer obtained from zkml_attestor_proof (B1). */
void zkml_attestor_free_proof(ZKML_Attestor* attestor,
                              uint8_t* proof, size_t proof_len);

/*
 * Destroy the attestor: frees the tree, all copied names and the handle
 * itself. Safe on unfinished attestors (mid-load destroy included).
 */
void zkml_attestor_destroy(ZKML_Attestor* attestor);

/*
 * Standalone proof verification — the auditor's path. Needs NEITHER
 * the model NOR the tree: only the published root and the proof bytes.
 * Returns ZKML_OK when the proof verifies against `expected_root`.
 *
 * `allocator`: pointer to a std.mem.Allocator used for scratch space.
 */
int zkml_proof_verify(void* allocator,
                      const uint8_t* proof, size_t proof_len,
                      const uint8_t* expected_root /* 32 bytes */);

/*
 * F1: derive a deterministic 32-byte sampling seed from a context.
 * Same context -> same seed, always (no clocks, no hidden state).
 * Use: reproducible/auditable sampling in ktransformers-zig.
 */
int zkml_transcript_seed(const uint8_t* context, size_t context_len,
                         uint8_t* seed_out /* 32 bytes */);

#ifdef __cplusplus
}
#endif

#endif /* ZKML_C_H */
