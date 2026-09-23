# ktransformers-zig reference glue (Stage 4)

The **reference adapter**: the simplest possible mapping from an engine's
C ABI to the engine-agnostic core, used to calibrate the checklist that the
llama.cpp, zig-ai and vLLM adapters are measured against. The engine repo is
never modified from here — this compiles against ktransformers-zig's own
`include/kt_kernel.h` and links its prebuilt shared library.

## Files

| File | Role |
|---|---|
| `kt_glue.h` | Public glue API (`kt_zkml_*`), built on `kt_kernel.h` + `zkml_c.h` |
| `kt_glue.c` | Implementation: generic attestor, typed MoE/LlamaMoe helpers, F1 seed, witness shim |
| `test_glue.c` | Gated test: positive, determinism, order-independence, **negative**, argument validation, engine-shaped configs, seed, witness |
| `CMakeLists.txt` | Standalone build of `libkt_zkml_glue.so` + `kt_glue_test` |

## Mapping: engine names → core calls

The engine's design doc (`zkML.md`) sketched the F0/F1 surface as
`kt_weights_merkle_root`, `kt_mla_weights_merkle_root` and
`kt_transcript_seed`. The glue prefixes everything with `kt_zkml_` instead
of claiming those names, so a future implementation inside the engine cannot
produce two shared objects exporting the same symbol.

| Engine-side (proposed) | This glue | Core ABI |
|---|---|---|
| `kt_weights_merkle_root(KT_MOE*, out)` | `kt_zkml_moe_weights_merkle_root(const kt_moe_config_t*, out)` | `zkml_attestor_create/add/finish/root` |
| `kt_mla_weights_merkle_root(KT_MLA*, out)` | *no typed helper* — see below | `zkml_attestor_*` |
| `kt_transcript_seed(ctx, len, out)` | `kt_zkml_transcript_seed` | `zkml_transcript_seed` |
| recorded mode (does not exist) | `kt_zkml_witness_*` | `zkml_witness_*` (ABI v2) |
| — (generic) | `kt_zkml_attest_tensors(views, n, out)` | `zkml_attestor_*` |

### Why the typed helpers take the *config*, not the handle

`KT_MOE` / `KT_MLA` are opaque and the engine exposes **no** weight getter
and no per-tensor callback; `kt_moe_load_weights` only copies from the
caller-owned `config.gate_proj` / `up_proj` / `down_proj`. The caller still
holds that config, so attestation runs on the config it is about to hand to
`kt_moe_new` — the exact bytes the engine will copy, hashed before the copy.

### Why there is no MLA/DSV3 typed helper

`kt_kernel.h` gives the dimensions (`hidden_size`, `q_lora_rank`,
`num_heads`, `nope_size`, `rope_size`, `kv_lora_rank`) but not the
per-projection shapes, and `v_head_dim` is absent entirely. Every extent
would be a guess, and a root over a guessed extent is worse than no root:
use `kt_zkml_attest_tensors` with views the caller knows to be right.

### Why scales and zero points are not attested

`kt_moe_config_t` carries `gate_scale` / `up_scale` / `down_scale` and zero
points, but their layout depends on `quant_config` (`group_size`,
`per_channel`, `zero_point`), which the C header does not pin. They are
excluded rather than guessed; a deployment that needs them covered passes
explicit views. This is a deliberate, documented gap, not an oversight.

### Sizes come from the engine

For GGUF-quantized `KT_LlamaMoe`, row extents come from the engine's own
`kt_type_row_bytes(n, type)` — no block-size table is duplicated in the
adapter. If the engine cannot size a row (unsupported type, or a row length
that is not a whole number of blocks) the helper returns 0 and the glue
fails with `ZKML_INVALID_ARGUMENT` instead of hashing a wrong number of
bytes. All multiplications are overflow-checked.

### Witness: the engine has no recorded mode

`kt_kernel.h` exposes no begin/end-recording lifecycle, no op-record type,
and no replay entry point. The only internal cache (`ForwardCache` in the
SFT path) exists for backpropagation, is unreachable from C, and is not
replayable. `kt_zkml_witness_*` is therefore the hook surface the engine
*should* call once a recorded mode lands; it is exercised by the tests today
and is a deliberate pass-through of `zkml_witness_*` (this is the reference
adapter — the mapping is meant to be trivially auditable).

## Build and test

```sh
zig build                       # libzkml.so
zig build kt-adapter            # CMake + test + independent Python audit

# or directly
cmake -S adapters/ktransformers -B .zig-cache/kt_adapter \
      -DKT_DIR=/path/to/ktransformers-zig -DKT_VARIANT=avx2
cmake --build .zig-cache/kt_adapter
.zig-cache/kt_adapter/kt_glue_test .zig-cache/kt_adapter/artifacts
python3 tools/integration/verify_adapter_root.py .zig-cache/kt_adapter/artifacts
```

`KT_VARIANT` must match the running CPU (`avx2`, `amx`, `avx512_*`); the
prebuilt kernels live in `<kt>/zig-out/lib/libkt_kernel_ext_<variant>.so`.

## Gate matrix

| Case | Expectation |
|---|---|
| Positive | 4 synthetic tensors → root, stable across two passes |
| Order independence | forward vs reversed views → identical root |
| **Negative** | one flipped tensor byte → **different** root |
| Argument validation | null views / zero count / empty name / null data rejected; duplicate names rejected |
| MoE (BF16) | config-shaped root, flip detected, non-BF16 `hidden_type` refused |
| LlamaMoe (Q8_0) | extents from `kt_type_row_bytes`, flip detected |
| Transcript seed | deterministic, context-sensitive |
| Witness | lifecycle, frozen-after-finalize, deterministic trace |
| Engine | `kt_get_cpu_variant()` reachable (logged next to the root) |
| Cross-check | root verified by `tools/verify_weights.py` |
| Symbols | `nm -D libkt_zkml_glue.so` shows `kt_zkml_*` |

A published root is only meaningful together with the kernel that will
consume it, which is why `kt_zkml_cpu_variant()` exists: log the variant
next to the root.
