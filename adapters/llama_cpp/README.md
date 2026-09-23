# llama.cpp attestation adapter (zero-fork)

Stage 1 of [`PLAN_MULTI_ENGINE.md`](../../PLAN_MULTI_ENGINE.md). Streams every
tensor of a GGUF file into zig-zkml's weights attestation
(`zkml_attestor_*`) and exposes the 32-byte Merkle root. Uses only the
public `gguf.h` reader API — the same container walk as
`examples/gguf-hash/` — so it survives upstream rebases with no fork.

## Pinned llama.cpp

| | |
|---|---|
| Commit | `1c3c9674de4d455f1e571bed808252af54932767` |
| Version | `0.0.10269` |
| Shared libs | `build/bin/libggml-base.so`, `libllama.so`, … |

Any recent llama.cpp with `gguf_init_from_file` / `gguf_get_tensor_*` works;
the commit above is the checkout this adapter was tested against.

## Build

```sh
# 1. zig-zkml core (installs zig-out/lib/libzkml.so)
zig build

# 2. adapter (LLAMA_DIR defaults to the sibling ../llama.cpp checkout)
cmake -S adapters/llama_cpp -B .zig-cache/llama_adapter
cmake --build .zig-cache/llama_adapter

# 3. run the gated test (positive + negative + artifacts)
.zig-cache/llama_adapter/zkml_llama_test .zig-cache/llama_adapter/artifacts

# 4. independent Python cross-check of the emitted root
python3 tools/verify_weights.py manifest \
    --root "$(tr -d '\n' < .zig-cache/llama_adapter/artifacts/root.hex)" \
    .zig-cache/llama_adapter/artifacts/manifest.json
```

Or through the build graph (requires CMake on `PATH`):

```sh
zig build llama-adapter
```

## API

```c
#include "zkml_llama.h"

uint8_t root[32];
int rc = zkml_llama_attest_gguf("model.gguf", root);        // root only
rc = zkml_llama_attest_gguf_manifest("model.gguf", root,    // + manifest.json
                                     "manifest.json");
```

`rc == 0` (`ZKML_OK`) on success; negative values are `zkml_c.h` status
codes. The manifest is the `{"name","data_b64"}` shape consumed by
`tools/verify_weights.py manifest`.

## Design notes

- **mmap / zero-copy**: the reader path streams each tensor once through
  `zkml_attestor_add` (Blake3 incremental, data never retained). No full
  model materialization for pure attestation.
- **`llama_model_init_from_user` + `cb_eval`** (plan path a/b for
  decode-time witness) is intentionally *not* used here: pure F0
  attestation needs only the container, not a loaded model. The witness
  ABI (v2) is already exported from `libzkml`; wiring `cb_eval` lands with
  F2 engine integration.
- **No upstream edits** — this lives entirely under `adapters/llama_cpp/`.

## Test matrix (Stage 1 gate)

| Case | Expectation |
|---|---|
| Positive | tiny 3-tensor GGUF → root, stable across two passes |
| Manifest | `attest_gguf_manifest` root == `attest_gguf` root |
| Negative | flip one byte in tensor payload → **different** root |
| Cross-check | `tools/verify_weights.py manifest --root … manifest.json` exits 0 |
| Symbols | `nm -D libzkml_llama.so` shows `zkml_llama_*` + inherited `zkml_*` |
