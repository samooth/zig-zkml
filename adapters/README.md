# adapters/ — per-engine integration

Each subdirectory is a reference adapter that wires one host inference
engine to the engine-agnostic zig-zkml core (`include/zkml_engine.h`
contract, `include/zkml_c.h` ABI). Adapters live **inside this repo**;
upstream engine repos are never edited — their required patches/diffs are
documented as READMEs next to the adapter code.

| Adapter | Stage | Mechanism |
|---|---|---|
| `llama_cpp/` | 1 ✅ | Zero-fork wrapper over the public `gguf.h` reader (streamed tensors → attestor) |
| `zig_ai/` | 2 ✅ | Direct Zig module import; `TensorSource` over `GgufFile` → attestor; `MetricHooks` → witness |
| `vllm/` | 3 | Shared `libzkml.so` + ctypes; `BaseModelLoader` subclass |
| `ktransformers/` | 4 | Reference glue (`kt_*` → `zkml_*` mapping) |

Staging, per-file work items and gates: [`../PLAN_MULTI_ENGINE.md`](../PLAN_MULTI_ENGINE.md).

Rules that apply to every adapter:

- **ABI is additive-only** — v1's nine `zkml_*` functions never change.
- **Each adapter ships its own negative test** (corrupt byte → root changes
  / verify rejects), cross-checked against `tools/verify_weights.py`.
- **Default `zig build test` stays green** — adapter tests that need extra
  toolchains (CMake, Python) run as separate build steps.
