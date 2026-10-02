# adapters/ — per-engine integration

Each subdirectory is a reference adapter that wires one host inference
engine to the engine-agnostic zig-zkml core (`include/zkml_engine.h`
contract, `include/zkml_c.h` ABI). Adapters live **inside this repo**;
upstream engine repos are never edited — their required patches/diffs are
documented as READMEs next to the adapter code.

| Adapter | Stage | Mechanism | Status | Gate |
|---|---|---|---|---|
| `llama_cpp/` | 1 | Zero-fork wrapper over the public `gguf.h` reader (streamed tensors → attestor) | **done** | `zig build llama-adapter` (local) |
| `zig_ai/` | 2 | Direct Zig module import; `TensorSource` over `GgufFile` → attestor; `MetricHooks` → witness | **done** | `zig build test` (CI) |
| `vllm/` | 3 | `ctypes` over `libzkml.so`; `zkml_attested` load format; witness recorder | **done** | `zig build vllm-adapter` (CI) |
| `ktransformers/` | 4 | Reference C glue: `kt_zkml_*` → `zkml_*`, sized by `kt_type_row_bytes` | **done** | `zig build kt-adapter` (local) |

All four ship a negative test and are cross-checked against
`tools/verify_weights.py`.

Two of these gates are **local only** — `llama-adapter` and `kt-adapter` need
`../llama.cpp` and `../ktransformers-zig` checked out outside this repository,
so CI cannot run them. They pass on a machine that has the siblings, so "CI
cannot run them" is not "they are broken". That is also why the status is
per adapter with its gate, rather than one "all four done" that would hide
which two nobody re-checks automatically.

Staging, per-file work items and gates: [`../docs/PLAN_MULTI_ENGINE.md`](../docs/PLAN_MULTI_ENGINE.md).

Rules that apply to every adapter:

- **ABI is additive-only** — v1's nine `zkml_*` functions never change.
- **Each adapter ships its own negative test** (corrupt byte → root changes
  / verify rejects), cross-checked against `tools/verify_weights.py`.
- **Default `zig build test` stays green** — adapter tests that need extra
  toolchains (CMake, Python) run as separate build steps.
