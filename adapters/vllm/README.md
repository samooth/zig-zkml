# vLLM adapter (ctypes over libzkml.so)

Stage 3 of [`PLAN_MULTI_ENGINE.md`](../../PLAN_MULTI_ENGINE.md). vLLM is
Python, so the adapter talks to the core through the shared library
(`zig-out/lib/libzkml.so`, added in Stage 1) using `ctypes` — no compiled
extension, no Cython, no build step for the plugin itself.

vLLM itself is **not** edited: the adapter ships as a package
(`zkml_vllm`) that registers a custom model loader through the documented
`vllm.general_plugins` entry point.

## Files

| File | Role |
|---|---|
| `zkml.py` | ctypes binding: `Attestor`, `Witness`, `SlotKey`, `verify_proof`, `transcript_seed`, `attest_items`; library discovery + explicit `argtypes`/`restype` |
| `model_loader.py` | `ZkmlAttestedLoader` (built on demand by `build_loader_cls`), plugin entry point `register()`, safetensors shard iteration |
| `witness.py` | `WitnessRecorder` + `LayerScope` (owns the begin/end boundary), custom-op recording helpers |
| `pyproject.toml` | package `zkml-vllm`; entry point `zkml_vllm.model_loader:register` |
| `test_zkml.py` | 15 gate tests, **zero Python dependencies** (see below) |

## Use

```sh
# 1. build the shared library
zig build

# 2. install the plugin (editable is fine)
pip install -e adapters/vllm

# 3. serve with attestation
vllm serve <model> --load-format zkml_attested
```

The loader logs and publishes the root:

```
INFO zkml.vllm: attested 291 tensors, root 3f9c…
model.weights_root == bytes.fromhex("3f9c…")
```

Nothing changes for vanilla vLLM: the `zkml_attested` load format is opt-in,
and the plugin's registration is a no-op if never selected.

### Direct use of the binding

```python
from zkml_vllm import zkml

with zkml.Attestor() as att:
    for name, tensor in loader:
        att.add(name, tensor)
    root = att.finish()
    proof = att.proof("blk.0.attn_q.weight")

assert zkml.verify_proof(proof, root)     # no model, no tree, no runtime
```

`$ZKML_LIB` overrides the library path; otherwise the adapter looks for
`zig-out/lib/libzkml.so` relative to itself.

## What is attested

The loader keeps vLLM's normal weight path intact and tees the same
`(name, tensor)` stream vLLM already produces
(`weight_utils.safetensors_weights_iterator`) into the attestor:

- **checkpoint names**, pre-`WeightsMapper` — `q_proj.weight` is hashed as
  `q_proj.weight`, not as the engine's fused `qkv_proj` shard. A published
  root has to describe the artifact, not the engine's parameter layout.
- **raw on-disk bytes** — tensors are read unconverted, so a dtype cast
  applied during load cannot change what is attested.
- **only what the model consumes** — the tee runs inside the generator, so
  tensors skipped by vLLM's filters (e.g. expert-parallel pruning) are not
  silently included in the root.

Order does not matter: leaves are name-sorted before folding, so a
multi-shard or hash-ordered iteration yields the same root.

## Witness recording: what vLLM does and does not offer

There is no first-class per-op I/O hook in vLLM. The options, and why the
adapter does not pretend otherwise:

| Mechanism | Verdict |
|---|---|
| PyTorch module hooks | skipped for `CompilationMode.STOCK_TORCH_COMPILE`; not called during CUDA-graph replays |
| Inductor passes (`inductor_passes`) | compile-time `fx.Graph` only — never runtime values |
| `set_forward_context` | batch/attention metadata, not op witnesses |
| `direct_register_custom_op` | a real runtime hook; usable, but the model must call the op |
| `CustomOp.register_oot` | per-target-layer replacement, platform-specific dispatch |

So `witness.py` provides the recording side (`WitnessRecorder`,
`LayerScope`, `register_custom_op`, `register_oot_op`) and documents that the
hook must be explicit. The recorder enforces that `begin_layer`/`end_layer`
come from one thread while `record_op` may be called concurrently — matching
the core's TraceRecorder contract.

## Gate

```sh
zig build vllm-adapter     # or: python3 adapters/vllm/test_zkml.py
```

The tests are written as plain functions and run **without pytest** (a
built-in runner executes them when the file is run directly), because the
adapter's design goal is having no Python dependencies; `pytest
adapters/vllm/test_zkml.py` also works when pytest is available. The
safetensors test skips itself when `safetensors` is not installed.

| Case | Expectation |
|---|---|
| Determinism | same tensors twice → same root |
| Order independence | forward vs reversed → same root |
| **Negative** | one flipped tensor byte → **different** root |
| Duplicates | rejected with `DUPLICATE_NAME` |
| Empty attestation | rejected with `INVALID_ARGUMENT` |
| Proof | `ZKMP` bytes verify; wrong root rejected |
| Transcript seed | deterministic, context-sensitive |
| **Cross-check** | root from ctypes == root from `tools/verify_weights.py` |
| Witness | per-layer trace deterministic; state-machine violations rejected |
| SlotKey | 8 bytes, offsets 0/4/6/7 (C layout parity with the Zig struct) |
| Concurrency | 4 threads × 16 `record_op` on one open layer, no error |
| vLLM-free | `zkml_vllm` imports and its tests run with no vLLM/torch present |

The cross-check is the load-bearing one: it proves the foreign-function
boundary marshals names and byte lengths correctly, not just that the
library is self-consistent.
