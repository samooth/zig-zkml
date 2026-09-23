# zig-ai adapter (same-language, no FFI)

Stage 2 of [`PLAN_MULTI_ENGINE.md`](../../PLAN_MULTI_ENGINE.md). zig-ai and
zig-zkml are both Zig 0.16.0 stable, so the adapter is a **module import** —
no C ABI, no FFI, no CMake. zig-ai is a sibling repo and is **not** edited
from here; the exact patches it needs are below.

## What is here

| File | Role |
|---|---|
| `root.zig` | Module root re-exporting the two compiled pieces |
| `gguf_attestation.zig` | `TensorSource` interface + `attestSource` (streamed tensor table → Merkle root), `ProofSession`, GGUF tensor-name classification |
| `witness_hooks.zig` | `Recorder` (witness ABI) + the `MetricHooks.on_layer` bridge (thread-local session) |
| `abi.zig` | The single place the adapter touches raw C-ABI pointer types |
| `integration.zig` | zig-ai-facing glue (`GgufSnapshot` over `GgufFile`) — compiled **inside zig-ai** only |
| `test_adapter.zig` | Gate tests, run by `zig build test` in this repo (11 tests) |

The adapter consumes a structural `TensorSource` (`count` / `name` / `data`)
rather than importing zig-ai's module, because zig-ai's build registers its
modules with `b.createModule` and never calls `b.addModule` — there is no
package-level module to depend on. `GgufSnapshot` supplies the three
functions from `GgufFile.tensors` + `tensorData()`, which is the same mmap
view the engine's own layer loaders read from (zero-copy: the attestor reads
each tensor once and never retains it).

## Patch 1 — dependency

`zig-ai/build.zig.zon`:

```zig
.dependencies = .{
    .httpx = .{ ... },          // unchanged
    .zig_zkml = .{ .path = "../zig-zkml" },
},
```

## Patch 2 — build.zig module

Next to the existing `gguf_mod` definition in zig-ai's `build.zig`
(≈line 597):

```zig
    // === zkml attestation adapter (Stage 2) ===
    const zkml_dep = b.dependency("zig_zkml", .{
        .target = target,
        .optimize = optimize,
    });
    const zkml_adapter_mod = b.createModule(.{
        .root_source_file = .{ .cwd_relative = "/absolute/path/to/zig-zkml/adapters/zig_ai/integration.zig" },
        .target = target,
        .optimize = optimize,
    });
    zkml_adapter_mod.addImport("gguf", gguf_mod);
    zkml_adapter_mod.addImport("zig_zkml", zkml_dep.module("zig_zkml"));
```

(`integration.zig` imports its two siblings by relative path, so they come
along automatically.) If you prefer to import the pure-Zig pieces without
the `gguf` binding, point the module at `root.zig` instead and drop the
`gguf` import.

## Patch 3 — attest at load

In `src/main.zig`, right after `GgufModel.load` (≈line 989):

```zig
    var model = try gguf_model.GgufModel.load(io, allocator, model_path);
    // --- zig-zkml: weights attestation over the loaded GGUF ---
    var zkml_snap = try adapter.GgufSnapshot.init(allocator, &model.file);
    defer zkml_snap.deinit();
    const zkml_att = try adapter.attestSource(allocator, zkml_snap.source());
    std.debug.print("[zkml] weights root: {x}\n", .{zkml_att.root});
```

Publish `zkml_att.root` wherever the engine already reports model identity
(startup log, `/v1/models` response). It is order-independent, so the
hash-map iteration order of `GgufFile.tensors` does not matter.

## Patch 4 — witness hook (optional)

`MetricHooks` in `src/engine_api/contract.zig` currently has **no call site**
in the inference loops, and its callbacks carry **no userdata**. The bridge
handles both: install a recorder on the inference thread and pass the hook
pointer.

```zig
    var zkml_rec = try adapter.Recorder.init(allocator);
    defer zkml_rec.deinit();
    adapter.install(&zkml_rec);   // thread-local
    defer adapter.uninstall();
    ...
    // inside the per-layer loop, after the layer forward:
    hooks.on_layer = &adapter.onLayer;   // needs @ptrCast of the fn type
```

Because `MetricHooks` is a plain struct of function pointers, wiring it
means a small cast at the call site:

```zig
    const hooks = contract.MetricHooks{
        .on_layer = @ptrCast(&adapter.onLayer),   // *const LayerMetrics -> *const LayerMetricsView
        ...
    };
```

The two structs are layout-identical; `adapter.assertLayerMetricsLayout(contract.LayerMetrics)`
turns any future upstream field reorder into a compile error. Note that
`on_layer` fires **after** the layer forward: with only the hook, the trace
holds one metrics entry per layer. To record real op payloads, open the
layer around the forward (`rec.beginLayer(i)` / `rec.endLayer()`) and call
`rec.recordOpen(.gemm_c, bytes)`; the hook's entry then lands inside it.

## Gate (this repo)

```sh
zig build test          # 70 core + 11 adapter tests
```

| Case | Expectation |
|---|---|
| Order independence | forward vs reversed tensor order → identical root |
| Negative | one flipped tensor byte → different root |
| Duplicates | rejected (`error.DuplicateName`), not silently folded |
| Core agreement | adapter root == `zkml_attestor_*` root for the same tensors |
| Proof | round-trips through `zkml_proof_verify`; wrong root rejected |
| Name parsing | `blk.N.{attn_q,ffn_down,mlp.gate_proj,feed_forward.w2}.{weight,bias}` |
| Witness | per-layer hook trace is deterministic; engine-opened layer is left open |
| State machine | double begin / record with no layer / finalize with open layer → rejected |
| Hook safety | no installed recorder → no-op, no trap |

Writing these tests found a real leak in the core
(`merkle.Builder.deinit` dropped the `ArrayList` backing buffers without
freeing them); fixed in `libs/merkle.zig`.
