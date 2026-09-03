# zig-zkml

Verifiable-inference (zkML) layer for [ktransformers-zig](../ktransformers-zig): prove that a MoE layer's output was produced by the committed model weights, using the native kernels as the witness generator.

**Status: F0/F1 foundations implemented.** The technical specification lives in [BLUE_PRINT.md](BLUE_PRINT.md) (Spanish). [zkML.md](zkML.md) is the earlier motivation document; BLUE_PRINT.md supersedes its open decisions. The L0/L2 foundations (field, merkle, transcript, tensor, trace, statement, attestation) are working — the C API integration with ktransformers-zig and the STARK prover (F2+) come next.

## What it does

zkML has three modes; this library targets the practical one for local/served inference:

| Mode | Proves | In scope |
|---|---|---|
| **(a) Integrity** | The output Y was produced by actually running the committed weights `root(W)` on input X | **v1 (F0–F3)** |
| (b) Weight privacy | Prover uses W without revealing it | future extension |
| (c) Both (zkVM-style) | (a) + (b) | out of scope (671B models) |

The core design principle: **native inference is the witness generator**. MoE kernels (`gemmExpert`, routing, SwiGLU) run at native speed and record their trace; the STARK prover only interpolates the trace and runs FRI. The model is never executed "inside the circuit".

Key design decisions (full rationale in BLUE_PRINT.md):

- **Exact-arithmetic contract**: in recorded mode, kernels run the gadget's exact arithmetic (integer accumulators, fixed-point scales, lookup-table nonlinearities) — a fast-path FP32 trace would not satisfy field-exact constraints.
- **Single backend (Goldilocks) in v1**: no cross-domain composition; Binius deferred to F4+.
- **GEMM as monolithic AIR in v1** (chunk-16 running sums); fingerprint sumcheck (zkLLM-style) as the v2 optimization.
- **Weights as committed AIR columns** bound to the F0 Merkle root via an in-trace Poseidon2 hash column.

## Architecture

```
L4  C API                     kt_prove_* / kt_verify_* (kt_kernel.h pattern)
L3  Model compiler            CircuitGraph → AirGraph
L2  zkML gadgets  ◄ this repo QuantTensor, gemm, quant/dequant, swiglu,
                              routing, lookups (LogUp)
L1  Proof systems (zig-zk)    STARK+FRI, AIR, transcript, sumcheck, recursion
L0  Algebra (zig-algebra)     Goldilocks, hash, merkle, NTT
```

Layout (single module rooted at `zkml.zig` — tests are only collected
from the root module's file set, so all libs are file imports):

```
zkml.zig           # module root — re-exports everything, collects all tests
libs/
├── field.zig      # [done] Goldilocks p = 2^61−1 (vendored L0)
├── merkle.zig     # [done] Blake3 tree (cached root, orphan self-pairing)
├── transcript.zig # [done] Fiat-Shamir transcript (absorb/challenge)
├── attestation.zig# [done] WeightsAttestor (F0)
├── tensor/        # [done] QuantTensor + Scheme + exact q4.22 dequant
├── trace/         # [done] TraceRecorder — canonical-order, thread-safe
├── statement/     # [done] StatementLayer — public inputs (§6.1)
├── gadgets/       # F2+: gemm, quant, nonlin, norm, routing
├── compile/       # F3+: CircuitGraph → AirGraph
└── prove/         # F3+: prove()/verify() orchestration
```

Build and test:

```
zig build --summary all test   # run the suite (check the test count!)
zig build fmt                  # formatting gate
```

## Roadmap

| Phase | Deliverable | Go/no-go |
|---|---|---|
| **F0** | `kt_weights_merkle_root` — weights attestation (Blake3 + Merkle at load time) — **lib done** (`libs/merkle.zig`, `libs/attestation.zig`), C API pending | load overhead < 5% |
| **F1** | `kt_transcript_seed` — deterministic, auditable sampling — **lib done** (`libs/transcript.zig`, `libs/trace/root.zig` incl. multi-thread determinism test) | zero kernel changes |
| **F2** | `tensor` lib + GEMM gadget (AIR) with positive+negative tests — **tensor lib done**; gadget pending | two spikes first: dep toolchain (semver `0.16.0-dev`), STARK/FRI over Goldilocks |
| **F3** | `kt_prove_moe_layer` / `kt_verify_moe_layer` for one expert (Qwen3-Next shapes) | proof < 1 MB, verify < 100 ms, tampered witness (±1 ulp) rejected |
| **F4** | fingerprint-sumcheck GEMM + multi-block recursion | product decision |

Realistic cost expectations (measured against DeepSeek-V3/Qwen3-Next shapes): weights attestation ~1x; one expert block ~10³x native (sub-second to seconds); full model **not viable today** — state of the art is ≤1B params with dedicated teams.

## Requirements

- Zig `0.16.0-dev.2535+` (toolchain lock shared with ktransformers-zig)
- [zig-algebra](https://github.com/samooth/zig-algebra) (L0) and [zig-zk](https://github.com/samooth/zig-zk) (L1) — vendored by default; see BLUE_PRINT.md §12 for the toolchain-semver risk and fork policy
- Linux x86_64 (primary)

## Documentation

- [BLUE_PRINT.md](BLUE_PRINT.md) — authoritative technical design (Spanish)
- [zkML.md](zkML.md) — motivation and prior-art survey (Spanish)

## License

Apache-2.0
