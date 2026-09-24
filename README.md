# zig-zkml

![toolchain](https://img.shields.io/badge/zig-0.16.0-f7a41d?logo=zig&style=flat-square)
![estado](https://img.shields.io/badge/estado-pre--alpha--F2-d08770?style=flat-square)
![gates](https://github.com/samooth/zig-zkml/actions/workflows/ci.yml/badge.svg)

Verifiable-inference (zkML) layer for host inference engines — **llama.cpp, vLLM, zig-ai, ktransformers-zig** — via per-engine adapters: prove that a layer's output was produced by the committed model weights, using the engine's native kernels as the witness generator.

**Status: F0/F1 + all four engine adapters implemented; F2 STARK backend, operand binding, 16-MAC chunking, compiled routing AIR, a format-generic bit-exact float multiply and exact widening to fp32 all landed. The plan was REORDERED by what was measured — see the architecture decision at the top of [TODO.md](TODO.md).** The technical specification lives in [BLUE_PRINT.md](BLUE_PRINT.md) (Spanish); the multi-engine compatibility plan (adapter matrix, staging, llama.cpp wrapper design) lives in [PLAN_MULTI_ENGINE.md](PLAN_MULTI_ENGINE.md). [zkML.md](zkML.md) is the earlier motivation document; BLUE_PRINT.md supersedes its open decisions. The L0/L2 foundations (field, merkle, transcript, tensor, trace, statement, attestation) and all four engine adapters (Stages 0–4) are working, and the F2 STARK backend (`libs/stark/`) proves a composed quotient. What remains is the real-engine witness, the per-format weight layer and the fingerprint/sumcheck prover, which is now the critical path rather than an optimisation: at the measured ~130 µs per MAC, per-output-element proving costs about 9 days for one 2048×1408 tile, and even bit-exact float multiplication costs 72–118 composed constraints per operation depending on the format — affordable only because the fingerprint statement touches O(m+n) values instead of O(mnk).

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
L4  C API                     zkml_* (attest/prove/verify + witness ABI v2)
                              — adapters per engine in adapters/<engine>/
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
include/
├── zkml_c.h       # C ABI (zkml_* — F0/F1 surface, BLUE_PRINT §8)
└── zkml_engine.h  # engine adapter contract (weight stream, witness,
                   #        prove/verify lifecycle, stage checklist)
adapters/
├── README.md      # per-engine integration matrix (llama.cpp, zig-ai,
│                  #        vLLM, ktransformers — PLAN_MULTI_ENGINE.md)
├── llama_cpp/     # [done] zero-fork GGUF attestation wrapper
│                  #        (zig build llama-adapter)
├── zig_ai/        # [done] same-language module import: TensorSource →
│                  #        root, MetricHooks → witness (11/11 tests)
├── vllm/          # [done] ctypes over libzkml.so, `zkml_attested` load
│                  #        format, witness recorder (15 hermetic tests)
└── ktransformers/ # [done] reference C glue kt_zkml_* → zkml_*
                   #        (zig build kt-adapter)
libs/
├── field.zig      # [done] Goldilocks p = 2^61−1 (vendored L0)
├── merkle.zig     # [done] Blake3 tree (cached root, orphan self-pairing,
│                  #        streaming Builder, "ZKMP" proof wire format)
├── transcript.zig # [done] Fiat-Shamir transcript (absorb/challenge)
├── attestation.zig# [done] WeightsAttestor (Zig-level F0 API)
├── api.zig        # [done] C ABI: zkml_attestor_* / zkml_proof_verify /
│                  #        zkml_transcript_seed / zkml_witness_* (ABI v2;
│                  #        allocator captured, B1)
├── tensor/        # [done] QuantTensor + Scheme + exact q4.22 dequant
├── trace/         # [done] TraceRecorder — canonical-order, thread-safe
├── statement/     # [done] StatementLayer — public inputs (§6.1)
├── gadgets/       # F2+: gemm, quant, nonlin, norm, routing
├── stark/         # [done] STARK backend: fft, constraint IR, column
│                  #        commitment, composition + quotient, verify;
│                  #        float multiply AIR (binary16/bf16/fp8),
│                  #        exact widening to fp32, barrel shifter
├── compile/       # F3+: CircuitGraph → AirGraph
└── prove/         # F3+: prove()/verify() orchestration
tools/
├── abi_check.zig  # end-to-end ABI runner (dumps artifacts for audit)
└── verify_weights.py  # INDEPENDENT Python auditor (root + proof wire)
```

Build, test and audit:

```
zig build --summary all test    # unit tests (check the test count!)
zig build --summary all abi     # C-ABI end-to-end + nm symbol gate
zig build --summary all verify  # tests + ABI + independent Python audit
zig build fmt                   # formatting gate
zig build vllm-adapter          # vLLM ctypes adapter tests (python3, no vLLM needed)
```

`zig build llama-adapter` and `zig build kt-adapter` are **local integration
gates, not CI**: both need a sibling checkout outside this repository
(`../llama.cpp`, and `../ktransformers-zig` already built). CI
(`.github/workflows/ci.yml`) runs everything a clean checkout can: `fmt`,
the test suite, `abi`, `verify` and the vLLM adapter.

The `verify` step is the F0 acceptance gate: the Zig library generates a
weights root and an inclusion proof through the exported C API, and
`tools/verify_weights.py` — which shares **no code** with the library —
re-derives the root from the manifest and verifies the proof bytes.

## Roadmap

| Phase | Deliverable | Go/no-go |
|---|---|---|
| **F0** | Weights attestation (Blake3 + Merkle at load time) — **lib + C ABI + independent auditor done** (`zig build verify` gate); **all four engine adapters done**: llama.cpp (`zig build llama-adapter`), zig-ai (module import), vLLM (ctypes, `zkml_attested`), ktransformers-zig (`zig build kt-adapter`) — each with a negative test and an independent cross-check | load overhead < 5% (not yet measured) |
| **F1** | Deterministic, auditable sampling — **`zkml_transcript_seed` done in the ABI**; adapters consume it via the engine contract | zero kernel changes |
| **Witness ABI v2** | `zkml_witness_*` session/record/finalize (additive) — **done** (Stage 5 of the multi-engine plan; `ZKML_ABI_VERSION = 2`) | determinism test green |
| **F2** | `tensor` lib + GEMM gadget (AIR) with positive+negative tests — **backend done** (`libs/stark/`: FFT over the norm-1 torus, constraint IR, column commitment, RLC composition, quotient, FRI, query-time verification) **plus** operand binding to the quantized weights (nibble range + dequant equation), 16-MAC chunking with its own binding, a compiled routing AIR proving the selected set IS the top-k, a LogUp core, a format-generic bit-exact float multiply with overflow to infinity (118/108/72/72 composed constraints for binary16/bf16/fp8-e4m3/fp8-e5m2, measured), exact widening to fp32 (48–62 constraints, no rounding to get wrong) and a tested barrel-shifter gadget; **still pending**: the real-engine witness, the per-format weight layer, subnormal results, zero/inf/NaN inputs and fp32 accumulation | backend negatives green (tampered trace/opening/quotient/root, random trace); every rounding witness of every format's AIR is flipped and rejected, and ~1465 pairs per format are swept against the reference with every constraint evaluated; all 65536 bfloat16 patterns are widened exhaustively |
| **F3** | `zkml_prove_layer` / `zkml_verify_layer` for one layer (Qwen3-Next shapes) | proof < 1 MB, verify < 100 ms, tampered witness (±1 ulp) rejected |
| **F4** | multi-block recursion over the fingerprint statement — **the sumcheck prover is promoted to the critical path (ROADMAP S3), because it is what makes the float AIR affordable**; only the Groth16 wrap stays optional | product decision |

Realistic cost expectations, measured rather than guessed: weights attestation ~1x; proving one output element of a 2048-deep reduction costs ~130 µs per MAC (`zig build bench`), so one 2048×1408 tile is ~9 days and the per-element STARK is a **reference implementation, not a product path** — the fingerprint statement is what changes that, at O(m+n). Full model **not viable today** — state of the art is ≤1B params with dedicated teams.

## Requirements

- Zig `0.16.0` (stable; `0.16.0-dev.2535+` also works)
- Python 3.8+ with `blake3` (`pip install blake3`) — only for the independent audit tool / `zig build verify`
- [zig-algebra](https://github.com/samooth/zig-algebra) (L0) and [zig-zk](https://github.com/samooth/zig-zk) (L1) — not needed until F2 (F0/F1 are self-contained in `libs/`); see BLUE_PRINT.md §12
- Linux x86_64 (primary)
- Per-adapter extras: a **built** llama.cpp checkout + CMake (Stage 1), vLLM + ctypes (Stage 3), a **built** ktransformers-zig checkout (Stage 4) — see `adapters/`

## Documentation

- [BLUE_PRINT.md](BLUE_PRINT.md) — authoritative technical design (Spanish)
- [PLAN_MULTI_ENGINE.md](PLAN_MULTI_ENGINE.md) — multi-engine adapter plan, staging, llama.cpp wrapper pros/cons
- [TODO.md](TODO.md) — actionable pending work
- [zkML.md](zkML.md) — motivation and prior-art survey (Spanish)

## License

Apache-2.0
