# zig-zkml

![toolchain](https://img.shields.io/badge/zig-0.16.0-f7a41d?logo=zig&style=flat-square)
![estado](https://img.shields.io/badge/estado-pre--alpha--F2-d08770?style=flat-square)
![gates](https://github.com/samooth/zig-zkml/actions/workflows/ci.yml/badge.svg)

Verifiable-inference (zkML) layer for host inference engines — **llama.cpp,
vLLM, zig-ai, ktransformers-zig** — via per-engine adapters: prove that a
layer's output was produced by the committed model weights, using the
engine's native kernels as the witness generator.

> **Design principle: native inference is the witness generator.** MoE kernels
> (`gemmExpert`, routing, SwiGLU) run at native speed and record their trace;
> the STARK prover only interpolates the trace and runs FRI. The model is never
> executed *inside* the circuit.

| | |
|---|---|
| **Landed** | 4 engine adapters (Stages 0–4) · weights attestation + independent Python auditor · witness ABI v2 · STARK backend (FRI, column commitments, quotient) · GEMM AIR with operand binding, fp16 scale provenance and 16-MAC chunking · bit-exact float multiply for 4 formats |
| **Next** | real-engine witness → per-format weight layer → fingerprint/sumcheck |
| **Gates** | `zig build verify` — 236 tests (221 core + 15 vLLM), C ABI, independent Python audit |

The plan was **reordered by what was measured** — the sumcheck prover became
the critical path, and the weight layer and the witness source became F2/F3
requirements rather than later work.

**Documentation.** [docs/BLUE_PRINT.md](docs/BLUE_PRINT.md) is the
authoritative technical design. [docs/soundness.md](docs/soundness.md) records
the F2 soundness reasoning and the measurements behind it (cyclic domain,
closing-row exemptions, scale provenance).
[docs/PLAN_MULTI_ENGINE.md](docs/PLAN_MULTI_ENGINE.md) covers the adapter
matrix and staging. Design decisions are logged under
[docs/decisions/](docs/decisions/). All of these are in Spanish.

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Building and testing](#building-and-testing)
- [Roadmap](#roadmap)
- [F2 — STARK backend](#f2-stark-backend)
- [Cost, measured](#cost-measured)
- [Requirements](#requirements)
- [Documentation](#documentation)
- [License](#license)

## What it does

zkML has three modes; this library targets the practical one for local and
served inference:

| Mode | Proves | Scope |
|---|---|---|
| **(a) Integrity** | The output Y was produced by actually running the committed weights `root(W)` on input X | **v1 (F0–F3)** |
| (b) Weight privacy | The prover uses W without revealing it | future extension |
| (c) Both (zkVM-style) | (a) + (b) | out of scope (671B models) |

Key design decisions, with full rationale in [docs/BLUE_PRINT.md](docs/BLUE_PRINT.md):

- **Exact-arithmetic contract** — in recorded mode the kernels run the gadget's
  exact arithmetic (integer accumulators, fixed-point scales, lookup-table
  nonlinearities). A fast-path FP32 trace would not satisfy field-exact
  constraints.
- **Single backend (Goldilocks) in v1** — no cross-domain composition; Binius
  deferred to F4+.
- **GEMM as a monolithic AIR in v1** (16-MAC chunked running sums);
  fingerprint sumcheck (zkLLM-style) is the v2 optimisation.
- **Weights as committed AIR columns**, bound to the F0 Merkle root through an
  in-trace Poseidon2 hash column.

## Architecture

```text
L4  C API                   zkml_* — attest / prove / verify, witness ABI v2
                            adapters per engine: adapters/<engine>/
L3  Model compiler          CircuitGraph → AirGraph
L2  zkML gadgets      ◄     QuantTensor, gemm, quant/dequant, swiglu,
    (this repo)             routing, lookups (LogUp)
L1  Proof systems           STARK + FRI, AIR, transcript, sumcheck, recursion
    (zig-zk)
L0  Algebra                 Goldilocks, hash, merkle, NTT
    (zig-algebra)
```

The L0–L2 layers are implemented in `libs/`; L1 and L0 come from `zig-zk` and
`zig-algebra`.

### Layout

Single module rooted at `zkml.zig` — tests are collected only from the root
module's file set, so every library is a file import:

```text
zkml.zig              module root: re-exports everything, collects all tests
include/
  zkml_c.h            C ABI (zkml_* — F0/F1 surface, docs/BLUE_PRINT.md §8)
  zkml_engine.h       engine adapter contract: weight stream, witness
                      hooks, prove/verify lifecycle, stage checklist
adapters/             per-engine integration matrix; llama.cpp, zig-ai,
                      vLLM, ktransformers (see docs/PLAN_MULTI_ENGINE.md)
libs/
  field.zig           Goldilocks p = 2^61−1 (vendored L0)
  merkle.zig          Blake3 tree: cached root, orphan self-pairing,
                      streaming Builder, "ZKMP" proof wire format
  transcript.zig      Fiat-Shamir transcript (absorb / challenge)
  attestation.zig     WeightsAttestor (Zig-level F0 API)
  api.zig             C ABI: zkml_attestor_*, zkml_proof_verify,
                      zkml_transcript_seed, zkml_witness_* (ABI v2,
                      allocator captured)
  tensor/             QuantTensor + Scheme + exact q4.22 dequant
  trace/              TraceRecorder — canonical order, thread-safe
  statement/          StatementLayer — public inputs (§6.1)
  gadgets/            gemm, quant, nonlin, norm, routing
  stark/              STARK backend: FFT, constraint IR, column
                      commitment, composition + quotient, verify; float
                      multiply AIR (binary16/bf16/fp8), exact widening
                      to fp32, barrel shifter
  compile/            F3+: CircuitGraph → AirGraph
  prove/              F3+: prove() / verify() orchestration
tools/
  abi_check.zig       end-to-end ABI runner, dumps artifacts for audit
  fri_audit.zig       independent FRI low-degree audit
  verify_weights.py   independent Python auditor (root + proof wire)
```

Adapter gates: `zig build vllm-adapter`, `zig build llama-adapter`,
`zig build kt-adapter`. The zig-ai adapter needs no separate step — it is a
module import and runs inside `zig build test`.

## Building and testing

```bash
zig build --summary all test    # unit tests — check the count
zig build --summary all abi     # C ABI end-to-end + exported-symbol gate
zig build --summary all verify  # tests + ABI + independent Python audit
zig build fmt                   # formatting gate
zig build spike                 # FRI low-degree audit
zig build bench -- --k 256      # GEMM proving cost
```

`zig build llama-adapter` and `zig build kt-adapter` are **local integration
gates, not CI**: both need a sibling checkout outside this repository
(`../llama.cpp`, and `../ktransformers-zig` already built). CI
(`.github/workflows/ci.yml`) runs everything a clean checkout can: `fmt`, the
test suite, `abi`, `verify` and the vLLM adapter.

`verify` is the F0 acceptance gate: the Zig library generates a weights root
and an inclusion proof through the exported C API, and
`tools/verify_weights.py` — which shares **no code** with the library —
re-derives the root from the manifest and verifies the proof bytes.

## Roadmap

| Phase | Deliverable | Status | Go/no-go |
|---|---|---|---|
| **F0** | Weights attestation (Blake3 + Merkle at load time) | lib + C ABI + independent auditor done, **all four adapters done** | load overhead < 5% (not yet measured) |
| **F1** | Deterministic, auditable sampling | `zkml_transcript_seed` in the ABI; adapters consume it | zero kernel changes |
| **Witness ABI v2** | `zkml_witness_*` session / record / finalize | done (`ZKML_ABI_VERSION = 2`) | determinism test green |
| **F2** | STARK backend + GEMM AIR | backend done, see [below](#f2-stark-backend) | backend negatives green |
| **F3** | `zkml_prove_layer` / `zkml_verify_layer` for one layer | pending | proof < 1 MB, verify < 100 ms, ±1 ulp rejected |
| **F4** | Multi-block recursion over the fingerprint statement | **arithmetic core done** and **commitment order enforced** (`fingerprint_bind.zig`: u, v drawn only after both roots are absorbed); measured 36–309× faster than the oracle. AIR and tile aggregation pending. **Sumcheck is the critical path** | product decision |

## F2 — STARK backend

`libs/stark/` proves a composed quotient. What is done:

- **Proving system** — FFT over the norm-1 torus of F(p²), constraint IR,
  column commitments, RLC composition, quotient, FRI, query-time verification,
  boundary openings, and transition exemptions so the synthetic closing row
  carries no operand binding.
- **Operand binding** — every operand is a raw quant value in range times a
  block scale, and the scale is pinned to the image of `fp16ToFixedQ4_22`
  (`u = ±(2^10 + m)·2^s`, 21 composed constraints and 29 columns per scale).
  Shared per chunk rather than per operand: 16 divides every Q4_K block size,
  so a chunk can never straddle a block boundary.
- **Formats** — `Q4_0` (with scale provenance), `Q8_0` (representation only;
  its scale is not an fp16 in q4.22 yet) and `Q4_1` (affine, signed minima per
  32-element block).
- **Chunking** — 16 MACs per row, 254 columns and 235 composed constraints for
  a 240-MAC reduction.
- **Float multiply** — format-generic and bit-exact, covering zero, infinity
  and NaN as inputs, subnormal *results* and the underflow to zero, with a
  single rounding across the whole range: 258 / 394 / 158 / 170 composed
  constraints for binary16 / bfloat16 / fp8-e4m3 / fp8-e5m2, measured.
- **Widening** — exact widening to fp32, 48–62 constraints, no rounding to
  get wrong. All 65536 bfloat16 patterns swept exhaustively.
- **Also** — a compiled routing AIR proving the selected set *is* the top-k,
  a LogUp core, and a tested barrel-shifter gadget.

Still pending: the real-engine witness, the per-format weight layer, and fp32
accumulation.

Every negative is a gate: tampered trace, opening, quotient and root are
rejected, a random trace is refused, every rounding witness of every format is
flipped and rejected, and **3249 reference pairs per format** are swept against
the reference with every constraint evaluated.

## Cost, measured

Not estimated — measured with `zig build bench`:

| Operation | Cost |
|---|---|
| Weights attestation | ~1× load time |
| Proving one output element, 2048-deep reduction | 338 µs/MAC (1 MAC per row) · 142 µs/MAC (16 MACs per chunk), best of 3 |
| One 2048×1408 layer, per-element STARK | 23 days (1 MAC/row) · 10 days (16 MACs/chunk) — derived from the µs/MAC above × 2.88M MACs × 2048 rows |
| Bit-exact float multiply | 158–394 composed constraints per operation, by format |

The chunked layout is the better one: 16× fewer trace rows and ~2.6× less
proving time, at the cost of ~2.9× slower verification and a 2.6× *larger*
proof. Absolute µs/MAC move with machine load — re-run `zig build bench` and
take the ratios, not the absolute numbers, as the stable claim. The
per-element STARK is a **reference implementation, not a product path** — the
fingerprint statement is what changes that, at O(m+n) instead of O(mnk), which
is why the sumcheck prover sits on the critical path. A full model is **not
viable today**; state of the art is ≤1B parameters with dedicated teams.

## Requirements

- Zig `0.16.0` (stable; `0.16.0-dev.2535+` also works)
- Python 3.8+ with `blake3` (`pip install blake3`) — only for the independent
  audit tool and `zig build verify`
- [zig-algebra](https://github.com/samooth/zig-algebra) (L0) and
  [zig-zk](https://github.com/samooth/zig-zk) (L1) — not needed until F2
  (F0/F1 are self-contained in `libs/`); see docs/BLUE_PRINT.md §12
- Linux x86_64 (primary)
- Per-adapter extras: a **built** llama.cpp checkout + CMake, vLLM + ctypes,
  and a **built** ktransformers-zig checkout — see `adapters/`

## Documentation

- [docs/BLUE_PRINT.md](docs/BLUE_PRINT.md) — authoritative technical design (Spanish)
- [docs/PLAN_MULTI_ENGINE.md](docs/PLAN_MULTI_ENGINE.md) — adapter matrix, staging,
  llama.cpp wrapper pros/cons (Spanish)
- [docs/soundness.md](docs/soundness.md) — F2 soundness design notes and the
  measurements behind the choices (Spanish)
- [docs/decisions/](docs/decisions/) — decision log (ADRs)

## License

Apache-2.0
