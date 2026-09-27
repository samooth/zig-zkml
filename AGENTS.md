# AGENTS.md

Working rules for agents in this repository. What the system is and how it is
built lives in `README.md` and `docs/`; this file is about how to work here.

Language: English, as in `zig-zk`'s and `zig-algebra`'s. The specification in
`docs/` is Spanish on purpose — it was written that way and translating it is
an open decision, not something to do in passing.

## The first rule, because it has caught this repository five times

**Nothing counts as exercised until a test calls it, and the test is what
calls it.** Not `refAllDecls`, not the address of a declaration, not an
`installArtifact`, not a `&&` that never runs. A test.

Every instance found in this repository, for reference:

| Where | What | Why it did not fail |
|---|---|---|
| `zig-zk` `libs/air/` | five constructors | "instantiated by nothing" — its own CHANGELOG |
| `zig-zk` `stark/binius/` | a re-exported surface | duck-typed, and the prover never asked for it |
| `zig-algebra` `rng/src/main.zig:44` | an assert that survived to 0.5.0 | built as an executable, never type-checked as a test root |
| `zkml` `float_air.zig` `diagnose` | public, wired, no caller | the early version had zero callers and zero tests |
| `zkml` `zkml.zig` refs | `&gadgets.nonlin.SiLULookup.airFragment` | the **worst** one: it compiles, answers a grep, and executes nothing |

That last one is the dangerous shape, because it makes unexercised code look
exercised. A reviewer searching for the reference finds it and concludes the
gadget is covered. `&f` is analysed for its type — so a type error is caught —
but never called, so no behaviour is. Conflating those two is the error.

### And the test must name the exact failure

A test that asserts "accept or reject" is weaker than one that asserts *which*
error. `UnsupportedCase` versus `SubnormalInput` is the difference between an
hour of debugging and a line of text — and a test that pinned the generic error
passed just as happily while being wrong, which is debt wearing a test's
clothes. If two failure modes exist, they are two names and two assertions.

The corollary: a guard must agree with the thing it guards, and that agreement
is a test. A guard that disagrees is worse than no guard, because it promises a
name the caller never gets.

## Rules of use

From `.private/TODO.md` §0, which is the working contract. The plan itself is
internal and not versioned; these rules are.

### Measuring, not calculating

**Every number that enters this repository comes out of a command or a test
that asserts it.** Costs are not recomputed by hand: doing so has already
produced two regressions (108 vs 109, 72 vs 73/70).

The same rule extends to claims. A statement in a document is a claim, and it
holds or it does not; the audit that caught four of them in `README.md` is the
model. Two corollaries that have earned their place:

- **A number needs a gate, or it does not go in the document.** The test count
  is in `README.md` because CI checks it. It is *not* in `docs/soundness.md`,
  because nothing checks that, and a copy nobody verifies goes stale silently —
  which is worse than no number at all.
- **Absolute timings move with machine load.** The stable claim is the ratio,
  not the µs/MAC. Quote the ratio.

### Statuses

- `HECHO` — implemented and covered by a reproducible gate.
- `CERRADO-POR-MEDICIÓN` — not built; a measurement shows it is the wrong path
  or that it does not fit.
- `ABIERTO` — work outstanding.
- `DECISIÓN` — an alternative has to be chosen before implementing.
- `CONDICIONAL` — only if the product or a measurement demands it.

The status vocabulary is the contract. A change that lands `HECHO` without a
gate is not `HECHO`.

## Git

- **The remote belongs to the person.** Prepare the work, verify it, and hand
  over the exact command. Do not publish, do not force, do not delete refs, do
  not open PRs.
- Commits and tags are always signed. `git log --format='%G?'` must show a
  valid signature on every line. If signing fails, the commit does not happen:
  an unsigned commit in a repository of signed ones is a supply-chain
  regression, and "it worked locally" is not an excuse. `gpgconf --kill all`
  fixes a wedged agent.
- Lowercase messages, Conventional Commits. The body explains **why**, not
  what.
- One topic per commit. Never leave a gate red for the next one to find.
- **Do not rebase or force-push `main` while another person is working.** A
  signed history rewrite removes commits for everyone: the old SHA stops
  existing, and only GitHub's archive cache still serves it. Tags survive that;
  a commit SHA does not. That is why this repository pins by tag.
- If you rewrite published history, tell whoever already cloned it how to get
  back in sync.

## Ownership and coordination

There are two work lines, because the float soundness work cannot be split
without a safety net.

- **Main line, one person:** `libs/stark/float_air.zig`, `float_air_test.zig`,
  `air_builder.zig`, `barrel.zig`, `barrel_test.zig`, `float_ref.zig`,
  `fp16_ref.zig`, and `libs/stark/root.zig` where it guards
  `max_composed_constraints`.
- **Parallel line, other person:** `quant_binding.zig`, `quant_test.zig`,
  `widen_air.zig`, `widen_air_test.zig`, `libs/api.zig`, `include/zkml_c.h`,
  `tools/verify_weights.py`.
- **Cross-cutting:** `ZKML_ABI_VERSION`; do not change it without saying so.
- Own named branch, do not reformat someone else's files.

## Toolchain

**Zig 0.16.0 stable.** The system `zig` may be `0.16.0-dev` and is not the
reference toolchain — it formats and type-checks differently, and a build that
passes on the dev build has not been verified. Use one of:

```sh
/home/t0m4s/.zvm/0.16.0/zig          # release
/home/t0m4s/bin/zig-0.16.0/zig       # fallback
```

The second path is a **dev build** (`0.16.0-dev.2535`), not a release, and it
happens to pass the suite today. That is not the same as being the reference:
it formats and type-checks differently, and CI installs the official release.
Do not treat a green run on it as a green run on the release.

Verify with `zig version` before trusting a result, and check `zig build fmt`'s
**exit code**: it prints the offending file and still returns 0 in some
invocations, so a `&&` does not see the failure.

## Gates

```sh
zig build fmt
zig build test --summary all
zig build -Doptimize=ReleaseFast test --summary all
zig build abi
zig build verify
zig build spike
zig build vllm-adapter
zig build bench-fingerprint
```

`llama-adapter` and `kt-adapter` need sibling checkouts and are local gates only;
CI cannot run them. The count in the summary line is the count CI asserts.

## Soundness rules that are not negotiable

From `.private/TODO.md` §8. These are the ones that have already cost a bug.

1. One rounding. A second step revives the 6,459,545-pair bug.
2. The reduction pin is relative to the exponent: `r·(r − (1−E₀)) = 0`.
   `r·(1−sub) = 0` does not close.
3. `sub` binds to the shift, not the output exponent.
4. `overflow·(1−not_sub) = 0` is mandatory.
5. `promote` is redundant — the `carry` already says it.
6. Input classification must be exhaustive. That is what makes an arbitrary
   input unsatisfiable, including a subnormal.
7. OR is an inequality and this IR has only equalities: use `sum·sum⁻¹ = or`.
8. A reference only exercised where the prover rejects is not a reference.
9. `max_composed_constraints` is a verifier resource limit, never a soundness
   parameter and never a prover input.

## Dependencies

Pin by tag, not by commit SHA, and consume only what is used. An unused
declared dependency is a supply-chain surface with no offsetting benefit — and
a stale one is worse, because the URL keeps resolving through GitHub's archive
cache after the commit it names is gone.
