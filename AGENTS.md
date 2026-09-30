# AGENTS.md

Working rules for agents in this repository. What the system is and how it is
built lives in `README.md` and `docs/`; this file is about how to work here.

Language: English, as in `zig-zk`'s and `zig-algebra`'s. The specification in
`docs/` is Spanish on purpose — it was written that way and translating it is
an open decision, not something to do in passing.

Two documents are in English on purpose, and the exception is declared here so
it can be checked rather than remembered: `docs/asserts.md` and
`docs/decisions/ADR-0002-fingerprint-rank-one.md`. Both mirror code
identifiers and a measurement table verbatim, and a translated copy of a
machine-checked list is a second thing to keep in sync for no reader. Anything
else under `docs/` is Spanish. The test
`every prose document is in the language AGENTS.md says it is` fails if an
exception is added here without listing it, so the list cannot grow quietly.

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

## Prose is audited like code

A claim in a document is a claim. The audit that caught five of them in
`README.md` is the model, and the failures were never wrong facts so much as
facts in the wrong place.

**Prominence has to match the audience, and the payload belongs next to the
trigger — not in the lede.** Three instances of the same shape, all of which
happened here:

| What was prominent | Where the decision is actually made |
|---|---|
| the name "Goldilocks", in 38 files | the M31-vs-Goldilocks distinction, nowhere |
| the *false* reason for the FRI fork, in a code comment | the true reason, nowhere |
| 24 lines of `expf` approximation detail, at line 11 | F3/F4 are scoped at line 184 |

The README lede used to open with the native-path verification argument —
twenty-four lines, before the reader knew what the library did. It is now three
lines that state the claim and point at [F2](README.md#f2--stark-backend), where
the detail now sits, directly above the Roadmap row that depends on it. That is
not deferral: deferred is not invisible, it is positioned. The risk of moving
something down is only that it lands where nobody looks.

Deferring it two sections is not the same as deferring it twelve.

**A test that pins the specific error beats one that pins accept-or-reject**, for
the same reason: specificity is what makes a failure diagnosable. See
[UnsupportedCase → SubnormalInput](libs/stark/float_ref.zig) below.

### The instrument has to exist before you look at the answer

**An oracle written after reading the implementation detects nothing.** It
encodes what the code already does, so agreement is guaranteed and the
measurement carries no information.

This is the mutation half of the rule above, and it is **prose** — there is no
test for it, and pretending otherwise is the mistake the test above exists to
catch. The three from one day: an `eql` check that compared `A.eql(A)` with
`TA.eql(TA)` and so compared yes to yes; a `grep` of a field's criterion that
measured the base field and labelled the output as the extension's; and a
`print` announcing that a find-and-replace had run, with no `assert` that the
pattern had matched. The first two are the object inside the data, covered
above. The third is a number reported without being measured, which is the
same shape one level down.

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

**Six run in CI**, and a report that lists gates lists these six:

```sh
zig build fmt
zig build test --summary all
zig build -Doptimize=ReleaseFast test --summary all
zig build abi
zig build verify
zig build spike
zig build vllm-adapter
```

**Seventh, and it is local only:**

```sh
zig build bench-fingerprint
```

`bench-fingerprint` is a real gate and it does run here, but `.github/workflows/ci.yml`
never calls it — `grep -c bench-fingerprint .github/workflows/ci.yml` is 0. It is
listed separately precisely because a report that says "the seven gates" against
a CI that runs six is a false claim, and the ratios it produces
(`~36× / ~168× / ~309× / ~834×`, in the README) have this as their only door. A
door that nothing executes is not a door.

`llama-adapter` and `kt-adapter` need sibling checkouts and are local gates
only; CI cannot run them. They do pass on a machine that has the siblings, so
"CI cannot run them" is not "they are broken".

The count in the summary line is the count CI asserts.

**Report the same seven every time.** When a status update lists the gates,
it lists all seven or none. A summary that says "the five gates pass" when
there are seven is the same genre of error as a stale test count in a
document: a number that is not derived from the list drifts.

## Reports carry their validity condition

**Every report states `valid at <commit-sha>`.** A report is a photograph, and
a photograph goes stale silently.

**Any claim about another repository is re-verified with a command before
acting on it.** This is not caution, it is a measured failure: three claims in
one session that `zig-algebra` had no `v0.5.1` were all written before it was
published, and were re-emitted verbatim afterwards instead of regenerated. The
local half did not change because nobody worked on it; the cross-repo half
changed because the other repository moved. That asymmetry is the whole cause.

The same shape as the test count: a copy of a checkable fact elsewhere
desynchronises, and the only cure is that nobody reads it without refreshing it.
That is why the test count left the README.

### Every localised claim carries its subject

A report's `valid at` is a condition; this is the same requirement applied to
each individual claim inside it. **Every localised claim — a number, a line, a
file — names the revision it was checked against, or it has not been checked.**
A citation without its SHA is a postal address, not a citation: it points at a
possible place.

Two from one day, and they are the same defect in both directions:

- `docs/asserts.md` said `the 26` in the title and `total 20` in the body of
  the same page. True at one revision, false at the next.
- A report cited `AGENTS.md:171` for a defect. At `origin/main` line 171 was
  `zig build bench-fingerprint`; after a commit added 46 lines above it, line
  171 was something else. **The citation was right and expired when I wrote
  over it** — which is the same failure, because a claim without its subject is
  not a claim.

What expires is not the figure, it is the revision it was checked against. That
is why the SHA goes in the citation and not in the conclusion.

The other half is the *object* being wrong rather than missing: an oracle that
compares `A.eql(A)` against `TA.eql(TA)` is well formed and passes forever, and
a measurement of the base field's criterion labelled as the extension's is a
true number about the wrong thing. Both are "the object goes inside the data".

The operational form of all three is the same: mutate the thing on purpose and
confirm the instrument notices. Every number in this repository that claims to
be measured has a mutation behind it, or it does not get to say it is.

**These are prose, and this section says so deliberately.** There is no test
that can check whether a citation carries its SHA — that would mean parsing
prose, and a prose parser that passes is worse than none. That is the same
argument as the language policy, and it is the reason a written rule is not a
gate.

### A pin bump changes a signature — re-read the errors

The exit code is not the gate. Concretely, `zig_algebra` v0.5.1 turned
`Domain.init`'s `std.debug.assert` into a returned error (its own reason: the
assert is compiled out in `ReleaseFast`, where `two_adicity - log_n` then
underflowed). Six of the seven gates still returned 0, and the one that failed
did so in a line I had walked past. The fix was one line and the commit was
amended, but **why it needed amending lived only in the reporting** — the
remote would have seen a green commit and no red one underneath.

So: after a dependency bump, re-read the compile output, not just the exit
codes. An amended commit is green by the time anyone sees it.

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
