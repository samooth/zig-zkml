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
  is in neither `README.md` nor `docs/soundness.md`, because CI does not check
  it: the gate is a **floor** (`2[0-9]{2}/2[0-9]{2} tests passed`), not the
  exact count, and a floor cannot justify printing the number it produces. A
  copy nobody verifies goes stale silently — which is worse than no number at
  all, and no number at all is what this repository decided to print.
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

## Skills

**Load the skill before writing the code, not after it fails.** Three are
available and this repository is exactly what they are for:

| skill | load it when |
|---|---|
| `zig` | any Zig is being written or read — language semantics, stdlib, allocators, `build.zig` |
| `zig-multios` | anything touches more than one target, or `build.zig` cross-compilation |
| `zkp` | the task is ZKP-shaped; it is a **router** that picks the specialist |

`zig-multios` covers Windows, Linux, macOS, Android and WASM, and it excludes
itself for single-OS projects. This one is not single-OS: it builds for the
host and for `wasm32-freestanding`, and the arithmetic in `libs/fri` and
`libs/field` has to be correct at 32-bit and 64-bit `usize` simultaneously.
Those are the targets that exist *here* — `wasi` appears in no `build.zig`, no
workflow and no tool, so a rule written about it would be a conclusion carrying
a tree it was not measured in.

**The Zig skill exists because 0.16 differs from what most of us remember**,
and that is not a hypothetical: it documents itself as covering the `std.Io`
interface, allocators and cross-compilation for 0.16.x and 0.17.x. Writing the
consumer shim without it cost six compile iterations, each on pure type
inference rather than on anything about consumers:

1. bare `null` does not infer to `*anyopaque`
2. address zero is rejected for `*anyopaque`
3. an optional return needs `.?` before it can be passed on
4. an `i32` result must be discarded, not left bare
5. a comptime `0` does not infer to `?[*]u8`
6. `?*usize` and `?[*]u8` are different types and one is not the other

Every one of those is a rule about how 0.16 infers, which is the category the
skill covers. None was a design problem; all six were avoidable, and the list is
checkable against the session rather than taken on trust.

The ZK router earns its place on the decisions rather than on the syntax.
Choosing between our own `libs/fri` and `zig-algebra`'s, or whether a torus of
order `2^61` is the right shape, are questions about FRI, soundness and
transcripts — not about Zig. `zkp` routes to `zkp-theory` (FRI, STARK, AIR,
sumcheck, soundness error terms) and `zkp-vm` (the other proving stacks, for the
"would a zkVM be better" question that `Roadmap` implies). Load the router, then
the one it names.

Scope it honestly: **not all three, always.** `zkp` on a formatting fix is
noise, and `zig-multios` on a one-line typo in a comment is worse. The rule is
that the skill for the thing you are about to write is loaded first, and that
choosing not to load one is a decision you can say out loud rather than a
default.

This is prose and it is **not** a gate. There is no test that can check whether
an agent loaded a skill before typing — that would mean instrumenting the
session, and the instrument would cost more than the rule. It is here because a
written rule is still the contract, exactly as the language policy is.

## Gates

**Six run in CI as blocking gates**, and a report that lists gates lists these
six. `zig build test -Doptimize=ReleaseFast` is the seventh and is run locally
with every change; the CI job runs the debug one.

```sh
zig build fmt
zig build test --summary all
zig build -Doptimize=ReleaseFast test --summary all
zig build abi
zig build verify
zig build spike
zig build vllm-adapter
```

**Eight more run here, and seven of them in CI as `continue-on-error` jobs** —
the weekly `wasm-portability` and `differentials` jobs. They are non-blocking on
purpose, and being non-blocking is exactly why they are written down here:

```sh
zig build bench-fingerprint   # local: shells out to a full second build
zig build wasm-consumer       # local + weekly CI, non-blocking
zig build wasm-test-sweep     # local + weekly CI, non-blocking
zig build doc-paths           # local + weekly CI, non-blocking
zig build masking-mutation    # local + weekly CI, non-blocking
zig build field-diff          # local + weekly CI, non-blocking
zig build domain-cost         # local + weekly CI, non-blocking
zig build fri-conv-diff       # local + weekly CI, non-blocking
zig build fri-diff            # local + weekly CI, non-blocking
```

### The three differentials are gates now, and their reach is measured

They were scripts with no entry in `build.zig` and no job, so their output
lived in a commit message and nobody could repeat it. Between them they
decided the two largest questions here — 176 lines of field and 828 of FRI — so
a measurement that decides and that nobody repeats is a claim with a manual
oracle.

| step | checks | what it decides |
|---|---|---|
| `field-diff` | 17,344 | `libs/field.zig` against the pinned M61 |
| `fri-conv-diff` | 484 | `Fp2 ↔ Torus61` identity, byte layout, canonicality guard |
| `fri-diff` | 108 | our FRI composition against the pin's, over 7 parameter points |

**A gate that has never been shown to fail is not a gate**, so each carries
named mutations that abort if the pattern did not land:

| mutation | discrepancies |
|---|---|
| `add-sin-reducir` | 1,587 |
| `sub-invertido` | 4,006 |
| `eql-reflexivo` | 4,006 |
| `mul-shift-60` | 3,848 |
| `inv-p-menos-3` | 58 |
| `neg-mas-uno` | 61 |
| `toBytes-coords-intercambiadas` | 3 |
| `fromBytes-sin-canonico` | 1 |
| `fri-diff`, one round less, **either side** | 10 |

**Two of those rows were zero until the mutations existed**, and both were in
`fri-conv-diff`, which is the instrument that took the conversion decision:

- Its byte-layout check compared **the pin against itself** — `Q.new(c0,c1)`
  against two separately computed torus values — and never called our own
  serializer, which is the one that would enter a transcript. Swapping `a` and
  `b` in `Fp2.toBytes` produced **0 discrepancies across 476 checks**.
- The canonicality guard in `fromBytes` was never exercised. Weakening `rb >= p`
  to `rb > p` also produced **0**. The code's own comment says the guard exists
  so rejection sampling is uniform: a documented property, not a verified one.

Both are now checked, and the canonicality cases include `b == p - 1`, which
must be *accepted* — otherwise the guard would be rejecting too much and the
weakened version would pass as good.

`tools/pin_dir.sh` unpacks the pin from Zig's global cache using the hash in
`build.zig.zon`. Nothing created `zig-pkg/` before: the three scripts demanded a
directory that only existed on one machine because somebody had unpacked it by
hand, and all three failed with "falta el pin" on a clean checkout.

### The portability gate is two gates, and the split is the point

`zig build` runs for one target, so "this compiles" had no destination in it.
The first attempt supplied one by pinning every file that failed under
`zig build test -Dtarget=wasm32-freestanding`. That list was not a gate about
portability, it was an inventory wearing one, and it mixed three unrelated
things: stdlib internals, repository code, and **the test harness itself**,
which takes `std.process.Init` and spawns threads and can never compile
freestanding on any platform for any reason connected to this library. The
harness in the list meant every future fix moved the list for an unrelated
reason, and a gate that cries wolf is worse than no gate.

So the question is now asked of a consumer, and the inventory kept its honest
name:

- **`zig build wasm-consumer`** compiles `tools/wasm_consumer_shim.zig`, which
  imports the public C ABI (`libs/api.zig`) and nothing else. That is
  literally what an embedder does, so the answer is a yes or a no. **It
  passes.**
- **`zig build wasm-test-sweep`** is the inventory, against
  `tools/wasm_test_sweep_expected.txt`, renamed because a file called
  `wasm_expected.txt` that lists `posix.zig` lies about its own reach. Data
  lines are bare paths; the file may carry `#` comments saying what each entry
  is and why, because a list that cannot explain itself is half a list of
  tickets. Five entries: four std/harness, and `libs/fri/root.zig`.

**The shim had to be shown failing before its green meant anything.** Its first
version held `&api.zkml_witness_session_create` and friends. Taking a
function's address analyses its *signature*, not its *body* — and the body is
where `std.heap.smp_allocator` lives. So it compiled cleanly against
freestanding while the one defect it exists to catch sat in the file it was
supposed to be reading. That is the `&f` shape, committed in the gate built to
prevent it. The shim now *calls* the entry points from an `export fn`, whose
body is always analysed, and the mutation behind it is measured: reverting
`process_allocator` to `smp_allocator` turns the step red on
`std/Thread.zig:493: Unsupported operating system freestanding`.

That one line — `std.heap.smp_allocator`, which needs threads and does not
exist without them — had blocked the web demo since the 27th, through twelve
commits of tests and documentation that never touched it. The fix asks
`builtin.single_threaded` rather than inferring from the OS. **A gate is only
worth its green if you have watched it go red on the defect it names.**

The `u5`/`u6` sites are a different matter and are mostly done: `@as(usize, 1)
<< log_n` needs `u5` of shift on wasm32 and `u6` on native, and `@intCast` on
the shift amount infers whichever the destination needs, so the destination
does not enter the library. `prove` and `stark` are done; `libs/torus/domain.zig`
is done; **the two that remain in `libs/fri/root.zig` wait on the architecture
decision**, so the sweep list cannot reach zero until that is settled — by
either two `@intCast`s or deleting the file.

`doc-paths` and `masking-mutation` run in the weekly `differentials` job and not
in the blocking six. `bench-fingerprint` is a real gate and it does run here,
but `.github/workflows/ci.yml`
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

### Uncommitted on purpose: `tools/fri_diff.{zig,sh}`

They are **deliberately uncommitted**, and that is a decision, not an oversight.
They are the differential that turns the `libs/fri` decision — 614 lines, own
FRI versus the `zig-algebra` pin — from an opinion into a measurement.

They were uncommitted because they did not run: on `zig-algebra` v0.5.2 the
pin's two domain paths never terminate at `log_n = 8`, so the harness hung.
**A harness that hangs is not a result, and committing one would be the inert
door this file keeps warning about.** Both bounds were fixed upstream in
v0.6.0 (`extension.zig:31 MAX_NON_RESIDUE_SEARCH = 1024`,
`torus.zig:181 max_candidates = 1 << 20`) and it now runs — see the ledger
below for the result and for what it does and does not cover.

Do not `git clean` them away. They are the instrument that decides it, and a
deleted instrument converts a measurement back into a judgement call.

### A conclusion travels; the tree it came from does not

**Before carrying a conclusion from one repository to another, check that the
file, the path and the revision exist in the destination.** A `file:line` is
not a location — it is a position in a tree, and it only means anything next
to the tree it was read from.

The ninth time this bit was distinct from the other eight. In those, an
instrument was asked the wrong question and returned a coherent zero. This
one the data belonged to one repository and the question to another: the
*conclusion* travels, and the tree it was measured in does not. `FieldTooSmall`
in three different repositories, a `stark.zig` that exists in none of them, and
`B.2.12` as an appendix — none of those were checked against the tree they
were being asserted about, and all of them read as findings.

This is the rule that makes the rest a system rather than nine anecdotes. It
costs one `ls` and it is the cheapest thing in this file.

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

**And the sister rule: normalise before you compare, compare the same object
on both sides, and name each outcome for what it means.** Both halves are
failures of the report rather than of the check, and both were found today in
the same gate.

*The order half.* A gate listing nine files failed while both lists held the
same nine: one side was sorted before the `/home/.../lib/std/` prefix was
stripped and the other after, and stripping it changes the collation of the
capitalised names. Two legitimate counts, different numbers, and no assertion
on the result sees it — it is the 8-bytes-against-16 shape again. The failure
is not *what* you compare but *after what* you compare it, which is why no
grep finds it.

*The naming half, which is the worse of the two.* The same gate's two output
sections were computed correctly and then labelled wrongly: a file that was
**still failing** and had merely been dropped from the expected list was
headed *"nuevo, o arreglado"*. Neither. And that label invites regenerating the
list without looking, which is moving the figure so the door stops complaining
— the silent version of everything above. **A label that misdescribes its own
output changes what you do with it, and that is worse than a broken gate,
because a broken gate is visible.**

Which is why the list is a contract: a file leaves `tools/wasm_test_sweep_expected.txt`
in the same commit that fixes it, and that commit says which file left and
why. A list that goes from nine to eight with no explanation is a claim that
moved on its own. The rule exists so that "move the list" stops being a
judgement call and becomes a procedure.

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

## Work that is stopped, and why

`docs/PLAN_PRACTICAL_VALUE.md` §1. These are not style preferences. Each is a
measurement that was taken, understood, and found to describe a statement the
project no longer has — so re-taking it produces a number about the wrong
thing, which is the same error as citing the right file and inferring the
content.

**1.1 · Do not re-run the per-MAC benchmarks.** `zig build bench` drives
`bench/gemm_bench.zig`, which measures 1 MAC/row against 16 MACs/row. That
granularity is not the statement: the statement is a whole model run, and the
fingerprint identity changes it to O(m+n) instead of O(m·n·k). The µs/MAC figures
stay in the README as **historical** and are not to be refreshed.
`zig build bench-fingerprint` is a different instrument — the fingerprint claim
against the oracle — and this rule does not cover it.

**1.2 · Do not start optimising the 10-day path.** Its cost is in the statement,
not in the code: 5.91 billion STARK constraints, plus FFT, RLC, quotient, FRI and
openings over the whole trace. There is nothing to optimise until the statement
changes, and an optimisation applied to the per-element path is an optimisation
of something the project has already decided not to ship.

**1.3 · Do not widen the trace to group MACs.** Grouping 16 per row buys ~2.6×
proving time and pays ~2.9× verification and ~2.6× proof size. The README
already measures the trade and `libs/stark/gemm_chunk.zig` implements the worse
side of it; the point is that the remaining headroom is not worth spending
there.

If you think one of these is wrong, say so in the commit message and change the
plan's box — do not quietly re-run the benchmark and update the number. That is
the failure this section exists to prevent: a figure that moves because someone
re-measured it, with no statement having changed.

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

### `patch-dep` on `libs/api.zig` fails on purpose, and the failure is the point

A consumer working around the freestanding allocator without touching this
library ended up applying the fix through `zig patch --dep`, which rewrites
`libs/api.zig` **textually** and refuses to proceed if the source does not have
the shape it expects — it fails with `UnrecognisedSource`. That is correct
behaviour and it is the class of detector this repository builds on its own
terms: a tool that refuses to no-op. A patcher that silently did nothing would
be worse than one that stops.

So expect it to break, and do not "fix" it by relaxing the matcher. Someone
will reformat that file, the patcher will refuse, and the error only makes
sense if you already know the refusal is intended. The same reasoning applies
to the allocator itself: it now asks `builtin.single_threaded`, so if someone
"simplifies" it back to a bare `smp_allocator`, `zig build wasm-consumer` goes
red on `std/Thread.zig:493` rather than waiting for an embedder to find out.
