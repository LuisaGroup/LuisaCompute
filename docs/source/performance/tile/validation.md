# Tile correctness and failure investigations

This record distinguishes executed correctness checks from performance claims.
See [current status](index.md) for the latest bounded conclusion.

## Expression producers join their first reduction traversal

The September 9 expression-fusion checkpoint uses `2634be45d` (including
`next@8911828eb`) plus an explicit eight-file C++ overlay in an isolated,
recursively pinned source export. It passes a full configured build,
**35 Tile CTests** and **79 XIR/SIMD CTests**. Additional exact-name tests at
W1/W2/W4/W8/W16 execute nonzero assertions and pass with expression fusion
alone and combined with the existing load/pointwise options. Snapshot
retention/elision, alias writes, strict folds, stage boundaries, zero-trip
and repeated scopes, fill, tails and permuted dimensions are covered.

The first width-test wildcard selected no cases; those empty successes are
excluded. An initial exact-name audit also incorrectly required a per-case
PASSED banner from a reporter that prints only the aggregate. The final
receipts require exactly one executed case and nonzero assertions; both
earlier invocation/audit mistakes remain in the evidence.

The [shared planner/lowering rule](../../internals/tile/xir.md#first-consumer-fusion-preserves-the-snapshot-contract)
fuses two producers in each of four masked-softmax cases and one in each of
four LayerNorm cases. The other 16 cases have byte-identical LLVM and ORC
objects. All 48 capture outputs and 72 native smoke outputs pass complete
FP64 checks; all 24 off/on pairs are bitwise equal. An independent audit
rereads 138 snapshots, including 18 from an excluded partial timing run,
and rejects eight in-memory corruptions of output or evidence records.

**No new performance claim is accepted.** Schedule blocks fall, but static
clone instruction counts rise and retained workspace sizes do not fall.
Background CPU activity survives the named-process preflight; foreign
rendering subsequently resumes. The owned benchmark is stopped, and all
96 visits from that partial timing cohort are excluded. No second cohort,
default change, cost calibration or automatic fusion selection is claimed.
The {download}`Chinese checkpoint report <../../../../scripts/benchmark/tile_torch/results/m1-max-20260909-expression-reduction/notes.md>`
records the scope, tradeoffs, exact source/binary lineage and remaining work.

## next integration keeps performance qualification separate

The later September 9 checkpoint includes `origin/next@03a0f5158`, with merge
commit `bc7b1df1f`. A fresh, no-overlay export pins 19 repositories and passes
a full configured build, **80 XIR/SIMD CTests**, **35 Tile CTests** and the
complete **Metal codegen fixture**. The shared XIR fixes cover loop-epoch exit
dispatches and ordered callable-swizzle copy-out. The only merge conflict is
a test comment; the argument guard remains intact. Original unfinished
worktree files and dependency checkouts are preserved. The
{download}`Chinese follow-up record <../../../../scripts/benchmark/tile_torch/results/m1-max-20260909-next-integration/xir-followup/notes.md>`
keeps the rejected empty CTest selection and successful registered-name rerun
separate. No native performance objects are recaptured, and no speedup,
default-policy change or cost calibration is claimed from this integration.
The new upstream investigation is owned by the existing
[validation archive](../validation.md), not a parallel documentation tree.

The earlier September 9 merge includes `origin/next` through `8911828eb`, with merge
commit `360d9791e`. A fresh source export includes the merge and recursively
pinned submodule commits: 19 repositories and 18,830 fingerprinted base files.
One subsequent, fingerprinted test-only overlay fixes the Metal fixture's
argument guard; no compiler, library or Tile source is changed by that fix.
The original worktree's 11 modified files and all dependency checkouts remain
unchanged; the isolated build excludes the unfinished matrix experiment.
The selected SIMD + Metal + TIRx configuration, with LLVM 21.1.8 and Metal4
disabled, passes a full build, **all 35 Tile CTests**, **79 XIR/SIMD CTests**
and both selected Metal regressions. The Runtime fusion A/B suite and the LLM
suite also pass with the recorded packet/pointwise configurations.
External TVM libraries are reused and fingerprinted, not rebuilt.

The upstream Metal test constructed `string_view(argv[2])` without a third
argument: Boost.UT's overloaded logical operator defeated the intended
short-circuit. Splitting the argument guard fixes the test entry; both the
complete and local-only fixture pass. Earlier Runtime metadata failures came
from the test supervisor forcing fusion on/off while fixtures explicitly
test both policies, not from a numerical failure. The corrected invocation
leaves those policy switches to the fixtures. All failed invocations remain
in the record. Changed C++ lines pass formatting and the translation unit
passes syntax checks; seven inherited full-file formatting diagnostics and
one unused-include warning remain, so this is not a warning-free claim.
The first strict documentation build also rejects 13 newly merged, unowned
validation pages. They now have one owner under the existing
[performance and validation archive](../validation.md), retaining their
original paths and content rather than weakening the ownership check.

The separate, premerge pointwise experiment completes two fixed 24-case
native replays. Their 864 timed visits check complete outputs, input
immutability and guards; an independent audit rereads 144 output snapshots
and rejects eight in-memory evidence/output mutations. Both attempts observe
concurrent foreign rendering, including the retry after a quiet preflight.
**Both complete timing cohorts are excluded** from performance rankings,
default selection and model calibration. No apparently quiet subset is
promoted. Five-second process observations do not prove exclusive hardware
use or a stable clock frequency.

The {download}`integration and exclusion record <../../../../scripts/benchmark/tile_torch/results/m1-max-20260909-next-integration/notes.md>`
separates the new source/build from the premerge native objects, retains failed
preparation and raw runs, and explains privacy-minimized coactivity records.
Correctness and source integration do not establish a new speedup or parity
result. Pointwise fusion remains opt-in, and its cost prior is not recalibrated.

## Per-operation reduction policy checkpoint

The September 8 checkpoint passes a **full selected build and all 35 Tile
CTests** (225.83 s), plus all 110 Python benchmark tests. The selected
configuration enables SIMD, Metal and TIRx, with Metal4 disabled. To avoid
mixing another unfinished matrix experiment into this result, the staged
source was exported to a separate source/build tree. Dependency submodules
retain their local checkout state; this is not an assertion that every
dependency or the original worktree is clean.

Coverage includes the default unordered tree, explicit ordered tree and both
fold directions, multidimensional/empty domains, non-identity and signed-zero
seeds, cancellation-sensitive FP32 data, mixed strict/relaxed reductions,
analysis invalidation and target capability rejection. TIRx CPU/Metal and
XIR/SIMD execute the fold fixtures. Tests also check the final LLVM/Metal
fast-math boundary and precise TVM Metal module serialization/reload. Closed
FP32 add/max/min Tiles use subgroup collectives inside composed programs;
ordered policies still retain their serial implementation.

The original worktree's separate full run passes **34/35**: the unfinished
matrix-initializer experiment changes four structural matrix-count
assertions in `test_tile_tirx_matrix_metal`. Those edits are excluded from
this checkpoint, not deleted or hidden by weakening tests. The earlier
memory/cooperative fence failures are repaired: without an applicable effect
proof, phase publication must cover device as well as threadgroup memory.
The {download}`raw build and regression logs <../../../../scripts/benchmark/tile_torch/results/m1-max-20260908-reduction-policy/notes.md>`
distinguish the two source states. No new performance claim follows from
these correctness results.

## Closed matrix epilogues: positive and fail-closed coverage

The September 7 extension exercises ordinary clamp, polynomial and tanh-GELU
graphs across aligned, ragged, tiny, transposed/offset output, masked, negative
origin, zero-K, nonzero carry, observed output and manual-memory cases. Two
non-dyadic input sets check complete outputs, including unchanged sentinels.
The proof-boundary tests retain fallback for unmarked/manual producers,
transposed reads, additional memory inputs and extra consumers. Typed MPP
element tests use a nonzero fragment ordinal, check read/write effects, and
reject invalid result/index/value types and fragment ordinals.

After a full build, 17/19 integration invocations pass, with **5,565 assertions
in the Metal matrix suite** and 14,411 in the planner suite. Both CPU/Metal
execution, basic/neural/algorithm PoCs, pipelines and native Runtime pass.
The same two existing Metal memory/cooperative suites still fail three
source-string assertions because the user-owned barrier flag is 2 rather
than the expected 3. That edit is present in both measured stacks but is not
part of the epilogue change. This is not an all-green worktree.

The [performance replay](results.md#closed-matrix-epilogues-general-legality-mixed-profitability)
passes all 288 complete native/Torch outputs. Ten Metal default-off controls
have byte-identical old/new source; four CPU controls differ only in bijective
TBAA object-address labels, preserving instructions, alias graph and offsets.
All 56 control outputs pass. These are compatibility checks, not CPU or
reduction speedups. The
{download}`receipts and corruption-tested auditor <../../../../scripts/benchmark/tile_torch/results/m1-max-20260907-fragment-epilogue/notes.md>`
retain original raw runs and test logs. The auditor rejects missing/duplicate
rows, changed artifacts, partial output receipts, altered tolerances, wrong
GPU divisors, worker retuning, unbalanced ordering and erased scalar cost.

## Bounded program traversal: structure and unchanged defaults

The September 7 [generic traversal emitter](../../internals/tile/planner.md#program-traversal-is-a-mapping-choice-not-a-memory-scope)
passes 480 independently enumerated grid/rectangle combinations, including
partial bands, plus invalid/overflowing input checks. Eight non-matrix Metal
executions combine batch axes, bounded three-tap input access, pipeline state
and optional local reduction. Twenty-four matrix programs each run two
non-dyadic input sets, spanning SIMD-group/MPP, staged/view inputs, transposes,
partial M/N/K, nonzero recurrence state and two program orders. Malformed
axis metadata and unsupported exact traversal requests reject before launch.

After a full build, **17/19 integration invocations pass**. The two retained
failures are existing Metal memory/cooperative source assertions requiring
`mem_flags(3)`, while the untouched user-owned barrier edit emits
`mem_flags(2)`. These are not silently waived or reported as a green suite.
All new layout/execution/matrix tests, both CPU/Metal basic/neural/algorithm
PoCs, pipeline tests and native Runtime checks pass. The
{download}`full receipt <../../../../scripts/benchmark/tile_torch/results/m1-max-20260906-program-order/correctness-v3/results.json>`
and {download}`syntax/Python receipt <../../../../scripts/benchmark/tile_torch/results/m1-max-20260906-program-order/syntax/results.json>`
retain the boundary: eleven changed C++ translation units pass syntax checks.
The subsequent {download}`Python receipt <../../../../scripts/benchmark/tile_torch/results/m1-max-20260906-program-order/python-final.json>`
passes all 102 tests, including exact traversal preservation in frozen replay.

A frozen pre-change executable/library set checks unchanged defaults across
three GEMMs plus add, paired GELU and softmax. Six Metal sources are
byte-identical; three CPU LLVM sources differ only in bijectively renamed
allocation-identity TBAA labels. Their instructions, alias graph and width/
offset suffixes are unchanged. All 36 native/Torch outputs pass complete
checks. The initial byte-only CPU comparison remains a recorded failure;
the {download}`independent source audit <../../../../scripts/benchmark/tile_torch/results/m1-max-20260906-program-order/defaults/audit.json>`
establishes the narrower equivalence without discarding metadata or changing
the raw source. These short runs are compatibility checks, not speed claims.

The earlier missing-header build failed before tests: the `/tmp` TVM checkout
had lost files and Git metadata. An isolated persistent-cache checkout at the
same pinned TVM/FFI revisions plus the four recorded MPP patches restores all
surviving source/header contents exactly. Existing compiler/runtime libraries
are unchanged; subsequent full builds and artifact fingerprints are retained.

## Multi-output pointwise ownership and effects

The September 6 extension passes the full selected build and ten native
suite/backend combinations: execution, matrix and the three PoC suites on
both CPU and Metal. The execution suite passes **803,488 Metal assertions**
and **82,357 CPU assertions**, in 36 tests per backend. These include existing
subgroup-reduction controls. Matrix tests retain 2,334/3,058 assertions on
Metal/CPU; the PoCs exercise losses, normalization, attention, convolution,
filters, sorting and Top-K. The
{download}`build and correctness receipts <../../../../scripts/benchmark/tile_torch/results/m1-max-20260906-element-multi-output/correctness/receipt.json>`
record exact commands and unchanged libraries. This is selected coverage,
not a whole-worktree all-green claim.

New tests check three outputs with an interleaved second shared producer,
ragged/negative input coordinates, nonzero loop minima, conditional stores,
and permuted output coordinates. Missing noalias, explicit worker bindings,
read-after-write, write-after-read, repeated writes to the same buffer and
different local domains retain the reference path. Full numerical checks
include all outputs and untouched sentinels. Nonconstant shared exponentials
must appear once in generated source, without thread-private Tile arrays.
The constant all-zero case instead verifies removal of the input parameter;
its first source-count assertion was overstrict and the failed log is retained.

There are 99 passing Python benchmark tests, including independent autograd
checks of the FP64 derivative formulas and corrupt-second-output rejection.
The six-round performance replay validates 192 full value/derivative pairs;
its independent audit rejects eight corrupted evidence variants. Old/new
controls validate another 64 native/Torch graph outputs. Four CPU raw source
hashes differ only in TBAA object-address labels: the separate audit preserves
the alias equivalence classes while renaming those labels, leaving all
instructions and other metadata unchanged. Neither those diagnostic control
timings nor passing broad PoCs are reported as new performance gains.

## Bounded-K MPP input-view checks

The September 6 bridge extension removes only a proved common zero-padded K
suffix from immutable A/B views. The full selected build passes the Metal
matrix suite: **1,857 assertions in 28 tests**. The same current binary also
passes with the frozen older TVM v2 libraries: **1,548 assertions in 28
registered tests**; optional bounded-K cases are skipped without the new
capability. Five related CPU/execution/pipeline/planner/target CTests and all
95 Python benchmark tests pass. These are selected tests, not a new
whole-worktree all-green claim.

The focused extension checks 69 complete outputs: 32 positive configurations
run with two changed non-dyadic input sets; one two-stage pipeline
configuration runs twice; and three semantic counterexamples run once each.
All A/B transpose combinations, nonzero initial accumulators, K=7/61/1033/11008,
nominal BK=1024, and physical K strides shorter than BK are covered. The
FP64 oracle uses `atol=1e-4, rtol=2e-5`. Nonzero fill, an extra K mask and
unequal A/B effective K intervals retain materialization and their original
numerical results. Malformed actual-K types, zero/oversized lengths and known
invalid leading strides fail before launch. Existing alias/manual-memory,
M/N-tail and recurrence regressions remain in the full matrix suite.

The prior local barrier-flag change remains present in both tested old and
new binaries but is not part of this patch. The two older source assertions
described below still prevent a general all-green-worktree claim. The
[matrix reference](../../internals/tile/matrix.md#bounded-k-views-avoid-nominal-padding-storage)
documents the capability and fallback boundary; fixed-request performance
is reported separately from these regression checks.

## Larger-matrix benchmark coverage

The September 5–6 scale cohort adds 560 passing complete GEMM replay outputs
(11,022,491,732 checked elements), eight passing pilot outputs, and 72 passing
softmax/RMSNorm/LayerNorm outputs. The view schedule also has two rejected
pilot shapes and 28 replay admission rejections; the initial six missing-TVM-
capability probes are separate. These failures are retained, not numerical
successes or silent fallback paths. See [large GEMM](results.md#larger-matrices-the-1024-cubed-win-does-not-generalize)
and [wide reductions](reductions.md#wide-rows-and-large-working-sets).

The {download}`independent scale audit
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-large-matrices/audit.py>`
checks complete-output validation records, all four timing metrics, balanced
GEMM order, reversed reduction precedence, fixed source identities and exact
automatic reduction choices. It also checks emitted SIMD-group collectives.
GEMM's 26-artifact inventory and reductions' separate 22-artifact before/after
inventory are unchanged. Outputs were checked in full during execution but
transient arrays are not saved for independent recomputation.

The scale orchestration passes 93 Python tests. No C++ rebuild or CTest run is
added here; the prior 31/33 boundary below still applies. This experiment
tests deterministic FP32 inputs and large dimensions, not arbitrary data
distributions, low precision or end-to-end LLM correctness.

## Correctness: common LLM operators now use both bridges

`test_tile_xir_llm` runs **24 captured kernel/shape combinations**, each through
XIR/SIMD and native-target TIRx CPU, with `atol=rtol=5e-5` against independent
FP64 formulas. Every output element is checked. XIR outputs begin as NaNs and
use an offset BufferView with guards before/after the writable range.

| Family | Shapes / edge cases | Reference |
|---|---|---|
| RMSNorm, LayerNorm | 17 rows × widths 7, 32, 65; shared gamma/beta | FP64 mean/variance, epsilon and affine transform |
| SwiGLU, GELU+residual | Same three widths; non-dyadic inputs | FP64 sigmoid/tanh formulas with the same float coefficients |
| Masked softmax | Same widths; row-dependent nonempty mask | Stable FP64 masked exponential normalization |
| Split-half RoPE | 17 rows × widths 6, 32, 66 | FP64 rotation using identical supplied sine/cosine tables |
| Online attention | `(B,Hq,Hkv,Q,K,D,Dv)` = `(1,2,2,4,5,4,3)`, `(2,4,2,7,11,8,7)`, `(2,4,2,1,17,8,7)` | Full-score FP64 causal softmax and value contraction, independent of the online recurrence |

Attention queries represent the final Q positions of the KV sequence. Local
query/key tiles are 2×3 for the original cases; three additional captures use
1×1, 2×4 and 3×5 on `(1,2,1,7,11,8,7)`. Invalid zero-head and zero-block
requests are rejected. These cases exercise tails, causal masks, online
max/sum/accumulator carries and grouped query heads. These tests do not measure
KV-cache paging, variable-length batches, long contexts or production hidden
dimensions. CNN, traditional filters, Top-K and sort remain available in the
language/earlier TIRx gallery; they are **not newly validated on XIR** here.

The September 7 partitioned-output checkpoint passes a full build, all three
selected CTests (LLM XIR/TIRx-CPU, CPU execution, Metal execution), and then
**33/35 tests in the full Tile CTest rerun**, plus 110 Python benchmark tests.
The execution tests add 64 width/partition/policy
configurations: touching disjoint intervals, reversed order, overlapping and
identical writes, disabled fusion, explicit worker scope and absent noalias.
All output elements, including untouched regions, are checked. Metal's whole
execution suite passes 818,965 assertions. The two retained failures are the
previously reported unrelated barrier-source assertions in
`test_tile_tirx_cooperative_metal` and `test_tile_tirx_memory_metal`; the local
barrier edit and those assertions are unchanged. This is not an all-green
worktree.

An initial expanded 8×16 attention-block unit case was stopped after 186.49 s;
the final unit suite uses bounded SSA sizes. Two SIMD decode benchmark attempts
separately hit their 90 s process limit. Both records remain in the
{download}`LLM investigation
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260907-llm-coverage/notes.md>`;
smaller passing tests do not certify large Tile compilation scalability.

Additional XIR tests cover transposed/ragged GEMM, nonzero accumulators, two
changed non-dyadic input sets, loop-carried swaps, zero-trip loops, view
offsets, read/write snapshots, move-only shader lifetime, negative origins and
signed-overflow rejection in the bounds proof. The dedicated SIMD PHI test
uses widths 1/2/4/8/16 and every active-lane count, independent of TileIR.

### Automatic cooperative program contracts

The next full build and CTest run again gives **33/35 Tile passes** (35/37
under `-R tile`, which also matches two passing sparse-resource tests). The
same two user-owned barrier source-string failures remain; no numerical
regression is reported in that run.

The new `tile_matrix_automatic_cooperative_semantics` test separately passes
17,189 assertions. It covers two singleton-axis placements, modularly permuted
program coordinates, nonzero accumulators, aliased same-instance C/D, disabled
cooperative admission, explicit worker constraints and missing MMA arithmetic permission.
Three attention block shapes also exercise ragged causal/GQA recurrence state.
These are runtime output tests, not a general alias proof.

The [six-order replay](results.md#automatic-cooperation-removes-the-attention-worker-fallback)
validates all 36 complete old/new/Torch outputs. Its independent auditor rejects
five types of corrupted evidence. The separate calculus reference has nine
finite tests including unsafe fusion and noncommutative reduction examples;
it does not invoke the production compiler. The
{download}`receipts and methods <../../../../scripts/benchmark/tile_torch/results/m1-max-20260907-cooperative-programs/notes.md>`
retain the worktree barrier distinction and exact artifact hashes.

## Metal reduction validation checkpoints

The later fixed-total-group benchmark adds 456 executed complete-output
validations, all passing. Its independent audit enumerates all 39 automatic
candidates for each of twelve cases and checks exact plans, source identities,
four timing metrics, balanced order and 21 unchanged binary artifacts. This
is new benchmark evidence, **not a new CTest run**. The
[two-baseline comparison](reductions.md#fixed-total-group-size-versus-automatic-execution)
keeps mixed results and regressions visible.

The latest cooperating-packing checkpoint completes a full build, all 89
Python benchmark tests and 31/33 Tile CTests. CPU/Metal execution and planner
tests pass, including 36 new numerical configurations and six typed raw-IR
admission/fence cases. Outputs are checked against independent FP64 formulas;
guard rows and unused output columns must retain sentinels after three
dispatches. The independent benchmark audit checks 240 executed output
validation records, frozen plans/sources and 21 unchanged binary artifacts.
The {download}`packing validation note and full CTest log
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cooperating-packing/notes.md>`
retain the same two unrelated source-assertion failures, not an all-green result.

### Earlier submitted-value and service-policy checkpoints

The submitted source preserves `metal::mem_flags(3)`. The earlier
{download}`submitted-value checkpoint <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-dual-timing-validation/notes.md>`
passed **33/33** `test_tile_*` entries: 30 unit-labeled tests and three
integration tests. The service-policy follow-up rebuilt
the full selected tree and reran all 33 entries without touching the user's
pre-existing local `mem_flags(2)` edit: **31/33 passed**. The two failures
are generated-source assertions requiring `3` in
`test_tile_tirx_cooperative_metal` and `test_tile_tirx_memory_metal`; their
numerical checks pass. Neither assertion was weakened and the local edit is
not submitted. At that checkpoint, benchmark Python contracts pass **89/89**; the planner
passes **5,988 assertions in ten tests**. The prior
24 ownership-layout cases, 14 wider/non-power-of-two layouts and 22 new
input-reuse numeric configurations also pass in the execution test. The
{download}`service-policy validation and full log <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-service-policy-validation/notes.md>`
keeps this dirty-worktree result separate from the earlier submitted-value
checkpoint. No whole-repository test pass is claimed.

### Earlier shared-Tile and resource checkpoints

The original shared-Tile run reported the following historical counts:

```text
complete CTest /^test_tile_/:            32 / 32 tests passed
guarded CPU view proof:               1,572 assertions passed
Metal subgroup LayerNorm:            12,297 assertions passed
Metal subgroup cross-entropy:            20 assertions passed
focused TIRx execution, CPU:          33,071 assertions passed
focused TIRx execution, Metal:        38,363 assertions passed
focused TIRx planner:                  5,891 assertions passed
Python benchmark contract discovery:    69 / 69 tests passed
```

That checkpoint temporarily tested the submitted memory-flag value and then
restored the local edit; the current run does not alter it. The
{download}`shared-Tile note <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-shared-tile-validation/notes.md>`
retains exact commands and the warning boundary. Do not relabel those counts
as a new clean-source run.

The subsequent target-width checkpoint adds 14 ragged Metal layouts and
passes 83 Python tests; input reuse adds 22 numerical configurations and
passes 84 Python tests. Access-demand validation then passes 87 Python tests,
89,942 focused input-reuse assertions and 5,941 planner assertions in nine
tests. Each rebuilds the selected tree and retains the 31/33 CTest boundary
without changing the local flag. The
{download}`target-width record <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/notes.md>`,
[input-reuse evidence](reductions.md#budgeted-immutable-input-reuse) and
[joint mapping evidence](reductions.md#joint-resource-and-execution-mapping)
own their original logs. Later tail-pack validation adds 28 full/partial-pack
configurations with 89 Python contracts and the same CTest boundary; its
[fixed replay](reductions.md#tail-packs-a-structural-repair-after-width-reuse-ablation)
separately validates 192 benchmark outputs.

## Five failure investigations changed the implementation

### LLVM coexistence is a build/runtime constraint

The first combined TIRx/SIMD process loaded LLVM 21 through TVM and LLVM 22
through SIMD, then crashed in LLVM analysis setup before kernel execution.
Configuring both stacks against LLVM 21.1.8 removed that crash. An XIR-only
LLVM-22 executable had worked, which helped isolate the combined-process
configuration. This does not mean LLVM 22 is intrinsically unsupported; the
tested combined stack uses matching versions.

### PHI transfers must be simultaneous

The Tile bridge emitted valid loop PHIs, but SIMD's edge assignment lowering
loaded/stored one assignment at a time. A cycle `a←b, b←a` could read an
already overwritten state slot. The fix snapshots all right-hand sides
before any destination update. A pure-XIR regression produced 62 failures
before the fix and passed afterwards, along with existing SIMD regression
tests. Slot-coloring policy was not relaxed as part of this repair.

### Bounds proofs belong before Schedule expansion

Initial LLM runs were stopped during excessive JIT work, not recorded as
numeric passes. A process sample located LLVM machine scheduling/register
pressure work. Shared SSA/CFG cleanup and avoiding an extra diagnostic
assembly compilation helped but were insufficient. Per-axis checked integer
range proofs now remove redundant bounds diamonds before XIR/Schedule
expansion; unknown/overflowing accesses retain the original guarded behavior.

The complete LLM test subsequently finished in approximately 12 seconds,
including capture, both JIT routes, launches and validation. This observation
is a compilation/verification usability result, **not a kernel speedup ratio**;
the interrupted runs are not comparable completed timing samples.

### Distributed initialization does not create shared private storage

The first cross-entropy subgroup run passed both reduction recognizers but
failed six of seven output rows. Generated Metal showed why: 256 workers each
had a private `float[4096]`, initialization wrote only the worker-owned stripe,
and thread zero later performed the dynamic label gather from its own mostly
uninitialized array. This is exactly the abstraction error that execution/
resource separation is meant to prevent.

The repair makes immutable-view analysis path-sensitive for guarded indirect
indices, producing a direct guarded Tensor read, and adds a separate
whole-program ownership audit for every distributed nonscalar local buffer.
An unknown ownership proof now declines the optimized mapping. A positive
cross-platform guarded-view test and a negative explicitly materialized Tile
test protect both decisions; the full explanation and diagram are in
[the reduction report](../../internals/tile/reductions.md).

### Shared SSA must survive until target resource planning

Fused residual LayerNorm exposed a different structural failure. The old
exporter preserved shared transcendental expressions but cloned cheap shared
arithmetic into every consumer. `combined = X + residual` was consequently
expanded four times in generated Metal, multiplying global input reads even
though the later subgroup mapper had enough ownership information to retain a
compact value.

The exporter now preserves every pure multi-consumer Tile definition by
default. This is logical SSA, not a source-level `Memory` allocation. The
Metal mapper may materialize it as bounded worker stripes after an affine
ownership proof; a target may instead choose the explicit `EXPENSIVE_ONLY`
recomputation candidate. A 64-scalar-per-worker software-state bound rejects
pathological candidates before code generation. Metal selects preservation
for all four measured shapes; LLVM CPU selects recomputation for all four.
The [language/layout design](../../tile/design.md) and
[formal reduction report](../../internals/tile/reductions.md) record the full
contract and shared-Tile planning diagram.
