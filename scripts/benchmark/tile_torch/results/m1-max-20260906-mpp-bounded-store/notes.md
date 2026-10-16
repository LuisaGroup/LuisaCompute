# Bounded MPP output: general legality, bounded performance evidence

## Technical summary

The TIRx bridge can now compose a proved rectangular output prefix with its
subgroup distribution and store a closed accumulator directly. No operator
name, benchmark size, new DSL entity, cost coefficient or solver is involved.
The existing resource planner consumes the new legality fact; it is not a
new general scheduler or a cost-model calibration.

On Apple M1 Max/macOS 26.6.2, September 6, 2026, six fixed-schedule paired
rounds validate **216 complete outputs (1,587,230,352 elements)**. The four
ragged cases remove 16 KiB of shared C. Median paired GPU time reductions are
1.36–10.33%; batched end-to-end reductions are 2.29–17.33%. Three ragged cases
lose one GPU pair each. All four still have median GPU time above Torch;
two beat direct MPS by median paired ratio. The two aligned control sources
are byte-identical across compiler versions and are not compiler speedups.
The [reader-facing result](../../../../../docs/source/performance/tile/results.md#bounded-output-removes-shared-c-not-the-whole-library-gap)
retains every shape and comparison.

## What is general, and what is not

The matcher consumes affine unit projections, canonical bounds, enclosing
execution domains, ownership and closed recurrence observations. The sink's
valid lengths are independent of A/B's input padding. Nonnegative offsets,
permuted memory axes, tails and empty subgroup outputs are supported; extra
masks, negative origins, manual/observed carry and missing TVM capability keep
the prior realization. A bounds conjunction is matched bidirectionally, so a
transpose cannot fail merely because it permutes equivalent guard clauses.
An old-output snapshot remains a separate resource at the original sink order.

The existing planner releases C's backing and derives one-shot overwrite mode
when the initializer and recurrence allow it. Consequently this A/B measures
the complete newly legal realization: direct store, released shared C and
derived overwrite/fill removal. It is not an isolated store-instruction test.
Full row-major interiors keep MPP bulk stores; edges use public cooperative
coordinates, not guessed lane ownership. This contract currently covers
FP32 rank-two compact positive-stride projections, not all layout algebra.
The independent native MPP and XIR/SIMD emitters are unchanged.

## Frozen measurement protocol

The [protocol](protocol.md) was written before implementation. Both variants
use one captured 128×32×4096 tile, 128 threads, four 32×32 subgroup outputs,
pipeline window 1 and copy batch 1. The six shapes include four ragged inputs
through 4097×4097×4096 and aligned 1024³/4096³ controls. There is no search,
retuning or selected minimum across rounds. A large nominal K is deliberately
held fixed even where it is unlikely to be the best schedule.

[replay.py](replay.py) reuses the repository benchmark/oracle but independently
selects the old/new executable and dynamic-library search path. The frozen old
stack predates this change; both stacks have the previous M/N/K extension.
**Both contain the same user-owned `mem_flags(2)` barrier edit**, which is not
part of this checkpoint commit. Absolute timings therefore describe this
fingerprinted experimental worktree, not an untouched branch build. The edit
is held constant in the A/B and remains uncommitted in the user's workspace.
Both full build gates precede timing; no builds overlap measurements.

Each of six fresh-JIT rounds rotates the starting case and reverses variant
order. Each variant/case gets all six native/Torch/MPS execution permutations.
Each measurement phase has nine samples, 30 ms calibrated batch windows and
100 ms warmup. Torch 2.14.0 and direct MPSMatrixMultiplication both use
preallocated FP32 outputs. Inputs are deterministic dyadic FP32; all saved
full-output FP64-oracle error receipts are zero at `atol=rtol=1e-4`. Unit
tests additionally use non-dyadic values, nonfinite values and signed zero.
Math implementation and framework/driver dispatch differences are retained;
this is not a proof of identical instruction-level arithmetic or all inputs.

The GPU comparison uses uninstrumented completed command-buffer intervals
divided by that phase's own repetition count. It includes GPU work and gaps,
not CPU encoding, and **is not isolated pure kernel time**. Instrumented
compute-pass samples remain diagnostics. Batched E2E, single-call E2E and
single-call GPU times are separately sampled. Do not subtract their medians
to estimate dispatch overhead.

## Evidence and validation

- [old-pilot](old-pilot/results.json) freezes the original schedule before
  implementation. [balanced-pilot](balanced-pilot/results.json) completes
  old/new/new/old; pilot timings are excluded from final statistics.
- [frozen-replay](frozen-replay/results.json) contains 72 rows, full numerical
  receipts, all raw phase samples, exact native commands, generated-source
  hashes, per-variant loaders and 35 unchanged fingerprinted artifacts.
- [audit.py](audit.py) imports neither the benchmark validator nor its
  aggregation helpers. It recomputes every p50 from raw samples, checks
  denominators, output counts, fixed geometry, resource/mode selection,
  case/order completeness, source hashes and both aligned controls. Its
  [audit receipt](audit.json) records ten rejected adversarial mutations and
  all four metrics with paired median/min/max ratios. Saved arrays are
  transient after full runtime validation; this is an independent receipt
  audit, not a second rerun of the original arrays.
- The full [matrix correctness run](full-matrix-correctness-v2/receipt.json)
  passes 3599 Metal and 3159 CPU assertions. Low-level bounded-store contracts
  cover five nominal rectangles, seven full/partial/empty prefixes, two
  orientations and static/dynamic lengths: 140 compiled contracts, each
  checked on five value classes, plus sixteen malformed ABI cases.
- [Old-TVM compatibility](old-tvm-compatibility/receipt.json) uses the current
  bridge with the frozen compiler lacking bounded-store-v1. All 2729 Metal
  matrix assertions pass; optional cases retain their expected fallback.
- [Cross-operator regression](cross-operator-correctness-v2/receipt.json) passes
  the execution, basic PoC, neural PoC and algorithm PoC suites on both CPU
  and Metal. These include reductions, attention, convolution/filter and
  sorting/Top-K numerical controls, not new performance comparisons.
  [Native Metal Runtime](runtime-correctness/receipt.json) also passes 364
  assertions after its own full build gate.
- The incremental TVM patch applies cleanly to copies of the frozen source.
  Resulting codegen/header SHA256 values exactly match the compiled external
  sources (`5be29800d9d6170a03c1b39fc308d211b966071f10c6f2d73bf2cf7a71be034b`
  and `2a5826db47767375915f7f2cc729a10f3a3d55c1e88dc84252638764c4f66118`).

The first matrix pilot failed on unsupported test-fixture predication and a
zero-K 8×24 MPP descriptor, an overly broad zero-shared-memory expectation,
and reordered transpose guards. [Diagnostic logs](bounded-output-diagnostics/receipt.json)
and the [protocol amendments](protocol.md) preserve the hypotheses and fixes.
Two earlier filtered runs executed zero assertions and were rejected by the
checker, not counted as successes. An initial cross-operator invocation also
passed `cpu` to the Luisa Runtime test, which expects an installed runtime
backend rather than TIRx's LLVM alias. It failed at backend loading, before a
kernel ran; the invalid invocation is retained, not counted as CPU coverage.

## Interpretation, uncertainty and next questions

The 1025×1025×1024 GPU improvement is small (paired ratio 0.986, range
0.972–1.004), and the 4097×4097×4096 case loses one GPU pair despite its
0.971 median ratio. The aligned controls range over 0.982–1.020 and
0.928–1.037 with **identical code**, indicating nontrivial system variation.
No rounds or outliers were removed. Six paired ranges are not confidence
intervals; desktop activity and device clock/thermal state were not isolated.
Single-call timings are particularly variable and remain in the audit rather
than being replaced with batch throughput.

The next planner work is to price input/output edge fractions, physical K
partition, output materialization and cross-program reuse jointly within a
backend-overridable policy. Add independently validated candidates before
calibrating those features; retain held-out shapes/operators/devices and
the unchanged-source controls. This checkpoint neither changes defaults nor
establishes a cross-operator, low-precision or cross-device performance win.

### Report structure and visual contract

This methods-first report uses the user's existing Sphinx hierarchy, not a
parallel report app. Technical summary, exact-shape evidence, definitions,
experimental design, limitations and next questions map to the technical
report specification. The reader-facing six-row table supports exact lookup
across old/new times and three paired comparisons; heterogeneous matrix sizes
and multiple metrics make a single ranked bar chart less useful. No new chart
is needed. Definitions precede that table; paired ranges and negative results
are adjacent. Existing matrix-planning diagrams carry the architecture context.
