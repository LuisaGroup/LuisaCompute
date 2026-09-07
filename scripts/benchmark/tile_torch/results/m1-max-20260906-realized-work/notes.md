# Realization-derived work: analysis and model-selection checkpoint

## Technical summary

The TIRx matrix model now charges the work realized by each candidate: a proved
mean bounded K for MPP, retained versus eliminated scalar recurrence domains,
and the global output transfer replacing a removed scalar sink. Candidate-
dependent work enters the Pareto objective before pruning. Coefficients,
default state budgets, numerical permissions and resource legality are unchanged.
The change is generic within the admitted matrix family, not a kernel-name or
shape dispatch rule. Native Metal, XIR/SIMD and non-matrix operators do not
inherit a performance claim.

The bound analysis uses native TVM C++ arithmetic services. Linear detection
uses the registered native FFI function: this TVMx build declares
`DetectLinearEquation` in a header but does not export that symbol from its
compiler dylib. There is no Python-source export or operation-name dispatch.

The existing Sphinx matrix reference owns the design; its route-results page
owns the performance interpretation. This is a supporting reproducibility
record, following the user's requested documentation structure rather than a
new report application. The exact mapping/work lookup and paired timing table
serve different questions; no chart of search minima is used as speed evidence.

## Scope, grain and methods

Baseline: commit `71bd1d6a5` plus the unchanged user-owned shared-only barrier
edit, copied after full TVM/Luisa builds to
`/tmp/luisa-mpp-realized-baseline.mbTNjX`. The candidate holds that barrier
constant; it is not committed with this change. Both executable/library sets
are kept separate because the public diagnostic/workload structs changed ABI.

The [protocol](protocol.md) registers eight FP32 GEMM shapes through 8192³ and
the same fifteen block choices for both models. The last four shapes are
protocol-held-out checks, not a claim that the project has never benchmarked
those dimensions. No coefficients are fitted, including to the held-out half.
Each candidate is recaptured/JITed and checks all native/Torch/direct-MPS
outputs against the complete deterministic FP64 oracle at atol=rtol=1e-4.
The benchmark data are dyadic; separate native tests cover non-dyadic values.
Outputs are preallocated. TVM's existing Metal fast-math setting is unchanged;
Torch uses its default eager MPS path with benchmark override variables removed.
This is not strict cross-library arithmetic-policy equivalence.

Selection uses only the reported whole-kernel model score. Its one-sample
host timings are diagnostics, not a search objective or acceptance evidence.
The fresh replay freezes both the selected block and the **solved** thread
width, not the original automatic-width request. Six fresh-JIT rounds balance
all six native/Torch/MPS orders and old/new precedence, with nine samples,
30 ms requested windows and 100 ms warmup. GPU batch/single command-buffer
intervals include GPU work and gaps; host batch/single E2E includes dispatch
and synchronization. Counter-instrumented passes remain diagnostics, never
substitutes for the no-counter GPU control or isolated-kernel timestamps.

## Analysis, source and numerical evidence

[Selection receipts](selection/results.json) and the
[independent audit](selection/audit.json) cover 240 accepted trials plus 16
fresh selections: **768 complete outputs, 12,225,970,560 elements**, all with
zero maximum error for these dyadic inputs. Of 120 old/new fixed-block pairs,
118 retain the same solved mapping and byte-identical Metal source; two change
mapping. All eight candidate-model plans correctly report physical-equivalent
versus nominal K work and exclude only their proved-eliminated scalar domains.

Three final model choices change: 1025×1025×1024, 4096×4096×11008 and
2049×4097×1025. The other five selected-source pairs are identical controls.
The auditor checks coverage, resource/state counts, codegen mode, score-based
selection, fresh source identity, recorded complete-output receipts and raw
GPU/E2E denominators. It does not import the benchmark's oracle, selection or
percentile functions. Its reusable timing checker is explicitly linked and
fingerprinted; rejected data mutations exercise missing/duplicate rows,
changed binaries and, for replay, output size, work counts, divisors, loaders
and provider order. [review.ipynb](review.ipynb) is the executable companion.

The [final full-build receipt](final-correctness/results.json) passes twelve
planner/CPU/Metal test invocations, including 14,406 planner and 4,307 Metal
matrix assertions. New coverage executes 28 matrix programs across staged/
global inputs, positive/nonzero initial state, retained/nonresident/direct
output, both K traversal directions, transposes and partial M/N/K. Four
oversized staging requests still reject at the shared-capacity boundary.
The two-matrix cost oracle independently enumerates all choices under four
scalar prices and three shared-capacity limits, testing the Pareto objective.
CPU/Metal execution and basic/neural/algorithm PoCs and native Metal Runtime
remain passing. All six changed C++ translation units pass the repository
syntax checker. Python benchmark contracts: 95 tests, one intentional skip.

## Retained diagnoses and limitations

- The [initial correctness attempt](correctness/results.json) failed new
  model-count assertions. Its independent score oracle omitted the existing
  subgroup parallelism prior. Neutralizing that prior **in the test only**
  fixed twelve assertions; no production coefficient was changed.
- The [normalization diagnostic](normalization-diagnostic/results.json)
  records TVM rewriting `min(45 - 16*k, 16)` as
  `45 - max(16*k, 29)`. Preserving the cap before simplifying its affine
  operand repairs the six unrecognized physical-K cases. Their numerical
  output checks passed in both failed attempts. The temporary log is removed.
- The first independent selection audit incorrectly assumed the unit tests'
  256-thread limit. Actual benchmark metadata records a 1024-thread target
  limit; the automatic solver legitimately uses 512 threads for some small
  grids. The auditor now checks the recorded target capacity. No experiment
  candidate, thread request or measurement was changed to fix the auditor.
- The first multi-file syntax invocation was rejected because the repository
  checker accepts one path per call. All six subsequent individual checks
  passed; the failed invocation is not counted as syntax coverage.

M/N edge fractions, hardware K rounding, cache traffic and MPP's opaque
internal registers/spills remain unmodeled. Physical-equivalent matrix atoms
are normalized work, not a claim about fractional hardware instructions.
Unknown dynamic/nonlinear/multi-axis K expressions keep nominal work; cost
facts never grant load-elision, reordering or aliasing permission.

Read-only process inspection during selection observed concurrent desktop/video
activity. `pmset` reported no recorded thermal/performance warning, which is
not proof of stable clocks or idle GPU load. No application was closed. The
unchanged-source controls and all round reversals must stay in the report;
paired ratios do not establish causality or a confidence interval.

## Performance interpretation

The [frozen replay](replay/results.json) and its [independent audit](replay/audit.json)
pass all 288 outputs (4,584,738,960 elements) and 48 paired rounds. All three
changed schedules win 6/6 GPU pairs. In mapping-table order, their median
paired GPU time reductions are 3.88%, 6.16%, 17.75%; E2E reductions are 5.53%,
5.06%, 18.39%. These are batched measurements, not single-call or isolated
kernel times. Both single-call metrics remain in the complete audit.

The five unchanged-source controls have median GPU new/old ratios
0.992742–1.004665, with individual pairs spanning 0.937795–1.015383. Their
variation is not an emitter change. The three consistent changed-schedule
wins support a bounded model-selection improvement, with the concurrent-load
and non-causal limitations above. **All eight new GPU medians still lose to
Torch.** MPS ratios are below one on only three shapes, including two unchanged
controls. No universal parity or calibrated general cost model is claimed.

Assessment: ready to share for analysis/correctness and qualified finite-cohort
paired observations, not for general performance guarantees. The model's
feature accounting is more faithful; its remaining coefficients, edge and
memory behavior still need independent validation. No timing-fitted coefficient
or per-shape dispatch table is promoted. The full cohort and four timing views
remain in the raw replay; the existing Sphinx results page owns the compact
reader-facing table. End-to-end LLM speed and the broad goal remain open.

## Documentation QA

[docs-qa.json](docs-qa.json) records a fresh Sphinx tree, exact rendered-table
comparison against the independent audit, desktop/mobile image inspection,
and reachable horizontally scrolling table columns. All 3,693 local links/
assets and 199 compatibility anchors pass across 48 pages. There are no new
Tile warnings. The full build exits 1 on ten existing missing-Doxygen-XML API
warnings; full generated API documentation is **not** certified. Visual review
caught a clipped long table header/caption; shortened headers, an external
caption and a compact vertical proof diagram were verified before handoff.
