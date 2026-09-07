# Multi-output pointwise fusion: evidence and limits

## Technical summary

The compiler change admits same-domain, independent multi-output graphs to
the existing fused element grid. It removes the serial program-per-worker
boundary without kernel-name/shape rules, new DSL primitives, a score refit,
or parameter search. This is an execution/resource realization extension;
it does not improve the CPU or reduction emitters by itself.

On September 6, 2026, Apple M1 Max/macOS 26.6.2, six frozen fresh-JIT rounds
validate 192 complete value/derivative pairs (384 planes). All 48 new/old
pairs improve in batched GPU and E2E time. The eight median new/Torch GPU
ratios are 0.309–0.952; small GELU loses one GPU batch round and ragged GELU
does not beat Torch's single-call E2E latency. This is fused Tile versus
**preallocated eager Torch**, not compiled Torch, direct MPS or a full
training step. See the [reader-facing route report](../../../../../docs/source/performance/tile/results.md#multi-output-pointwise-fusion-removes-a-mapping-boundary).

## Measurement definitions and population

- Graphs: sigmoid and tanh-approximate GELU, each returning value and derivative
  in separate native parameters. Both are named `benchmark_activation_pair`
  in the capture API; the compiler uses IR structure, not this label.
- Dimensions: 1×127, 37×1537, 1024×4096 and 4096×4096 for both graphs. The
  existing element block is fixed at 1×256; both plans request automatic
  execution and the same preserved shared-SSA policy. No tuning occurs.
- Native: FP32 input/output, full FP64 formula check with `atol=2e-6,
  rtol=2e-5`. Value and derivative arrays are concatenated after timing.
  Torch: 2.14.0, output preallocation on both sides. Sigmoid uses three out
  calls; GELU uses forward and backward out calls, with a preallocated unit
  gradient. Input/compilation/allocation/upload/download are outside timing.
- Timing: nine samples per phase, 30 ms calibrated batch window, 100 ms
  warmup. Mapper order, native/Torch order and starting case are rotated
  across six rounds. Every invocation group is recaptured/JIT-compiled.
- Reported GPU comparison: no-counter completed command-buffer intervals,
  including GPU work/gaps, with a separate repetition denominator. This is
  not an isolated instruction/kernel timestamp. Compute-pass instrumentation
  is retained separately and never substituted into this comparison.
- E2E throughput is synchronized host batch time per invocation; single-call
  E2E and GPU latency are separate phases. Do not subtract phase medians to
  invent CPU dispatch cost. Desktop activity is not controlled.
- FP64 inputs are deterministic non-dyadic values over approximately [-1,1].
  Python oracle tests additionally compare analytic derivatives against
  independent autograd at saturation and near zero. The native benchmark
  does not establish every input distribution, dtype or device.

## Evidence map and independent validation

The [pre-change protocol](protocol.md) predates the compiler edit.
[Old automatic](old-auto/results.json) and
[old reference](old-reference/results.json) pilot sources match on the four
small cases. A frozen copy of the old executable and Tile/bridge libraries
also reproduces the four large reference sources in
[large controls](control-large-old-old/results.json). Thus the current
disabled-fusion control reproduces the old automatic Metal code on all eight
cases, not merely an identical option name. Preliminary pilots do not have a
full before/after loaded-library receipt; their times are not final rankings.

[Current automatic](new-auto/results.json) and
[current reference](new-reference/results.json) freeze the schedules used by
the [six-round replay](replay/results.json). The accompanying
[four-metric report](replay/results.md) includes paired medians, full ranges,
and single-call results. All 23 recorded compiler/runtime/helper/executable
artifacts are unchanged. Twelve distinct generated Metal sources are saved;
each case/variant retains one source across all six JITs. Aligned large
shapes differ in launch size, not shader source.

[audit.py](audit.py) independently recomputes p50s from raw samples, verifies
phase denominators, exact coverage/order, both-output counts, tolerances,
source hashes and realized scalar/grid plans. Its [receipt](audit.json)
also records eight adversarial checks: missing, duplicate, failed, partial
output, changed artifact, wrong denominator, missing source and unbalanced
framework order. Invalid variants must be rejected. Output arrays themselves
are transient after complete runtime validation; the saved receipt is not
a second independent rerun of those original numerical arrays.

## Scope controls and failure investigations

[controls.py](controls.py) executes 32 additional native/Torch cases under
frozen old and current bridge libraries, with 27 unchanged artifacts:

- Add and single-output GELU(A+B), plus Softmax/RMSNorm/LayerNorm, at 37×1537
  and 1024×4096: all ten old/new Metal sources are byte-identical.
- CPU value/derivative graphs at the two small shapes: all eight native/Torch
  comparisons validate. Raw LLVM hashes differ because TVM writes object
  addresses into TBAA metadata labels, not because the CPU emitter changed.
- Four large old-automatic sources match the current disabled-fusion control.

The raw [controls receipt](controls.json) retains its failed byte-identity
check for CPU. [audit_controls.py](audit_controls.py) permits only bijective
renaming of those specific complete TBAA object-label definitions, preserving
alias equivalence classes, every instruction and all other metadata. Its
[separate receipt](controls-audit.json) verifies four normalized CPU pairs and
fourteen raw Metal pairs, with mutations of instructions/alias classes
rejected. These short pilots support source/correctness control, not new CPU
or reduction speed claims.

The [full-build correctness receipt](correctness/receipt.json) covers the
execution, matrix, basic PoC, neural PoC and algorithm PoC suites on both CPU
and Metal. All ten pass with unchanged test/compiler libraries. Coverage
includes three interleaved outputs, nonzero minima, ragged and negative input
origins, conditional and permuted output maps, explicit-scope/noalias controls,
and RAW/WAR/WAW/different-domain rejection. Existing losses, attention,
convolution/filter and sorting/Top-K PoCs are numerical controls only.

The [initial source-assertion failure](constant-source-assertion.log) is
preserved. A test required exactly one textual `exp` even when all inputs
were zero. TIRx substituted constant `exp(0)` expressions for the Metal
compiler to fold; input storage was entirely absent. The first hypothesis
mistakenly attributed the failure to a missing `exp`. Printing the exact
source showed repeated constant expressions instead. The final test checks
input removal for that case and exactly one nonconstant `exp` for the other
two shapes; all numerical checks are unchanged.

99 Python tests pass, including complete second-output rejection and FP64
analytic derivatives checked against autograd. clangd/clang-tidy report no
errors in the three changed C++ translation units; style/performance hints
remain, including by-value DSL signature parameters.

## Interpretation and next questions

Legality generalizes across the tested expression graphs and beyond two
outputs. Profitability is established only for this finite Metal cohort:
constant/shared-producer graphs, more output/register pressure, irregular
layouts and different devices need their own checks. Every output still
requires one injective compact store and no global reads/escapes of written
buffers. Unknown cases retain their original mapping. Neither `parallel`
independence nor shape equality is used to erase inter-domain dependencies.

The most useful next step is an IR-level fusion/partition and live-state
candidate model, then reuse of those facts in CPU vector and reduction
realizations. A future solver should compare emit-able distributions with
traffic/reuse/spill costs on held-out graphs; it should not learn a table of
the eight measurements here. Direct XIR/SIMD GEMM, large-MPS parity, native
MPP epilogues and performance of attention/CNN/sort remain separate open work.

Validation assessment: **share with caveats**, not general-performance
completion. No MPS capture or new hardware-counter attribution is claimed.

## Reproduce and report QA

Complete the selected CMake build and run `check.py --tag NEW_TAG` before
native timing. `repeat.py` accepts `new-reference/results.json` and
`new-auto/results.json`, `--operations sigmoid_pair,gelu_pair`,
`--rounds 6 --samples 9 --sample-ms 30 --warmup-ms 100 --capture-sources`,
the prebuilt Metal timing helper and five `--compiler-artifact` paths for
TVM compiler/runtime/runtime_metal/runtime_extra/ffi. Its saved metadata and
per-row commands are the exact argument/source/loader specification.
Run `audit.py --check-local-artifacts` on the original machine, or omit that
flag for a source-and-receipt-only audit on another machine.

The reader-facing surface is the user's existing Sphinx documentation, not
a second report app. The technical-report roles are integrated into the
route section: answer, definitions, exact comparison, limitations and next
work. Detailed methods/failures remain here. The visualization contract is
an eight-row exact-lookup table: neutral colors, µs units, paired-ratio
denominators and near-table caveats. A chart spanning the 2–2045 µs range
would obscure the small-case lookup; no extra chart is added. Final rendered
Sphinx QA checks the new table, lowering map and navigation at desktop and
narrow widths. The fresh build renders 48 HTML pages; 3,665 local targets and
199 compatibility anchors pass. It has exactly ten existing missing-Doxygen-
XML warnings in `api_reference.rst`, no new Tile warnings. Doxygen is not
available locally, so the warnings-as-errors build exits 1: full API docs are
**not** certified. No warnings are suppressed to present an all-green build.
