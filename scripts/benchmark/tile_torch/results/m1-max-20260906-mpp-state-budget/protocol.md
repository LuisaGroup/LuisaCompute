# MPP realization-state budget — protocol before implementation

The MPP scorer already counts only the output cooperative tensor as explicit
fragment state. Its candidate gate still charges A and B fragments as if it
were emitting the SIMD-group reference realization. Correct that inconsistency
without changing numerical permission, device limits, scalar budget, cost
coefficients, or the Tile DSL. Memory operands do not promise zero internal
registers: this budget bounds compiler-explicit logical tensor state only.

The full selected TVM and Luisa builds completed before freezing the current
benchmark, Tile/bridge and TVM libraries in
`/tmp/luisa-mpp-budget-baseline.ISAho9`. Luisa is built in
`/tmp/luisa-tvm-mpp.VaKmzx/luisa-build`; TVM is in the adjacent `build`.
Baseline is commit `52e17083a` plus the unchanged, user-owned shared-only
barrier edit in `cooperative.cpp`. Both variants retain that edit; do not
attribute it to this change. Other dirty files are out of scope.

## Correctness and admission

- Independently enumerate rectangular coverage and the MPP descriptor rule
  over small/tall/wide domains, subgroup widths and scalar-budget boundaries.
  Check exact-budget acceptance and one-below rejection. SIMD-group A/B/C
  storage accounting must remain unchanged.
- Compile and execute newly admitted output rectangles with direct global
  inputs and shared staging, single/multiple K steps, nonzero initial state,
  ragged bounds, transposes and sentinel outputs. Preserve existing matrix,
  cross-operator CPU/Metal and native Metal Runtime regressions.
- A full selected-tree build precedes every native test/benchmark phase. No
  measurements overlap builds. A valid timer requires a complete output check
  against the existing FP64 oracle, not a sampled correctness check.

## Search and acceptance

This is a candidate-admission change. First compare a fixed exploratory set:
output blocks `128x32`, `128x64`, `64x128`, `64x64`, K block `4096`, threads
`64,128,256`, copy batch 1 and pipeline window 1, explicit MPP global views.
Retain old-stack rejections; they are evidence of missing candidates, not
timings. Use the same set for both stacks. Initial shapes are `1024^3`,
`4096^3`, `1025x1025x1024`, and `4096x4096x11008`. Held-out evaluation uses
`257x769x113`, `2049x4097x1025`, `4097x4097x4096`, and `8192^3`.

Do not change this cohort or fit coefficients after looking at its labels.
If a measured candidate is worth accepting, freeze its selection and rerun
old incumbent/new winner with fresh JIT, alternating variant/case order and
balancing native/Torch/direct-MPS order over six rounds. Preserve all losses.
Report score-selected and timing-selected picks separately. No claim of
universal selection or cross-device performance follows from this experiment.

Record batched and single-call E2E separately from no-counter command-buffer
GPU intervals (not isolated kernel timestamps); instrumented pass timings are
diagnostic only. Record commands, software/hardware identities, source and
library fingerprints, raw samples and full validation. Use preallocated FP32
outputs for all GEMM implementations and no new fast-math permission.

## Reader-facing report plan

Update the existing Sphinx matrix/cost-policy and performance references, not
a separate report application. The user explicitly requested the existing
documentation structure. Put the result and scope first, then candidate
admission evidence, defined multi-metric comparisons, methodology, limitations,
and the next generic modeling work. Preserve prior sections. Use an exact
comparison table because admission status, selected mapping, GPU/E2E and
library ratios require row-level lookup; do not imply a time-series trend.
Recalculate reported numbers independently and inspect the rendered table at
desktop/mobile widths before claiming the report validated.
