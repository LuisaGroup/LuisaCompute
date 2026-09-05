# Tile implementation checkpoints

Historical record through September 5, 2026. These checkpoints preserve
their original cohorts, baselines and limitations; they are not a single
current benchmark run. Start with [current status](index.md).

## Recorded checkpoints

**Overall verdict: the architecture is executable and several bounded cohorts
beat Torch or MPS, but the general performance objective is not complete.**
The strongest recorded Metal GEMM result is the TIRx-to-MPP view path, not
the native MPP route. In the same 14-round FP32 1024³ replay, their median
host-wall batch times are 270.675 and 287.137 µs respectively, versus MPS at
272.572 µs and Torch at 284.654 µs. The paired TIRx-view/MPS ratio is 0.9938:
near parity with a small measured advantage, not broad MPS dominance. These
historical host-wall measurements have different recorded fast-math settings
across routes and must not be presented as matched pure-kernel timings.

The packaged [XIR pilot report](../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-xir-simd/report.html)
is an earlier bounded snapshot with charts, audit tables and canonical sources.
Its canonical validator and structural verifier pass. The packager found no
compatible Chromium headless-shell, so responsive/source-dialog browser QA is
not claimed; the self-contained semantic chart and table fallbacks remain in
the file. This Markdown page and its reading map are the current
repository-native report across all Tile routes.

The execution-first C++ Tile language now has three distinct working compiler
routes: the maintained C++ TIRx bridge (including a typed Metal MPP contract),
a limited native Metal MPP realization, and a direct TileIR→XIR→SIMD CPU
realization on ordinary Luisa Runtime.
The XIR route includes a real, bounded execution-map solver; it does not
mechanically preserve the root axis order. Its model is still an uncalibrated
prior over a narrow candidate space, not a solved general CPU scheduler.

Correctness coverage now extends beyond GEMM to normalization, activations,
RoPE, masked softmax and online causal prefill/decode/GQA. The same captured
program is tested through XIR and TIRx against an independent FP64 reference.
Those are small multi-shape correctness cases, not production-size LLM
performance claims.

Metal MPP contract v2 now recognizes a proved positive-zero, single-iteration
accumulator as overwriting `D=A*B`. It removes the generated cooperative-tensor
zero-fill and C input instead of asking MPP to multiply-accumulate into zero.
MPP cost-model v2 now ranks legal rectangles by subgroup critical-path work
and whole-device subgroup waves. On the finite 8-shape calibration cohort it
reduces mean model regret from 74.18% to 8.82%; this is in-cohort evidence, not
held-out calibration. In the independent 14-round replay, the selected TIRx
MPP-view plans beat Torch and MPS on all eight GEMMs. At 1024³ they are 4.87%
faster than Torch and 0.62% faster than MPS by paired time ratio.

The TIRx CPU route now also has two explicit, structurally proved provider
families. A whole compact FP32 `C=A*B` contract can become one CBLAS call;
shared compact FP32 exponentials and exact add/max/min recurrences can become
Accelerate array operations. A six-order GEMM replay is faster than eager
Torch on seven of eight shapes and within 0.5--10.5% of direct CBLAS on seven
of eight; the 32³ wrapper-dominated case is 25.4% slower. A separate six-round
policy A/B shows no systematic add change, 2.71--6.12× reduction speedups and
2.10--5.46× softmax speedups. All 192 replayed outputs pass their complete
oracles and the fingerprinted artifacts remain unchanged.

The TIRx Metal route now has a third, non-library realization family for
proved FP32 row reductions. It maps one logical program to a target-legal
number of 32-lane SIMD groups, packs independent short programs, and compacts
eligible compiler-owned Tiles to worker-private stripes through an affine
ownership proof. Path-sensitive guarded-view analysis and an independent
distributed-local ownership audit now also make dynamic label gathers safe.
Across the current 24-case sum/softmax/RMSNorm/LayerNorm/cross-entropy/residual-
LayerNorm reports, every complete output passes and Tile/Torch host-wall
throughput ranges from 0.032× to 0.902×. Sum and softmax use preallocated
output on both sides; PyTorch's functional normalization/loss calls include
returned-output allocation. Same-binary four-round native A/B replays measure
21.19×--49.87× for RMSNorm, 14.04×--75.54× for LayerNorm/cross-entropy, and up
to 1.421× for preserving shared arithmetic in residual LayerNorm. A paired
CPU search selects the opposite recomputation policy, demonstrating that
logical SSA sharing and physical storage must remain separate decisions. This
is a narrow M1 Max cohort, not a production LLM suite or pure-GPU-event claim.

Automatic TIRx GPU elementwise mapping now fuses a logical program and its
independent Tile element coordinates into a physical thread grid. A four-round
frozen-binary add A/B validates all 32 outputs and measures 34.49×, 79.17×,
8.50× and 4.20× over the previous program-per-worker realization at 1×127,
17×257, 128×1024 and 4096×256. New times are 2.55, 2.74, 5.20 and 18.62 µs;
paired new/Torch time ratios are 0.704--0.832. This combines proved immutable
input forwarding with coordinate fusion; it is not just a launch-width tweak.

The same family now accepts bounded, pointwise shared Tile SSA chains and
turns each producer into one per-worker scalar definition. A same-binary
four-round GELU(A+B) A/B measures 32.70×, 52.64×, 9.91× and 3.83× over the
unfused program-per-worker map at those four shapes, with all 64 native/Torch
outputs valid. Both variants keep identical input-view and shared-SSA policy.
This is storage scalarization plus execution-grid fusion, not recomputation.
Its 2.59--19.93 µs times are still **host wall time**, and Torch is a two-op
eager graph with preallocated intermediate/output, not a compiled fused graph.

The benchmark now additionally supports **real Metal compute-pass timestamps
in a separate measurement phase**, alongside uninstrumented batched and
single-dispatch end-to-end times. A two-case integration smoke confirms that
GPU execution and dispatch latency differ substantially; it is not a stable
performance comparison. The M1 Max supports pass-boundary rather than arbitrary
per-dispatch counters, so multi-dispatch passes are labeled batch time. No
existing result has been relabeled as pure kernel time. The helper uses public
Metal APIs without changing the pinned TVMx build or Runtime interfaces.

The follow-up **audits the observer itself**: same-sized, alternating-order
samples also collect command-buffer GPU timestamps with no encoder probes or
counter attachments. Torch softmax's counter/control GPU time ratio reaches
3.08–5.85× in the diagnostic cohort. Accordingly, counter samples are retained
as diagnostics, not used as an uninstrumented kernel-speed ranking. Reports
now show the no-counter GPU control beside E2E timing for native/Torch/MPS.
This control includes all GPU work/gaps inside each command buffer, not just
one kernel. Six reduction cases, three SIMD-group/Torch/MPS GEMMs and three
native-MPP GEMMs pass complete output validation with the new control. Current
background-load variability rules out a stable ranking or cost-model update
from this cohort; prior host-wall A/B results keep their original scope.

Reduction collaboration width, independent-program packing and ordered stripe
unrolling are now separately controllable and jointly searchable through
ordinary staged/JIT runs. The TIRx cost model has a backend-overridable abstract
policy, while proofs and candidate legality remain bridge-owned. A separate
four-round, 80-output replay finds stable 1.107× for 1024×4096 sum, 1.062× for
1024×4096 softmax and 1.051× for 1024×257 softmax over the existing automatic
reduction mapper. The other seven shapes are flat/noisy or slightly worse;
the unchanged 17×257 sum is a noise control. Therefore the default unrolling
factor and coefficient prior remain unchanged. Sixteen additional norm/loss
outputs validate the new unrolled codegen, not a universal performance win.

The next structural reduction candidate is **consecutive elements per worker**:
`i=(chunk*workers+worker)*V+element`, V=1/2/4/8. Reducers, elementwise
consumers and compiler-local Tile storage share this ownership map; full packs
are separated from guarded tails. V defaults to one and is exposed to the
backend cost policy and staged/JIT search, not hardcoded for RMSNorm. An
explicit GPU-control JIT objective now complements the unchanged host-wall
default. Its score uses no-counter command-buffer GPU throughput, never the
perturbing compute-pass probe. The
{download}`implementation/validation checkpoint <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-lane-validation/notes.md>`
records 24 new Metal layout cases, 82 Python tests and the full 31/33 Tile
regression: two existing source assertions conflict with the untouched local
`mem_flags(2)` edit. A separate four-round, 64-output frozen replay measures
**1.208× / 1.469× / 1.156× GPU-throughput gains** over the current V=1 mapper
for 1×127 / 17×257 / 64×4096 RMSNorm. At 64×4096 native GPU time is
9.216 µs versus Torch 9.172 µs: approximately parity, not a win. The
1024×4096 case is flat (1.015× GPU, 0.992× E2E throughput), and synchronized
single-call E2E latency still trails Torch at both wider sizes. Forty-eight
additional outputs validate six operators at V=4 without establishing a
cross-operator optimum. All figures keep no-counter GPU, instrumented probe
and E2E scopes distinct. Default V and coefficients remain unchanged pending
held-out mapping-model validation.

The
{download}`target-complete width checkpoint <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/notes.md>`
also repairs two omitted parts of the candidate space: TVM's benchmark target
inherited a 256-thread default, and the bridge restricted automatic
cooperation to powers of two through eight subgroups. The adapter now queries
the device limit and the solver includes every legal subgroup count up to the
32-partial collective bound. Fourteen new ragged softmax configurations pass,
including 96 and 1024 threads on this device. Physical group counts and useful
lane-work fractions reach backend cost policies and reports. A separate
15-case, four-round frozen replay holds V=4/P=1/U=1 fixed and compares the
best of six measured widths with the best of the restricted {32,128,256}
subfamily, not the old automatic planner. At 1024×4096, sum/softmax/RMSNorm
gain **1.051× / 1.141× / 1.101× GPU throughput** and
**1.045× / 1.156× / 1.092× E2E throughput**, with all four pairs positive.
RMSNorm takes 64.210 µs versus Torch 68.802 µs in the no-counter GPU phase;
the median paired time ratio is 0.931. Single-call RMSNorm GPU/E2E remains
approximately parity. Two other search winners regress in every replay round:
128×8192 sum costs 6.79% more GPU time and 17×257 softmax costs 25.88% more.
Five same-plan controls and all 240 replay outputs are retained; the search
adds 202 validated outputs and four explicit resource rejections. GPU here
means command-buffer execution, not an isolated kernel timestamp. Default V
and scoring coefficients remain unchanged; independent incumbent acceptance
and a calibrated full-device cost policy remain necessary.

The
{download}`input-reuse checkpoint <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-input-cache-validation/notes.md>`
retains proved immutable snapshots across reduction/element domains using
the existing private-stripe ownership proof and cumulative budget, with no
new Tile DSL syntax. A 25-case, four-round cache/reload replay fixes
W=512/V=4/U=1/P=1. At 1024×4096 softmax/RMSNorm/LayerNorm gain
**1.378× / 1.265× / 1.221× GPU throughput** and
**1.381× / 1.279× / 1.229× E2E throughput**, with all four pairs positive.
RMSNorm GPU time is 55.863 µs versus Torch 69.108 µs; paired native/Torch
time ratio is 0.807. This is not a comparison with earlier tuned widths.
Three smaller changed-source cases have mixed individual GPU pairs; ten
identical-source controls are retained. All 400 replay and 100 pilot outputs
pass executed validation and the independent timing/source audit. The default
stays reload: the current score counts extra private traversals but cannot
price reduced global reads, so joint mapping/resource cost calibration is
still needed. GPU here remains command-buffer, not isolated-kernel timing.

The subsequent {download}`access-demand implementation checkpoint
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-access-demand-validation/notes.md>`
adds per-program/per-worker global and private payload access facts to the
backend-overridable policy, plus joint cache/reload × execution-mapping JIT
search. Optional access coefficients remain zero until calibrated. These are
conservative logical IR features, not physical traffic measurements. Tests
verify same-expression load deduplication, cross-phase accounting, budget
rejections and fresh-winner capture. The following 12-case, four-round replay
compares joint selection against the best measured reload width in the same
five-width family, not the earlier fixed-W=512 reference. At 1024×4096,
softmax/RMSNorm/LayerNorm gain **1.200× / 1.214× / 1.234× GPU throughput**
and **1.199× / 1.221× / 1.248× E2E throughput**, all four pairs positive.
Their GPU times are 49.179 / 52.826 / 61.316 µs versus eager Torch
122.653 / 69.742 / 205.799 µs. Seven changed-source cases improve in every
GPU pair; four smaller cases have mixed pairs, and one identical-source
control is retained. All 192 replay and 226 search/fresh-winner outputs pass
executed validation and the independent raw-timing/source/plan audit. GPU is
command-buffer, not isolated-kernel time. Defaults remain unchanged; this
demonstrates joint JIT selection, not a fitted replacement cost model.

The {download}`whole-launch cost-policy checkpoint <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-service-policy-validation/notes.md>`
separates local program work from the complete machine objective. Backends can
override the latter without a hidden program-wave multiplication; an optional
typed service model combines subgroup launch demand and payload access facts.
The old policy remains the default. Six nonnegative coefficients are frozen
from the preceding 101 valid trials before four disjoint shapes are measured.
Model-only resource selection and independent four-round replay now validate
all 288 outputs, with unchanged implementation/calibration artifacts. At
768×6144, softmax/RMSNorm/LayerNorm gain **1.360×/1.287×/1.231× GPU** and
**1.372×/1.280×/1.233× E2E throughput** over the old automatic planner.
Nine cases improve in every GPU pair, but 37×1537 softmax and LayerNorm regress
in every GPU/E2E-throughput pair; small RMSNorm GPU is mixed and approximately
Torch parity. This is shape-held-out, not operator/device-held-out evidence.
The profile remains opt-in. The subsequent fixed width/reuse ablation finds
416 workers slower than 192 in every fixed-reuse E2E pair, and motivates a
generic tail-pack emission repair rather than kernel-specific width rules.

The {download}`tail-pack lowering A/B
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-tail-pack-replay/notes.md>`
guards full worker packs once and isolates the unique partial worker, keeping
ownership, recurrence order, resource choices and all twelve plans unchanged.
At 37×1537 softmax/RMSNorm/LayerNorm gain **1.134×/1.207×/1.210× E2E batch
throughput** over the previous emitter, with every pair positive. GPU control
gains are 1.157×/1.222×/1.175× by paired median, but both norms have a losing
pair; six identical-source controls also show substantial timing variability.
All 192 replay outputs and 28 new tail-pack numeric configurations validate.
The full cohort, ranges, GPU/E2E distinction and source controls are retained.
No noisy GPU labels refit the cost model, and the general performance objective
remains incomplete.

**The general library-performance goal is still not complete.** These CPU
wins are legal provider realizations for narrow proved contracts, not evidence
that the portable loop family or direct XIR route has acquired BLAS-class
matrix and vector-math atoms. Direct XIR GEMM remains 38--55× behind eager
Torch in its pilot; Metal has not been benchmarked across a production LLM
operator suite; and the MPP model has no held-out device/operator validation.
No Metal result is relabeled as XIR performance.

## Reading map and scope

| Document / artifact | What it answers |
|---|---|
| [Language and layout design](../../tile/design.md) | Minimal primitives, lexical Nests, assignment capture, layout algebra and proof boundaries |
| [Executable kernel gallery](../../tile/kernels.md) | GEMM, reductions, attention, CNN/filter and Top-K/sort syntax/composition examples |
| [Execution planner](../../internals/tile/planner.md) | General binding/distribution/atom/resource/time formulation; current Metal implementation versus future calibrated model |
| [TIRx Metal reductions](../../internals/tile/reductions.md) | Formal subgroup mapping, ownership proof, finite solver, staged/JIT interface, tests and complete performance evidence |
| [Runtime and native lowering](../../internals/tile/runtime.md) | Factory, backend ownership, shader handles, Metal native/TIRx limits and ABI |
| [XIR execution planning](../../internals/tile/xir.md) | Exact implemented CPU candidate space, cost equations, alias constraints, SSA/CFG lowering and extension roadmap |
| {download}`Benchmark tools <../../../../scripts/benchmark/tile_torch/README.md>` | Reproduction interfaces, timing definitions, controls and saved runs |
| {download}`Validation evidence <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-xir-validation/notes.md>` | Test commands, failure investigations, logs and audit details |
| {download}`Metal MPP cost v2 search <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-search/notes.md>` | v1 failure, v2 equations, hard legality and in-cohort model regret |
| {download}`Metal MPP cost v2 replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-replay/notes.md>` | Frozen schedules, 784 full-output checks and balanced MPP/MPS/Torch evidence |
| {download}`Metal reduction cohort <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-reductions/notes.md>` | Sum, softmax and RMSNorm plans, 12 complete outputs, timings, hashes and exact command |
| {download}`Balanced RMSNorm A/B <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-rmsnorm-replay/notes.md>` | Same-binary reference/subgroup causality check across four shapes and four balanced rounds |
| {download}`Metal row-program extension <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions/notes.md>` | LayerNorm and cross-entropy plans, guarded-gather repair, eight complete outputs and API-level timings |
| {download}`Balanced row-program A/B <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions-replay/notes.md>` | Same-binary LayerNorm/cross-entropy causality check: 64 valid native variants and unchanged artifacts |
| {download}`Residual LayerNorm materialization search <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-materialization-search/notes.md>` | Separate preserve/recompute JIT candidates, complete Metal outputs and exposed v1 cost-model regret |
| {download}`Residual LayerNorm materialization A/B <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-materialization-replay/notes.md>` | Four balanced same-binary rounds isolating the shared-SSA decision |
| {download}`Bounded Metal thread search <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-bounded-thread-search/notes.md>` | Exact width candidates, rejected over-budget stripes and fresh winner validation |
| {download}`CPU materialization search <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-residual-layernorm-materialization-search/notes.md>` | Opposite target choice under fixed LLVM/input-view/vectorization/stack policies |
| {download}`CPU CBLAS replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-cblas-v2-replay/notes.md>` | Eight frozen GEMMs, six implementation orders, direct CBLAS overhead and Torch comparison |
| {download}`CPU array-math replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-accelerate-ops-replay/notes.md>` | Causal reference/Accelerate A/B over add controls, row reductions and softmax |
| {download}`Earlier CPU/provider validation <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-provider-validation/notes.md>` | Historical provider checkpoint, with its then-current build/test counts and documentation QA |
| {download}`Shared-Tile validation <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-shared-tile-validation/notes.md>` | Previous shared-SSA checkpoint, its 32/32 submitted-source Tile cohort and 69/69 benchmark contracts |
| {download}`Element-grid A/B <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-element-grid-replay/notes.md>` | Frozen old/new binaries, four balanced rounds, complete add outputs and the combined forwarding/fusion gain |
| {download}`Reduction joint-map replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-joint-map-replay/notes.md>` | Collaboration/packing/unrolling choices, three stable gains, seven qualified outcomes and raw per-round evidence |
| {download}`Execution-map validation <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-execution-map-validation/notes.md>` | Latest build, exact-constraint failures, CPU/Metal regressions, benchmark contracts and documented limitations |
| {download}`Shared element-SSA A/B <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-shared-element-replay/notes.md>` | Four balanced GELU(A+B) rounds, same-binary fusion isolation, complete outputs and eager-graph caveats |
| {download}`GPU/dispatch integration smoke <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-device-timing-smoke-v2/notes.md>` | Real pass-boundary GPU counters versus uninstrumented host dispatch; raw calibration and explicitly preliminary evidence |
| {download}`Dual-timing validation <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-dual-timing-validation/notes.md>` | Full 33-test Tile checkpoint, then-77 Python contracts, timestamp tests and submitted/worktree distinction |
| {download}`GPU timing observer audit <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-device-timing-counter-control/notes.md>` | No-counter GPU control, probe perturbation, reduction/GEMM/native-MPP integration and the preceding 80-test Python checkpoint |
| {download}`Consecutive-worker reduction layout <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-lane-validation/notes.md>` | Generic ownership map, GPU-objective JIT, four-round RMSNorm GPU/E2E replay, six-operator coverage and current verification boundaries |
| {download}`Target-complete reduction widths <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/notes.md>` | Queried device limit, all whole-subgroup widths, cost-policy features, 14 new GPU layouts, 83 Python tests, 15-case GPU/E2E replay, rejected winners and source-hashed audit |

```{figure} ../../../_static/tile/xir-planning-pipeline.svg
:alt: The same execution-first TileIR feeds XIR/SIMD, Metal-native MPP and TIRx compiler routes with separate ownership.
:width: 100%

One language and Runtime contract do not imply identical target schedules or interchangeable benchmark results.
```
