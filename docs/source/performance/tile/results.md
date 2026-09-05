# Tile performance by compiler route

Saved comparisons through September 5, 2026. These are separate experiments,
not a cross-route leaderboard with one matched timing and math policy.
See [current status](index.md) for the conclusion and remaining goal.

## Performance: preserve the measurement basis

Unless explicitly labeled otherwise, reported times are **warm synchronized
host-wall time per invocation**, amortized over a batch. They include each
runtime's dispatch/encoding/submission and synchronization. They exclude JIT,
setup allocations/uploads and cold-call setup; returned-output allocation
stays inside timing where the recorded operator API requires it. They are not GPU hardware-event times,
and CPU thread requests are not measurements of actual library worker use.

Report tables use medians of within-round p50s. A paired ratio is the median
of same-round numerator/denominator ratios, **not** a ratio of the displayed
medians. Ranges and counts of slower rounds are descriptive, not confidence
intervals. No slow or failed row is discarded to improve the headline.

### Metal subgroup reductions close the measured normalization defect

The [lowering reference](../../internals/tile/reductions.md)
documents the new opt-in TIRx Metal realization. It structurally revalidates
canonical FP32 add/max/min reductions, searches whole-SIMD-group cooperating
widths, and derives private/shared storage from the selected execution
map. For softmax width 4096, a logical compiler-owned 4096-element Tile becomes
a compact private stripe (16 values at 256 threads) only after every access proves the same affine
owner; the old per-thread `float[4096]` form is rejected by source tests.

The two original current-binary cohorts use 11 samples, 100 ms calibrated
sample windows and 100 ms warmup. All 20 complete FP64 checks pass:

| Family | Shapes | Tile/Torch range | Fastest absolute Tile | Slowest relative Tile |
|---|---|---:|---:|---:|
| row sum | 1×127, 17×257, 128×1024, 64×4096 | 0.293×--0.716× | 3.106 µs | 0.716× |
| softmax | same widths/row counts | 0.124×--0.286× | 3.305 µs | 0.286× |
| RMSNorm | same widths/row counts | 0.546×--0.902× | 3.904 µs | 0.902× |
| LayerNorm | same widths/row counts | 0.511×--0.648× | 4.500 µs | 0.648× |
| cross-entropy | same widths/row counts | 0.032×--0.052× | 3.449 µs | 0.052× |

The four additional residual-LayerNorm cases search both shared-Tile policies
with separate capture/JIT/full validation, then recapture the winner. Metal
selects `PRESERVE` in every case:

| Rows×width | Tile µs | eager Torch MPS µs | Tile/Torch | Worker stripe scalars |
|---|---:|---:|---:|---:|
| 1×127 | 3.426 | 10.671 | 0.321× | 4 |
| 17×257 | 3.655 | 11.705 | 0.312× | 6 |
| 128×1024 | 6.321 | 18.592 | 0.340× | 8 |
| 64×4096 | 8.324 | 27.046 | 0.308× | 32 |

The independent same-binary replays rotate variant and case order for four
rounds. The subgroup path is 21.19×--49.87× faster for RMSNorm and
14.04×--75.54× for LayerNorm/cross-entropy by median paired ratio. Native uses
preallocated output; PyTorch's functional normalization/loss calls allocate
their returned output inside timing, so only the native reference/candidate
A/B is the clean causal comparison. The saved
{download}`cohort report <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-reductions/notes.md>`,
{download}`balanced replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-rmsnorm-replay/notes.md>`,
{download}`row extension <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions/notes.md>`
and
{download}`extension replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions-replay/notes.md>`
and the
{download}`residual materialization search <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-materialization-search/notes.md>`
retain every sample, plan, output error, artifact hash and exact command. The
separate
{download}`materialization A/B <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-materialization-replay/notes.md>`
isolates the Metal decision: median paired preservation speedup is 1.057×,
1.008×, 1.354× and 1.421× from smallest to largest shape. The analytic v1
model does not count duplicated global loads or expression depth and incurs up
to 43.66% regret; the report preserves that miss rather than calling it a
model success.

This closes the diagnosed scalar-worker realization for the admitted subset.
It is not production attention, training-loss/backward coverage,
low-precision evidence, held-out device calibration or pure Metal kernel
timing. In particular, the very large cross-entropy advantage includes
PyTorch's general eager API and returned-output overhead; it is not presented
as an isolated MPS-kernel ratio.

### New XIR/SIMD planner pilot

The {download}`XIR pilot <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-xir-simd/notes.md>`
compares automatic planning, fixed `{order=[0,1], block=64}`, and eager
Torch on the same CPU, separately from TIRx. The Tile specialization is fixed
at 1×1×8. Six rounds balance all three implementation orders for 32³, 128³
and 127×193×61. Raw outputs, LLVM source, actual plans and hashes are retained.

This pilot asks whether the initial mapping prior helps that specific
specialization; it is not a production GEMM schedule search or a benchmark
of the LLM family. The report preserves negative results and names the exact
comparison baseline. Neither a better model score nor a passing test is
reported as a speedup.

| Shape | Planned µs | Fixed map µs | Torch µs | Paired planned/fixed | Paired planned/Torch |
|---|---:|---:|---:|---:|---:|
| 32³ | 50.822 | 50.037 | 0.978 | 1.0157× | 51.851× |
| 128³ | 272.410 | 278.352 | 4.979 | 0.9755× | 54.543× |
| 127×193×61 | 255.711 | 281.315 | 6.696 | 0.9142× | 38.186× |

All automatic plans chose root order `[0,1]`; worker packing differed from the
fixed 64-worker control. Automatic planning was slower in 4/6, 3/6 and 1/6
rounds respectively. The fixed comparison therefore shows a modest, noisy
mapping effect, while the 38–55× Torch gap diagnoses a missing realization
family. This is direct evidence for Tile/lane distribution, register blocking,
cache-aware reuse and vector/matrix microkernels before cost-model polishing.

### Balanced Metal evidence: MPP cost v2 closes this GEMM cohort

The {download}`cost-model study <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-search/notes.md>`
first preserves the failed v1 ranking. Across the same 8 shapes and 45 requested
block/thread candidates, v1's mean/median/maximum finite-set regret is
74.18/43.05/239.58%; v2's is 8.82/2.59/34.37%. Exact measured-winner picks
increase from 1/8 to 4/8. Those 3-sample, 10 ms values are **in-cohort** and
diagnostic. They neither establish held-out prediction nor replace final timing.

The independent {download}`v2 replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-replay/notes.md>`
freezes the measured schedules, then uses 14 balanced rounds, 8 shapes and
7 compiler/library paths. All 784 complete outputs passed the same FP64 oracle;
all 21 fingerprinted benchmark/compiler/runtime artifacts retained their hashes.
No schedule was searched or selected during replay.

```{figure} ../../../_static/tile/mpp-cost-model.svg
:alt: MPP planning generates and proves legal candidates, applies a target-specific relative-work model, searches the bounded space, and finally defers to correctness-checked JIT measurement.
:width: 100%

The analytic plan is a shortlist prior. The independently validated measured winner is what the replay freezes.
```

| Shape | Frozen block @ threads | TIRx MPP views | Hand MPP | MPS | Torch | Paired view/MPS | Paired view/Torch |
|---|---|---:|---:|---:|---:|---:|---:|
| 32³ | 32×32×32 @ 128t | 2.982 | 2.809 | 10.081 | 26.899 | 0.2794× | 0.1105× |
| 128³ | 32×32×128 @ 256t | 5.335 | 5.441 | 16.904 | 27.218 | 0.3174× | 0.1943× |
| 512³ | 32×64×32 @ 128t | 42.413 | 46.802 | 52.428 | 47.745 | 0.8285× | 0.8919× |
| 1024³ | 128×32×1024 @ 128t | 270.675 | 266.105 | 272.572 | 284.654 | 0.9938× | 0.9513× |
| 256×1024×128 | 64×64×128 @ 256t | 16.025 | 17.286 | 20.350 | 28.668 | 0.8189× | 0.5554× |
| 1024×128×256 | 32×32×32 @ 128t | 16.500 | 18.508 | 26.270 | 28.655 | 0.5946× | 0.5596× |
| 127×193×61 | 32×32×32 @ 256t | 8.861 | 7.127 | 16.915 | 26.997 | 0.5172× | 0.3266× |
| 513×257×129 | 32×32×32 @ 256t | 20.607 | 24.424 | 35.043 | 34.002 | 0.5874× | 0.6057× |

At 1024³ the new 128×32×1024, 4×1-subgroup schedule is 4.87% faster than
Torch, 5.76% faster than native Tile→MPP and 0.62% faster than MPS by paired
ratio; it remains 1.68% slower than handwritten MPP. It was slower than MPS in
only 1/14 rounds. Across all eight rows the TIRx-view path beats both external
baselines. This closes the measured FP32 GEMM cohort, not the general library-
performance goal: model v2 still needs held-out shapes/operators and residual
regret shows that cache/layout, edge and launch features are incomplete.

Native/handwritten MPP use fast math off; TVM's Metal runtime uses fast math
on. All values above are synchronized host-wall batched times, not GPU-event
durations. The original TIRx, staged TIRx MPP, native MPP, handwritten MPP,
MPS and Torch controls remain in the raw report.

### CPU TIRx: reference gaps and proved provider realizations

The original six-round, eight-shape reference-loop cohort remains useful as a
negative control. At 1024³ it measured 5919.062 µs versus Torch at 1020.527 µs
and direct Accelerate at 1027.681 µs: a paired 5.769× TIRx/Torch gap. Changing
only `target-cpu` did not change the emitted 4×16 register-blocked loop body or
close that gap. Cache-aware panels, packing/reuse and a matrix microkernel were
absent from the reachable realization family, so a better solver score could
not help.

The new solution preserves the same TileIR semantics but adds a target
realization boundary. Structural TileIR matching proves a whole compact FP32
GEMM or an exact reduction recurrence; structural export preserves every pure
multi-consumer Tile SSA by default. The CPU pass then revalidates the actual
TIRx body, buffer ABI, layout, alias contract and target policy before choosing
a resource or provider atom. It never matches a diagnostic operation name,
and an explicit unsupported request fails rather than silently changing
semantics.

```{figure} ../../../_static/tile/tirx-realization-pipeline.svg
:alt: TileIR is structurally exported once, then portable, CPU-provider and Metal matrix families are selected behind a second proof firewall.
:width: 100%

Provider calls are target realizations selected from proved semantic
contracts; direct CBLAS/MPS benchmark programs remain independent baselines.
```

#### Whole-GEMM CBLAS realization

The {download}`current single-session plan <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-cblas-v2-plan/notes.md>`
verifies that each generated LLVM kernel has exactly one external matrix call.
The {download}`six-order replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-cblas-v2-replay/notes.md>`
then freezes those schedules and remeasures Tile, eager PyTorch and a separate
direct Accelerate CBLAS executable. There are 48 valid complete-output rows,
zero failures, and stable binary/library hashes.

| FP32 shape M×N×K | Tile→TIRx→CBLAS µs | eager Torch µs | direct CBLAS µs | paired Tile / CBLAS [range] |
|---|---:|---:|---:|---:|
| 32×32×32 | 0.503 | 0.918 | 0.390 | 1.254× [1.071, 1.484] |
| 128×128×128 | 4.518 | 4.961 | 4.073 | 1.105× [1.085, 1.148] |
| 512×512×512 | 130.099 | 139.469 | 131.055 | 0.995× [0.988, 1.002] |
| 1024×1024×1024 | 984.515 | 930.311 | 965.743 | 1.031× [0.893, 1.234] |
| 256×1024×128 | 65.597 | 65.877 | 64.332 | 1.020× [1.007, 1.026] |
| 1024×128×256 | 62.717 | 63.152 | 61.323 | 1.023× [1.019, 1.028] |
| 127×193×61 | 6.287 | 6.791 | 6.030 | 1.047× [1.012, 1.075] |
| 513×257×129 | 43.612 | 43.701 | 43.356 | 1.005× [0.990, 1.035] |

The Tile path beats the displayed Torch median on seven of eight shapes. The
comparison against direct CBLAS answers a different question: wrapper and TVM
packed-ABI overhead are visible, especially at 32³ and 128³. The wide 1024³
range also shows why one lucky run must not be used as the headline.

#### Shared SSA and reduction realization

The structural exporter preserves a shared `exp` Tile once when its SSA result
has multiple consumers, instead of expanding the lazy expression into both a
reduction and an output consumer. The same default preserves cheap shared
arithmetic, but only a structurally revalidated `exp` contract can select the
provider below. The opt-in
`CpuMathBackend::ACCELERATE` policy can then realize that exact compact map with
vForce and exact FP32 add/max/min recurrence contracts with vDSP. The reference
path remains available. Unrelated add kernels are a negative control.

The {download}`six-round policy replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-accelerate-ops-replay/notes.md>`
contains 144 freshly captured/JIT-compiled rows, all valid. Times are medians
of per-round p50 synchronized host-wall measurements. The speedup is the median
of paired reference/candidate ratios, not a ratio of selected best runs.

| Case | Reference µs | Accelerate realization µs | Paired speedup [range] | candidate-run Torch µs |
|---|---:|---:|---:|---:|
| add 1×127 | 0.068 | 0.068 | 1.001× [0.978, 1.067] | 0.548 |
| add 17×257 | 0.421 | 0.418 | 1.001× [0.998, 1.018] | 0.934 |
| add 128×1024 | 4.698 | 4.713 | 1.004× [0.885, 1.226] | 38.289 |
| add 4096×256 | 32.807 | 32.207 | 1.022× [0.958, 1.437] | 84.070 |
| sum 1×127 | 0.064 | 0.024 | 2.708× [2.587, 2.774] | 0.772 |
| sum 17×257 | 2.186 | 0.375 | 5.828× [5.564, 5.939] | 1.060 |
| sum 128×1024 | 16.640 | 3.703 | 4.581× [3.534, 4.978] | 37.578 |
| sum 64×4096 | 33.738 | 5.512 | 6.123× [5.267, 7.228] | 40.651 |
| softmax 1×127 | 0.551 | 0.126 | 4.357× [4.276, 4.370] | 0.619 |
| softmax 17×257 | 5.436 | 2.555 | 2.098× [1.980, 2.286] | 33.428 |
| softmax 128×1024 | 79.242 | 14.527 | 5.460× [5.159, 5.609] | 88.699 |
| softmax 64×4096 | 156.785 | 41.876 | 3.753× [3.524, 4.113] | 128.818 |

The independent add control staying near 1× is evidence that the policy does
not broadly rewrite unrelated code. The single-session candidate report also
records zero provider calls for add, one dynamic reduction operation per row,
and three semantic call sites for softmax (`max`, `exp`, `sum`). Static LLVM
call-site counts can be larger when a small serial root is unrolled; they are
not dynamic-call counters.

This policy has a deliberately different numerical contract. vDSP may choose
a different FP32 reduction order, while vForce documents different denormal
and floating-exception behavior from scalar libm. The benchmark accepts it
only through the explicit target option and checks all outputs with recorded
tolerances; it is not silently enabled by the Tile DSL or execution hierarchy.

#### Target-specific residual-LayerNorm materialization

The
{download}`CPU materialization search <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-residual-layernorm-materialization-search/notes.md>`
holds native LLVM code generation, input-view forwarding, automatic element
packing, eight host threads and a 64 KiB compiler-local stack budget fixed.
Every one of its four measured winners uses `EXPENSIVE_ONLY`: 0.252, 8.799,
36.271 and 70.599 µs for widths 127, 257, 1024 and 4096. The corresponding
Tile/Torch ratios are 0.109×, 0.225×, 0.382× and 0.643×.

Metal selects `PRESERVE` on the identical semantic kernel because its mapped
worker stripes avoid repeated global reads. CPU benefits from recomputation
and LLVM fusion. This is direct evidence that preserving SSA in Candidate
TileIR does not imply a universal physical allocation; materialization belongs
beside binding, distribution and atom selection in the target plan.

Two other CPU scheduling repairs matter independently of providers. Automatic
roots below 64 cheap tasks stay serial unless the source explicitly requests a
worker scope; small roots containing transcendental/opaque work retain
parallel execution. Ragged SIMD packs are binary-versioned into a proved
all-lanes fast arm and an unchanged guarded slow arm. This removed full-pack
store scalarization: the 17×257 add control is now about 0.42 µs instead of the
earlier 2.84 µs observation. Both policies preserve the original tail and
parallel semantics and have dedicated structural/numerical tests.
