# Tile performance by compiler route

Saved comparisons through September 6, 2026. These are separate experiments,
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

(new-xir-simd-planner-pilot)=
### Initial XIR/SIMD planner pilot

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

### SIMD packet-index proof closes a codegen disconnect

The September 6 [bounded packet proof](../../internals/tile/xir.md#proven-packet-accesses-not-estimated-slopes)
lets Schedule retain value-preserving integer casts and prove aligned
quotient/remainder relationships. The existing memory emitter now uses eight
A broadcasts and eight contiguous B reads per static 1×1×8 K chunk, instead
of sixteen gathers. The Tile program, root plan, cost coefficients and strict
math policy are unchanged. This is an index-analysis/codegen change, not a
new GEMM DSL primitive or a BLAS substitution.

The final six-round frozen old/new/Torch comparison validates all 108 full
outputs and 38 unchanged artifacts. Values below are synchronized host-wall
batched dispatch microseconds; they are **not CPU kernel-only timings**.

```{table} Final SIMD compiler comparison, fixed Tile 1×1×8 and eight CPU workers
:class: benchmark-table

| M×N×K | Old µs | New µs | Torch µs | Paired new/old | New slower rounds | New/Torch |
|---|---:|---:|---:|---:|---:|---:|
| 32³ | 51.739 | 38.913 | 0.978 | 0.756 | 0/6 | 39.986 |
| 128³ | 301.521 | 118.771 | 4.936 | 0.398 | 0/6 | 24.215 |
| 512³ | 12528.875 | 4142.333 | 146.611 | 0.326 | 0/6 | 28.267 |
| 1024³ | 109013.021 | 39793.646 | 985.279 | 0.367 | 0/6 | 40.493 |
| 128×2048×512 | 13117.948 | 5592.222 | 158.803 | 0.427 | 0/6 | 35.075 |
| 127×193×61 | 246.743 | 247.816 | 6.549 | 1.006 | 5/6 | 38.045 |
```

The four nontrivial aligned shapes improve in every throughput and single-call
latency pair; small-shape latency is mixed. The ragged control has identical
LLVM in both arms and retains its small throughput regression and mixed
latency, rather than being dropped. The
{download}`complete report <../../../../scripts/benchmark/tile_torch/results/m1-max-20260906-xir-packet/notes.md>`
keeps both metrics, observed ranges, compile times, the earlier replay and
source/binary boundaries. These gains are relative to the old implementation:
**every final shape still loses to Torch**, with aligned nontrivial throughput
ratios of 24.2–40.5. Cache/register blocking and local-Tile/lane distribution
remain the larger CPU realization gap. Multi-operator correctness tests pass;
this cohort makes no new LLM-operator performance claim.

A separate new 8192³ MPS capture passes complete FP64 validation. Xcode window
inspection timed out, so its launch/counter attribution is not available yet;
the capture is excluded from performance rankings. No Metal default changes
follow from this CPU checkpoint.

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

### Larger matrices: the 1024-cubed win does not generalize

The new six-shape scale test freezes existing schedules before timing; it
does not tune a new winner at each size. Native and handwritten MPP retain
their old 32×32 control; ordinary and non-forwarding TIRx use 32×32×32,
128 threads. The view path transfers the old 128×32×1024, 128-thread winner
unchanged. Fourteen rounds balance all seven positions and pair precedence.
The {download}`predeclared protocol
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-large-matrices/protocol.md>`
and {download}`complete scale report
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-large-matrices/notes.md>`
retain all paths, round ranges, single-call timings and failures.

**8192³ still has a substantial gap.** Native MPP's paired GPU/Torch time
ratio is 1.985 [1.916, 2.786], slower in all 14 rounds. TIRx→MPP views reduces
that to 1.125 [1.019, 1.251], but also loses all 14 GPU pairs. Its E2E/Torch
ratio is 1.096 [0.887, 1.597], with only two faster rounds. This is not general
MPS/Torch parity; the nearby 2048³/4096³ medians have much wider mixed ranges.

GPU batch times below are **milliseconds**, from no-counter command-buffer
intervals, not isolated kernel timestamps. Native/handwritten MPP keep fast
math off; TVM Metal's existing fast-math behavior is unchanged. “TIRx” means
ordinary SIMD-group matrices, “MPP” means TIRx→MPP without forwarded inputs,
and “Views” means TIRx→MPP with proved input views. MPS is the direct matrix
API, not MPSGraph; Torch is eager MPS. All outputs are preallocated for GEMM.

```{table} Large GEMM GPU batch time (ms)
:class: benchmark-table

| M×N×K | Native | TIRx | Hand MPP | MPS | Torch | MPP | Views |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048×2048×2048 | 3.201 | 4.089 | 2.832 | 3.018 | 2.989 | 3.713 | 3.012 |
| 4096×4096×4096 | 29.978 | 39.556 | 27.169 | 30.148 | 27.074 | 33.065 | 29.273 |
| 8192×8192×8192 | 476.117 | 438.198 | 412.663 | 248.050 | 237.612 | 421.804 | 271.092 |
| 256×11008×4096 | 6.581 | 6.128 | 6.902 | 3.944 | 3.810 | 5.718 | 4.264 |
| 4096×4096×11008 | 102.115 | 156.372 | 90.646 | 94.582 | 82.550 | 150.774 | rejected |
| 2049×4097×1025 | 3.626 | 10.898 | 3.748 | 3.645 | 3.350 | 10.581 | rejected |
```

Separately measured **batched E2E milliseconds**, including warm Runtime/
framework dispatch, submission and synchronization:

```{table} Large GEMM end-to-end batch time (ms)
:class: benchmark-table

| M×N×K | Native | TIRx | Hand MPP | MPS | Torch | MPP | Views |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048×2048×2048 | 3.254 | 4.208 | 2.883 | 3.012 | 3.196 | 3.596 | 3.144 |
| 4096×4096×4096 | 30.026 | 35.439 | 27.701 | 30.651 | 27.377 | 32.446 | 29.808 |
| 8192×8192×8192 | 461.686 | 369.682 | 412.923 | 295.217 | 278.915 | 399.020 | 300.684 |
| 256×11008×4096 | 6.529 | 6.214 | 7.156 | 4.101 | 3.896 | 6.005 | 4.276 |
| 4096×4096×11008 | 95.074 | 146.511 | 91.488 | 85.497 | 77.179 | 131.272 | rejected |
| 2049×4097×1025 | 3.984 | 11.906 | 3.865 | 3.811 | 3.520 | 11.755 | rejected |
```

GPU and host phases are independent; do not subtract these medians to infer
dispatch cost. Large GEMM times drift substantially despite balanced order;
the experiment does not identify the cause. No noisy labels refit the model.

The two rejected view requests have K tails. MPP's unguarded address contract
prevents forwarding a region that is not proved fully in bounds; materializing
the fixed large tiles then fails the bounded resource/geometry planner.
This is a fixed-schedule admission failure, not general MPP unavailability:
the other six paths validate both shapes. At that checkpoint, bounded full/tail
realization and shape-aware resource search remained open; no substitute block
hid the rejection. The later bounded-K result below does not rewrite those
historical failures. The {download}`loader preflight
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-large-matrices/environment-preflight.md>`
also preserves the initial unpatched-TVM capability failures separately.

All 560 executed GEMM replay outputs pass complete FP64 comparison; 28 view
admission failures remain failures. All 26 inventoried artifacts are unchanged.
The {download}`independent audit and all paired metrics
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-large-matrices/audit.json>`
checks recorded validation, balanced order, fixed plans, sources and all four
timing metrics. These deterministic FP32 tests are not low-precision or
end-to-end model coverage. The separate
[wide-row reduction cohort](reductions.md#wide-rows-and-large-working-sets)
extends normalization to width 16384 and a 512 MiB input/output payload.

### Bounded-K MPP views: legal tails, remaining library gap

The September 6 [bounded-K proof](../../internals/tile/matrix.md#bounded-k-views-avoid-nominal-padding-storage)
admits a common zero-padded K suffix as two immutable physical input views.
The previously fixed 128×32×1024 schedule now needs **zero shared allocation**
in these cases, instead of nominal 640 KiB A/B staging. The frozen old v2
binary rejects the three K-tail requests; there is no old execution time to
divide by. M/N tails, extra masks, nonzero fill and unequal A/B K intervals
remain outside this forwarding capability. No model coefficients or DSL
entities change.

The fixed four-shape, seven-route, 14-order replay validates **392/392 full
outputs** (8,325,201,920 checked elements) and 26 unchanged artifacts. The
small shape wins all host-throughput pairs against MPS/Torch, but has mixed
GPU pairs against MPS. Every nontrivial shape still has a paired GPU time
ratio above one against both libraries. **This is not general library parity.**

```{table} Bounded-K cohort: GPU command-buffer batch microseconds
:class: benchmark-table

| M×N×K | TIRx MPP views µs | MPS µs | Torch µs | Paired view/MPS | Paired view/Torch |
|---|---:|---:|---:|---:|---:|
| 128×128×61 | 8.592 | 8.938 | 13.889 | 0.962 | 0.658 |
| 1024×1024×1537 | 511.423 | 433.547 | 437.150 | 1.180 | 1.171 |
| 4096×4096×11008 | 60221.083 | 53208.125 | 54887.771 | 1.097 | 1.124 |
| 8192³ | 241151.958 | 220077.896 | 210055.750 | 1.075 | 1.182 |
```

Views lose 5/14, 14/14, 10/14 and 8/14 GPU-throughput pairs to MPS, and
0/14, 14/14, 12/14 and 12/14 to Torch, respectively. At 8192³ the generated
view source is **identical to the earlier scale cohort**. Its current
192.820–285.891 ms range and cross-session differences are not a compiler
speedup or evidence of a particular thermal/cache cause.

Separate **batched E2E** view times are 8.845, 521.213, 57790.291 and
193842.396 µs. Paired view/MPS ratios are 0.889/1.169/1.119/1.204;
view/Torch ratios are 0.302/1.141/1.141/1.161. Single-call latency remains
separate and retains regressions. The no-counter GPU interval includes
command-buffer work/gaps, not isolated shader instruction time; instrumented
compute-pass samples remain diagnostic.

Against the retained **materialized 32×32×32 TIRx MPP** control, paired GPU
ratios are 0.677/0.439/0.739/0.776, all 14 pairs improving at every shape.
This comparison includes both view realization and different geometry;
it is not a same-schedule ablation of the new proof. The large K-tail case
still has one host-throughput regression against that control.

The {download}`complete seven-route report
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260906-mpp-bounded-k/notes.md>`
and {download}`independent audit
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260906-mpp-bounded-k/audit.json>`
retain every path, all four metrics, ranges, orders, source hashes and old
admission failures. A separate 36-output
{download}`operation-scope screen
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260906-mpp-scope/notes.md>`
finds no universally better collective width; no single-order minimum becomes
a default. M/N-edge atoms, physical K chunking, reuse and distribution remain
the next realization work, followed by independent model/search validation.

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
