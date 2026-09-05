# Metal reduction measurements

This page records measured cohorts and known regressions. The
[lowering reference](../../internals/tile/reductions.md) owns execution
mapping, correctness admission, storage and target intrinsics.

```{contents} On this page
:local:
:depth: 2
```

## Outcome at a glance

The old TIRx Metal path mapped one logical row program to one scalar worker.
A width-4096 RMSNorm therefore performed a serial 4096-element recurrence in
one thread; softmax could additionally allocate one private `float[4096]` per
thread. Launch-width tuning could not repair that execution structure.

The new lowering proves the reduction program, then maps one logical program
to one or more 32-lane SIMD groups. It packs independent short programs into a
threadgroup, cooperates across up to 32 SIMD groups for wide programs, and
compacts eligible compiler-owned Tiles to worker-private stripes. The source
C++ kernel and logical TileIR remain unchanged.

Across the saved 24-case Apple M1 Max row-program cohort, all complete FP64 checks pass and
Tile/TIRx is faster than eager PyTorch MPS in every row. Tile/Torch ranges from
0.032× to 0.902× in synchronized device-resident host-wall throughput. Sum and
softmax use preallocated output on both sides; PyTorch's functional RMSNorm,
LayerNorm and cross-entropy allocate returned outputs inside timing, so those
external comparisons are explicitly qualified below. Separate four-round,
same-binary native A/B replays measure 21.19×--49.87× for RMSNorm and
14.04×--75.54× for the LayerNorm/cross-entropy extension. Those causal
native-to-native results are unaffected by PyTorch's output policy.

The final four cases in that cohort are fused residual LayerNorm. They expose
and repair a second structural gap: cloning a cheap shared Tile expression into each
consumer duplicated device reads even after the execution hierarchy was
mapped correctly. The structural exporter now preserves all multi-consumer
pure Tile SSA by default, and the target mapper can compact it to bounded
worker-private stripes. A staged/JIT candidate can still choose recomputation.

These results close two identified structural gaps. They do **not** establish
all-operator, all-shape, low-precision, cross-device or pure-kernel parity.
The later [target-width experiment](#target-aware-widths-gpu-and-dispatch-acceptance)
adds separately measured GPU and E2E evidence, including search winners that
regress in independent replay. The fixed-width input-reuse experiment
([input reuse](#budgeted-immutable-input-reuse)) then isolates a resource-planning improvement: caching audited
immutable inputs improves the three large normalization cases in every
paired round, while smaller mixed results and identical-source controls
remain visible. The subsequent [joint resource/width replay](#joint-resource-and-execution-mapping)
compares against the best measured reload width in the same family, rather
than a fixed width. The three 1024×4096 cases gain 1.200×/1.214×/1.234×
GPU throughput, while seven stable changed-source cases, four mixed cases
and one unchanged-source control remain separated. Input caching stays opt-in.

The [frozen whole-launch model](#whole-launch-policy-shape-held-out-gains-and-small-case-failures) then tests four disjoint shapes
without using their timings for selection. At 768×6144 it improves GPU
throughput by 1.360×/1.287×/1.231× over the old automatic planner. However,
37×1537 softmax and LayerNorm regress in every GPU/E2E-throughput pair.
These held-out failures keep the new cost profile opt-in as well.

The tail-pack repair improves the three small-case batched E2E times
against the old emitter, with mixed GPU pairs for the two norms. The
[tail-pack replay](#tail-packs-a-structural-repair-after-width-reuse-ablation)
retains that limitation and unchanged-source variability controls. It does not
establish that the held-out mapping regression is fully closed.

The newest [cooperating-program packing experiment](#cooperating-program-packing)
also separates expressibility from profitability. With a fixed 256 workers
per row, packing two cooperating programs improves GPU throughput in every
pair for two LayerNorm cases, but regresses in every pair for eight other
cases. The explicit candidate family is implemented; automatic packing and
cost coefficients remain unchanged.

## Performance evidence

### Base reductions versus eager PyTorch

The complete report is
{download}`Metal subgroup reductions <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-reductions/notes.md>`;
raw samples are in its adjacent `results.json`.

| Operator / rows×width | Tile µs | Torch µs | Tile/Torch |
|---|---:|---:|---:|
| sum 1×127 | 3.268 | 7.211 | 0.453× |
| sum 17×257 | 3.106 | 4.340 | 0.716× |
| sum 128×1024 | 3.387 | 5.604 | 0.604× |
| sum 64×4096 | 4.721 | 16.119 | 0.293× |
| softmax 1×127 | 3.578 | 26.111 | 0.137× |
| softmax 17×257 | 3.305 | 26.594 | 0.124× |
| softmax 128×1024 | 5.385 | 30.376 | 0.177× |
| softmax 64×4096 | 8.881 | 31.029 | 0.286× |
| RMSNorm 1×127 | 3.904 | 7.155 | 0.546× |
| RMSNorm 17×257 | 5.335 | 6.154 | 0.867× |
| RMSNorm 128×1024 | 6.673 | 8.707 | 0.766× |
| RMSNorm 64×4096 | 11.177 | 12.392 | 0.902× |

These are p50 warm synchronized host-wall times across 11 samples with
100 ms calibrated sample windows and 100 ms warmup. Inputs remain
device-resident and native outputs are preallocated. Torch sum/softmax use
preallocated `out=` storage; the public functional RMSNorm has no `out=`
overload, so its returned-output allocation remains inside the Torch warm
timing and is recorded per row. Capture, compilation, transfers and cold calls
are separately recorded. PyTorch is eager and no `torch.compile` path is
claimed.

### LayerNorm and cross-entropy versus eager PyTorch

The independent eight-case extension is
{download}`LayerNorm/cross-entropy <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions/notes.md>`;
its adjacent JSON retains every sample, plan, error, setup phase and generated
Metal source.

| Operator / rows×width | Tile µs | Torch µs | Tile/Torch |
|---|---:|---:|---:|
| LayerNorm 1×127 | 4.500 | 8.400 | 0.536× |
| LayerNorm 17×257 | 5.714 | 8.821 | 0.648× |
| LayerNorm 128×1024 | 7.542 | 13.726 | 0.549× |
| LayerNorm 64×4096 | 12.413 | 24.313 | 0.511× |
| cross-entropy 1×127 | 4.513 | 107.246 | 0.042× |
| cross-entropy 17×257 | 3.449 | 107.695 | 0.032× |
| cross-entropy 128×1024 | 4.290 | 110.171 | 0.039× |
| cross-entropy 64×4096 | 5.838 | 112.263 | 0.052× |

These use the same synchronized host-wall protocol, now with 11 samples and
100 ms windows. PyTorch's functional LayerNorm and cross-entropy calls return
new output tensors, so their allocation is inside timing. Cross-entropy also
includes the general eager operator's dispatch and semantic machinery. The
table is therefore a real API-level comparison, not evidence that the Tile
kernel is 19--31× faster than an isolated MPS kernel. The native A/B below is
the causal lowering comparison.

### RMSNorm causal A/B against the old lowering

The independent
{download}`RMSNorm replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-rmsnorm-replay/notes.md>`
uses one current executable for both variants and changes only the explicit
subgroup policy. It rotates case and implementation order over four rounds and
freshly captures/JIT-compiles every row.

| Rows×width | Old reference µs | New subgroup µs | Paired speedup median [range] |
|---|---:|---:|---:|
| 1×127 | 103.180 | 3.792 | 27.216× [24.924, 28.020] |
| 17×257 | 268.202 | 5.366 | 49.871× [49.180, 54.574] |
| 128×1024 | 144.082 | 6.805 | 21.192× [20.989, 21.207] |
| 64×4096 | 524.444 | 11.160 | 47.096× [46.344, 50.864] |

All 32 reference/candidate outputs pass. Ranges are observed paired-round
minima/maxima, not confidence intervals. The result demonstrates a structural
execution-mapping gain; it does not prove the chosen map globally optimal.

### LayerNorm and cross-entropy causal A/B

The
{download}`balanced extension replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-metal-subgroup-row-extensions-replay/notes.md>`
uses the same executable for both policies, counterbalances order over four
rounds, recaptures/JIT-compiles every variant and checks every output.

| Operator / rows×width | Old reference µs | New subgroup µs | Paired speedup median [range] |
|---|---:|---:|---:|
| LayerNorm 1×127 | 131.413 | 4.577 | 28.675× [27.944, 29.063] |
| LayerNorm 17×257 | 337.366 | 5.693 | 58.942× [57.105, 64.220] |
| LayerNorm 128×1024 | 280.352 | 7.517 | 37.333× [36.900, 37.661] |
| LayerNorm 64×4096 | 928.945 | 12.306 | 75.536× [74.338, 82.088] |
| cross-entropy 1×127 | 62.412 | 4.446 | 14.042× [13.737, 14.854] |
| cross-entropy 17×257 | 191.603 | 3.228 | 59.357× [53.681, 61.339] |
| cross-entropy 128×1024 | 74.350 | 4.370 | 17.015× [16.097, 17.463] |
| cross-entropy 64×4096 | 355.493 | 5.774 | 60.879× [59.618, 63.291] |

All 64 native variant measurements pass, and all fingerprinted artifacts are
unchanged across the replay. This attributes the gain to the execution/view/
resource realization family rather than PyTorch output allocation or a
different binary. The ranges are observed paired-round extrema, not confidence
intervals.

### Historical launch plans

These are the exact plans from the original 20-case reduction/loss report,
not a current schedule recommendation. `threads/group` is the whole physical
group: at 17 rows, the S=1 plans pack eight independent programs rather than
cooperating on one row.

| Case | Threads/group | SIMD groups/program | Shared bytes | Private stripe/worker | Reductions | Model score |
|---|---:|---:|---:|---:|---:|---:|
| sum 1×127 | 32 | 1 | 0 | 0 | 1 | 23 |
| sum 17×257 | 256 | 1 | 0 | 0 | 1 | 14 |
| sum 128×1024 | 128 | 4 | 16 | 0 | 1 | 33 |
| sum 64×4096 | 256 | 8 | 32 | 0 | 1 | 49 |
| softmax 1×127 | 64 | 2 | 16 | 2 | 2 | 32 |
| softmax 17×257 | 256 | 1 | 0 | 9 | 2 | 42 |
| softmax 128×1024 | 128 | 4 | 32 | 8 | 2 | 64 |
| softmax 64×4096 | 256 | 8 | 64 | 16 | 2 | 112 |
| RMSNorm 1×127 | 64 | 2 | 8 | 0 | 1 | 24 |
| RMSNorm 17×257 | 256 | 1 | 0 | 0 | 1 | 22 |
| RMSNorm 128×1024 | 128 | 4 | 16 | 0 | 1 | 40 |
| RMSNorm 64×4096 | 256 | 8 | 32 | 0 | 1 | 64 |
| LayerNorm 1×127 | 64 | 2 | 16 | 0 | 2 | 30 |
| LayerNorm 17×257 | 256 | 1 | 0 | 0 | 2 | 33 |
| LayerNorm 128×1024 | 128 | 4 | 32 | 0 | 2 | 56 |
| LayerNorm 64×4096 | 256 | 8 | 64 | 0 | 2 | 96 |
| cross-entropy 1×127 | 32 | 1 | 0 | 0 | 2 | 31 |
| cross-entropy 17×257 | 256 | 1 | 0 | 0 | 2 | 27 |
| cross-entropy 128×1024 | 128 | 4 | 32 | 0 | 2 | 51 |
| cross-entropy 64×4096 | 256 | 8 | 64 | 0 | 2 | 83 |

LayerNorm's independent element count is the row width because its affine
output is distributed. Cross-entropy has only three scalar independent
elements after immutable logits/label views are forwarded; its two width-sized
loops are the reductions already counted separately.

The current default structural exporter additionally preserves LayerNorm's
shared `centered` Tile, so newly compiled width-4096 LayerNorm reports bounded
worker stripes rather than zero. The exact size follows its selected width,
not this historical 256-thread entry. Saved artifacts are never rewritten to
pretend they were produced by a newer policy.

### Fused residual LayerNorm and materialization choice

The current
{download}`Metal materialization search <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-materialization-search/notes.md>`
JIT-compiles both shared-Tile policies for every shape. All measured winners
use `PRESERVE`:

| Rows×width | Tile µs | Eager Torch MPS µs | Tile/Torch | Stripe scalars/worker |
|---|---:|---:|---:|---:|
| 1×127 | 3.426 | 10.671 | 0.321× | 4 |
| 17×257 | 3.655 | 11.705 | 0.312× | 6 |
| 128×1024 | 6.321 | 18.592 | 0.340× | 8 |
| 64×4096 | 8.324 | 27.046 | 0.308× | 32 |

PyTorch evaluates eager `layer_norm(X + residual)` and allocates its returned
output, so this is an API-level comparison. The clean policy attribution is
the separate
{download}`four-round A/B replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-materialization-replay/notes.md>`:

| Rows×width | Expensive-only µs | Preserve µs | Paired preserve speedup [range] |
|---|---:|---:|---:|
| 1×127 | 3.692 | 3.506 | 1.057× [1.027, 1.137] |
| 17×257 | 3.648 | 3.632 | 1.008× [0.957, 1.039] |
| 128×1024 | 8.244 | 6.084 | 1.354× [1.313, 1.366] |
| 64×4096 | 13.548 | 9.591 | 1.421× [1.392, 1.471] |

All 32 native A/B measurements pass and use unchanged fingerprinted artifacts.
The independent
{download}`bounded thread search <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-residual-layernorm-bounded-thread-search/notes.md>`
also records the 64-scalar legality bound and rejected wide-shape candidates.

The
{download}`CPU materialization search <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-residual-layernorm-materialization-search/notes.md>`
selects the other legal policy, `EXPENSIVE_ONLY`, with native/Torch ratios
0.109×, 0.225×, 0.382× and 0.643×. That run includes native LLVM codegen,
proved input views, automatic element packing and the explicit 64 KiB local
stack budget. It demonstrates target-dependent resource planning, not a
universal preference for either materialization policy.

### Target-aware widths: GPU and dispatch acceptance

The latest
{download}`width evidence and independent audit <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/notes.md>`
cover 15 FP32 sum/softmax/RMSNorm cases: 17×257, 64×4096, 1024×4096,
7×1537 and 128×8192. A six-width GPU-objective search compares
{32,96,128,256,512,1024}; its reference is the best valid member of the
restricted {32,128,256} subfamily. Both frozen variants retain V=4, P=1 and
U=1. This is neither a full-width optimum nor an old/new default-policy test.

All 240 replayed native/Torch outputs pass in four order-balanced rounds,
with unchanged fingerprinted executable, bridge and TVM compiler/runtime
libraries. GPU values below are **no-counter command-buffer execution
intervals**, not isolated kernel timestamps. Host batches and single-call
dispatch latency are separate uninstrumented phases. Times are medians of
per-round p50s; gains are median paired reference/candidate ratios.

At 1024×4096, all columns below are GPU measurements. Ref/new are the
frozen native reference/candidate; Torch is measured with the candidate.

| Op | Ref µs | New µs | Gain [min, max] | Torch µs |
|---|---:|---:|---:|---:|
| sum | 24.837 | 23.625 | 1.051× [1.031, 1.080] | 26.511 |
| softmax | 67.543 | 59.174 | 1.141× [1.111, 1.157] | 121.056 |
| RMSNorm | 70.910 | 64.210 | 1.101× [1.092, 1.132] | 68.802 |

All three wider-row gains are positive in every pair. Their separate
batched E2E throughput gains are 1.045×, 1.156× and 1.092×. RMSNorm's paired
native/Torch GPU time ratio is 0.931, but single-call GPU medians are
93.417/92.458 µs and E2E medians 316.479/318.855 µs: approximately parity,
not a general dispatch-latency win. Torch's functional RMSNorm retains its
returned-output allocation; sum and softmax use preallocated output on both
sides. The reference-to-candidate comparison uses the same native API and
allocation policy.

The full cohort also contains necessary counterexamples. W=1024 sum at
128×8192 and W=96 softmax at 17×257 were search winners but regress in all
four independent GPU pairs, costing 6.79% and 25.88% more time. Five
identical-plan controls quantify observed variability, including a
0.848--1.131 apparent gain range for the shortest sum control. These are not
confidence intervals or correction factors. The four-round finite cohort
does not justify a universal winner, V default, or fitted cost coefficients.
It does justify retaining an incumbent in independent measured acceptance.

At N=4096/W=1024/V=4 the generated MSL has one straight-line four-element
pack; at W=128 it has eight chunk iterations. Softmax's private stripe also
shrinks from 32 to four scalars. These are real code-shape/resource features,
not evidence of physical register counts or the exact performance cause.
The remaining policy needs memory/issue/collective service and whole-device
subgroup demand, followed by held-out ranking validation. A more elaborate
solver cannot compensate for missing features or noisy timing labels.

### Budgeted immutable-input reuse

The optional `PlannerOptions::cache_reduction_inputs` restores a scheduling
choice that unconditional input forwarding erased: a proved immutable input
Tile used in distinct element/reduction domains may remain materialized.
The existing reduction ownership audit then maps it to worker-private
stripes and charges the same cumulative scalar budget as computed Tiles.
This does not add a DSL entity, change logical execution coordinates, or ask the user
to place input memory manually. The default remains reload/forward.

The view analysis still requires noalias, immutable source/address/guard/fill,
complete initialization, dominance, bounds, and non-escape. It counts distinct
consumer domains, so `x*x` in one recurrence alone does not cache `x`.
Preserved copies carry compiler provenance; manual memory is not relabeled.
The mapping must also prove that every later access belongs to the same
worker. Cross-worker dynamic gathers and over-budget requests fail closed;
enabling this option is an exact request for the reduction mapping family,
not permission to silently fall back to a replicated Tile.

The benchmark switch `--cache-reduction-inputs` and frozen replay preserve
this choice explicitly. The
{download}`complete input-cache evidence <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-input-cache-validation/notes.md>`
contains a fixed W=512, V=4, U=1, P=1 experiment for sum, softmax, RMSNorm,
LayerNorm and residual LayerNorm at five shapes. Four-round counterbalanced
replay validates all 400 outputs; the two unscreened pilots add 100 outputs.
All realized source hashes and complete plans match their frozen pilots.

At 1024×4096 the cache candidate reduces GPU execution time for all three
affected normalizations in every paired round. GPU means no-counter
command-buffer time, not an isolated kernel timestamp. Times are medians of
per-round p50s; gain is the paired ratio median, with observed min–max:

| Op | Reload µs | Cache µs | Gain [min, max] | Torch µs |
|---|---:|---:|---:|---:|
| softmax | 74.198 | 53.949 | 1.378× [1.373, 1.395] | 121.715 |
| RMSNorm | 70.668 | 55.863 | 1.265× [1.246, 1.345] | 69.108 |
| LayerNorm | 79.150 | 64.704 | 1.221× [1.213, 1.251] | 206.598 |

E2E-throughput gains are 1.381×, 1.279× and 1.229× in the same order.
These compare cache/reload at the **same fixed mapping**, not the default
planner, earlier tuned widths or an exhaustive hardware optimum. Torch uses
the recorded eager/output-allocation policy. All 15 changed-source cases
have positive median GPU gains, but RMSNorm 17×257/64×4096 and LayerNorm
17×257 include an individual pair at or below parity. All ten unchanged
sum/residual LayerNorm cases are retained as identical-source controls;
their timing variation is not credited as an optimization.

The unchanged analytic score exposes the next problem concretely: caching
raises the 4096-column RMSNorm score from 64 to 72 because it adds a private
copy traversal, while the measured GPU time falls. It does not distinguish
global from private access service or price eliminated cross-phase input
reads. Cache/reload candidates must not be pruned by that uncalibrated score.
Resource-sensitive features and held-out ranking validation remain necessary;
defaults and cost coefficients are unchanged.

The implementation checkpoint passes 22 new Metal numeric configurations
(including packed-program tails, V=1/V=4, three-way unrolling, non-power-of-two
cooperation and zero-padded input). Tests also reject unsupported targets,
missing noalias, cross-worker gathers and over-budget caches. The full Tile
suite remains 31/33: only the two pre-existing source assertions against the
untouched local `mem_flags(2)` edit fail. Benchmark Python tests pass 84/84.

### Joint resource and execution mapping

The {download}`access-demand and joint-search report
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-access-demand-validation/notes.md>`
adds global/private payload read/write facts to backend-overridable policies
and cache/reload as a staged/JIT Cartesian dimension. The facts are conservative
logical IR demand per program and per longest worker stripe, not physical
DRAM/register traffic. Identical loads count once within an evaluation, not
across phases. Unsupported constructs mark the feature unavailable; optional
access-service coefficients remain zero until calibrated.

The experiment searches W={32,128,256,512,1024} × {reload,cache}, fixing
V=4/U=1/P=1 and the 64-scalar private budget, for softmax/RMSNorm/LayerNorm
at 23×769, 128×2048, 1024×4096 and 128×8193. It retains 101 valid trials,
19 resource/mapping rejections and 12 fresh winner JITs. The reference is the
best valid reload width in that same family, not the default planner. Three
shapes are new relative to the [input-reuse cohort](#budgeted-immutable-input-reuse), but are tuned before acceptance and
therefore are not held-out model validation.

Four frozen, counterbalanced replay rounds validate all 192 outputs; search
and fresh winner measurements validate another 226. Complete plans and source
hashes match their frozen catalogs, with identical binaries/compiler artifacts.
The following values are medians of per-round p50 no-counter GPU
command-buffer times, not isolated-kernel timestamps. Gains are paired-ratio
medians and observed min–max, not confidence intervals.

| 1024×4096 op | Reload GPU µs | Joint GPU µs | GPU gain [range] | E2E gain |
|---|---:|---:|---:|---:|
| softmax | 59.162 | 49.179 | 1.200× [1.195, 1.210] | 1.199× |
| RMSNorm | 64.211 | 52.826 | 1.214× [1.198, 1.221] | 1.221× |
| LayerNorm | 75.660 | 61.316 | 1.234× [1.210, 1.240] | 1.248× |

All anchor pairs improve. Softmax keeps W=1024, LayerNorm keeps W=128;
RMSNorm changes W=1024→256, so its improvement combines width and reuse.
Candidate-run eager Torch GPU medians are 122.653 / 69.742 / 205.799 µs;
native/Torch paired time ratios are 0.401 / 0.761 / 0.297. Torch softmax has
preallocated output, while its functional norms allocate returned outputs.

Seven changed-source cases improve in every GPU pair: these anchors, all
three 128×8193 cases, and LayerNorm 128×2048. Softmax and RMSNorm at
23×769/128×2048 have mixed individual GPU pairs despite positive medians.
All 11 changed-source cases improve in every E2E-throughput pair. The unchanged
23×769 LayerNorm control has 0.924× apparent GPU gain [0.909, 1.023], exposing
measurement variability rather than a code regression. No control is used
as a correction factor, and no universal cache default is inferred.

Independent batch/single-call GPU and E2E phases are retained in the full
report. At 1024×4096 RMSNorm, native/Torch E2E batch time is 53.938/74.218 µs
and E2E single-call latency is 303.354/323.521 µs; GPU single-call time is
71.708/79.979 µs. Their phase medians must not be subtracted to estimate host
overhead. These measurements use TIRx/TVM runtime, not native MPP/MPS or XIR.

The new cost facts expose the actual tradeoff: at N=8193/W=256,
softmax/LayerNorm caching needs 66 private scalars and is rejected; W=512
needs 34 and is legal. RMSNorm admits W=256 with 33 scalars. At N=4096/W=512,
caching adds eight rounds but removes 32 global-read bytes per worker; the
old score cannot reward that service change. The optional resource terms
enable calibrated backend policy, but full-device demand, live state and
independent acceptance still need to guide any future pruning/default.

### Whole-launch policy: shape-held-out gains and small-case failures

The {download}`service-policy report
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-service-policy-validation/notes.md>`
records the first shape-held-out check of a calibrated reduction objective.
`reduction_cost` separates local program work from the complete machine
score, which the solver now minimizes without a hidden second wave multiplier.
An optional typed service model prices subgroup launch demand and the new
payload-access facts; the legacy policy remains default.

The implementation, six-coefficient nonnegative fit and protocol were frozen
in `47314e616` before measuring 37×1537, 256×3072, 768×6144 and 64×12289.
The same three operators run on M1 Max, so this is not an operator/device
holdout. Reference: legacy automatic width with reload. Candidate: automatic
service-policy width plus separately captured reload/cache choices selected
only by model cost. V=4/U=1/P=1 is fixed on both sides. No new timing label is
used to select the candidate, and no per-shape winner table enters the model.

All 96 plan-collection and 192 replay outputs pass executed validation. An
independent audit reconstructs all 32 width scores and verifies exact plans,
source hashes and 21 unchanged compiler/runtime/calibration artifacts. The
four-round GPU results below are no-counter command-buffer execution, **not
isolated-kernel time**. Gains are paired-ratio medians with observed min–max,
not confidence intervals; displayed times are medians of per-round p50s.

| 768×6144 op | Old GPU µs | Model GPU µs | GPU gain [range] | E2E gain |
|---|---:|---:|---:|---:|
| softmax | 78.017 | 57.414 | 1.360× [1.353, 1.364] | 1.372× |
| RMSNorm | 78.097 | 60.753 | 1.287× [1.279, 1.289] | 1.280× |
| LayerNorm | 92.931 | 75.251 | 1.231× [1.218, 1.243] | 1.233× |

All anchor pairs improve; W changes from 384 to 192/256/256 with caching.
Across the full cohort, nine cases improve in every GPU pair and ten in
every E2E-throughput pair. But **37×1537 softmax and LayerNorm regress in every
GPU and E2E-throughput pair**: GPU gains are 0.904× [0.846, 0.932] and
0.876× [0.818, 0.928]. Both change W=192/reload to W=416/cache. Small RMSNorm
GPU is mixed, 0.987× [0.894, 1.043], and its native/Torch time ratio is
1.024 [0.949, 1.070]. These failures prevent promoting the profile to default.
That A/B does not isolate width, reuse or their interaction. The subsequent
fixed 2×2 diagnostic and code-generation repair are recorded in the
[tail-pack section](#tail-packs-a-structural-repair-after-width-reuse-ablation).

Eleven of twelve candidates beat eager Torch GPU throughput in every pair;
all twelve beat its E2E batch throughput. Torch softmax uses preallocated
output, while its functional norms allocate returned output inside timing.
At 768×6144, native/Torch E2E batch times are 58.321/136.194,
62.432/83.321 and 76.873/290.837 µs. Synchronized single-call E2E times are
285.563/439.895, 321.375/338.583 and 340.417/532.854 µs. Large batch gains
do not establish equivalent single-call improvements: only one case improves
E2E single-call latency in every A/B pair. Do not subtract independently
sampled GPU/host medians. Instrumented probe/control throughput ratios span
0.895–4.357 for Torch, so those probe samples remain diagnostic only.

This is a useful whole-launch planning improvement with explicit negative
evidence, not a general reduction scheduler or production-network claim.
The code checkpoint passes 89 Python contracts, 5,988 planner assertions and
the same 31/33 Tile CTest boundary: two untouched local `mem_flags(2)` source
assertion conflicts, with the new execution/numerical tests passing.

### Tail packs: a structural repair after width/reuse ablation

The {download}`fixed 2×2 diagnostic
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-width-cache-ablation/notes.md>`
keeps 192/416 workers and reload/cache separate at 37×1537. Holding reuse
fixed, 416 is slower in every E2E pair for all three operators; caching helps
five of six fixed-width cases in every pair (192-worker softmax is mixed).
All 120 output validations pass. Its GPU-control samples are strongly
variable and are not used to refit the model.

The reduction emitter had guarded every scalar in the final blocked-cyclic
chunk, even when most workers owned full packs. It now separates full worker
packs from the unique partial worker. For remainder `r`, workers below
`floor(r/V)` execute V elements under one guard; at most one additional worker
executes `r mod V` elements. The sets are disjoint, cover the same domain, and
preserve each worker's recurrence and private slots. Collective/barrier
placement, memory/resource policy and planner scores are unchanged.

The {download}`four-round fixed-plan code A/B
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-tail-pack-replay/notes.md>`
retains all twelve service-policy plans and validates all 192 outputs. At
37×1537, source `if` counts fall from 24 to 14 for softmax/LayerNorm and
14 to 8 for RMSNorm. This is a source metric, not an ISA/register claim.
The paired old/new gains below compare the emitter only, **not Torch**.

| 37×1537 operator | E2E batch gain [range] | GPU-control gain [range] |
|---|---:|---:|
| softmax | 1.134× [1.053, 1.250] | 1.157× [1.118, 1.224] |
| RMSNorm | 1.207× [1.188, 1.229] | 1.222× [0.578, 1.260] |
| LayerNorm | 1.210× [1.142, 1.252] | 1.175× [0.508, 1.216] |

Every small-case E2E batch pair improves. GPU norms are mixed, and six
identical-source controls show substantial background variability; every
case/round remains in the linked report. GPU is command-buffer, not isolated
kernel time; ranges are observed min–max, not confidence intervals. Small
softmax/RMSNorm single-call E2E latency remains mixed. Do not compose this
A/B with earlier model ratios to claim that the original regression is closed.

The repair passes 28 new full/partial-pack numerical configurations plus the
existing execution suite, 89 Python contracts and the same 31/33 CTest local
source-assertion boundary. Both C++ files pass selected-database clangd and
changed-line formatting. The model stays opt-in; the next mapping comparison
must use the repaired emitter and stable independent timings before refitting.

### Cooperating-program packing

The mapper can now place several independently cooperating programs in one
threadgroup. The [lowering reference](../../internals/tile/reductions.md#explicit-packing-of-cooperating-programs)
defines the uniform-fence, private-stripe and tail-replay proof. This experiment
asks whether the new freedom helps at a **fixed** 256 workers per row, V=4,
U=1 and cached immutable inputs: reference P=1/T=256 versus P=2/T=512.
Neither variant is an exhaustive or model-selected optimum.

Four counterbalanced fresh-JIT rounds use nine samples, 30 ms windows and
200 ms warm-up. All 192 replay outputs pass full validation; pilots add 48.
The independent audit reconstructs ownership, private/shared resources, access
facts, analytic costs and all four timing phases. It verifies frozen sources,
exact plans, balanced order and 21 unchanged executable/compiler/runtime
artifacts. Pilot timings are excluded from these comparisons.

**The fixed wider grouping usually loses.** Eight of twelve cases regress in
every GPU-throughput pair and nine in every E2E-throughput pair. Only LayerNorm
256×3072 and 768×6144 improve in every GPU pair; the latter has mixed E2E pairs.
The table reports median paired P1/P2 time ratios with observed min–max,
not confidence intervals. Above 1 favors P2. GPU means the no-counter Metal
command-buffer control, not an isolated-kernel timestamp.

| Operator / rows×width | GPU gain [range] | E2E batch gain [range] |
|---|---:|---:|
| softmax 37×1537 | 0.884× [0.824, 0.898] | 0.852× [0.688, 0.949] |
| softmax 256×3072 | 0.935× [0.796, 1.182] | 0.962× [0.880, 0.968] |
| softmax 768×6144 | 0.918× [0.895, 0.928] | 0.917× [0.823, 0.965] |
| softmax 1024×4096 | 0.953× [0.937, 0.964] | 0.972× [0.958, 1.010] |
| RMSNorm 37×1537 | 0.853× [0.480, 0.976] | 0.911× [0.883, 0.928] |
| RMSNorm 256×3072 | 0.981× [0.878, 1.027] | 0.976× [0.973, 0.990] |
| RMSNorm 768×6144 | 0.920× [0.876, 0.938] | 0.935× [0.908, 0.982] |
| RMSNorm 1024×4096 | 0.837× [0.812, 0.885] | 0.813× [0.801, 0.822] |
| LayerNorm 37×1537 | 0.814× [0.597, 0.925] | 0.900× [0.862, 0.956] |
| LayerNorm 256×3072 | 1.058× [1.019, 1.210] | 1.009× [1.003, 1.044] |
| LayerNorm 768×6144 | 1.044× [1.026, 1.346] | 1.031× [0.995, 1.067] |
| LayerNorm 1024×4096 | 0.818× [0.811, 0.838] | 0.815× [0.809, 0.827] |

The incumbent analytic score prefers P2 in all twelve cases because its
setup term is divided by P. The measured losses expose a missing cost of
physical grouping, not another absent warp intrinsic. Reduced occupancy and
cross-program barrier coupling are plausible contributors, **not measured
causes** in this run. The absolute timing regime also changes between rounds;
all balanced pairs are retained, without noise correction or sample pruning.

Ten candidates beat eager Torch GPU throughput in every pair and all twelve
beat its E2E batch throughput, but that does not make P2 better than P1.
Torch softmax uses preallocated output; functional RMSNorm/LayerNorm allocate
returned outputs. Single-call E2E remains separately sampled and mixed.
Instrumented Torch probe/control ratios span 0.762–4.558, so probes are retained
only as diagnostics. No direct MPS/MPP or cross-device comparison is implied.

The full build, 36 new numerical configurations, six typed proof cases and
89 Python tests pass; the full Tile CTest result retains the known 31/33 local
source-assertion boundary. Automatic packing and all coefficients stay
unchanged. A later experiment should compare narrower cooperating/packed
factorizations against the established single-subgroup family at a fixed
total group size before attempting a calibrated grouping model.

Evidence: {download}`protocol <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cooperating-packing/protocol.md>`,
{download}`validation and methods <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cooperating-packing/notes.md>`,
{download}`complete replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cooperating-packing/replay/results.json>`,
{download}`independent audit <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cooperating-packing/audit.py>` and
{download}`audit receipt <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cooperating-packing/audit.json>`.

## What this closes, and what remains

This work closes the specific defect “logical reduction hierarchy is exported
but mechanically scalarized on Metal” for the admitted FP32 row-program
subset. It also demonstrates the intended architecture:

- execution structure is primary;
- execution distribution is a target-chosen map, not a source memory level;
- resource layout follows a proved ownership correspondence;
- a thin mutable semantic IR can feed TVMx without becoming a serialization
  format;
- the target bridge can add specific analyses/passes incrementally; and
- finite analytic planning and staged/JIT measurement compose naturally.

The next honest milestones are:

1. add typed reduction policy for deterministic tree shape, accuracy, NaN and
   signed-zero behavior;
2. extend the atom catalog to FP16/BF16 and pair/tuple reducers such as
   Welford, argmax and online attention state;
3. share target-independent reduction/ownership facts between the TIRx and XIR
   bridges rather than re-deriving them from target IR;
4. repeat the width/reuse comparison after the tail-pack repair, then add
   missing issue/live-state features and validate a revised profile with stable
   timings on a new holdout and another Apple GPU; retain exact JIT overrides;
5. measure cross-entropy backward, decode and prefill attention, Top-K/sort
   and representative end-to-end LLM blocks;
6. add equivalent CUDA and CPU realization families without pretending their
   binding, memory or collective costs are Metal's; and
7. introduce a general Machine TileIR only when multiple backends need the
   same scheduled atom/resource representation and its invariants can be
   stated more cleanly than bridge-local plans.

Until those milestones are measured, the correct claim is narrow but useful:
the TIRx route now has a proof-driven, cost-ranked, high-performance Metal
reduction realization, and the previously measured RMSNorm, LayerNorm and
forward cross-entropy gaps plus the fused residual LayerNorm shared-SSA defect
are closed on the recorded M1 Max cohort.
