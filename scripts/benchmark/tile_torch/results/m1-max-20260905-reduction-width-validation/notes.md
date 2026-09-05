# Target-aware reduction widths: gains and rejected winners

September 5, 2026; Apple M1 Max; FP32; TileIR to TIRx Metal.

## Technical summary

Expanding the legal reduction-width family unlocks useful implementations,
but does **not** make every search winner a better schedule. In a four-round
frozen-plan replay, 1024x4096 sum, softmax and RMSNorm improve no-counter GPU
throughput by **1.051x, 1.141x and 1.101x**, respectively, over a restricted
three-width reference search. Each improvement is positive in all four
rounds. Large RMSNorm now takes 64.210 us versus Torch's 68.802 us in that
GPU phase; the median paired native/Torch ratio is 0.931, or 6.93% less time.
Its separately measured single-call GPU/E2E latency is still approximately
parity. This is not a blanket kernel or dispatch win.

Two other winners reverse in independent replay: 128x8192 sum and 17x257
softmax regress in every round. Five identical-plan comparisons are retained
as noise controls. The repaired target capacity and legal candidate family
are implemented; the default V=1 layout and scalar-round coefficients are
unchanged. No shape-specific winner table is installed as a default policy.

The [raw search](../m1-max-20260905-target-width-search/results.json) and
[raw replay](../m1-max-20260905-target-width-replay/results.json) contain
**442 complete native/Torch output checks**: 202 during search/fresh winner
measurement and 240 during independent replay. Four rejected candidates
produced no measurements and are not counted as validated outputs.

## Comparison scope and measurement contract

The cohort is sum, softmax and RMSNorm at 17x257, 64x4096, 1024x4096,
7x1537 and 128x8192, all device-resident FP32. Both variants use the same
binary, V=4 consecutive elements per worker, U=1 ordered unrolling, and
one logical row per threadgroup. Only exact cooperating width differs.

- Reference: measured best valid width in **{32, 128, 256}**.
- Candidate: measured best valid width in **{32, 96, 128, 256, 512, 1024}**.
- Both frozen catalogs are derived from the same six-width search, before
  replay. They contain unmodified trial rows and selection/source hashes,
  not additional measurements. See [reference](reference-plan.json) and
  [candidate](candidate-plan.json).
- The reference is a deliberately restricted three-width subfamily, **not
  the best of every previously legal width**: 64 is not measured, and the
  old exact-width interface could already request 96 when within its cap.
  The six measured widths also do not exhaust the new 32-width automatic
  family. This is not an old-binary/new-binary default-planner comparison.

Primary GPU values use completed command-buffer GPU timestamps with no
encoder hooks or counter attachments. They include all GPU work/gaps inside
the command buffer, including any encoded non-compute work, and exclude CPU
encoding/completion notification. They are **not individual-kernel time**.
Each GPU phase uses its own recorded repetition count. The counter-instrumented
compute-pass probe remains diagnostic only; neither selection nor gains below
use it. Host throughput and synchronized single-dispatch E2E latency are
separate, uninstrumented phases, not a subtraction-based overhead estimate.

Sum/softmax use preallocated outputs on both sides. Native RMSNorm is
preallocated; Torch's recorded eager functional RMSNorm includes returned
output allocation. JIT, initial upload and final validation/download are
outside all warm timing phases. Torch is the installed MPS implementation,
not a compiled fused graph. MPS/MPP GEMM, XIR/CPU, other dtypes and production
LLM operators are outside this experiment.

## Independent replay: large rows improve, two winners regress

Times are medians of four per-round p50s in microseconds per operation.
Gains are medians of paired **reference/candidate** ratios; brackets are
observed min--max, not confidence intervals. Each case has four complete
pairs. W denotes cooperating threads per row; both sides retain V=4/P=1.

| Changed case | W ref -> candidate | Reference GPU us | Candidate GPU us | Paired GPU gain [range] | Candidate-run Torch GPU us |
|---|---:|---:|---:|---:|---:|
| sum 1024x4096 | 128 -> 1024 | 24.837 | 23.625 | 1.051 [1.031, 1.080] | 26.511 |
| sum 7x1537 | 128 -> 96 | 2.917 | 2.875 | 0.995 [0.896, 1.068] | 4.754 |
| sum 128x8192 | 128 -> 1024 | 8.462 | 9.063 | **0.936 [0.902, 0.987]** | 20.562 |
| softmax 17x257 | 32 -> 96 | 2.695 | 3.363 | **0.796 [0.722, 0.838]** | 14.106 |
| softmax 64x4096 | 256 -> 1024 | 8.307 | 7.972 | 1.048 [0.991, 1.100] | 22.383 |
| softmax 1024x4096 | 128 -> 1024 | 67.543 | 59.174 | 1.141 [1.111, 1.157] | 121.056 |
| rmsnorm 17x257 | 32 -> 96 | 3.203 | 3.196 | 0.989 [0.969, 1.076] | 3.484 |
| rmsnorm 64x4096 | 256 -> 512 | 8.841 | 8.795 | 1.006 [0.957, 1.032] | 9.601 |
| rmsnorm 1024x4096 | 128 -> 1024 | 70.910 | 64.210 | 1.101 [1.092, 1.132] | 68.802 |
| rmsnorm 7x1537 | 256 -> 1024 | 4.330 | 4.034 | 1.037 [1.018, 1.099] | 4.036 |

The three 1024x4096 gains survive all four order-balanced pairs. Their
batched E2E gains are 1.045x, 1.156x and 1.092x. Conversely, sum 128x8192
and softmax 17x257 have paired candidate/reference GPU ratios of 1.068 and
1.259: **6.79% and 25.88% more GPU time**, respectively. Their batched E2E
results also regress. Independent acceptance must retain a known incumbent;
selecting a search minimum is insufficient.

Small RMSNorm 7x1537 is positive in all four GPU pairs, but only 1.003x in
E2E throughput, and its native/Torch GPU ratios span 0.854--1.117. It does
not establish a Torch win. Softmax 64x4096 and the other changed RMSNorm
shapes have GPU gain ranges crossing one. They do not establish a consistent
width benefit either. Per-round records and all host/single-call tables are
in the [complete replay report](../m1-max-20260905-target-width-replay/results.md).

### Same-plan controls bound the interpretation of small differences

The following rows selected identical widths and have identical generated
source hashes on both sides. Any measured difference is not an optimization.
This exact lookup is why the report uses tables rather than a ranking chart
that could make small noisy ratios look like independent wins.

| Control case | Fixed W | Reference GPU us | Candidate GPU us | Paired apparent gain [range] |
|---|---:|---:|---:|---:|
| sum 17x257 | 32 | 2.569 | 2.712 | 0.919 [0.848, 1.131] |
| sum 64x4096 | 128 | 4.156 | 4.111 | 1.024 [0.975, 1.042] |
| softmax 7x1537 | 128 | 4.122 | 4.143 | 0.990 [0.986, 1.011] |
| softmax 128x8192 | 256 | 22.168 | 22.347 | 0.999 [0.986, 1.024] |
| rmsnorm 128x8192 | 128 | 21.342 | 21.324 | 0.999 [0.990, 1.002] |

The shortest sum control spans 0.848--1.131; other controls are tighter.
Four rounds do not estimate a confidence interval or eliminate background
load, clocks, cache state and process-level dispatch variability. Do not
reuse this spread as a universal correction factor or noise threshold.

### Throughput is not single-call latency

At 1024x4096 RMSNorm, GPU throughput improves relative to both the
restricted reference and Torch. Single-call values tell a narrower story:

| RMSNorm 1024x4096, separate phase | Native us | Candidate-run Torch us |
|---|---:|---:|
| No-counter GPU throughput, amortized per op | 64.210 | 68.802 |
| Uninstrumented host throughput, amortized per op | 66.326 | 73.113 |
| No-counter single-call GPU latency | 93.417 | 92.458 |
| Synchronized single-call E2E latency | 316.479 | 318.855 |

The single-call GPU medians are effectively parity/slightly slower for
native, and E2E medians are approximately parity. Separate-phase medians
must not be subtracted to infer a precise CPU-overhead value. These are
TIRx runtime dispatches, not measurements of Luisa native Metal launch.

## Structural findings

The test/benchmark Runtime previously selected Metal with only its 32-lane
subgroup attribute. TVM's installed target definition supplied the remaining
default `max_num_threads=256`. Independently, the bridge restricted its
reduction family to eight subgroups and automatic powers of two. Thus legal
96-thread candidates were omitted automatically, and 512/1024-thread
candidates were rejected before device compilation.

The Runtime now queries `DeviceAPI::kMaxThreadsPerBlock` and forwards the
result as both Metal target thread-capacity attributes. The native benchmark
records it as `metal_max_threads`. The installed Metal runtime implements
that query with the device's `maxThreadsPerThreadgroup`; a compiled pipeline
may still impose a tighter resource-dependent constraint.

The bridge enumerates every whole-subgroup cooperating width through
`min(32, target_max_threads/32)`. The 32-subgroup cap is algorithmic: the
second collective assigns one lane to each partial. Packed independent rows
retain the existing separate 1..8-program family, so there are at most 39
automatic candidates. Search-budget exhaustion fails explicitly; exact
widths do not need the full automatic budget. The reduction-tree numerical
permission, ownership proofs, resource bounds and no-fallback exact requests
are unchanged.

## Cost-policy features and remaining work

Backend policies now receive physical threadgroup count, useful scalar
element count and useful lane-work fraction, in addition to worker width,
packing, consecutive elements, scalar rounds and private/shared storage.
Plans/JSON expose corresponding facts. They are not measured occupancy.
The default scalar-round coefficients are unchanged: this checkpoint repairs
the admissible family and target information, not its active-group/issue cost
model. Widths alone do not prove a performance win.

## Verification

Full selected-tree builds succeeded. CPU/Metal execution tests pass,
including 14 new V=1/4 softmax cases covering automatic mapping and exact
96/160/224/288/512/1024-thread layouts with ragged tails. Independent ownership
counts check private storage and useful-lane fractions; a custom backend
policy proves every legal width is presented, including non-powers of two.
Budget exhaustion and over-limit exact requests fail closed.

The old auto-layout assertions are replaced with independent exhaustive
evaluation of the documented objective. A separate matrix reference fixture
now explicitly requests its original 256 threads so it continues exercising
the atom-wave loop on devices supporting 1024 threads; its source/numerical
assertions are retained. Benchmark metadata tests pass **83/83**. The final
full Tile rerun passes **31/33** in 101.98 seconds; the only failures are the
two pre-existing cooperative/memory source-assertion conflicts with the user's
unowned `mem_flags(2)` edit. Their numerical checks pass. See [full log](tests.log).
The local edit and those assertions were not changed or submitted. Project
clangd checks pass for all four changed translation units (both shared headers
are checked through their consumers).

The updated repository-native report also passes a full Sphinx HTML build.
Only the two pre-existing unrelated toctree warnings remain. The rendered
reduction chapter was inspected in the in-app browser at its default laptop
viewport; the new table headers were shortened so all timing columns fit.
This is document QA, not additional benchmark or mobile-browser validation.
Report updates preserve the existing Markdown/Sphinx delivery, with scope,
methods, uncertainty, controls and next questions adjacent to the evidence.

## Reproduction and evidence audit

The search starts from code commit `62bac524e1ddd82432189a90c85af8c12b47f172`.
Search and replay use the same executable and adjacent Tile libraries. Replay
also fingerprints the installed TVM compiler/runtime libraries before and
after measurement; `artifacts_unchanged=true`. This is the local worktree
snapshot containing the untouched `mem_flags(2)` edit noted above, not a
clean submitted-source build. A commit alone is insufficient provenance.

| Artifact | SHA-256 |
|---|---|
| Native benchmark | `8140c9109e35c5e7e2b0a16d666cb8794c333b3625c1fea27d25110c0f759a2d` |
| TIRx bridge | `256dadacc8d9b3f3a8c0a9f2d128b87aa0f39cd114205c09ce612bc1ada61cf3` |
| Metal timing helper | `eb568b9177926113e8af2ea1ec995b9c90c13623fc1e73d483864c0b8217eae0` |
| TVM compiler | `44a277c13f8400925b6eb7170148b0c0e03ca727a70d29f33713d0bc0c8d5c89` |
| TVM runtime | `bf5e2c96f4f946e27d590f99d859860d3c6a8a7803b20bbd17cd3534bb759f9d` |
| TVM Metal runtime | `83dc06e34c4a531f0eb0ab1912579c8ee7c2a05f2736d7cdfd7d7d9ce700ac60` |

The installed Torch is 2.14.0, commit
`08187d9e0fba026dc8217405802ab5381dc88d90`; installed TVM source is
`c7b458e946bc4266915da582457476bdcd9705ae`.
Search has 5 samples/10 ms batches/100 ms warmup per configuration. Replay
has four rounds, 9 samples/30 ms batches/200 ms warmup. Case order rotates;
each case gets each A/B order twice, and each variant gets native-first and
Torch-first twice. All 120 replay measurements are freshly captured/JIT
compiled, checked against independent FP64 formulas, and source-hashed.
No search parameters change during replay.

The standard-library [audit script](analyze.py) independently recomputes all
GPU and host p50s from raw samples, checks each phase's own denominator,
selection scores and frozen catalog provenance, full cohort coverage, exact
W/V/P/U and physical-plan metadata, source hashes and balanced ordering.
It validates 86 accepted trials plus 15 fresh winners and 120 replay rows.
The four resource-rejected trials are softmax W=32 at 64x4096/1024x4096 and
W=32/96 at 128x8192: their required worker-private stripes exceed 64 scalars.
It prints paired gains and native/Torch ratios without importing the
benchmark's selection or statistics helpers.
Full-array numerical comparisons were executed by the benchmark harness;
temporary output buffers were not archived. The offline audit checks their
recorded validation results and cannot rerun those array comparisons without
executing the kernels again.

Commands below were run from the repository root. Output directories and
plan files are creation-only: use fresh paths for a new experiment and
update the audit's constants when intentionally defining a new cohort.

```sh
cmake --build cmake-build-tirx -j8

uv run --no-project --python 3.13 --with torch --with numpy python scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-target-width-search \
  --backends metal --operations sum,softmax,rmsnorm \
  --row-shapes 17x257,64x4096,1024x4096,7x1537,128x8192 \
  --metal-subgroup-reductions --reduction-lane-elements 4 \
  --tune-group-threads 32,96,128,256,512,1024 \
  --tuning-metric gpu-control --max-tuning-candidates 6 \
  --samples 5 --sample-ms 10 --warmup-ms 100 \
  --metal-device-timing cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib \
  --capture-sources

python3 scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/analyze.py select

uv run --no-project --python 3.13 --with torch --with numpy python scripts/benchmark/tile_torch/repeat.py \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/reference-plan.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/candidate-plan.json \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-target-width-replay \
  --operations sum,softmax,rmsnorm --rounds 4 --samples 9 --sample-ms 30 --warmup-ms 200 \
  --metal-device-timing cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib \
  --compiler-artifact cmake-build-tirx/bin/libluisa-tile-bridge-tirx.dylib \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_compiler.dylib \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_runtime.dylib \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_runtime_metal.dylib \
  --capture-sources

python3 scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/analyze.py audit
```

## Next model and solver work

1. Keep a verified incumbent in measured selection, and independently accept
   a proposed winner before caching/promoting it. The two reversing search
   winners above are concrete counterexamples to minimum-only promotion.
2. Price whole-device subgroup demand and separate memory/issue/collective
   service. The current per-program scalar-round score does not model how
   width changes active-group supply or issued memory operations. Queried
   thread capacity is not an occupancy model; useful lane work is not a
   measured issue rate.
3. Treat full-pack code shape and tail utilization as candidate features.
   At 4096 elements, W=1024/V=4 emits one straight-line pack; W=128/V=4
   retains eight stripe chunks. Softmax's compiler-owned stripe shrinks from
   32 to 4 scalars per worker. The archived
   [W=128 MSL](../m1-max-20260905-target-width-search/sources/50854affc840ae96f481a126d8f69e4aba49297753032133e47cacecd44a7615.metal)
   and [W=1024 MSL](../m1-max-20260905-target-width-search/sources/7f843b26d10e0c35678cab6dd4c6aec72f3ad4a26aa3dfc60b99a17837b2c50c.metal)
   show this difference, but do
   not prove physical register allocation, vector transactions or the cause
   of every measured improvement.
4. Evaluate any revised backend policy on held-out shapes/operators with
   both timing scopes. This cohort is search/diagnostic evidence, not a
   held-out model benchmark. The bounded family still needs only enumeration;
   a more complex solver cannot fix missing cost features or noisy labels.

Open questions are the crossover between narrow and wide rows, remaining
memory traffic/alias-driven reloads, and TIRx single-dispatch overhead. None
requires a kernel-specific public DSL entity or a change to Tile semantics.
