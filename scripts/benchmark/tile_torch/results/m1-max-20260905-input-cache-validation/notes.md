# Immutable input reuse in Metal reductions

## Technical summary

The TIRx bridge now has an explicit, default-off scheduling choice to keep
immutable input values across reduction phases. It reuses the existing
worker-private ownership proof and cumulative stripe budget instead of
requiring manual memory in the Tile DSL. Implementation and regression
evidence were committed as `bfe14c675`.

The independent four-round, 25-case replay confirms **1.378×, 1.265× and
1.221× GPU-throughput gains** for 1024×4096 softmax, RMSNorm and LayerNorm
at the same W=512/V=4/U=1/P=1 mapping. Their E2E-throughput gains are
1.381×, 1.279× and 1.229×. All four paired rounds improve for these cases.
The 15 changed-source cases all have positive median GPU gains, but three
have an individual pair at or below parity. Ten identical-source controls
are retained, not advertised as improvements. All 400 replay outputs and
100 pilot outputs pass executed validation; the independent timing/source
audit passes. No default, cost coefficient or winner table has been changed.

## Fixed-width GPU and dispatch results

GPU here is **uninstrumented command-buffer execution**, not an isolated
kernel timestamp. The reference and candidate have the same W=512/V=4/U=1/P=1
mapping; only input caching changes. Values below are medians of four
per-round p50s. Gains are medians of the four paired ratios, which need not
equal a ratio of the displayed time medians. Brackets are observed min–max,
not confidence intervals. Independent E2E phases include dispatch.

| 1024×4096 operation | Reload GPU µs | Cache GPU µs | GPU gain [range] | E2E gain |
|---|---:|---:|---:|---:|
| softmax | 74.198 | 53.949 | 1.378× [1.373, 1.395] | 1.381× |
| RMSNorm | 70.668 | 55.863 | 1.265× [1.246, 1.345] | 1.279× |
| LayerNorm | 79.150 | 64.704 | 1.221× [1.213, 1.251] | 1.229× |

These are gains over the fixed-width **reload implementation**, not over the
old planner's default or previously tuned W=1024 candidates. In the same
candidate runs Torch GPU throughput is 121.715, 69.108 and 206.598 µs;
paired native/Torch time ratios are 0.444, 0.807 and 0.313 respectively.
Torch is the eager operator/allocation policy described below, not a fused
compiled graph. No cross-cohort comparison is used to claim the gains.

All other changed-source cases are retained here to expose sensitivity:

| Operation / shape | GPU gain [range] | E2E gain [range] |
|---|---:|---:|
| softmax 17×257 | 1.082× [1.037, 1.143] | 1.163× [1.148, 1.177] |
| softmax 7×1537 | 1.058× [1.010, 1.111] | 1.160× [1.035, 1.183] |
| softmax 64×4096 | 1.099× [1.053, 1.125] | 1.072× [1.041, 1.106] |
| softmax 128×8192 | 1.246× [1.223, 1.255] | 1.251× [1.226, 1.262] |
| RMSNorm 17×257 | 1.092× [0.999, 1.123] | 1.077× [0.983, 1.100] |
| RMSNorm 7×1537 | 1.138× [1.066, 1.277] | 1.114× [1.104, 1.141] |
| RMSNorm 64×4096 | 1.074× [0.972, 1.100] | 1.086× [1.034, 1.121] |
| RMSNorm 128×8192 | 1.090× [1.017, 1.133] | 1.102× [1.046, 1.126] |
| LayerNorm 17×257 | 1.253× [0.977, 1.380] | 1.192× [0.991, 1.239] |
| LayerNorm 7×1537 | 1.168× [1.070, 1.313] | 1.136× [1.123, 1.143] |
| LayerNorm 64×4096 | 1.054× [1.033, 1.149] | 1.082× [1.037, 1.130] |
| LayerNorm 128×8192 | 1.176× [1.173, 1.182] | 1.203× [1.150, 1.217] |

RMSNorm 17×257/64×4096 and LayerNorm 17×257 have mixed individual GPU
pairs; the smallest RMSNorm and LayerNorm also have a slightly negative
E2E pair. This finite fixed-width cohort does not establish a universal cache
default or exclude regressions at narrower widths with larger private state.

The identical-source controls quantify variation without pretending to
estimate a universal noise correction. Their GPU apparent gains are:

| Control / shape | Median apparent gain | Observed range |
|---|---:|---:|
| sum 17×257 | 0.919× | [0.855, 1.091] |
| sum 7×1537 | 1.010× | [0.908, 1.083] |
| sum 64×4096 | 0.987× | [0.960, 1.096] |
| sum 1024×4096 | 0.993× | [0.959, 1.021] |
| sum 128×8192 | 1.010× | [0.965, 1.188] |
| residual LayerNorm 17×257 | 1.037× | [0.870, 1.042] |
| residual LayerNorm 7×1537 | 0.980× | [0.915, 1.008] |
| residual LayerNorm 64×4096 | 0.953× | [0.857, 1.004] |
| residual LayerNorm 1024×4096 | 0.988× | [0.909, 1.086] |
| residual LayerNorm 128×8192 | 1.006× | [0.990, 1.037] |

Single-call latency is a separate acceptance dimension. At 1024×4096,
candidate/Torch single-call GPU and E2E medians are respectively
69.354/130.812 and 295.792/479.355 µs for softmax;
60.646/87.604 and 324.417/341.459 µs for RMSNorm; and
82.375/204.708 and 325.396/459.771 µs for LayerNorm. These separate-phase
medians are not subtractable CPU-overhead estimates. Small RMSNorm 17×257
and 64×4096 cases have approximately E2E parity with Torch despite lower GPU time.
The complete 25-case GPU, E2E-throughput and single-call tables are in the
sibling replay's `results.md`; all paired raw samples are in `results.json`.

## What the transformation actually preserves

The prior pipeline correctly proved that input snapshots could be forwarded
to immutable external reads, but always erased the option to keep those
values across subsequent element/reduction traversals. Input caching retains
only audited snapshots with at least two distinct consumer domains.
Occurrences inside one domain, such as the two operands of `x*x`, are not a
reason to retain an input stripe.

```text
immutable input ── once per owned element ──► worker-private stripe
                                                  │
                            ┌─────────────────────┴──────────────────┐
                            ▼                                        ▼
                       reduction phase                       later element map
                            │                                        │
                            └──────── collective scalar ─────────────┘

execution: e = (chunk * W + worker) * V + v
resource:  stripe_slot = chunk * V + v
```

At fixed W/V/P, execution coordinates, numerical recurrence, barriers and ABI
are unchanged. Automatic width selection can change when resource demands
change; that is not the comparison performed here.
No alias freedom is inferred from a pointer: the caller must supply noalias.
The existing whole-function audit proves immutable source, address, guard
and fill; complete initialization; lexical dominance; bounds and non-escape.
The reduction audit separately requires same-worker access and charges every
retained stripe against the aggregate 64-scalar default budget. A dynamic
gather that can cross workers or an over-budget cache is rejected, not
silently reloaded or materialized per worker in full. Manual memory is never
marked as a compiler-pure input snapshot.

At N=4096/W=512/V=4, the retained RMSNorm input is an eight-float private
array used by both the sum-of-squares phase and output map. Softmax and
LayerNorm add one such input array to their existing eight-float computed
stripe. Their planned private totals are therefore 8, 16 and 16 respectively.
Sum has no cross-phase input use; residual LayerNorm already retains its
computed residual sum. All five shapes of these two operators generate
identical source under both settings and serve as controls.

These are IR/source-level resource and traversal facts, not hardware register
counts, a proof of DRAM transaction counts, or a claim that the Metal compiler
cannot hoist/CSE other repeated expressions. In particular, denominator math
is unchanged. The installed TVMx CSE pass excludes expressions containing
calls or buffer loads; the downstream Metal optimizer may still optimize
them. This experiment does not attribute a measured gain to denominator
hoisting or physical occupancy.

## Scope and measurement definitions

- Device: Apple M1 Max, macOS 26.6.2; FP32, device-resident data.
- Operations: sum, softmax, RMSNorm, LayerNorm and residual LayerNorm.
- Shapes: 17×257, 7×1537, 64×4096, 1024×4096 and 128×8192.
- Fixed realization: W=512 cooperating workers, V=4 consecutive elements,
  U=1 chunk unrolling, P=1 program per threadgroup. These are **not** claimed
  optimal widths, the default planner, or the earlier width-search winners.
- Reference: input forwarding/reloading. Candidate: the same executable,
  same mapping, plus `cache_reduction_inputs=true`. No parameter search.
- Primary GPU metric: no-counter command-buffer GPU throughput time, using
  each phase's own repetitions. It includes GPU work/gaps and any blits
  inside the command buffer, excludes CPU encoding/completion notification,
  and is **not an individual-kernel timestamp**.
- E2E throughput and synchronized single-call E2E latency are measured in
  separate uninstrumented host phases. Single-call GPU timestamps are also
  retained. Do not subtract unrelated phase medians to estimate CPU cost.
- Instrumented compute-pass probes remain diagnostic only. They neither
  select configurations nor supply a correction factor for GPU times.
- Native is the TIRx/TVM runtime route, not the native Luisa Metal runtime.
  Torch is the recorded eager operator sequence, not a compiled fused graph.
  Native outputs are preallocated; Torch sum/softmax also use preallocated
  outputs, while normalization uses the functional API's allocation behavior.
- JIT, upload, validation and download are outside warm timing. Complete
  output arrays are checked by the harness against an FP64 oracle. No CPU
  fallback is enabled. No MPS/MPP/XIR, other dtype or other-device conclusion
  follows from this cohort.

## Experiment and provenance

The two sequential pilots contain all 25 configurations each, with five
samples, 10 ms batches and 100 ms warmup. The entire pilot reports become
frozen input catalogs, without filtering by gain. Their timings do not enter
the replay's summary statistics.

The acceptance run uses four rounds, nine samples, 30 ms batches and 200 ms
warmup per measurement. It balances reference/candidate order and
native/Torch-first order independently, rotates case order, and freshly
captures/JIT-compiles each measurement. Times are medians of per-round p50s;
gains are medians of paired reference/candidate ratios. Min–max across four
ratios is a finite observed range, not a confidence interval.

The pilots ran on the work-in-progress source after `5091de036`. Before
acceptance, one rejection diagnostic string changed and a padded-input test
was added, followed by a full build and `bfe14c675`. Consequently pilot and
replay library hashes need not match. The independent audit requires every
realized source hash and complete plan to match its frozen pilot, and both
acceptance variants to use the same unchanged fingerprinted artifacts.

Raw sources and results are in sibling directories:

- `m1-max-20260905-input-cache-reference/`
- `m1-max-20260905-input-cache-candidate/`
- `m1-max-20260905-input-cache-replay/`

`analyze.py` independently recomputes GPU and E2E medians from raw samples,
checks all requested cache/geometry/resource fields, full source hashes,
catalog identity, complete coverage and counterbalancing, and prints paired
GPU/E2E ratios and single-call comparisons. It uses only the standard library,
not benchmark statistics helpers. Raw numeric arrays are not archived; the
offline audit validates recorded executed correctness, not the arrays again.

## Regression validation

The full selected CMake tree built successfully before tests. New coverage
executes 20 softmax configurations (five sizes × two V values × cache on/off),
one zero-padded generic statistic and one single-domain sum-of-squares case.
It includes three-way ordered unrolling, non-power-of-two cooperative width,
three packed programs with a tail, and independent FP64 softmax checks.
Separate rejection checks cover unsupported policy, noalias omission,
cross-worker gather, and cumulative stripe budget exhaustion.

The full Tile suite is **31/33** in 110.64 seconds. The only failures are the
pre-existing source assertions in `test_tirx_cooperative.cpp:100` and
`test_tirx_memory.cpp:168` against the untouched local `mem_flags(2)` change.
No failing test is skipped or edited to pass. Benchmark Python tests pass
**84/84**. clangd checks pass for views, compiler, benchmark and execution-test
translation units. An initial Python invocation without NumPy could not
import the MPP comparison tests; rerunning with its dependency passed all 84.

## Next decisions and open questions

Keep the independently measured candidate available, but default-off.
Profitable input reuse must eventually be ranked jointly with cooperation
width, element packs, private live state and memory/issue service. This
fixed-width experiment is not held-out calibration of that cost model.
Cache lifetime compaction and mixed cache/reload choices per input remain
future work; current admission conservatively sums all stripe allocations.

The old score has a concrete blind spot here. At N=4096/W=512 it charges
RMSNorm 64 relative units without caching and 72 with caching. Softmax and
LayerNorm similarly rise from 112 to 120. It counts an extra input-copy
domain's scalar rounds but does not price the removed cross-phase global
reads or distinguish private/global service. A cache/reload search should
not use that uncalibrated score to prune the new candidate. This observation
is not a fitted replacement model: effects, unique same-phase reads,
register/live-state pressure and full-device scheduling still need their
own features and held-out validation.

The report uses the repository's existing Markdown/Sphinx technical surface.
Exact multi-metric case lookup, raw paired ranges and identical-source
controls are the evidence format; no time-series or regression chart is
implied by five deliberately chosen shapes. Definitions precede results;
methods, uncertainty, regression evidence and open questions remain visible.
Sphinx HTML builds successfully with only the two existing unrelated
not-in-toctree warnings (`custom_agility_sdk.md` and
`coro_suspend_extensions.md`). The rendered reduction section and all five
headline table columns were visually checked at the default 1280×720 browser
viewport; no mobile or alternate-breakpoint certification is implied.

## Reproduction

From the repository root, after a full build (each output directory must be
new):

```sh
cmake --build cmake-build-tirx -j 8
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-input-cache-reference \
  --backends metal --operations sum,softmax,rmsnorm,layernorm,residual_layernorm \
  --row-shapes 17x257,7x1537,64x4096,1024x4096,128x8192 \
  --metal-subgroup-reductions --group-threads 512 --reduction-lane-elements 4 \
  --samples 5 --sample-ms 10 --warmup-ms 100 \
  --metal-device-timing cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib \
  --capture-sources
```

Repeat that pilot command with `--cache-reduction-inputs` and output suffix
`input-cache-candidate`, then:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260905-input-cache-reference/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260905-input-cache-candidate/results.json \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output scripts/benchmark/tile_torch/results/m1-max-20260905-input-cache-replay \
  --operations sum,softmax,rmsnorm,layernorm,residual_layernorm \
  --rounds 4 --samples 9 --sample-ms 30 --warmup-ms 200 \
  --metal-device-timing cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib \
  --compiler-artifact cmake-build-tirx/bin/libluisa-tile-bridge-tirx.dylib \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_compiler.dylib \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_runtime.dylib \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_runtime_metal.dylib \
  --capture-sources
uv run --no-project --python 3.13 python \
  scripts/benchmark/tile_torch/results/m1-max-20260905-input-cache-validation/analyze.py
```
