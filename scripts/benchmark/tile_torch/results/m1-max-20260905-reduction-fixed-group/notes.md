# Fixed group size does not make a universal reduction mapping

## Result

On September 5, 2026, Apple M1 Max FP32 Metal, the predeclared S=2/P=4/T=256
candidate beats a fixed S=1/P=8/T=256 control in every GPU-throughput pair
for four of twelve cases. Against the **current automatic execution family**,
it improves only two cases in every GPU/E2E-throughput pair and regresses
eight in every pair. The two improvements are many short rows: softmax and
LayerNorm at 16384×257. This is not evidence for an unconditional S=2/P=4
default. No implementation, coefficient or automatic policy changed.

All **456 executed complete-output validations** pass: 72 pilot outputs plus
192 in each of two independent four-round replays. The independent
[audit](audit.py) and [receipt](audit.json) reconstruct all 39 automatic
candidates for each case, ownership/resource/access facts, selected costs,
source identities, timing samples, round balance and 21 unchanged artifacts.
The audit checks the runner's full-output validation records; output arrays
were checked during execution but were not archived.

## Scope and comparison definitions

The [protocol](protocol.md) was committed as `b288baf2a` before timing.
All three variants use input views, opt-in immutable input caching, V=4,
ordered U=1, preserved shared Tile SSA and the existing 64-scalar private
bound. S is cooperating SIMD groups per program, P independent programs per
group, W=32*S workers per program and T=W*P total threads per group.

The automatic baseline keeps the unchanged analytic solver and its established
candidate family, **with caching enabled**, not all-default CompileOptions.
For widths 769/1024/257/1024 it selects S7/P1, S8/P1, S3/P1, S8/P1 for
softmax/LayerNorm; RMSNorm differs only at width 257, selecting S1/P8. Every
one of the 39 candidates is legal in this cohort. The audit explicitly
enumerates element ownership rather than copying planner feature functions.

Below, gain means **baseline time / candidate time** within each round;
above 1 favors the candidate. Values are medians of four paired ratios with
observed min/max, not confidence intervals or ratios of displayed medians.
GPU denotes the no-counter completed-command-buffer control, not an isolated
kernel timestamp. E2E includes warm host dispatch and amortized synchronization.
The experiments remain separate; do not pool their timings or multiply gains.

## Fixed-T256 comparison

The analytic model prefers S2/P4 over S1/P8 in all twelve cases. Measurements
instead show four all-positive, two all-negative and six mixed GPU cases.
The two 1024-wide softmax cases consistently lose; reducing per-worker work
does not by itself offset communication and grouping costs.

| Operator / rows×width | GPU gain [range] | E2E batch gain [range] |
|---|---:|---:|
| softmax 37×769 | 1.132× [1.051, 1.192] | 1.138× [1.093, 1.192] |
| softmax 1024×1024 | 0.862× [0.722, 0.923] | 0.873× [0.816, 0.881] |
| softmax 16384×257 | 0.997× [0.963, 1.085] | 1.020× [0.986, 1.071] |
| softmax 4096×1024 | 0.929× [0.869, 0.970] | 0.920× [0.800, 0.954] |
| RMSNorm 37×769 | 1.188× [1.078, 1.530] | 1.165× [0.933, 1.191] |
| RMSNorm 1024×1024 | 1.026× [0.993, 1.081] | 1.063× [1.014, 1.115] |
| RMSNorm 16384×257 | 1.030× [0.997, 1.149] | 0.957× [0.826, 0.996] |
| RMSNorm 4096×1024 | 1.055× [1.018, 1.130] | 1.055× [0.995, 1.068] |
| LayerNorm 37×769 | 1.161× [1.079, 1.200] | 1.143× [1.107, 1.191] |
| LayerNorm 1024×1024 | 0.998× [0.792, 1.098] | 0.969× [0.924, 1.010] |
| LayerNorm 16384×257 | 0.985× [0.931, 1.075] | 0.996× [0.955, 1.023] |
| LayerNorm 4096×1024 | 1.048× [0.923, 1.129] | 0.979× [0.941, 1.022] |

E2E has three all-positive, three all-negative and six mixed cases. In
particular, the slightly positive RMSNorm 16384×257 GPU median does not imply
an E2E benefit. Raw source: [fixed replay](fixed-replay/results.json).

## Comparison against automatic execution

The unchanged automatic family is the stronger practical control. Softmax and
LayerNorm at 16384×257 gain 1.220×/1.210× GPU throughput with S2/P4;
eight cases consistently lose and two have mixed pairs. An explicit family
extension is useful, but making it expressible is not enough to rank it well.

| Operator / rows×width | GPU gain [range] | E2E batch gain [range] |
|---|---:|---:|
| softmax 37×769 | 0.716× [0.666, 0.726] | 0.683× [0.664, 0.701] |
| softmax 1024×1024 | 0.872× [0.852, 0.916] | 0.890× [0.885, 0.911] |
| softmax 16384×257 | 1.220× [1.208, 1.277] | 1.236× [1.233, 1.258] |
| softmax 4096×1024 | 0.903× [0.882, 0.933] | 0.910× [0.905, 0.922] |
| RMSNorm 37×769 | 0.635× [0.584, 0.662] | 0.648× [0.643, 0.650] |
| RMSNorm 1024×1024 | 0.896× [0.863, 0.906] | 0.884× [0.864, 0.885] |
| RMSNorm 16384×257 | 0.992× [0.978, 1.016] | 0.999× [0.964, 1.022] |
| RMSNorm 4096×1024 | 0.923× [0.917, 0.928] | 0.949× [0.902, 0.963] |
| LayerNorm 37×769 | 0.667× [0.643, 0.723] | 0.687× [0.649, 0.704] |
| LayerNorm 1024×1024 | 0.959× [0.918, 0.993] | 0.975× [0.932, 0.981] |
| LayerNorm 16384×257 | 1.210× [1.182, 1.230] | 1.187× [1.163, 1.208] |
| LayerNorm 4096×1024 | 0.996× [0.974, 1.036] | 1.007× [0.977, 1.021] |

In this independent replay, automatic execution beats eager Torch GPU and
batched E2E throughput in every pair for all twelve cases. The explicit
candidate loses to Torch GPU throughput in every pair for RMSNorm 37×769
(paired candidate/Torch ratio 1.294×), even though its E2E remains faster.
The automatic RMSNorm GPU ratio is 0.772×. Do not conflate beating a fixed
native control with beating either the automatic compiler or a library.
Raw source: [automatic replay](automatic-replay/results.json).

## Single-call timings and robustness

The receipt retains all per-round absolute medians and paired ratios for
GPU throughput, E2E throughput, GPU single-call time and synchronized E2E
single-call latency. Counts below summarize latency without hiding its mixed
results; “mixed” means the four ratios straddle 1.

| Experiment / single-call metric | All pairs improve | All pairs regress | Mixed |
|---|---:|---:|---:|
| Fixed-T256 / GPU | 2 | 1 | 9 |
| Fixed-T256 / E2E | 2 | 0 | 10 |
| Automatic / GPU | 0 | 5 | 7 |
| Automatic / E2E | 1 | 1 | 10 |

Some absolute timing regimes shift between rounds, including both native and
Torch in the fixed-control replay. Keep all paired data; no clock/thermal
cause is asserted without telemetry. Instrumented Torch probe/control ratios
span 0.531–6.777 in the fixed experiment and 0.861–5.915 in the automatic
experiment. Probes remain diagnostics only. Do not subtract independently
measured GPU and host medians to estimate dispatch overhead.

Torch softmax uses preallocated `out=`. Functional RMSNorm and LayerNorm
return allocated tensors in timing; native output is preallocated. All are
eager FP32 device-resident API comparisons with unchanged math policies and
operator tolerances, not direct MPS/MPP or compiled-Torch kernel comparisons.
CPU/CUDA, other dtypes, arbitrary data distributions and production LLM blocks
are not covered by this experiment.

## Reproduction and artifact boundary

The pilots (`reference/`, `candidate/`, `automatic/`) use the exact protocol
parameters with `run.py`, `--metal-subgroup-reductions --input-views
--cache-reduction-inputs --reduction-lane-elements 4 --capture-sources`.
Their three-sample, 10 ms/100 ms host timings only freeze plans. The reference
adds `--group-threads 256 --reduction-programs-per-group 8`; the candidate
uses packing 4; automatic requests zero for both controls.

Each replay uses:

```sh
uv run --no-project --python 3.13 --with numpy --with torch python \
  scripts/benchmark/tile_torch/repeat.py \
  --reference BASELINE/results.json --candidate candidate/results.json \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --metal-device-timing cmake-build-tirx/bin/libluisa-benchmark-metal-timing.dylib \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_compiler.dylib \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_runtime.dylib \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm/lib/libtvm_runtime_metal.dylib \
  --compiler-artifact /tmp/luisa-tvmx-venv/lib/python3.14/site-packages/tvm_ffi/lib/libtvm_ffi.dylib \
  --output NEW_REPLAY_DIRECTORY --operations softmax,rmsnorm,layernorm \
  --rounds 4 --samples 9 --sample-ms 30 --warmup-ms 200 --capture-sources
```

Resolve BASELINE and candidate under this evidence directory; use `reference`
then `automatic` in the two separate invocations. All plans and full native
commands are in the corresponding JSON. Environment: macOS 26.6.2 arm64,
Python 3.13.7, Torch 2.14.0 commit
`08187d9e0fba026dc8217405802ab5381dc88d90`, eight requested host threads.
The unchanged native executable SHA-256 is
`9cbdc7873355118a9874c58eec499cdb0692dd9286c14b52afceae620c62ad87`.
Bridge/library/source hashes, recorded environment and both source-report
hashes are in the receipt. Torch build identity is recorded, not a full hash
of every Torch binary. Code validation reuses the prior `356c5d53c` artifact
boundary after hash verification; no fresh CTest pass is claimed.

Run `uv run --no-project --python 3.13 python audit.py
--check-current-artifacts` from this directory to recheck the saved evidence
and current binaries. The original receipt uses an exclusive-create path and
must not be overwritten when artifacts later change.

## Implications and report QA contract

Keep automatic defaults unchanged. Extend the execution family only together
with group/concurrency costs and independent held-out validation; the existing
opt-in service policy already includes launch/access-demand terms, so do not
claim all such features are absent. Group barriers and private recurrence
costs remain hypotheses to inspect, not hardware diagnoses from timings alone.
The next coverage step increases GEMM dimensions and reduction row widths.

The selected surface remains the existing Sphinx performance chapter. The
technical-report roles map to Result, comparison definitions, the two evidence
sections, robustness, reproduction, and implications/open questions here.
Neutral exact-lookup tables preserve all shapes and both denominators; no
ranking chart or pooled headline hides counterexamples. Render the complete
Sphinx page and verify links after adding the chapter summary. Share with
caveats: this is a bounded cohort with an audited evidence trail, not general
MPS/Torch parity or measured occupancy/ISA attribution.
