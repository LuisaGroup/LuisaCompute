# First execution planner: structural gains and model errors

This is an initial, uncalibrated relative-work model, not a claim of
PyTorch-equivalent performance. The implemented solver is exact only for its
supported rectangular matrix realization family and stated additive objective.

Design and formulas: [execution planner](../../../../docs/source/tile_execution_planner.md).
The implementation checkpoint is `91332e038`; measurements were collected
before committing, so their metadata correctly records a dirty worktree based
on `0fc45b943` and fingerprints the actual executable and Tile libraries.

## Matched implementation comparison

The [four-round report](m1-max-20260903-planner-repeat-verified/results.md)
contains 64 fully validated native/PyTorch pairs: eight FP32 GEMM shapes, two
implementations, four counterbalanced rounds. All use the same 32x64x32 block,
window 1, explicit group execution, and cooperative-matrix capability. No
per-shape parameter search is included in this comparison.

Hardware/software: Apple M1 Max, macOS 26.6.2, PyTorch 2.14.0. Inputs and
preallocated outputs are device-resident. These are warm synchronized host-wall
throughput times, including dispatch, not GPU hardware-event times. Each round
uses nine approximately 40 ms batches after at least 200 ms warmup. Full
outputs are checked against the same FP64 reference; no failed measurements
occurred in the completed run. Build/profiling work did not overlap timing.

The reference is the pre-planner `0fc45b943` executable plus frozen Luisa
libraries in `/tmp/luisa-tile-before-planner-pBzxcr`. The candidate includes
multi-fragment input reuse, closed-recurrence accumulator residency, and the
first planner prior. Binary and Tile-library hashes are recorded in
[raw JSON](m1-max-20260903-planner-repeat-verified/results.json); an executable
hash alone does not identify a dynamically linked implementation.

| M x N x K | Old lowering us | First planner us | Paired old/new speedup | Candidate-run PyTorch us |
|---|---:|---:|---:|---:|
| 32 x 32 x 32 | 6.306 | 8.136 | 0.792x | 27.316 |
| 128 x 128 x 128 | 15.616 | 12.171 | 1.277x | 27.514 |
| 512 x 512 x 512 | 124.155 | 77.361 | 1.609x | 48.393 |
| 1024 x 1024 x 1024 | 998.477 | 474.079 | 2.106x | 293.380 |
| 256 x 1024 x 128 | 35.814 | 25.468 | 1.401x | 30.300 |
| 1024 x 128 x 256 | 34.228 | 23.891 | 1.432x | 30.876 |
| 127 x 193 x 61 | 14.238 | 20.937 | 0.687x | 27.308 |
| 513 x 257 x 129 | 53.952 | 41.769 | 1.296x | 34.380 |

Times are medians of round medians; speedups are medians of paired ratios.
The complete report includes the min/max paired-ratio range, not a confidence
interval. The gains at 512-cubed and 1024-cubed are versus **our old lowering**;
the candidate still takes about 1.60x and 1.62x the corresponding PyTorch time.
The 32-cubed and small ragged cases are real regressions and are not excluded.

## What changed structurally

For every shape in this fixed-block run, the first prior selects 128 threads,
a 1x4 subgroup grid, 4x2 resident fragments per subgroup, and a proved resident
accumulator. The emitted native TIRx layout agrees with the core IndexMap.

For 1024-cubed, per program, dynamic matrix atom issues stay at 4096; modeled
shared fragment transfers fall from 10240 to 3136. The compact shared footprint
falls from 28 KiB to 20 KiB by eliminating the result/carry-copy buffer. Live
fragment scalars per lane rise from 6 to 28. These counts describe the selected
realization; they are not measured hardware-register counts or cycle timings.
Generated source has eight static MMA call sites instead of one call site in a
wave loop; this does not mean eight times the dynamic computation.

Correctness guards include full atom coverage, complete subgroups, typed MMA
body/operand checks, alias and arithmetic policy, and absence of other observers
of the promoted accumulator. The compiler still uses conservative barriers and
shared initial/final accumulator storage.

## What the measurements say about the model

The structural optimization is useful, but the prior is not yet a reliable
universal selector. It selects the same mapping for all eight shapes despite
different program counts, ragged accesses, and contraction lengths. It does
not yet model layout-dependent copy concurrency, actual register allocation,
global tails, or whole-device occupancy. The regressions demonstrate missing
profitability information; they are not solved merely by choosing a more
sophisticated search algorithm.

The benchmark now accepts `--group-threads N` to vary only the exact group
worker count. Native output includes requested and realized counts, work
features, and the selected fragment distribution. This enables a separate
ranking experiment, rather than attributing every realization improvement to
the cost model. Zero leaves selection automatic; explicit constraints are
checked against the real target limit, independently of the reference launch
width and compiler search budget.

## Same-binary thread-count ranking test

Separate pilot runs at [64](m1-max-20260903-planner-threads64/results.md),
[128](m1-max-20260903-planner-threads128/results.md), and
[256](m1-max-20260903-planner-threads256/results.md) threads use one unchanged
binary/library build and the same block, window, inputs, and timing settings.
All 24 native/PyTorch pairs pass the full correctness oracle. The 64-thread
variant is slower on every pilot case; these pilot minima are not the published
repeat results below.

A fresh [four-round 128-versus-256 comparison](m1-max-20260903-planner-thread-repeat/results.md)
then freezes those two thread-count constraints, alternates ordering, and
validates all 64 native/PyTorch pairs. It has zero failures. This experiment
isolates the planner's mapping choice from changes to the compiler emitter.

| M x N x K | 128 threads us | 256 threads us | Paired 128/256 speedup [range] | 256-run PyTorch us |
|---|---:|---:|---:|---:|
| 32 x 32 x 32 | 7.511 | 5.673 | 1.312x [1.291, 1.420] | 27.142 |
| 128 x 128 x 128 | 12.151 | 11.115 | 1.083x [0.999, 1.222] | 27.331 |
| 512 x 512 x 512 | 76.885 | 74.039 | 1.037x [1.022, 1.053] | 48.168 |
| 1024 x 1024 x 1024 | 474.298 | 465.241 | 1.018x [0.986, 1.022] | 287.830 |
| 256 x 1024 x 128 | 24.889 | 23.840 | 1.044x [1.029, 1.094] | 30.207 |
| 1024 x 128 x 256 | 23.032 | 22.031 | 1.042x [0.963, 1.072] | 30.390 |
| 127 x 193 x 61 | 20.382 | 12.114 | 1.670x [1.579, 1.847] | 27.689 |
| 513 x 257 x 129 | 41.175 | 28.400 | 1.453x [1.423, 1.477] | 34.329 |

The 256-thread plan uses a 2x4 subgroup grid with 2x2 local fragments,
16 live fragment scalars/lane, and the same 20 KiB shared footprint. For
1024-cubed, it has *more* modeled shared fragment transfers than the 128-thread
plan (4160 versus 3136), while matrix issues remain 4096. The prior therefore
scores it worse (13792 versus 11712), yet its median is slightly faster. The
large ragged-case improvements are stable across these rounds; the 1024-cubed
gain is small and reverses in one round, so it is not a robust large-kernel win.

The same model already considered these mappings and deliberately selected
128 threads. This is a ranking/model failure, not an integer-search omission.
Register pressure, copy concurrency, and tail handling are hypotheses for the
missing costs, not measured causal explanations. No prior coefficient or
default is changed to fit these eight observations. They motivate calibrated
realization features and a diverse measured shortlist. The 512-cubed and
1024-cubed 256-thread variants still take about 1.54x and 1.62x PyTorch time;
the overall performance goal remains open.

## Correctness and dependency validation

The completed native bridge build passes all 137 unit tests, including 27
Tile tests and actual CPU/Metal execution. The Python benchmark/replay policy
tests pass 19/19. A separate clean default configuration, without any TVM
package paths and with the bridge off, builds the whole enabled project and
passes all 116 unit tests, including the six dependency-free Tile tests. This
is unit and focused device coverage, not a claim that every graphics integration
test was executed.

## Reproduction and audit

The [initial single-run comparison](m1-max-20260903-planner-candidate/results.md)
is a separate pilot, not the source of the four-round results above. To replay
the matched configurations using the two corresponding binaries:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --native /tmp/luisa-tile-before-planner-pBzxcr/benchmark_tile_tirx \
  --candidate-native cmake-build-tirx/bin/benchmark_tile_tirx \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260903-planner-candidate/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260903-planner-candidate/results.json \
  --output /tmp/tile-planner-new-repeat --rounds 4
```

To reproduce the same-binary thread mapping comparison:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260903-planner-threads128/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260903-planner-threads256/results.json \
  --output /tmp/tile-planner-new-thread-repeat --rounds 4
```

The first attempted repeat was [aborted](m1-max-20260903-planner-repeat/RUN_STATUS.md)
because the frozen baseline bundle lacked `libglslang.16.dylib`. Those partial
records are retained but excluded. The missing unchanged third-party binary
was copied from the build; both copies have SHA-256
`bbdbf823956fb94d1131177bb87c90b82209fdcbaced1466bbf337386048459d`
and a September 2 modification time preceding planner work. The corrected
baseline then executed successfully before the completed repeat was started.
