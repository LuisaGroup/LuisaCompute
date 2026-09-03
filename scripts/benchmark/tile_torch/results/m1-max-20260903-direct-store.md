# Proved direct accumulator output on M1 Max

The new realization removes an unnecessary shared accumulator when its whole
lifetime is proved: literal fill, closed MMA recurrence, and one fully in-bounds
global output. The output remains at its original program position. This is a
resource/lowering change, not a new DSL primitive or a GEMM-name shortcut.

At the same 64x64x32 tile, 256 threads, pipeline window 1, and copy batch 4,
four counterbalanced rounds show **1.242x speedup over the previous lowering**
for 1024-cubed GEMM. It is still **1.137x the PyTorch time** on that shape.
512-cubed changes little and remains slower than PyTorch. The full task of
matching PyTorch across CPU and Metal is not complete.

## What changed and what was proved

- A closed resident recurrence already removed the intermediate D buffer.
  The new proof also removes C's shared allocation, literal fill, and final
  shared publication. Native fragments are filled directly and stored directly
  to the global destination using Metal's FP32 matrix store.
- Every destination coordinate and predicate must be proved valid over the
  enclosing execution domains and local matrix domain, using native TVMx
  `StmtSimplify`. Unknown bounds, a second observer, an escaped pointer, or a
  manual resource annotation retains the previous path.
- The global store stays at the original sink. A test reads old output between
  the recurrence and the final store; moving the write to the recurrence's end
  would be wrong even without another C consumer.
- A native-IR counterexample exposed an existing residency bug: if C also
  supplies an A/B multiplicand, keeping only the new fragment value makes the
  next iteration read stale shared C. The unoptimized control was correct;
  enabled residency failed six assertions before the new rejection check.
  Both operand positions now have numerical regressions.
- A break/continue between MMA and yield must discard that iteration's D,
  rather than update retained C. Metal executes both oracles with residency
  enabled and disabled. The pinned TVMx LLVM emitter lacks Break/Continue
  visitors; CPU checks that specific compile rejection, not numerical support.

The planner reports the proof-based choice and accounts for both removed
buffers. Aligned output at 64x64x32 uses **16 KiB** of shared storage instead
of **32 KiB**, with the same subgroup distribution and 28 live fragment
scalars/lane proxy. Ragged output and the 32-cubed case keep 32 KiB and the
guarded reference output. These are allocation/work facts, not measured
register counts or an occupancy claim.

## Controlled implementation comparison

Reference: the frozen `e0093f959` executable and adjacent Luisa libraries in
`/tmp/luisa-tile-before-direct-store-WoKzpC`. Candidate: the direct-output work
based on that commit. The raw report identifies both executables and bridge
libraries by SHA-256; the bridge-library hashes are respectively
`73a4b465fd6a1c255ba5a1deb6ab71d5dae5ef007be1bbfef16d1fe014bf4827`
and `cfbdc3f28b0484bd13789eaa7dd0af7e6d36a569a0a7e863611a0a421ca1b949`.

FP32 inputs and preallocated device-resident output are unchanged. Every run
checks the complete output against the existing FP64 oracle. Each of four
rounds freshly captures/JIT-compiles both variants, rotates shape order, and
counterbalances both implementation and framework order. There are nine 40 ms
timing batches after 200 ms warmup, with no concurrent build, test, or profiler.
Times are synchronized amortized **host-wall** time including dispatch, not
GPU-event kernel time. Cold JIT/call measurements are retained separately.

| M x N x K | Old lowering us | Direct-output lowering us | Paired speedup median [range] | Candidate-run PyTorch us |
|---|---:|---:|---:|---:|
| 32 x 32 x 32 | 6.201 | 6.176 | 1.004 [0.969, 1.039] | 27.006 |
| 128 x 128 x 128 | 14.681 | 14.054 | 1.046 [1.040, 1.050] | 26.786 |
| 512 x 512 x 512 | 57.605 | 57.121 | 1.008 [1.005, 1.011] | 48.301 |
| 1024 x 1024 x 1024 | 406.999 | 327.762 | 1.242 [1.241, 1.242] | 288.270 |
| 256 x 1024 x 128 | 20.508 | 20.005 | 1.026 [1.016, 1.049] | 29.587 |
| 1024 x 128 x 256 | 26.056 | 25.435 | 1.025 [0.996, 1.046] | 29.563 |
| 127 x 193 x 61 | 12.265 | 12.275 | 1.000 [0.980, 1.019] | 26.848 |
| 513 x 257 x 129 | 28.031 | 28.077 | 0.998 [0.989, 1.010] | 34.569 |

All 64 measurements validated; there are four valid pairs per shape and zero
failed measurements. Values are medians of per-round medians. The speedup
range is the paired min-max range, **not a confidence interval**. No shape or
round was removed for being slow.

The large-square improvement with nearly unchanged 512-cubed time is evidence
that per-tile operation count alone is insufficient for ranking. Lower shared
storage and different program counts are relevant hypotheses for the change;
this experiment does not independently identify occupancy, cache behavior, or
barrier/issue throughput. The richer resource-constrained cost-model design
must be calibrated before claiming those mechanisms quantitatively.

## Small, bounded JIT exploration

Separate eight-shape pilots used 32x64x32 tiles at
[128 threads](m1-max-20260903-direct-store-32x64x32-128/results.md) and
[256 threads](m1-max-20260903-direct-store-32x64x32-256/results.md).
The smaller tile benefits small/ragged cases but does not replace 64x64x32 for
the large square: its 1024-cubed timings were 399.166 and 394.937 us.

An [explicit four-candidate JIT run](m1-max-20260903-direct-store-jit256/results.md)
then tried 32x64x32, 64x64x32, 64x128x32, and 64x64x64, all at 256 threads and
window 1. There were 26 validated candidates, six shared-capacity rejections,
and eight freshly recaptured/validated final selections. The larger candidates
fit only when the output proof removes C; a small/ragged shape cannot trade
correctness for a lower resource estimate. All rejected candidates and errors
remain in the raw report.

The final selection used 64x64x64 for 512-cubed (54.502 us versus PyTorch
48.370 us), and retained 64x64x32 for 1024-cubed (328.169 versus 288.333 us).
These are exploration plus fresh-selection timings, **not** a second
counterbalanced implementation comparison or evidence of a global optimum.
They support keeping shape/resource legality separate from ranking and using
JIT measurements for uncertain finalists. Neither integer programming nor
annealing would make the rejected storage plans legal.

## Reproduction and validation

The pilot is in [fixed 64x64x32 results](m1-max-20260903-direct-store-64x64x32/results.md).
The controlled comparison is in [four-round results](m1-max-20260903-direct-store-repeat/results.md)
and [raw measurements](m1-max-20260903-direct-store-repeat/results.json).

With a fully built candidate and the frozen reference bundle:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260903-copy4-64x64x32/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260903-direct-store-64x64x32/results.json \
  --native /tmp/luisa-tile-before-direct-store-WoKzpC/benchmark_tile_tirx \
  --candidate-native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/tile-direct-store-fresh-repeat \
  --operations gemm --rounds 4 --samples 9 --sample-ms 40 --warmup-ms 200
```

The output directory must not already exist. Build all configured targets
before tests or measurements; never overlap a build/test/profiler with timing.

The latest complete unit run before these measurements passed **137/137** in
51.01 seconds, including CPU and Metal Tile suites. Direct-output tests cover
eleven shape/layout/observation configurations, both policy settings, and two
input repetitions, as well as alias/control-flow regressions. The optional
bridge-off build and its six Tile tests also passed; no default TVM dependency
was introduced. Per-file syntax checks and `git diff --check` passed.

The [planner design](../../../../docs/source/tile_execution_planner.md) separates
hard eligibility/resource constraints from ranking, and retains exact finite
enumeration/Pareto DP as the reference solver. A more elaborate solver cannot
correct an uncalibrated cost model or select an implementation the emitter
cannot realize.
