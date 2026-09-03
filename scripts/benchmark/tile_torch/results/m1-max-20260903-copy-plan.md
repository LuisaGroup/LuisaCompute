# Cooperative copy plan: useful on ragged inputs, not the large-GEMM bottleneck

Apple M1 Max, macOS 26.6.2, PyTorch 2.14.0, contiguous FP32 GEMM. All results
use the full FP64 output oracle, resident buffers, preallocated outputs, and
warm synchronized host-wall timing including dispatch. No build or profiler
overlapped timing. The overall CPU/Metal performance goal remains open.

## Larger tiles are a separate experiment

The previous planner could not fit larger accumulator/result storage before
closed-recurrence residency. That residency now permits a 64x64x32 tile with
32 KiB of shared memory: C plus A/B, with the D carry-copy allocation removed.
At 256 threads the distribution is 2x4 subgroups, each retaining 4x2 fragments.

The [64x64x32 pilot](m1-max-20260903-planner-64x64x32/results.md) measures
512-cubed at 56.579 us and 1024-cubed at 407.675 us; the smaller
[64x64x16 K-block pilot](m1-max-20260903-planner-64x64x16/results.md) is slower
on both. Both include all eight shapes and full validation. These parameter
experiments are separate from the same-configuration lowering comparison below.
The earlier 32x64x32, 256-thread repeat measured about 465 us at 1024-cubed;
cross-experiment ratios are not a paired confidence claim.

## What changed

`PlannerOptions::max_copy_batch` controls a supported cooperative realization:
load/compute several independent values into TIRx bindings, then issue their
shared-memory stores. The worker/element correspondence, stage order, and all
barriers are unchanged. This uses scalar accesses, not aligned vector loads
or an asynchronous engine.

The transform requires an independent element domain and a compiler-owned
shared destination, rejects destination read/modify/write and opaque calls,
and preserves short-circuit bounds. Only complete worker chunks are grouped;
the remaining full/partial chunks use the original guarded path. The default
is one (disabled), and the compiler accepts limits from 1 to 16. Reports expose
the chosen maximum and the number of actually transformed operations.

## Controlled four-round batch=1 versus batch=4

The [raw repeated comparison](m1-max-20260903-copy-batch-repeat/results.md)
uses one unchanged executable/library build, a 64x64x32 tile, window 1,
256 threads, and cooperative FP32 matrix atoms. Only copy batching differs.
Eight shapes, two policies, four counterbalanced rounds produce 64 fully
validated native/PyTorch pairs with zero failures. Each uses nine approximately
40 ms batches after at least 200 ms warmup. Times are medians of round medians;
speedups are medians of paired ratios. Ranges are not confidence intervals.

| M x N x K | Batch 1 us | Batch 4 us | Paired speedup [range] | Candidate-run PyTorch us |
|---|---:|---:|---:|---:|
| 32 x 32 x 32 | 6.490 | 5.683 | 1.149x [1.105, 1.162] | 27.080 |
| 128 x 128 x 128 | 13.663 | 13.723 | 0.995x [0.990, 1.005] | 27.062 |
| 512 x 512 x 512 | 56.736 | 56.676 | 1.002x [0.999, 1.007] | 47.976 |
| 1024 x 1024 x 1024 | 407.528 | 407.691 | 1.000x [0.996, 1.002] | 288.183 |
| 256 x 1024 x 128 | 19.490 | 19.469 | 1.005x [0.994, 1.039] | 29.915 |
| 1024 x 128 x 256 | 24.867 | 25.043 | 0.994x [0.988, 1.002] | 30.197 |
| 127 x 193 x 61 | 17.619 | 11.392 | 1.536x [1.505, 1.595] | 27.125 |
| 513 x 257 x 129 | 36.400 | 26.939 | 1.349x [1.336, 1.357] | 34.363 |

Batching reliably helps the two ragged cases and the smallest square in this
configuration. It does not establish a speedup for aligned large GEMM; 512-
and 1024-cubed still take about 1.18x and 1.41x PyTorch time. Improved memory
overlap and simplified guarded address work are possible explanations for
the ragged gains, not measured causal attributions. No default is changed
and no global optimum is claimed.

## Verification and reproduction

The full native build passes all 137 unit tests. New CPU/Metal tests cover
negative origins, both-sided bounds, multi-axis/reversed accesses, 32/48/256
threads, and batch limits 1/4/16. They check actual batched-source emission
and full output values. All five modified C++ translation units pass individual
syntax checks. The benchmark/replay policy suite passes 21 tests. A complete
bridge-off build and its six dependency-free Tile tests also pass.

To replay the two recorded policies without selecting fresh parameters:

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/repeat.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --reference scripts/benchmark/tile_torch/results/m1-max-20260903-planner-64x64x32/results.json \
  --candidate scripts/benchmark/tile_torch/results/m1-max-20260903-copy4-64x64x32/results.json \
  --output /tmp/tile-copy-new-repeat --rounds 4
```

The [batch-4 pilot](m1-max-20260903-copy4-64x64x32/results.md) is retained but
not used as the repeated result. JSON records executable/Tile-library hashes
and the dirty source checkout, plus every requested and realized policy.
