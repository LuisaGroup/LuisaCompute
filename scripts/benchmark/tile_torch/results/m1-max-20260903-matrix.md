# M1 Max: guarded native matrix realization

All **24 native/PyTorch GEMM pairs passed**: eight Metal reference cases,
eight Metal matrix-enabled cases, and eight CPU reference cases. Both Metal
runs used the same binary, source, 8×8×16 tile shape, group mapping, and
two-window pipeline. Only the cooperative-matrix capability option changed.

For 1024³, Metal improved from **4.618 to 3.244 ms (1.42×)**, but remained
**9.41× slower than PyTorch MPS**. CPU measured **12.425 versus 1.103 ms**
(11.27× slower). This establishes a working, measurably useful native matrix
path, not competitive GEMM performance or an optimized limit for this hardware.

## Provenance and measurement contract

- Implementation: `37e1337fd943168b872f31802033561ae61fa298`.
- Executable SHA-256: `58d7c25140fd529a0adcdfb4070e5df3130b27c9cfae72adedc0ca7d0334bd0d`
  in all three runs.
- Apple M1 Max, macOS 26.6.2 arm64, PyTorch 2.14.0; FP32; eight CPU threads.
- Recorded on 2026-09-03 UTC. Runs were sequential, after this task's builds
  and correctness tests finished. `worktree_dirty=true` is retained in the
  raw metadata because pre-existing changes outside the Tile implementation
  were preserved; the Tile implementation and benchmark source were committed.
- Nine samples after at least 150 ms warmup, calibrated approximately 20 ms
  batches; native/PyTorch timing order alternates by case.
- Timings include host dispatch/binding and synchronization, but exclude
  compilation, allocation, and transfers. They are not GPU-event timings.
  Individually synchronized latency, p90, raw samples, and all setup phases
  remain in each complete report.
- Inputs and preallocated outputs stay on their selected device. Both full
  outputs are checked against the same CPU FP64 oracle and tolerances.
  These deterministic dyadic inputs produced zero maximum error in all three
  native runs; separate matrix correctness tests also cover non-dyadic data.
- Matrix permission and actual emitted instructions are separate fields.
  Every Metal reference/CPU case had zero matrix call sites; every enabled
  Metal case had two static `simdgroup_multiply_accumulate` call sites. These
  counts are generated-source evidence, not dynamic instruction counts.

## Matched Metal results

Warm batched p50 values are microseconds; the final column is compiler-call
time in milliseconds, reference / matrix-enabled. Speedup is reference divided
by matrix-enabled native time. The PyTorch column is the matrix-enabled run's
matched comparison; the reference run's PyTorch samples are also preserved.

| M×N×K | Reference | Matrix | Speedup | PyTorch | Matrix / PyTorch | Compile ref / matrix |
|---|---:|---:|---:|---:|---:|---:|
| 32×32×32 | 6.989 | 6.410 | 1.09× | 29.298 | 0.22× | 47.587 / 47.433 |
| 128×128×128 | 18.570 | 16.170 | 1.15× | 30.717 | 0.53× | 48.496 / 48.070 |
| 512×512×512 | 583.395 | 427.523 | 1.36× | 59.895 | 7.14× | 48.651 / 47.694 |
| 1024×1024×1024 | 4617.558 | 3244.304 | 1.42× | 344.738 | 9.41× | 48.819 / 49.368 |
| 256×1024×128 | 144.465 | 109.533 | 1.32× | 32.808 | 3.34× | 49.177 / 49.258 |
| 1024×128×256 | 145.985 | 114.866 | 1.27× | 34.434 | 3.34× | 48.063 / 48.795 |
| 127×193×61 | 13.553 | 11.839 | 1.14× | 30.830 | 0.38× | 55.101 / 55.665 |
| 513×257×129 | 107.070 | 78.058 | 1.37× | 41.485 | 1.88× | 54.267 / 55.588 |

Small-case ratios include different framework dispatch overheads and must not
be presented as superior GPU arithmetic throughput. These are single-run
comparisons without confidence intervals, not evidence that every small
difference is statistically significant. No tile-shape autotuning or
best-of-candidate selection was performed.

## CPU reference at the same implementation

CPU retains the worker contraction path. These fresh measurements are a
current baseline, not an off/on matrix comparison. Values are microseconds.

| M×N×K | Native p50 | PyTorch p50 | Native / PyTorch |
|---|---:|---:|---:|
| 32×32×32 | 5.636 | 0.919 | 6.14× |
| 128×128×128 | 58.146 | 4.961 | 11.72× |
| 512×512×512 | 1728.875 | 144.225 | 11.99× |
| 1024×1024×1024 | 12425.417 | 1102.796 | 11.27× |
| 256×1024×128 | 585.056 | 70.366 | 8.31× |
| 1024×128×256 | 630.605 | 64.846 | 9.72× |
| 127×193×61 | 68.115 | 6.479 | 10.51× |
| 513×257×129 | 667.339 | 43.591 | 15.31× |

## What is and is not established

The guarded selector works with the execution/resource model: kernels retain
the same Tile SSA `mma`, bounded subtiles, explicit stores, and inferred
pipeline carries. The compiler selects native FP32 matrices without reducing
input precision. Ordered math, incompatible types/layouts, worker-local
execution, incomplete participants, and stale provenance markers retain a
reference realization; CPU and physical Metal correctness tests cover those
boundaries, transposes, multiple matrix atoms, and zero-length contraction.

This first selector changes the contraction realization. It does not optimize
accumulator residency across pipeline iterations, eliminate conservative
group fences, or choose profitable tile/participant shapes. Those are concrete
next scheduling opportunities, not bottlenecks proven by hardware counters.
Hardware-asynchronous transfers, warp specialization, and other target atom
families remain future work. The default matrix option stays off.

Validation before measurement: full TVMx-enabled build, **136/136 CTests**;
full TVMx-disabled build, **6/6 Tile tests**; **8/8 benchmark contract tests**;
focused C++ syntax checks and formatting. The registered kernel gallery covers
GEMM, elementwise, statistics/losses/norms, cross entropy, causal online
attention, convolution/depthwise/pooling, Sobel/median, stable sort/Top-K,
segmented accumulation, and nested temporal composition on CPU and Metal.
This closes the current runnable POC scope, not production-performance tuning.

## Raw reports and reproduction

- [Metal reference report](m1-max-20260903-matrix-reference/results.md),
  [raw samples](m1-max-20260903-matrix-reference/results.json).
- [Metal matrix report](m1-max-20260903-matrix-native/results.md),
  [raw samples](m1-max-20260903-matrix-native/results.json).
- [CPU reference report](m1-max-20260903-matrix-cpu-reference/results.md),
  [raw samples](m1-max-20260903-matrix-cpu-reference/results.json).

Run from the repository root with the already-built executable. Output
directories must not already exist. Use a compatible Apple GPU for the
capability opt-in; this is not automatic device detection.

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/tile-matrix-reference \
  --backends metal --operations gemm --execution-scope group \
  --pipeline-window 2 --gemm-block 8,8,16 --threads 8

uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/tile-matrix-native \
  --backends metal --operations gemm --execution-scope group \
  --pipeline-window 2 --gemm-block 8,8,16 --threads 8 --cooperative-matrix

uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/tile-matrix-cpu-reference \
  --backends cpu --operations gemm --execution-scope worker \
  --pipeline-window 2 --gemm-block 8,8,16 --threads 8
```
