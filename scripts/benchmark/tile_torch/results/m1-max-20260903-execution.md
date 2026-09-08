# M1 Max: cooperative execution and loop-structure validation

Implementation: `2f8f2a04459b537bd21714bdcb2be6cbb47ad81e` (2026-09-03).
Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O; PyTorch 2.14.0; FP32; 8 CPU threads.

The final implementation passed all 129 configured CTest entries, including
19 Tile tests on CPU and physical Metal. The TVMx-disabled configuration
completed its full build and passed its 5 Tile tests. Both C++20 comma-adapter
and C++23 native-subscript versions compiled and passed; the benchmark kernels
retain the same explicit load/store semantics.

The three final full matrices contain 60 matched native/PyTorch cases; an
independent repeat of all 8 CPU GEMM sizes brings this version to 68 checked
pairs. All passed the common CPU-FP64 numerical oracle. The earlier 60 pairs
from `faf71dfce` are also retained, including the CPU performance regression
that prompted the loop-structure change. No failed numerical case or slower
timing was removed.

## Measurement contract

Each variant uses the same source, deterministic inputs, preallocated outputs,
and block sizes. GEMM is 8×8×16; only the root execution-scope request differs
between the final Metal runs. These are separate sequential runs, not a
best-of-autotuning search.

There are 9 warm samples, at least 150 ms warmup,
and calibrated approximately 20 ms batches.
All times below are per-call batched p50 in **microseconds**, including host
dispatch and final synchronization but excluding compilation and transfers.
They are not GPU hardware-event times. Raw samples, p90, individually
synchronized latency, setup phases, and numerical errors are in each linked
report. No builds or other benchmark runs overlapped these measurements.

## Metal: worker versus cooperative group

Group partitions independent Tile elements among workers and shares compiler
temporaries at their group execution level. It does not use matrix hardware,
parallel reduction trees, or asynchronous pipeline overlap yet.

The final 1024³ GEMM measured **14.725 ms worker → 4.275 ms group**, a
**3.44×** improvement at the same tile shape. PyTorch in the group run measured
**0.357 ms**, so this is still approximately **12× slower than PyTorch**.
Small-kernel wins include different framework dispatch costs and must not be
presented as hardware-kernel superiority. The PyTorch column below always uses
the group-run observation, not the better of the two runs.

| Case | Block | Worker | Group | Worker / group | PyTorch |
|---|---|---:|---:|---:|---:|
| gemm_32x32x32 | 8×8×16 | 89.506 | 6.779 | 13.20× | 30.415 |
| gemm_128x128x128 | 8×8×16 | 425.349 | 19.678 | 21.62× | 31.215 |
| gemm_512x512x512 | 8×8×16 | 2786.298 | 573.736 | 4.86× | 57.839 |
| gemm_1024x1024x1024 | 8×8×16 | 14725.208 | 4274.933 | 3.44× | 357.187 |
| gemm_256x1024x128 | 8×8×16 | 766.850 | 142.921 | 5.37× | 31.476 |
| gemm_1024x128x256 | 8×8×16 | 1412.935 | 146.748 | 9.63× | 32.061 |
| gemm_127x193x61 | 8×8×16 | 324.215 | 13.702 | 23.66× | 30.096 |
| gemm_513x257x129 | 8×8×16 | 973.129 | 107.106 | 9.09× | 42.222 |
| add_1x127 | 1×256×1 | 102.402 | 3.817 | 26.83× | 3.940 |
| add_17x257 | 1×256×1 | 223.624 | 3.517 | 63.59× | 4.661 |
| add_128x1024 | 1×256×1 | 51.603 | 6.560 | 7.87× | 8.122 |
| add_4096x256 | 1×256×1 | 94.407 | 24.938 | 3.79× | 29.529 |
| sum_1x127 | 1×127×1 | 56.858 | 13.787 | 4.12× | 7.300 |
| sum_17x257 | 1×257×1 | 169.255 | 25.533 | 6.63× | 5.442 |
| sum_128x1024 | 1×1024×1 | 63.287 | 13.150 | 4.81× | 5.991 |
| sum_64x4096 | 1×4096×1 | 228.875 | 39.456 | 5.80× | 20.323 |
| softmax_1x127 | 1×127×1 | 88.971 | 25.739 | 3.46× | 30.698 |
| softmax_17x257 | 1×257×1 | 249.953 | 49.700 | 5.03× | 32.788 |
| softmax_128x1024 | 1×1024×1 | 245.325 | 27.496 | 8.92× | 39.728 |
| softmax_64x4096 | 1×4096×1 | 787.807 | 86.553 | 9.10× | 36.290 |

Full reports: [group](m1-max-20260903-axes-group-metal/results.md),
[worker](m1-max-20260903-axes-worker-metal/results.md).
The earlier group run measured 4.300 ms for 1024³ GEMM; the retained-axis
implementation preserves that cooperative improvement.

## CPU: why axis preservation matters, and what remains unresolved

The first cooperative implementation flattened element loops in the common
exporter. Its 1024³ CPU GEMM measured 68.326 ms, versus the historical
31.571 ms baseline. The common exporter now retains rectangular loops and
marks their rank; only cooperative target binding performs flattening.
A three-dimensional 5×7×11 regression checks both the exported loop axes and
real cross-worker results, including a partial final worker chunk.

This change materially recovers the measured CPU GEMM performance, but the
largest case is **not fully closed**: final-run p50 was 48.226 ms and the
independent repeat was 38.508 ms, both above the old 31.571 ms observation.
Other sizes also vary between processes. The source of the remaining
large-case difference and variability has not been isolated; it must not be
dismissed as noise or reported as a completely fixed regression. The old
baseline predates intervening dataflow fixes, so it is a historical reference,
not an isolated comparison of only this patch.

| GEMM | Historical `8591c599d` | Flat exporter `faf71dfce` | Retained axes, full run | Retained axes, repeat |
|---|---:|---:|---:|---:|
| gemm_32x32x32 | 7.914 | 9.228 | 6.012 | 6.013 |
| gemm_128x128x128 | 126.422 | 215.421 | 85.638 | 116.958 |
| gemm_512x512x512 | 4465.938 | 9349.041 | 4139.025 | 4451.625 |
| gemm_1024x1024x1024 | 31571.125 | 68325.958 | 48226.167 | 38508.292 |
| gemm_256x1024x128 | 1351.819 | 2782.863 | 1134.336 | 1445.833 |
| gemm_1024x128x256 | 1639.056 | 2420.526 | 1449.984 | 1454.734 |
| gemm_127x193x61 | 129.002 | 203.369 | 124.742 | 123.412 |
| gemm_513x257x129 | 1031.794 | 2134.759 | 1177.586 | 834.086 |

Full reports: [CPU matrix](m1-max-20260903-axes-worker-cpu/results.md),
[CPU GEMM repeat](m1-max-20260903-axes-worker-cpu-repeat/results.md).
Both observations are shown; no best-of-run value is substituted into the
full matrix.

## Reproduction

First complete the full build and correctness suite as described in the
[benchmark guide](../README.md). For each command below choose a new output
directory; existing results are never overwritten.

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/tile-group-metal --threads 8 \
  --backends metal --execution-scope group

uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/tile-worker-metal --threads 8 \
  --backends metal --execution-scope worker

uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/tile-worker-cpu --threads 8 \
  --backends cpu --execution-scope worker
```

For the independent CPU repeat, use another new output directory and add
`--operations gemm`. No thread-affinity or clock-frequency controls were
applied, and these runs do not establish a hardware performance limit.

## Provenance and retained experiments

All final timing runs used unchanged Tile and benchmark implementation paths
at `2f8f2a044`. Existing unrelated working-tree changes were preserved, so
the raw metadata correctly records `worktree_dirty: true`.

The executable SHA256 is
`4d7a80ac02f2201305e12840bac4a75e2217f455b24a92fd40dd899b9790942e`.
The executable is dynamically linked: **its hash alone does not identify the
Tile compiler implementation**. Current Tile library fingerprints were
collected after the final runs, with no intervening rebuild:

| Library | SHA256 |
|---|---|
| `libluisa-tile.dylib` | `8aa7a0c24868114153eb7a40b31384a61f8e79e1fda3594b44486f437ec8a6bd` |
| `libluisa-tile-bridge-tirx.dylib` | `eeede421ae5cc9320c70c0f8aace21b18624fe8b84ffc57912d4047f92a75cd3` |

The native bridge uses the optional `cmake-build-tirx` configuration. The
checked header checkout is `/tmp/apache-tvm-tirx` at
`c7b458e946bc4266915da582457476bdcd9705ae`; the linked TVMx/FFI libraries
come from `/tmp/luisa-tvmx-venv/lib/python3.14/site-packages`. This records
the local setup, not a claim that the header revision proves the wheel's
build revision.

Earlier full matrices at `faf71dfce`:
[group Metal](m1-max-20260903-group-metal/results.md),
[worker Metal](m1-max-20260903-worker-metal/results.md),
[worker CPU](m1-max-20260903-worker-cpu/results.md).
They remain available to audit the effect of moving flattening out of the
common exporter.

The default mapping remains unchanged. Hardware MMA, parallel reductions,
asynchronous pipelines, selective/lifetime-aware shared-memory planning,
manual Memory bridge support, and the remaining CPU performance issue are
still implementation work.
