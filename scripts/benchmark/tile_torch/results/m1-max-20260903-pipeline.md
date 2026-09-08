# M1 Max pipeline-window comparison — 2026-09-03

The two-window schedule improved CPU GEMM in this run, but did not provide a
consistent Metal benefit. Across the eight shapes, the geometric mean of
ordered/two-window p50 ratios was **2.190× on CPU worker, 0.999× on Metal worker,
and 0.983× on Metal group**. These are equally weighted per-shape ratios, not
aggregate throughput. All **48 native/PyTorch pairs** passed full-output
numerical checks. A legal pipeline transformation is not automatically a
profitable one, and large GEMMs remain substantially slower than PyTorch.

## Controlled configurations

- Source: `aeeee8eaf42952c4869d114360bc89ef92437738`, including the native
  pipeline implementation in `0bea8b0a0`. All runs used the same executable,
  SHA256 `2c062bad17e2dd4ba670356ba51a01d1cb0f883433af3895b1eefaaa29dd8e74`.
  The worktree was dirty from unrelated files and generated artifacts; the
  benchmark and library sources were committed before measurement.
- Apple M1 Max, macOS 26.6.2, native arm64; PyTorch 2.14.0, Python 3.13.7;
  eight CPU threads. Metal and PyTorch MPS ran on the physical GPU, without
  CPU fallback.
- The same GEMM source, deterministic FP32 inputs, and **8×8×16** block shape
  were used in every variant. Each host configuration captures/JITs a fresh
  kernel. No best-of-tuning search or block-shape change was applied.
- `--pipeline-window 1` retains ordered execution. Window 2 permits the
  dependence-checked two-window software-prefetch schedule. Execution scope
  was held fixed within each comparison. Neither window implies
  hardware-asynchronous copies or matrix-atom lowering.
- Full output from both implementations was checked against the same CPU
  FP64 reference, with `atol=rtol=1e-4`; maximum absolute error was zero for
  these deterministic inputs. This does not imply exactness for all inputs.
- Device-resident inputs and preallocated outputs; at least 150 ms warmup,
  calibrated approximately 20 ms batches, and nine samples. Times below are
  warm batched per-call p50 in **microseconds**, including host dispatch and
  synchronization. They exclude transfers and compilation and are **not GPU
  hardware-event times**. Individually synchronized latency, p90, all raw
  samples, and setup phases remain in the per-run reports.
- Runs were sequential, without concurrent builds or tests, in the order
  listed below. Native/PyTorch timing order alternates by case. This is one
  pass per configuration, not a randomized repeated experiment; small
  differences should not be treated as statistically established wins.

The complete per-run records are:

1. [Window 1, CPU/Metal worker](m1-max-20260903-pipeline-window1-worker/results.md)
   ([raw JSON](m1-max-20260903-pipeline-window1-worker/results.json)).
2. [Window 2, CPU/Metal worker](m1-max-20260903-pipeline-window2-worker/results.md)
   ([raw JSON](m1-max-20260903-pipeline-window2-worker/results.json)).
3. [Window 1, Metal group](m1-max-20260903-pipeline-window1-group-metal/results.md)
   ([raw JSON](m1-max-20260903-pipeline-window1-group-metal/results.json)).
4. [Window 2, Metal group](m1-max-20260903-pipeline-window2-group-metal/results.md)
   ([raw JSON](m1-max-20260903-pipeline-window2-group-metal/results.json)).

## CPU worker

`Ordered / two-window > 1` means window 2 was faster. `Two-window / Torch > 1`
means native was slower than PyTorch. The Torch column uses the matching
window-2 run; each window-1 Torch baseline remains in its raw record.

| M×N×K | Ordered µs | Two-window µs | Ordered / two-window | Torch µs | Two-window / Torch |
|---|---:|---:|---:|---:|---:|
| 32×32×32 | 6.702 | 6.110 | 1.097× | 0.875 | 6.98× |
| 128×128×128 | 122.500 | 57.974 | 2.113× | 4.892 | 11.85× |
| 512×512×512 | 5835.639 | 1631.628 | 3.577× | 148.157 | 11.01× |
| 1024×1024×1024 | 37870.208 | 12576.604 | 3.011× | 1010.839 | 12.44× |
| 256×1024×128 | 1299.836 | 573.947 | 2.265× | 69.572 | 8.25× |
| 1024×128×256 | 1302.502 | 484.454 | 2.689× | 65.185 | 7.43× |
| 127×193×61 | 103.397 | 57.878 | 1.786× | 6.536 | 8.85× |
| 513×257×129 | 977.721 | 502.464 | 1.946× | 44.898 | 11.19× |

The 1024³ case improved from **37.870 ms to 12.577 ms**, an observed 3.011×
speedup at the same block shape. It still took 12.44× the matching PyTorch
time. The measurements establish the configuration difference, not a
microarchitectural explanation such as asynchronous memory overlap.

## Metal worker

| M×N×K | Ordered µs | Two-window µs | Ordered / two-window | Torch µs | Two-window / Torch |
|---|---:|---:|---:|---:|---:|
| 32×32×32 | 89.360 | 96.720 | 0.924× | 29.425 | 3.29× |
| 128×128×128 | 418.207 | 458.503 | 0.912× | 33.619 | 13.64× |
| 512×512×512 | 2811.571 | 2613.012 | 1.076× | 57.713 | 45.28× |
| 1024×1024×1024 | 15195.666 | 14571.875 | 1.043× | 355.905 | 40.94× |
| 256×1024×128 | 762.543 | 732.673 | 1.041× | 34.895 | 21.00× |
| 1024×128×256 | 1416.985 | 1339.015 | 1.058× | 33.229 | 40.30× |
| 127×193×61 | 320.880 | 346.016 | 0.927× | 30.936 | 11.19× |
| 513×257×129 | 978.167 | 948.492 | 1.031× | 42.403 | 22.37× |

Some larger shapes improved, while three smaller/tail cases regressed. The
geometric mean ratio is effectively unchanged. This is not evidence for
unconditionally preferring two windows on Metal.

## Metal group

| M×N×K | Ordered µs | Two-window µs | Ordered / two-window | Torch µs | Two-window / Torch |
|---|---:|---:|---:|---:|---:|
| 32×32×32 | 7.168 | 6.959 | 1.030× | 29.803 | 0.23× |
| 128×128×128 | 18.914 | 20.260 | 0.934× | 32.696 | 0.62× |
| 512×512×512 | 593.536 | 588.039 | 1.009× | 60.171 | 9.77× |
| 1024×1024×1024 | 4333.927 | 4438.979 | 0.976× | 347.806 | 12.76× |
| 256×1024×128 | 145.301 | 145.678 | 0.997× | 34.233 | 4.26× |
| 1024×128×256 | 145.296 | 151.408 | 0.960× | 31.830 | 4.76× |
| 127×193×61 | 12.892 | 13.267 | 0.972× | 29.149 | 0.46× |
| 513×257×129 | 105.535 | 106.319 | 0.993× | 43.000 | 2.47× |

Two windows did not yield a consistent benefit. The 1024³ case moved from
**4.334 ms to 4.439 ms** and remained 12.76× slower than PyTorch. The native
small-shape batched throughput wins do not imply equivalent synchronized
latency wins: for example, 32³ window-2 latency was 692.458 µs native versus
289.041 µs Torch. Launch/binding behavior matters at these sizes.

## Compilation cost and remaining work

CPU native compilation ranged from **143.167–163.004 ms** for window 1 to
**209.771–291.441 ms** for window 2. Metal group compilation ranged from
41.240–48.049 ms to 44.347–52.551 ms. Capture, allocation/upload, first call,
and download are recorded separately. First calls are process-cold, not a
guarantee of empty OS/driver caches.

The compiler now realizes source-stage boundaries, versions eligible
temporaries, and preserves ordering for unsupported or unsafe schedules.
That closes a semantic lowering gap; it does not close performance planning.
Profitability-based schedule selection or JIT autotuning, matrix-atom
selection, hardware-asynchronous pipelines, and parallel reduction trees
remain separate work. No target mapping or compiler policy default was
changed on the strength of this measurement.

## Reproduction

After completing the full build and correctness tests, run this command
sequentially for `(BACKENDS, SCOPE, WINDOW)` = `(cpu,metal, worker, 1)`,
`(cpu,metal, worker, 2)`, `(metal, group, 1)`, and `(metal, group, 2)`.
Use a different, nonexistent output directory for each run.

```sh
uv run --no-project --python 3.13 --with torch --with numpy python \
  scripts/benchmark/tile_torch/run.py \
  --native cmake-build-tirx/bin/benchmark_tile_tirx \
  --output /tmp/tile-pipeline-UNIQUE \
  --backends BACKENDS --operations gemm \
  --execution-scope SCOPE --pipeline-window WINDOW --threads 8
```
