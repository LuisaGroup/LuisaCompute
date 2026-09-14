# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T04:00:13.939655+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×64×32 / 1 | 8 | 7.723 | 27.279 | 8.516 | 28.694 | 0.28× | 212.125 | 276.459 |
| metal | gemm_128x128x128 | 32×64×32 / 1 | 8 | 12.079 | 27.698 | 13.241 | 28.375 | 0.44× | 232.042 | 269.125 |
| metal | gemm_512x512x512 | 32×64×32 / 1 | 8 | 78.358 | 49.720 | 78.859 | 50.442 | 1.58× | 303.000 | 304.084 |
| metal | gemm_1024x1024x1024 | 32×64×32 / 1 | 8 | 477.726 | 295.766 | 488.927 | 298.292 | 1.62× | 672.125 | 521.417 |
| metal | gemm_256x1024x128 | 32×64×32 / 1 | 8 | 26.279 | 28.878 | 26.575 | 30.049 | 0.91× | 270.125 | 257.625 |
| metal | gemm_1024x128x256 | 32×64×32 / 1 | 8 | 25.068 | 30.408 | 25.986 | 34.063 | 0.82× | 270.833 | 257.625 |
| metal | gemm_127x193x61 | 32×64×32 / 1 | 8 | 19.670 | 26.805 | 20.580 | 27.767 | 0.73× | 234.583 | 266.292 |
| metal | gemm_513x257x129 | 32×64×32 / 1 | 8 | 42.079 | 35.397 | 43.130 | 35.775 | 1.19× | 261.791 | 282.417 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.074 | 43.247 | 1.427 | 3.935 | 77.895 | 42.358 | 0.248 | 0.621 |
| metal / gemm_128x128x128 | 0.054 | 42.971 | 1.620 | 2.299 | 76.511 | 5.004 | 0.302 | 0.340 |
| metal / gemm_512x512x512 | 0.058 | 43.502 | 1.857 | 1.236 | 79.011 | 3.923 | 0.424 | 0.306 |
| metal / gemm_1024x1024x1024 | 0.061 | 43.699 | 2.747 | 2.090 | 79.746 | 3.972 | 0.789 | 0.421 |
| metal / gemm_256x1024x128 | 0.053 | 44.949 | 1.779 | 1.505 | 79.419 | 5.455 | 0.596 | 0.362 |
| metal / gemm_1024x128x256 | 0.079 | 45.746 | 1.764 | 2.179 | 77.518 | 4.204 | 0.383 | 0.335 |
| metal / gemm_127x193x61 | 0.057 | 48.417 | 1.575 | 1.103 | 82.743 | 4.538 | 0.344 | 0.287 |
| metal / gemm_513x257x129 | 0.053 | 48.165 | 1.486 | 0.953 | 83.341 | 4.167 | 0.415 | 0.304 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
