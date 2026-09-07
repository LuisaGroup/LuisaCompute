# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T04:24:42.067797+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×64×32 / 1 | 16 | 15.329 | 26.295 | 15.887 | 26.470 | 0.58× | 236.541 | 277.958 |
| metal | gemm_128x128x128 | 32×64×32 / 1 | 16 | 18.131 | 26.262 | 19.571 | 27.730 | 0.69× | 282.250 | 275.833 |
| metal | gemm_512x512x512 | 32×64×32 / 1 | 16 | 117.474 | 48.156 | 118.179 | 49.114 | 2.44× | 382.333 | 297.750 |
| metal | gemm_1024x1024x1024 | 32×64×32 / 1 | 16 | 662.563 | 289.230 | 672.063 | 294.472 | 2.29× | 885.792 | 522.667 |
| metal | gemm_256x1024x128 | 32×64×32 / 1 | 16 | 36.244 | 29.784 | 36.471 | 30.075 | 1.22× | 284.125 | 244.875 |
| metal | gemm_1024x128x256 | 32×64×32 / 1 | 16 | 33.495 | 29.545 | 34.124 | 30.659 | 1.13× | 245.042 | 259.292 |
| metal | gemm_127x193x61 | 32×64×32 / 1 | 16 | 41.277 | 27.477 | 41.693 | 28.173 | 1.50× | 275.166 | 280.958 |
| metal | gemm_513x257x129 | 32×64×32 / 1 | 16 | 74.754 | 35.181 | 76.353 | 35.598 | 2.12× | 301.250 | 285.084 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.071 | 44.489 | 1.486 | 4.616 | 111.435 | 51.081 | 0.253 | 0.359 |
| metal / gemm_128x128x128 | 0.052 | 46.681 | 1.629 | 0.779 | 100.840 | 4.307 | 0.396 | 0.300 |
| metal / gemm_512x512x512 | 0.053 | 44.711 | 1.437 | 2.241 | 100.551 | 3.906 | 0.513 | 0.361 |
| metal / gemm_1024x1024x1024 | 0.064 | 45.821 | 3.594 | 1.997 | 103.498 | 3.756 | 0.941 | 0.477 |
| metal / gemm_256x1024x128 | 0.057 | 45.421 | 1.654 | 1.119 | 99.320 | 4.375 | 0.525 | 0.319 |
| metal / gemm_1024x128x256 | 0.060 | 44.776 | 1.358 | 1.045 | 100.720 | 4.044 | 0.370 | 0.294 |
| metal / gemm_127x193x61 | 0.060 | 49.397 | 1.542 | 1.218 | 107.621 | 5.314 | 0.315 | 0.306 |
| metal / gemm_513x257x129 | 0.061 | 48.794 | 1.275 | 1.035 | 105.244 | 3.793 | 0.388 | 0.363 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
