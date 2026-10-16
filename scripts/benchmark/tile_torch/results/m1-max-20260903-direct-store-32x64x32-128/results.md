# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T05:43:15.990784+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×64×32 / 1 | 8 | 6.443 | 26.586 | 6.628 | 27.254 | 0.24× | 244.625 | 272.334 |
| metal | gemm_128x128x128 | 32×64×32 / 1 | 8 | 12.699 | 26.629 | 12.991 | 26.732 | 0.48× | 270.709 | 266.416 |
| metal | gemm_512x512x512 | 32×64×32 / 1 | 8 | 58.999 | 48.258 | 59.358 | 48.603 | 1.22× | 289.208 | 300.667 |
| metal | gemm_1024x1024x1024 | 32×64×32 / 1 | 8 | 399.166 | 287.810 | 405.343 | 291.975 | 1.39× | 608.042 | 528.625 |
| metal | gemm_256x1024x128 | 32×64×32 / 1 | 8 | 20.438 | 30.385 | 20.511 | 31.230 | 0.67× | 233.083 | 254.208 |
| metal | gemm_1024x128x256 | 32×64×32 / 1 | 8 | 23.997 | 29.131 | 24.408 | 30.299 | 0.82× | 243.292 | 253.542 |
| metal | gemm_127x193x61 | 32×64×32 / 1 | 8 | 12.614 | 26.390 | 12.850 | 27.927 | 0.48× | 264.833 | 276.708 |
| metal | gemm_513x257x129 | 32×64×32 / 1 | 8 | 27.114 | 34.831 | 28.630 | 34.988 | 0.78× | 261.083 | 269.292 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.051 | 44.579 | 1.297 | 3.244 | 79.501 | 44.332 | 0.305 | 0.329 |
| metal / gemm_128x128x128 | 0.126 | 43.897 | 1.312 | 2.205 | 71.909 | 4.703 | 0.322 | 0.293 |
| metal / gemm_512x512x512 | 0.063 | 43.897 | 1.846 | 1.187 | 72.347 | 3.856 | 0.490 | 0.339 |
| metal / gemm_1024x1024x1024 | 0.056 | 44.074 | 2.840 | 1.961 | 75.062 | 3.512 | 1.003 | 0.408 |
| metal / gemm_256x1024x128 | 0.077 | 44.889 | 1.595 | 1.074 | 72.077 | 4.047 | 0.500 | 0.350 |
| metal / gemm_1024x128x256 | 0.057 | 46.229 | 1.762 | 1.157 | 72.600 | 3.769 | 0.379 | 0.304 |
| metal / gemm_127x193x61 | 0.062 | 58.029 | 1.565 | 1.241 | 88.879 | 6.622 | 0.315 | 0.300 |
| metal / gemm_513x257x129 | 0.054 | 62.766 | 1.681 | 0.585 | 89.617 | 3.784 | 0.452 | 0.307 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
