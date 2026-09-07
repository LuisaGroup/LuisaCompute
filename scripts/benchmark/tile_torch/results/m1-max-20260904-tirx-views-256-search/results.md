# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T06:53:24.365776+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×32×32 / 1 | 1 | 3.333 | 26.791 | 3.536 | 26.881 | 0.12× | 196.167 | 229.708 |
| metal | gemm_128x128x128 | 32×32×128 / 1 | 1 | 5.787 | 27.383 | 6.081 | 27.423 | 0.21× | 223.125 | 296.000 |
| metal | gemm_512x512x512 | 64×64×32 / 1 | 1 | 44.509 | 49.212 | 47.196 | 49.225 | 0.90× | 270.000 | 276.334 |
| metal | gemm_1024x1024x1024 | 64×64×32 / 1 | 1 | 300.342 | 293.054 | 315.081 | 308.782 | 1.02× | 493.750 | 524.500 |
| metal | gemm_256x1024x128 | 64×64×32 / 1 | 1 | 15.763 | 29.606 | 16.733 | 29.980 | 0.53× | 209.209 | 249.125 |
| metal | gemm_1024x128x256 | 32×32×32 / 1 | 1 | 18.167 | 29.644 | 19.015 | 30.047 | 0.61× | 248.750 | 246.333 |
| metal | gemm_127x193x61 | 32×32×32 / 1 | 1 | 6.572 | 27.705 | 6.751 | 27.940 | 0.24× | 243.208 | 249.167 |
| metal | gemm_513x257x129 | 32×64×32 / 1 | 1 | 20.692 | 35.279 | 21.914 | 37.478 | 0.59× | 220.000 | 289.959 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.049 | 27.516 | 1.474 | 1.304 | 0.711 | 0.405 | 0.233 | 0.230 |
| metal / gemm_128x128x128 | 0.050 | 27.850 | 1.508 | 1.143 | 1.685 | 0.457 | 0.250 | 0.287 |
| metal / gemm_512x512x512 | 0.051 | 28.003 | 1.598 | 1.184 | 5.029 | 0.449 | 0.362 | 0.346 |
| metal / gemm_1024x1024x1024 | 0.053 | 28.237 | 2.883 | 1.353 | 4.779 | 0.710 | 0.730 | 0.414 |
| metal / gemm_256x1024x128 | 0.052 | 28.106 | 1.466 | 1.066 | 4.266 | 0.413 | 0.536 | 0.334 |
| metal / gemm_1024x128x256 | 0.048 | 28.013 | 1.724 | 1.178 | 3.953 | 0.481 | 0.322 | 0.259 |
| metal / gemm_127x193x61 | 0.051 | 34.761 | 1.267 | 1.051 | 1.115 | 0.435 | 0.288 | 0.263 |
| metal / gemm_513x257x129 | 0.057 | 43.456 | 1.748 | 1.060 | 1.460 | 1.310 | 0.324 | 0.267 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

| Device / case | Valid / attempted candidates | Selection wall ms |
|---|---:|---:|
| metal / gemm_32x32x32 | 8 / 12 | 5290.857 |
| metal / gemm_128x128x128 | 10 / 12 | 5162.248 |
| metal / gemm_512x512x512 | 10 / 12 | 5103.348 |
| metal / gemm_1024x1024x1024 | 12 / 12 | 6211.677 |
| metal / gemm_256x1024x128 | 10 / 12 | 5148.648 |
| metal / gemm_1024x128x256 | 10 / 12 | 5128.264 |
| metal / gemm_127x193x61 | 6 / 12 | 4343.589 |
| metal / gemm_513x257x129 | 6 / 12 | 4300.525 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
