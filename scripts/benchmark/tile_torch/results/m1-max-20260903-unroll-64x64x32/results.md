# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T06:55:56.382164+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 64×64×32 / 1 | 32 | 5.432 | 29.049 | 5.709 | 30.207 | 0.19× | 258.458 | 368.417 |
| metal | gemm_128x128x128 | 64×64×32 / 1 | 32 | 11.986 | 29.561 | 12.221 | 30.636 | 0.41× | 273.875 | 303.667 |
| metal | gemm_512x512x512 | 64×64×32 / 1 | 32 | 55.268 | 48.575 | 55.809 | 50.729 | 1.14× | 290.667 | 410.417 |
| metal | gemm_1024x1024x1024 | 64×64×32 / 1 | 32 | 382.680 | 285.550 | 384.693 | 286.028 | 1.34× | 621.875 | 542.375 |
| metal | gemm_256x1024x128 | 64×64×32 / 1 | 32 | 18.112 | 31.611 | 18.152 | 32.275 | 0.57× | 232.875 | 284.709 |
| metal | gemm_1024x128x256 | 64×64×32 / 1 | 32 | 22.008 | 31.521 | 22.154 | 32.286 | 0.70× | 239.625 | 291.875 |
| metal | gemm_127x193x61 | 64×64×32 / 1 | 32 | 10.886 | 29.253 | 11.450 | 30.131 | 0.37× | 247.833 | 307.875 |
| metal | gemm_513x257x129 | 64×64×32 / 1 | 32 | 27.032 | 34.548 | 27.409 | 35.821 | 0.78× | 261.625 | 352.334 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.070 | 52.532 | 1.286 | 3.883 | 100.142 | 48.146 | 0.288 | 0.326 |
| metal / gemm_128x128x128 | 0.061 | 49.404 | 2.479 | 0.808 | 99.162 | 5.099 | 0.350 | 0.358 |
| metal / gemm_512x512x512 | 0.068 | 49.834 | 2.047 | 1.512 | 92.077 | 7.180 | 0.435 | 0.409 |
| metal / gemm_1024x1024x1024 | 0.064 | 49.149 | 3.703 | 2.793 | 98.455 | 3.551 | 1.504 | 0.422 |
| metal / gemm_256x1024x128 | 0.059 | 51.640 | 1.403 | 1.128 | 93.266 | 4.707 | 0.431 | 0.338 |
| metal / gemm_1024x128x256 | 0.056 | 51.888 | 1.625 | 0.943 | 93.776 | 3.888 | 0.470 | 0.314 |
| metal / gemm_127x193x61 | 0.060 | 62.468 | 2.112 | 1.065 | 107.351 | 4.569 | 0.322 | 0.339 |
| metal / gemm_513x257x129 | 0.058 | 66.704 | 1.514 | 0.796 | 111.585 | 3.576 | 0.425 | 0.322 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
