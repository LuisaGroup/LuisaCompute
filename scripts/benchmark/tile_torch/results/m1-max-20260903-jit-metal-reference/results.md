# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T01:49:41.464338+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 8×8×16 / 2 | 2 | 5.446 | 30.060 | 5.968 | 31.574 | 0.18× | 250.416 | 286.583 |
| metal | gemm_128x128x128 | 8×8×16 / 2 | 2 | 14.373 | 29.748 | 14.785 | 30.734 | 0.48× | 289.750 | 362.250 |
| metal | gemm_512x512x512 | 8×8×16 / 2 | 2 | 381.377 | 52.892 | 391.977 | 53.720 | 7.21× | 562.584 | 359.750 |
| metal | gemm_1024x1024x1024 | 8×8×16 / 2 | 2 | 2725.524 | 316.579 | 2886.683 | 323.054 | 8.61× | 3028.459 | 657.542 |
| metal | gemm_256x1024x128 | 8×8×16 / 2 | 2 | 94.483 | 31.514 | 97.379 | 32.325 | 3.00× | 348.916 | 295.000 |
| metal | gemm_1024x128x256 | 8×8×16 / 2 | 2 | 102.838 | 32.194 | 104.075 | 32.701 | 3.19× | 402.083 | 299.291 |
| metal | gemm_127x193x61 | 8×8×16 / 2 | 2 | 10.386 | 29.486 | 11.728 | 31.253 | 0.35× | 274.208 | 355.917 |
| metal | gemm_513x257x129 | 8×8×16 / 2 | 2 | 67.556 | 37.999 | 69.363 | 39.377 | 1.78× | 318.167 | 316.667 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.082 | 47.138 | 2.124 | 5.123 | 1.196 | 51.603 | 0.288 | 1.074 |
| metal / gemm_128x128x128 | 0.062 | 50.058 | 1.691 | 1.027 | 1.261 | 7.677 | 0.352 | 0.275 |
| metal / gemm_512x512x512 | 0.061 | 48.976 | 1.917 | 0.896 | 2.220 | 3.381 | 1.525 | 0.345 |
| metal / gemm_1024x1024x1024 | 0.058 | 46.858 | 3.640 | 1.996 | 5.020 | 3.575 | 0.858 | 0.426 |
| metal / gemm_256x1024x128 | 0.068 | 49.026 | 1.337 | 1.159 | 1.227 | 4.624 | 0.561 | 0.361 |
| metal / gemm_1024x128x256 | 0.056 | 48.283 | 1.905 | 1.177 | 1.351 | 6.291 | 0.377 | 0.353 |
| metal / gemm_127x193x61 | 0.053 | 53.780 | 2.460 | 0.992 | 1.233 | 4.444 | 0.284 | 0.300 |
| metal / gemm_513x257x129 | 0.058 | 55.134 | 1.533 | 0.942 | 1.298 | 3.799 | 0.370 | 0.307 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
