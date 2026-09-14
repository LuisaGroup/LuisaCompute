# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T00:00:58.414157+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Pipeline window: `2`; 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 8×8×16 | 0 | 6.989 | 30.200 | 7.059 | 33.261 | 0.23× | 220.167 | 282.208 |
| metal | gemm_128x128x128 | 8×8×16 | 0 | 18.570 | 33.527 | 19.739 | 34.364 | 0.55× | 253.958 | 289.750 |
| metal | gemm_512x512x512 | 8×8×16 | 0 | 583.395 | 59.861 | 622.430 | 60.364 | 9.75× | 683.667 | 311.291 |
| metal | gemm_1024x1024x1024 | 8×8×16 | 0 | 4617.558 | 353.615 | 4857.558 | 376.636 | 13.06× | 4935.958 | 543.250 |
| metal | gemm_256x1024x128 | 8×8×16 | 0 | 144.465 | 32.544 | 156.409 | 35.496 | 4.44× | 324.083 | 321.584 |
| metal | gemm_1024x128x256 | 8×8×16 | 0 | 145.985 | 33.147 | 158.320 | 34.847 | 4.40× | 361.708 | 272.542 |
| metal | gemm_127x193x61 | 8×8×16 | 0 | 13.553 | 29.571 | 13.652 | 31.209 | 0.46× | 241.500 | 385.958 |
| metal | gemm_513x257x129 | 8×8×16 | 0 | 107.070 | 45.286 | 110.522 | 46.151 | 2.36× | 329.542 | 305.542 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.083 | 47.587 | 1.200 | 4.068 | 3.652 | 55.340 | 0.308 | 0.566 |
| metal / gemm_128x128x128 | 0.050 | 48.496 | 1.790 | 0.655 | 1.369 | 5.928 | 0.294 | 0.320 |
| metal / gemm_512x512x512 | 0.052 | 48.651 | 1.668 | 0.718 | 1.961 | 3.812 | 0.448 | 0.334 |
| metal / gemm_1024x1024x1024 | 0.060 | 48.819 | 2.993 | 2.023 | 6.133 | 7.108 | 1.420 | 0.456 |
| metal / gemm_256x1024x128 | 0.060 | 49.177 | 2.316 | 1.613 | 1.819 | 8.343 | 0.503 | 0.321 |
| metal / gemm_1024x128x256 | 0.054 | 48.063 | 1.828 | 0.721 | 3.920 | 3.836 | 0.380 | 0.288 |
| metal / gemm_127x193x61 | 0.057 | 55.101 | 1.521 | 1.082 | 2.809 | 6.657 | 0.359 | 0.297 |
| metal / gemm_513x257x129 | 0.058 | 54.267 | 2.553 | 0.928 | 1.497 | 3.836 | 0.404 | 0.340 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
