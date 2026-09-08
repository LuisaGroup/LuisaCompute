# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T04:25:17.467224+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×64×32 / 1 | 4 | 5.933 | 26.515 | 6.264 | 28.044 | 0.22× | 225.291 | 292.500 |
| metal | gemm_128x128x128 | 32×64×32 / 1 | 4 | 10.949 | 26.711 | 12.085 | 28.580 | 0.41× | 278.666 | 258.500 |
| metal | gemm_512x512x512 | 32×64×32 / 1 | 4 | 74.780 | 48.572 | 75.177 | 49.004 | 1.54× | 298.584 | 304.333 |
| metal | gemm_1024x1024x1024 | 32×64×32 / 1 | 4 | 464.600 | 288.503 | 474.989 | 292.945 | 1.61× | 669.500 | 518.500 |
| metal | gemm_256x1024x128 | 32×64×32 / 1 | 4 | 25.012 | 29.879 | 25.223 | 31.192 | 0.84× | 249.250 | 282.458 |
| metal | gemm_1024x128x256 | 32×64×32 / 1 | 4 | 22.526 | 30.022 | 23.280 | 30.222 | 0.75× | 279.542 | 246.750 |
| metal | gemm_127x193x61 | 32×64×32 / 1 | 4 | 13.029 | 27.450 | 13.498 | 28.810 | 0.47× | 235.208 | 267.542 |
| metal | gemm_513x257x129 | 32×64×32 / 1 | 4 | 28.756 | 35.024 | 29.008 | 35.262 | 0.82× | 282.500 | 273.583 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.053 | 43.198 | 1.180 | 3.030 | 64.612 | 43.602 | 0.292 | 0.355 |
| metal / gemm_128x128x128 | 0.053 | 42.260 | 1.434 | 0.676 | 68.400 | 7.426 | 0.383 | 0.318 |
| metal / gemm_512x512x512 | 0.055 | 42.754 | 1.953 | 1.142 | 66.564 | 4.495 | 0.432 | 0.355 |
| metal / gemm_1024x1024x1024 | 0.063 | 42.341 | 2.794 | 1.975 | 66.415 | 4.431 | 1.026 | 0.402 |
| metal / gemm_256x1024x128 | 0.055 | 42.683 | 1.431 | 1.054 | 68.655 | 8.767 | 0.504 | 0.327 |
| metal / gemm_1024x128x256 | 0.057 | 42.408 | 2.546 | 0.763 | 68.143 | 3.425 | 0.428 | 0.305 |
| metal / gemm_127x193x61 | 0.058 | 47.976 | 1.393 | 1.184 | 70.131 | 4.458 | 0.325 | 0.311 |
| metal / gemm_513x257x129 | 0.054 | 47.570 | 1.578 | 0.732 | 70.101 | 4.119 | 0.438 | 0.342 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
