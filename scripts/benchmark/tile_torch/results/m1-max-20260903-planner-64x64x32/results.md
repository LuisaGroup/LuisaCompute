# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T04:35:33.347120+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 64×64×32 / 1 | 8 | 6.450 | 27.475 | 6.504 | 28.042 | 0.23× | 238.167 | 272.125 |
| metal | gemm_128x128x128 | 64×64×32 / 1 | 8 | 13.567 | 27.046 | 13.880 | 27.902 | 0.50× | 253.084 | 284.000 |
| metal | gemm_512x512x512 | 64×64×32 / 1 | 8 | 56.579 | 48.078 | 57.504 | 48.763 | 1.18× | 288.542 | 296.875 |
| metal | gemm_1024x1024x1024 | 64×64×32 / 1 | 8 | 407.675 | 298.286 | 414.841 | 299.983 | 1.37× | 615.208 | 545.875 |
| metal | gemm_256x1024x128 | 64×64×32 / 1 | 8 | 19.447 | 29.534 | 19.851 | 30.100 | 0.66× | 253.917 | 264.625 |
| metal | gemm_1024x128x256 | 64×64×32 / 1 | 8 | 24.851 | 30.391 | 25.053 | 30.808 | 0.82× | 273.875 | 291.167 |
| metal | gemm_127x193x61 | 64×64×32 / 1 | 8 | 17.905 | 27.664 | 18.040 | 29.357 | 0.65× | 285.042 | 278.250 |
| metal | gemm_513x257x129 | 64×64×32 / 1 | 8 | 36.581 | 34.756 | 36.683 | 35.010 | 1.05× | 278.833 | 298.042 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.059 | 46.956 | 1.365 | 4.204 | 77.368 | 43.106 | 0.254 | 0.334 |
| metal / gemm_128x128x128 | 0.053 | 44.827 | 2.203 | 0.581 | 78.756 | 3.877 | 0.304 | 0.287 |
| metal / gemm_512x512x512 | 0.061 | 45.086 | 1.850 | 1.118 | 81.315 | 8.022 | 0.482 | 0.333 |
| metal / gemm_1024x1024x1024 | 0.052 | 48.998 | 3.976 | 1.967 | 87.211 | 3.300 | 0.992 | 0.406 |
| metal / gemm_256x1024x128 | 0.056 | 44.579 | 1.426 | 1.141 | 78.688 | 4.122 | 0.607 | 0.292 |
| metal / gemm_1024x128x256 | 0.061 | 44.855 | 1.764 | 1.070 | 81.093 | 3.410 | 0.431 | 0.312 |
| metal / gemm_127x193x61 | 0.054 | 50.405 | 1.492 | 1.210 | 84.901 | 4.718 | 0.276 | 0.311 |
| metal / gemm_513x257x129 | 0.056 | 51.580 | 1.220 | 0.952 | 84.711 | 8.133 | 0.440 | 0.340 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
