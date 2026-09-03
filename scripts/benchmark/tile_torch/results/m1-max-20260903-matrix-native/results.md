# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T00:01:39.010979+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Pipeline window: `2`; 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 8×8×16 | 2 | 6.410 | 29.298 | 7.040 | 33.705 | 0.22× | 246.875 | 432.042 |
| metal | gemm_128x128x128 | 8×8×16 | 2 | 16.170 | 30.717 | 18.640 | 34.430 | 0.53× | 260.167 | 293.083 |
| metal | gemm_512x512x512 | 8×8×16 | 2 | 427.523 | 59.895 | 434.717 | 61.221 | 7.14× | 561.208 | 321.083 |
| metal | gemm_1024x1024x1024 | 8×8×16 | 2 | 3244.304 | 344.738 | 3494.556 | 374.186 | 9.41× | 3174.625 | 576.625 |
| metal | gemm_256x1024x128 | 8×8×16 | 2 | 109.533 | 32.808 | 121.147 | 36.263 | 3.34× | 358.584 | 280.334 |
| metal | gemm_1024x128x256 | 8×8×16 | 2 | 114.866 | 34.434 | 122.318 | 35.029 | 3.34× | 347.833 | 273.209 |
| metal | gemm_127x193x61 | 8×8×16 | 2 | 11.839 | 30.830 | 12.066 | 33.758 | 0.38× | 260.875 | 296.375 |
| metal | gemm_513x257x129 | 8×8×16 | 2 | 78.058 | 41.485 | 81.175 | 45.423 | 1.88× | 281.792 | 390.000 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.061 | 47.433 | 3.995 | 3.699 | 67.239 | 52.393 | 0.225 | 0.380 |
| metal / gemm_128x128x128 | 0.052 | 48.070 | 2.273 | 0.578 | 67.751 | 4.361 | 0.333 | 0.317 |
| metal / gemm_512x512x512 | 0.054 | 47.694 | 1.579 | 1.828 | 68.666 | 7.703 | 0.420 | 0.334 |
| metal / gemm_1024x1024x1024 | 0.058 | 49.368 | 4.376 | 2.116 | 73.485 | 5.950 | 0.884 | 0.422 |
| metal / gemm_256x1024x128 | 0.075 | 49.258 | 3.307 | 1.327 | 67.638 | 4.935 | 0.510 | 2.083 |
| metal / gemm_1024x128x256 | 0.058 | 48.795 | 1.694 | 1.083 | 66.812 | 4.238 | 0.346 | 0.306 |
| metal / gemm_127x193x61 | 0.068 | 55.665 | 1.556 | 1.198 | 67.841 | 8.654 | 0.358 | 0.316 |
| metal / gemm_513x257x129 | 0.051 | 55.588 | 2.450 | 1.942 | 69.167 | 8.096 | 1.607 | 0.328 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
