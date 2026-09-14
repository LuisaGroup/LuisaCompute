# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T00:02:48.039293+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Pipeline window: `2`; 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 8×8×16 | 0 | 5.636 | 0.919 | 7.512 | 0.935 | 6.14× | 3.125 | 0.958 |
| cpu | gemm_128x128x128 | 8×8×16 | 0 | 58.146 | 4.961 | 77.103 | 8.131 | 11.72× | 26.792 | 5.041 |
| cpu | gemm_512x512x512 | 8×8×16 | 0 | 1728.875 | 144.225 | 1795.276 | 150.289 | 11.99× | 1551.000 | 139.041 |
| cpu | gemm_1024x1024x1024 | 8×8×16 | 0 | 12425.417 | 1102.796 | 12917.442 | 1224.067 | 11.27× | 11450.625 | 1058.625 |
| cpu | gemm_256x1024x128 | 8×8×16 | 0 | 585.056 | 70.366 | 776.651 | 72.313 | 8.31× | 554.208 | 65.750 |
| cpu | gemm_1024x128x256 | 8×8×16 | 0 | 630.605 | 64.846 | 1763.264 | 65.576 | 9.72× | 1395.167 | 63.083 |
| cpu | gemm_127x193x61 | 8×8×16 | 0 | 68.115 | 6.479 | 92.673 | 6.552 | 10.51× | 33.917 | 6.667 |
| cpu | gemm_513x257x129 | 8×8×16 | 0 | 667.339 | 43.591 | 748.920 | 44.404 | 15.31× | 413.708 | 47.292 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.044 | 292.161 | 0.003 | 0.069 | 0.139 | 0.048 | 0.007 | 0.015 |
| cpu / gemm_128x128x128 | 0.048 | 235.313 | 0.007 | 0.012 | 0.267 | 0.014 | 0.004 | 0.006 |
| cpu / gemm_512x512x512 | 0.058 | 235.485 | 0.166 | 0.034 | 2.391 | 0.342 | 0.175 | 0.006 |
| cpu / gemm_1024x1024x1024 | 0.058 | 241.993 | 0.767 | 0.037 | 12.448 | 1.424 | 0.514 | 0.007 |
| cpu / gemm_256x1024x128 | 0.048 | 241.021 | 0.031 | 0.025 | 0.975 | 0.119 | 0.055 | 0.006 |
| cpu / gemm_1024x128x256 | 0.048 | 233.773 | 0.180 | 0.019 | 0.431 | 0.122 | 0.067 | 0.005 |
| cpu / gemm_127x193x61 | 0.049 | 209.579 | 0.010 | 0.021 | 0.249 | 0.031 | 0.014 | 0.006 |
| cpu / gemm_513x257x129 | 0.070 | 211.352 | 0.013 | 0.019 | 0.784 | 0.060 | 0.130 | 0.006 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
