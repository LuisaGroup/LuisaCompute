# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T01:52:37.491145+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 8×8×16 / 2 | 0 | 5.160 | 0.930 | 5.688 | 0.949 | 5.55× | 7.000 | 0.958 |
| cpu | gemm_128x128x128 | 8×8×16 / 2 | 0 | 60.924 | 4.919 | 81.911 | 5.015 | 12.39× | 52.625 | 4.708 |
| cpu | gemm_512x512x512 | 8×8×16 / 2 | 0 | 1739.992 | 166.799 | 1822.160 | 172.073 | 10.43× | 1666.666 | 185.542 |
| cpu | gemm_1024x1024x1024 | 8×8×16 / 2 | 0 | 12339.459 | 1308.905 | 13475.617 | 1435.733 | 9.43× | 13304.042 | 1138.959 |
| cpu | gemm_256x1024x128 | 8×8×16 / 2 | 0 | 703.314 | 68.775 | 963.468 | 70.099 | 10.23× | 592.416 | 65.792 |
| cpu | gemm_1024x128x256 | 8×8×16 / 2 | 0 | 562.402 | 65.500 | 915.198 | 67.209 | 8.59× | 396.292 | 64.042 |
| cpu | gemm_127x193x61 | 8×8×16 / 2 | 0 | 129.701 | 6.903 | 221.612 | 7.996 | 18.79× | 85.500 | 6.375 |
| cpu | gemm_513x257x129 | 8×8×16 / 2 | 0 | 830.488 | 46.059 | 909.966 | 54.851 | 18.03× | 684.500 | 43.542 |
| cpu | add_1x127 | 1×256×1 / 2 | 0 | 0.431 | 0.598 | 0.434 | 0.603 | 0.72× | 0.458 | 0.708 |
| cpu | add_17x257 | 1×256×1 / 2 | 0 | 8.696 | 1.002 | 12.123 | 1.013 | 8.68× | 4.084 | 1.125 |
| cpu | add_128x1024 | 1×256×1 / 2 | 0 | 20.717 | 43.652 | 32.013 | 46.023 | 0.47× | 14.709 | 51.542 |
| cpu | add_4096x256 | 1×256×1 / 2 | 0 | 110.392 | 93.063 | 120.559 | 97.310 | 1.19× | 122.042 | 88.000 |
| cpu | sum_1x127 | 1×127×1 / 2 | 0 | 2.152 | 0.856 | 2.161 | 0.862 | 2.51× | 2.125 | 1.041 |
| cpu | sum_17x257 | 1×257×1 / 2 | 0 | 22.336 | 1.185 | 32.874 | 1.191 | 18.85× | 21.500 | 1.292 |
| cpu | sum_128x1024 | 1×1024×1 / 2 | 0 | 813.724 | 41.899 | 1019.142 | 42.892 | 19.42× | 775.083 | 42.625 |
| cpu | sum_64x4096 | 1×4096×1 / 2 | 0 | 1028.050 | 49.374 | 1220.971 | 51.405 | 20.82× | 1330.042 | 44.667 |
| cpu | softmax_1x127 | 1×127×1 / 2 | 0 | 5.966 | 0.655 | 6.021 | 0.667 | 9.11× | 5.833 | 0.709 |
| cpu | softmax_17x257 | 1×257×1 / 2 | 0 | 73.384 | 39.250 | 77.403 | 42.197 | 1.87× | 40.541 | 44.125 |
| cpu | softmax_128x1024 | 1×1024×1 / 2 | 0 | 1916.958 | 89.788 | 2744.622 | 91.636 | 21.35× | 1614.416 | 87.417 |
| cpu | softmax_64x4096 | 1×4096×1 / 2 | 0 | 2847.512 | 152.126 | 3298.448 | 154.018 | 18.72× | 3804.625 | 176.791 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.047 | 290.438 | 0.004 | 0.061 | 0.148 | 0.042 | 0.002 | 0.029 |
| cpu / gemm_128x128x128 | 0.047 | 235.403 | 0.007 | 0.013 | 0.154 | 0.010 | 0.006 | 0.006 |
| cpu / gemm_512x512x512 | 0.064 | 233.532 | 0.219 | 0.029 | 1.527 | 0.379 | 0.118 | 0.006 |
| cpu / gemm_1024x1024x1024 | 0.047 | 240.189 | 0.834 | 0.040 | 11.598 | 3.047 | 0.493 | 0.008 |
| cpu / gemm_256x1024x128 | 0.051 | 238.960 | 0.017 | 0.038 | 0.603 | 0.178 | 0.040 | 0.006 |
| cpu / gemm_1024x128x256 | 0.046 | 236.237 | 0.039 | 0.026 | 0.580 | 0.094 | 0.024 | 0.006 |
| cpu / gemm_127x193x61 | 0.048 | 216.423 | 0.011 | 0.026 | 0.198 | 0.043 | 0.006 | 0.006 |
| cpu / gemm_513x257x129 | 0.087 | 211.901 | 0.046 | 0.017 | 0.470 | 0.061 | 0.055 | 0.006 |
| cpu / add_1x127 | 0.045 | 42.155 | 0.003 | 0.028 | 0.012 | 0.075 | 0.002 | 0.006 |
| cpu / add_17x257 | 0.049 | 47.820 | 0.006 | 0.014 | 0.112 | 0.003 | 0.009 | 0.006 |
| cpu / add_128x1024 | 0.042 | 98.095 | 0.032 | 0.025 | 0.164 | 0.052 | 0.053 | 0.007 |
| cpu / add_4096x256 | 0.042 | 97.217 | 0.823 | 0.028 | 0.627 | 0.151 | 0.453 | 0.006 |
| cpu / sum_1x127 | 0.055 | 34.493 | 0.003 | 0.019 | 0.018 | 0.059 | 0.001 | 0.027 |
| cpu / sum_17x257 | 0.045 | 46.772 | 0.006 | 0.015 | 0.136 | 0.004 | 0.002 | 0.005 |
| cpu / sum_128x1024 | 0.041 | 39.798 | 0.029 | 0.017 | 0.907 | 0.118 | 0.001 | 0.005 |
| cpu / sum_64x4096 | 0.044 | 38.469 | 0.093 | 0.017 | 0.866 | 0.040 | 0.001 | 0.006 |
| cpu / softmax_1x127 | 0.054 | 40.139 | 0.006 | 0.027 | 0.019 | 0.062 | 0.001 | 0.006 |
| cpu / softmax_17x257 | 0.064 | 54.158 | 0.006 | 0.013 | 0.327 | 0.069 | 0.010 | 0.007 |
| cpu / softmax_128x1024 | 0.056 | 45.663 | 0.052 | 0.032 | 1.965 | 0.104 | 0.030 | 0.006 |
| cpu / softmax_64x4096 | 0.060 | 44.690 | 0.102 | 0.015 | 2.922 | 0.139 | 0.132 | 0.008 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
