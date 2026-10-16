# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T01:40:23.262584+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `False`. When enabled, CPU independent-element domains are packed into SIMD without changing inner serial/reduction order. Disabling this does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 8×8×16 / 2 | 0 | 4.626 | 0.884 | 5.627 | 0.917 | 5.23× | 2.916 | 1.042 |
| cpu | gemm_128x128x128 | 8×8×16 / 2 | 0 | 40.700 | 4.970 | 57.039 | 5.127 | 8.19× | 23.833 | 5.166 |
| cpu | gemm_512x512x512 | 8×8×16 / 2 | 0 | 1717.091 | 142.835 | 2161.374 | 146.572 | 12.02× | 1307.167 | 139.875 |
| cpu | gemm_1024x1024x1024 | 8×8×16 / 2 | 0 | 11094.562 | 1026.233 | 12622.937 | 1052.705 | 10.81× | 10910.000 | 947.416 |
| cpu | gemm_256x1024x128 | 8×8×16 / 2 | 0 | 541.768 | 68.357 | 564.277 | 68.846 | 7.93× | 545.750 | 67.292 |
| cpu | gemm_1024x128x256 | 8×8×16 / 2 | 0 | 470.432 | 65.449 | 644.678 | 65.836 | 7.19× | 416.959 | 62.958 |
| cpu | gemm_127x193x61 | 8×8×16 / 2 | 0 | 50.853 | 6.846 | 74.059 | 6.925 | 7.43× | 92.375 | 6.916 |
| cpu | gemm_513x257x129 | 8×8×16 / 2 | 0 | 430.237 | 45.089 | 535.469 | 46.486 | 9.54× | 348.333 | 43.417 |
| cpu | add_1x127 | 1×256×1 / 2 | 0 | 0.425 | 0.565 | 0.431 | 0.573 | 0.75× | 0.458 | 0.666 |
| cpu | add_17x257 | 1×256×1 / 2 | 0 | 8.649 | 0.969 | 11.941 | 0.979 | 8.93× | 4.750 | 1.083 |
| cpu | add_128x1024 | 1×256×1 / 2 | 0 | 21.688 | 41.332 | 38.854 | 43.766 | 0.52× | 29.000 | 37.792 |
| cpu | add_4096x256 | 1×256×1 / 2 | 0 | 119.781 | 87.642 | 129.181 | 88.897 | 1.37× | 76.583 | 84.125 |
| cpu | sum_1x127 | 1×127×1 / 2 | 0 | 2.100 | 0.820 | 2.117 | 0.830 | 2.56× | 2.125 | 0.917 |
| cpu | sum_17x257 | 1×257×1 / 2 | 0 | 29.657 | 1.114 | 32.759 | 1.129 | 26.62× | 17.916 | 1.250 |
| cpu | sum_128x1024 | 1×1024×1 / 2 | 0 | 476.102 | 40.730 | 518.769 | 41.596 | 11.69× | 593.834 | 46.709 |
| cpu | sum_64x4096 | 1×4096×1 / 2 | 0 | 1302.926 | 45.093 | 1521.177 | 45.972 | 28.89× | 1235.084 | 42.917 |
| cpu | softmax_1x127 | 1×127×1 / 2 | 0 | 5.892 | 0.633 | 6.007 | 0.636 | 9.31× | 5.833 | 0.708 |
| cpu | softmax_17x257 | 1×257×1 / 2 | 0 | 90.822 | 38.860 | 137.330 | 41.344 | 2.34× | 79.542 | 39.125 |
| cpu | softmax_128x1024 | 1×1024×1 / 2 | 0 | 1250.642 | 90.031 | 1760.992 | 93.984 | 13.89× | 1071.875 | 86.458 |
| cpu | softmax_64x4096 | 1×4096×1 / 2 | 0 | 1935.750 | 122.810 | 2091.888 | 123.596 | 15.76× | 1684.958 | 132.792 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.067 | 290.451 | 0.004 | 0.310 | 0.114 | 0.379 | 0.001 | 0.032 |
| cpu / gemm_128x128x128 | 0.063 | 230.990 | 0.008 | 0.011 | 0.217 | 0.007 | 0.011 | 0.006 |
| cpu / gemm_512x512x512 | 0.054 | 233.491 | 0.180 | 0.035 | 2.155 | 0.337 | 0.146 | 0.006 |
| cpu / gemm_1024x1024x1024 | 0.045 | 237.423 | 0.869 | 0.036 | 9.893 | 1.395 | 0.528 | 0.006 |
| cpu / gemm_256x1024x128 | 0.042 | 232.272 | 0.027 | 0.029 | 0.532 | 0.170 | 0.128 | 0.006 |
| cpu / gemm_1024x128x256 | 0.065 | 231.099 | 0.089 | 0.023 | 0.433 | 0.098 | 0.032 | 0.006 |
| cpu / gemm_127x193x61 | 0.046 | 206.967 | 0.006 | 0.018 | 0.170 | 0.041 | 0.055 | 0.007 |
| cpu / gemm_513x257x129 | 0.042 | 209.719 | 0.018 | 0.025 | 0.475 | 0.071 | 0.019 | 0.005 |
| cpu / add_1x127 | 0.037 | 40.106 | 0.005 | 0.022 | 0.010 | 0.085 | 0.001 | 0.005 |
| cpu / add_17x257 | 0.043 | 47.417 | 0.006 | 0.013 | 0.118 | 0.003 | 0.010 | 0.005 |
| cpu / add_128x1024 | 0.039 | 97.817 | 0.026 | 0.030 | 0.118 | 0.075 | 0.057 | 0.006 |
| cpu / add_4096x256 | 0.035 | 94.654 | 0.749 | 0.032 | 0.480 | 0.150 | 0.440 | 0.006 |
| cpu / sum_1x127 | 0.065 | 33.334 | 0.004 | 0.023 | 0.013 | 0.545 | 0.001 | 0.006 |
| cpu / sum_17x257 | 0.050 | 45.148 | 0.004 | 0.025 | 0.147 | 0.012 | 0.001 | 0.005 |
| cpu / sum_128x1024 | 0.044 | 37.906 | 0.053 | 0.024 | 0.399 | 0.122 | 0.001 | 0.006 |
| cpu / sum_64x4096 | 0.046 | 37.764 | 0.114 | 0.029 | 1.162 | 0.059 | 0.001 | 0.006 |
| cpu / softmax_1x127 | 0.053 | 39.242 | 0.006 | 0.021 | 0.020 | 0.760 | 0.002 | 0.006 |
| cpu / softmax_17x257 | 0.067 | 50.785 | 0.004 | 0.012 | 0.157 | 0.047 | 0.006 | 0.007 |
| cpu / softmax_128x1024 | 0.062 | 44.571 | 0.030 | 0.024 | 1.297 | 0.138 | 0.035 | 0.007 |
| cpu / softmax_64x4096 | 0.058 | 44.536 | 0.139 | 0.018 | 1.642 | 0.149 | 0.101 | 0.006 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
