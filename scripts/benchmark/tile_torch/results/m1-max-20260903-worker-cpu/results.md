# TileIR/TVMx vs PyTorch

Generated: 2026-09-02T19:56:52.835291+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR, but the current TVMx schedule lowers it to loops: no tensor-core claim. Pipeline stage markers currently run serially. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 8×8×16 | 9.228 | 0.898 | 12.485 | 0.918 | 10.28× | 3.875 | 1.125 |
| cpu | gemm_128x128x128 | 8×8×16 | 215.421 | 4.919 | 311.983 | 4.976 | 43.79× | 124.542 | 4.709 |
| cpu | gemm_512x512x512 | 8×8×16 | 9349.041 | 140.667 | 9731.175 | 141.785 | 66.46× | 9546.000 | 143.167 |
| cpu | gemm_1024x1024x1024 | 8×8×16 | 68325.958 | 1020.056 | 70056.309 | 1048.734 | 66.98× | 70936.125 | 985.708 |
| cpu | gemm_256x1024x128 | 8×8×16 | 2782.863 | 68.682 | 3187.652 | 69.319 | 40.52× | 3065.125 | 65.791 |
| cpu | gemm_1024x128x256 | 8×8×16 | 2420.526 | 64.478 | 2531.743 | 65.116 | 37.54× | 2325.208 | 63.375 |
| cpu | gemm_127x193x61 | 8×8×16 | 203.369 | 6.622 | 219.996 | 6.685 | 30.71× | 149.500 | 6.500 |
| cpu | gemm_513x257x129 | 8×8×16 | 2134.759 | 45.000 | 2596.882 | 45.348 | 47.44× | 2023.542 | 43.541 |
| cpu | add_1x127 | 1×256×1 | 0.431 | 0.566 | 0.433 | 0.574 | 0.76× | 0.458 | 0.708 |
| cpu | add_17x257 | 1×256×1 | 7.041 | 0.967 | 7.632 | 0.986 | 7.28× | 4.000 | 1.083 |
| cpu | add_128x1024 | 1×256×1 | 21.870 | 42.573 | 26.240 | 44.515 | 0.51× | 50.792 | 59.917 |
| cpu | add_4096x256 | 1×256×1 | 133.690 | 84.501 | 140.267 | 88.066 | 1.58× | 91.750 | 97.000 |
| cpu | sum_1x127 | 1×127×1 | 2.144 | 0.848 | 2.159 | 0.849 | 2.53× | 2.209 | 0.917 |
| cpu | sum_17x257 | 1×257×1 | 36.708 | 1.137 | 43.198 | 1.156 | 32.28× | 37.500 | 1.209 |
| cpu | sum_128x1024 | 1×1024×1 | 405.393 | 41.780 | 449.781 | 42.427 | 9.70× | 380.667 | 41.625 |
| cpu | sum_64x4096 | 1×4096×1 | 1257.042 | 44.603 | 1391.474 | 45.847 | 28.18× | 1236.125 | 37.208 |
| cpu | softmax_1x127 | 1×127×1 | 5.967 | 0.646 | 6.026 | 0.649 | 9.24× | 5.833 | 0.750 |
| cpu | softmax_17x257 | 1×257×1 | 52.212 | 38.311 | 62.548 | 39.523 | 1.36× | 38.500 | 33.708 |
| cpu | softmax_128x1024 | 1×1024×1 | 2095.426 | 89.137 | 2926.367 | 90.241 | 23.51× | 1937.291 | 87.375 |
| cpu | softmax_64x4096 | 1×4096×1 | 3842.573 | 126.125 | 4244.561 | 128.107 | 30.47× | 3095.083 | 114.792 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.048 | 73.229 | 0.003 | 0.048 | 0.210 | 0.034 | 0.002 | 0.022 |
| cpu / gemm_128x128x128 | 0.046 | 71.771 | 0.006 | 0.015 | 0.362 | 0.009 | 0.005 | 0.006 |
| cpu / gemm_512x512x512 | 0.046 | 74.521 | 0.215 | 0.032 | 10.267 | 0.370 | 0.106 | 0.006 |
| cpu / gemm_1024x1024x1024 | 0.044 | 73.074 | 0.941 | 0.032 | 65.434 | 1.342 | 0.521 | 0.006 |
| cpu / gemm_256x1024x128 | 0.069 | 73.264 | 0.043 | 0.045 | 2.128 | 0.149 | 0.050 | 0.006 |
| cpu / gemm_1024x128x256 | 0.050 | 73.730 | 0.114 | 0.018 | 2.088 | 0.165 | 0.017 | 0.006 |
| cpu / gemm_127x193x61 | 0.052 | 76.256 | 0.007 | 0.025 | 0.339 | 0.038 | 0.014 | 0.006 |
| cpu / gemm_513x257x129 | 0.052 | 74.029 | 0.014 | 0.023 | 1.959 | 0.071 | 0.032 | 0.006 |
| cpu / add_1x127 | 0.037 | 39.692 | 0.004 | 0.023 | 0.015 | 0.044 | 0.002 | 0.006 |
| cpu / add_17x257 | 0.041 | 45.296 | 0.008 | 0.012 | 0.196 | 0.004 | 0.002 | 0.006 |
| cpu / add_128x1024 | 0.050 | 97.053 | 0.045 | 0.034 | 0.159 | 0.063 | 0.069 | 0.007 |
| cpu / add_4096x256 | 0.039 | 98.000 | 0.726 | 0.059 | 0.472 | 0.135 | 0.492 | 0.006 |
| cpu / sum_1x127 | 0.059 | 33.786 | 0.003 | 0.024 | 0.013 | 0.059 | 0.003 | 0.005 |
| cpu / sum_17x257 | 0.049 | 46.148 | 0.007 | 0.012 | 0.129 | 0.004 | 0.001 | 0.005 |
| cpu / sum_128x1024 | 0.064 | 38.906 | 0.054 | 0.025 | 0.400 | 0.115 | 0.001 | 0.006 |
| cpu / sum_64x4096 | 0.043 | 36.560 | 0.090 | 0.020 | 1.081 | 0.055 | 0.002 | 0.006 |
| cpu / softmax_1x127 | 0.061 | 40.291 | 0.003 | 0.023 | 0.022 | 0.052 | 0.001 | 0.005 |
| cpu / softmax_17x257 | 0.049 | 53.033 | 0.004 | 0.024 | 0.178 | 0.034 | 0.014 | 0.006 |
| cpu / softmax_128x1024 | 0.053 | 44.883 | 0.057 | 0.030 | 1.246 | 0.105 | 0.039 | 0.026 |
| cpu / softmax_64x4096 | 0.053 | 44.308 | 0.091 | 0.022 | 2.388 | 0.158 | 0.139 | 0.005 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
