# TileIR/TVMx vs PyTorch

Generated: 2026-09-02T20:05:23.757612+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR, but the current TVMx schedule lowers it to loops: no tensor-core claim. Pipeline stage markers currently run serially. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 8×8×16 | 6.012 | 0.916 | 6.448 | 0.930 | 6.57× | 6.209 | 0.917 |
| cpu | gemm_128x128x128 | 8×8×16 | 85.638 | 5.022 | 95.741 | 7.691 | 17.05× | 53.083 | 4.958 |
| cpu | gemm_512x512x512 | 8×8×16 | 4139.025 | 141.977 | 4359.217 | 150.020 | 29.15× | 3989.416 | 149.250 |
| cpu | gemm_1024x1024x1024 | 8×8×16 | 48226.167 | 1007.788 | 53823.092 | 1055.610 | 47.85× | 49494.792 | 968.416 |
| cpu | gemm_256x1024x128 | 8×8×16 | 1134.336 | 68.229 | 1287.792 | 68.729 | 16.63× | 1161.333 | 69.458 |
| cpu | gemm_1024x128x256 | 8×8×16 | 1449.984 | 65.394 | 1534.635 | 65.839 | 22.17× | 1423.625 | 63.125 |
| cpu | gemm_127x193x61 | 8×8×16 | 124.742 | 6.637 | 150.659 | 6.738 | 18.79× | 104.791 | 6.791 |
| cpu | gemm_513x257x129 | 8×8×16 | 1177.586 | 44.923 | 1215.874 | 45.336 | 26.21× | 1149.125 | 43.375 |
| cpu | add_1x127 | 1×256×1 | 0.427 | 0.570 | 0.432 | 0.577 | 0.75× | 0.458 | 0.667 |
| cpu | add_17x257 | 1×256×1 | 7.248 | 0.981 | 10.901 | 0.996 | 7.39× | 4.000 | 1.083 |
| cpu | add_128x1024 | 1×256×1 | 18.513 | 41.893 | 20.700 | 42.725 | 0.44× | 20.750 | 39.833 |
| cpu | add_4096x256 | 1×256×1 | 125.139 | 83.722 | 157.866 | 85.024 | 1.49× | 125.583 | 76.875 |
| cpu | sum_1x127 | 1×127×1 | 2.130 | 0.802 | 2.146 | 0.807 | 2.66× | 2.125 | 0.958 |
| cpu | sum_17x257 | 1×257×1 | 27.671 | 1.107 | 29.265 | 1.122 | 25.00× | 54.416 | 1.250 |
| cpu | sum_128x1024 | 1×1024×1 | 692.014 | 41.263 | 749.148 | 42.521 | 16.77× | 725.083 | 42.625 |
| cpu | sum_64x4096 | 1×4096×1 | 1097.457 | 44.567 | 1321.535 | 47.121 | 24.63× | 1313.083 | 59.166 |
| cpu | softmax_1x127 | 1×127×1 | 5.956 | 0.644 | 6.026 | 0.653 | 9.25× | 5.833 | 0.750 |
| cpu | softmax_17x257 | 1×257×1 | 50.596 | 38.440 | 53.827 | 41.880 | 1.32× | 41.833 | 38.000 |
| cpu | softmax_128x1024 | 1×1024×1 | 1295.205 | 89.257 | 1445.045 | 90.239 | 14.51× | 1252.792 | 87.000 |
| cpu | softmax_64x4096 | 1×4096×1 | 3147.896 | 128.084 | 4454.604 | 138.774 | 24.58× | 3375.000 | 129.000 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.065 | 145.447 | 0.006 | 0.075 | 0.177 | 0.051 | 0.002 | 0.015 |
| cpu / gemm_128x128x128 | 0.048 | 141.558 | 0.006 | 0.017 | 0.223 | 0.009 | 0.005 | 0.006 |
| cpu / gemm_512x512x512 | 0.047 | 143.668 | 0.238 | 0.032 | 4.192 | 0.304 | 0.102 | 0.008 |
| cpu / gemm_1024x1024x1024 | 0.048 | 145.768 | 0.709 | 0.040 | 48.894 | 1.471 | 0.391 | 0.006 |
| cpu / gemm_256x1024x128 | 0.053 | 143.133 | 0.021 | 0.026 | 1.665 | 0.133 | 0.152 | 0.006 |
| cpu / gemm_1024x128x256 | 0.042 | 140.057 | 0.149 | 0.032 | 1.852 | 0.133 | 0.093 | 0.006 |
| cpu / gemm_127x193x61 | 0.050 | 138.934 | 0.009 | 0.020 | 0.193 | 0.038 | 0.016 | 0.005 |
| cpu / gemm_513x257x129 | 0.048 | 137.445 | 0.042 | 0.016 | 1.922 | 0.048 | 0.053 | 0.006 |
| cpu / add_1x127 | 0.041 | 39.488 | 0.003 | 0.022 | 0.018 | 0.058 | 0.002 | 0.005 |
| cpu / add_17x257 | 0.037 | 44.791 | 0.007 | 0.015 | 0.135 | 0.003 | 0.002 | 0.006 |
| cpu / add_128x1024 | 0.038 | 95.474 | 0.038 | 0.028 | 0.137 | 0.102 | 0.040 | 0.005 |
| cpu / add_4096x256 | 0.037 | 96.156 | 1.047 | 0.029 | 0.618 | 0.113 | 0.711 | 0.006 |
| cpu / sum_1x127 | 0.041 | 32.734 | 0.003 | 0.023 | 0.012 | 0.059 | 0.001 | 0.005 |
| cpu / sum_17x257 | 0.044 | 45.748 | 0.005 | 0.014 | 0.194 | 0.005 | 0.002 | 0.005 |
| cpu / sum_128x1024 | 0.040 | 38.357 | 0.062 | 0.023 | 0.761 | 0.095 | 0.001 | 0.005 |
| cpu / sum_64x4096 | 0.046 | 36.979 | 0.164 | 0.023 | 0.903 | 0.120 | 0.001 | 0.005 |
| cpu / softmax_1x127 | 0.055 | 40.450 | 0.004 | 0.021 | 0.021 | 0.068 | 0.001 | 0.006 |
| cpu / softmax_17x257 | 0.061 | 51.965 | 0.006 | 0.027 | 0.193 | 0.066 | 0.005 | 0.006 |
| cpu / softmax_128x1024 | 0.055 | 44.465 | 0.046 | 0.025 | 0.988 | 0.110 | 0.051 | 0.006 |
| cpu / softmax_64x4096 | 0.054 | 42.556 | 0.092 | 0.017 | 2.780 | 0.116 | 0.116 | 0.006 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
