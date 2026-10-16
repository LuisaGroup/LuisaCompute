# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T16:20:55.041444+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `True`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | add_1x127 | 1×256×1 / 2 | 0 | 0.095 | 0.570 | 0.096 | 0.572 | 0.17× | 0.125 | 0.750 |
| cpu | add_17x257 | 1×256×1 / 2 | 0 | 6.038 | 0.964 | 8.432 | 0.976 | 6.26× | 5.708 | 1.042 |
| cpu | add_128x1024 | 1×256×1 / 2 | 0 | 11.301 | 41.573 | 11.556 | 41.880 | 0.27× | 5.334 | 39.750 |
| cpu | add_4096x256 | 1×256×1 / 2 | 0 | 51.555 | 88.779 | 98.410 | 89.444 | 0.58× | 34.250 | 88.292 |
| cpu | sum_1x127 | 1×127×1 / 2 | 0 | 0.068 | 0.812 | 0.068 | 0.816 | 0.08× | 0.041 | 0.958 |
| cpu | sum_17x257 | 1×257×1 / 2 | 0 | 4.801 | 1.139 | 6.192 | 1.149 | 4.21× | 2.292 | 1.209 |
| cpu | sum_128x1024 | 1×1024×1 / 2 | 0 | 29.384 | 40.560 | 39.571 | 40.954 | 0.72× | 16.167 | 35.458 |
| cpu | sum_64x4096 | 1×4096×1 / 2 | 0 | 58.703 | 43.703 | 61.236 | 44.427 | 1.34× | 33.750 | 38.917 |
| cpu | softmax_1x127 | 1×127×1 / 2 | 0 | 0.651 | 0.640 | 0.659 | 0.646 | 1.02× | 0.625 | 0.709 |
| cpu | softmax_17x257 | 1×257×1 / 2 | 0 | 19.156 | 36.888 | 33.507 | 37.883 | 0.52× | 17.208 | 28.500 |
| cpu | softmax_128x1024 | 1×1024×1 / 2 | 0 | 208.676 | 89.157 | 226.136 | 89.835 | 2.34× | 121.000 | 88.750 |
| cpu | softmax_64x4096 | 1×4096×1 / 2 | 0 | 360.394 | 123.458 | 540.170 | 125.146 | 2.92× | 459.666 | 107.959 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / add_1x127 | 0.043 | 34.647 | 0.005 | 0.054 | 0.010 | 0.045 | 0.002 | 0.018 |
| cpu / add_17x257 | 0.052 | 38.598 | 0.006 | 0.013 | 0.112 | 0.004 | 0.010 | 0.006 |
| cpu / add_128x1024 | 0.037 | 29.678 | 0.134 | 0.038 | 0.161 | 0.084 | 0.022 | 0.006 |
| cpu / add_4096x256 | 0.038 | 30.205 | 0.798 | 0.031 | 0.436 | 0.143 | 0.565 | 0.011 |
| cpu / sum_1x127 | 0.051 | 23.477 | 0.004 | 0.022 | 0.015 | 0.329 | 0.002 | 0.006 |
| cpu / sum_17x257 | 0.049 | 26.651 | 0.007 | 0.022 | 0.106 | 0.006 | 0.002 | 0.003 |
| cpu / sum_128x1024 | 0.044 | 24.672 | 0.068 | 0.019 | 0.196 | 0.132 | 0.004 | 0.006 |
| cpu / sum_64x4096 | 0.049 | 25.498 | 0.145 | 0.033 | 0.149 | 0.088 | 0.006 | 0.007 |
| cpu / softmax_1x127 | 0.052 | 31.093 | 0.003 | 0.017 | 0.011 | 0.067 | 0.004 | 0.005 |
| cpu / softmax_17x257 | 0.055 | 30.982 | 0.006 | 0.013 | 0.388 | 0.107 | 0.010 | 0.006 |
| cpu / softmax_128x1024 | 0.071 | 30.852 | 0.138 | 0.027 | 0.394 | 0.117 | 0.054 | 0.006 |
| cpu / softmax_64x4096 | 0.059 | 31.091 | 0.097 | 0.017 | 0.379 | 0.138 | 0.112 | 0.006 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
