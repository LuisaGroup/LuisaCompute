# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T09:08:44.022872+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `True`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | add_1x127 | 1×256×1 / 2 | 0 | 0.215 | 0.592 | 0.217 | 0.597 | 0.36× | 0.208 | 0.708 |
| cpu | add_17x257 | 1×256×1 / 2 | 0 | 8.814 | 0.997 | 11.155 | 1.006 | 8.84× | 5.958 | 1.125 |
| cpu | add_128x1024 | 1×256×1 / 2 | 0 | 7.634 | 41.461 | 8.376 | 41.549 | 0.18× | 4.375 | 38.667 |
| cpu | add_4096x256 | 1×256×1 / 2 | 0 | 69.187 | 87.836 | 107.141 | 89.550 | 0.79× | 21.417 | 85.834 |
| cpu | sum_1x127 | 1×127×1 / 2 | 0 | 0.074 | 0.868 | 0.075 | 0.869 | 0.08× | 0.125 | 1.042 |
| cpu | sum_17x257 | 1×257×1 / 2 | 0 | 4.251 | 1.164 | 8.809 | 1.177 | 3.65× | 4.917 | 1.292 |
| cpu | sum_128x1024 | 1×1024×1 / 2 | 0 | 30.368 | 41.113 | 33.202 | 41.775 | 0.74× | 18.958 | 44.959 |
| cpu | sum_64x4096 | 1×4096×1 / 2 | 0 | 59.028 | 44.066 | 110.530 | 44.637 | 1.34× | 36.500 | 45.041 |
| cpu | softmax_1x127 | 1×127×1 / 2 | 0 | 0.667 | 0.665 | 0.674 | 0.672 | 1.00× | 0.625 | 0.750 |
| cpu | softmax_17x257 | 1×257×1 / 2 | 0 | 13.454 | 36.524 | 24.054 | 37.723 | 0.37× | 8.583 | 47.250 |
| cpu | softmax_128x1024 | 1×1024×1 / 2 | 0 | 162.978 | 87.598 | 167.337 | 90.397 | 1.86× | 159.292 | 83.084 |
| cpu | softmax_64x4096 | 1×4096×1 / 2 | 0 | 418.706 | 120.642 | 581.751 | 124.328 | 3.47× | 596.834 | 109.417 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / add_1x127 | 0.038 | 48.934 | 0.004 | 0.048 | 0.011 | 0.017 | 0.002 | 0.015 |
| cpu / add_17x257 | 0.039 | 60.213 | 0.009 | 0.014 | 0.126 | 0.003 | 0.009 | 0.006 |
| cpu / add_128x1024 | 0.059 | 49.102 | 0.092 | 0.031 | 0.149 | 0.054 | 0.021 | 0.007 |
| cpu / add_4096x256 | 0.058 | 49.016 | 1.001 | 0.024 | 0.449 | 0.149 | 0.393 | 0.006 |
| cpu / sum_1x127 | 0.050 | 26.249 | 0.003 | 0.023 | 0.010 | 0.055 | 0.002 | 0.005 |
| cpu / sum_17x257 | 0.045 | 27.470 | 0.004 | 0.014 | 0.107 | 0.009 | 0.002 | 0.005 |
| cpu / sum_128x1024 | 0.046 | 26.465 | 0.054 | 0.017 | 0.182 | 0.092 | 0.002 | 0.005 |
| cpu / sum_64x4096 | 0.077 | 28.249 | 0.106 | 0.020 | 0.190 | 0.047 | 0.001 | 0.005 |
| cpu / softmax_1x127 | 0.063 | 33.091 | 0.004 | 0.024 | 0.011 | 0.048 | 0.002 | 0.006 |
| cpu / softmax_17x257 | 0.057 | 33.295 | 0.007 | 0.015 | 0.133 | 0.043 | 0.007 | 0.006 |
| cpu / softmax_128x1024 | 0.063 | 32.170 | 0.050 | 0.027 | 0.248 | 0.134 | 0.023 | 0.006 |
| cpu / softmax_64x4096 | 0.057 | 32.411 | 0.125 | 0.015 | 0.601 | 0.137 | 0.094 | 0.005 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
