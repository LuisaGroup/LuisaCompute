# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T09:39:17.377295+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `True`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | add_1x127 | 1×256×1 / 2 | 0 | 0.097 | 0.598 | 0.098 | 0.599 | 0.16× | 0.125 | 0.750 |
| cpu | add_17x257 | 1×256×1 / 2 | 0 | 4.722 | 0.995 | 5.762 | 0.999 | 4.74× | 2.167 | 1.084 |
| cpu | add_128x1024 | 1×256×1 / 2 | 0 | 9.335 | 40.827 | 15.332 | 41.169 | 0.23× | 5.417 | 40.208 |
| cpu | add_4096x256 | 1×256×1 / 2 | 0 | 53.130 | 84.926 | 68.087 | 85.450 | 0.63× | 44.625 | 84.291 |
| cpu | sum_1x127 | 1×127×1 / 2 | 0 | 0.069 | 0.840 | 0.069 | 0.848 | 0.08× | 0.083 | 0.959 |
| cpu | sum_17x257 | 1×257×1 / 2 | 0 | 3.906 | 1.139 | 4.189 | 1.153 | 3.43× | 2.250 | 1.250 |
| cpu | sum_128x1024 | 1×1024×1 / 2 | 0 | 26.368 | 40.420 | 36.019 | 41.119 | 0.65× | 16.167 | 37.042 |
| cpu | sum_64x4096 | 1×4096×1 / 2 | 0 | 61.224 | 43.041 | 83.427 | 43.831 | 1.42× | 33.333 | 43.708 |
| cpu | softmax_1x127 | 1×127×1 / 2 | 0 | 0.660 | 0.669 | 0.661 | 0.675 | 0.99× | 0.625 | 0.750 |
| cpu | softmax_17x257 | 1×257×1 / 2 | 0 | 17.524 | 36.902 | 22.938 | 37.866 | 0.47× | 8.500 | 34.041 |
| cpu | softmax_128x1024 | 1×1024×1 / 2 | 0 | 165.434 | 87.881 | 172.183 | 88.205 | 1.88× | 235.791 | 82.375 |
| cpu | softmax_64x4096 | 1×4096×1 / 2 | 0 | 385.092 | 121.685 | 519.394 | 123.877 | 3.16× | 248.833 | 131.958 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / add_1x127 | 0.042 | 35.031 | 0.006 | 0.050 | 0.010 | 0.016 | 0.002 | 0.020 |
| cpu / add_17x257 | 0.046 | 38.965 | 0.047 | 0.016 | 0.112 | 0.003 | 0.008 | 0.006 |
| cpu / add_128x1024 | 0.046 | 51.095 | 0.108 | 0.033 | 0.260 | 0.072 | 0.060 | 0.010 |
| cpu / add_4096x256 | 0.045 | 30.500 | 0.840 | 0.031 | 0.581 | 0.153 | 0.508 | 0.006 |
| cpu / sum_1x127 | 0.048 | 23.599 | 0.003 | 0.022 | 0.010 | 0.089 | 0.006 | 0.005 |
| cpu / sum_17x257 | 0.042 | 26.817 | 0.006 | 0.014 | 0.119 | 0.006 | 0.002 | 0.006 |
| cpu / sum_128x1024 | 0.053 | 26.389 | 0.053 | 0.017 | 0.215 | 0.099 | 0.002 | 0.009 |
| cpu / sum_64x4096 | 0.058 | 25.519 | 0.092 | 0.019 | 0.183 | 0.054 | 0.002 | 0.005 |
| cpu / softmax_1x127 | 0.059 | 32.319 | 0.006 | 0.021 | 0.010 | 0.055 | 0.003 | 0.008 |
| cpu / softmax_17x257 | 0.058 | 31.413 | 0.019 | 0.011 | 0.129 | 0.041 | 0.007 | 0.007 |
| cpu / softmax_128x1024 | 0.062 | 31.740 | 0.050 | 0.025 | 0.456 | 0.102 | 0.058 | 0.006 |
| cpu / softmax_64x4096 | 0.054 | 30.877 | 0.097 | 0.017 | 0.525 | 0.140 | 0.229 | 0.006 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
