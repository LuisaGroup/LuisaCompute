# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T08:36:54.976993+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `True`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | add_1x127 | 1×256×1 / 2 | 0 | 0.218 | 0.571 | 0.220 | 0.574 | 0.38× | 0.250 | 0.708 |
| cpu | add_17x257 | 1×256×1 / 2 | 0 | 7.050 | 0.981 | 7.903 | 0.988 | 7.19× | 3.334 | 1.083 |
| cpu | add_128x1024 | 1×256×1 / 2 | 0 | 9.159 | 41.798 | 12.631 | 42.111 | 0.22× | 5.375 | 41.208 |
| cpu | add_4096x256 | 1×256×1 / 2 | 0 | 54.078 | 85.441 | 55.949 | 86.569 | 0.63× | 21.417 | 90.125 |
| cpu | sum_1x127 | 1×127×1 / 2 | 0 | 0.075 | 0.795 | 0.076 | 0.799 | 0.09× | 0.167 | 0.917 |
| cpu | sum_17x257 | 1×257×1 / 2 | 0 | 5.346 | 1.094 | 6.071 | 1.100 | 4.89× | 2.167 | 1.208 |
| cpu | sum_128x1024 | 1×1024×1 / 2 | 0 | 46.444 | 41.260 | 48.801 | 41.969 | 1.13× | 30.125 | 47.959 |
| cpu | sum_64x4096 | 1×4096×1 / 2 | 0 | 65.082 | 44.470 | 77.847 | 47.205 | 1.46× | 37.125 | 37.250 |
| cpu | softmax_1x127 | 1×127×1 / 2 | 0 | 0.664 | 0.636 | 0.665 | 0.642 | 1.04× | 0.667 | 0.709 |
| cpu | softmax_17x257 | 1×257×1 / 2 | 0 | 19.431 | 38.599 | 23.791 | 38.946 | 0.50× | 15.250 | 39.833 |
| cpu | softmax_128x1024 | 1×1024×1 / 2 | 0 | 200.691 | 89.443 | 240.644 | 91.934 | 2.24× | 121.709 | 88.292 |
| cpu | softmax_64x4096 | 1×4096×1 / 2 | 0 | 417.564 | 132.102 | 442.501 | 135.947 | 3.16× | 262.667 | 149.625 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / add_1x127 | 0.055 | 51.448 | 0.004 | 0.616 | 0.012 | 0.040 | 0.002 | 0.026 |
| cpu / add_17x257 | 0.045 | 60.637 | 0.008 | 0.012 | 0.097 | 0.002 | 0.011 | 0.005 |
| cpu / add_128x1024 | 0.069 | 49.239 | 0.157 | 0.037 | 0.183 | 0.077 | 0.075 | 0.006 |
| cpu / add_4096x256 | 0.040 | 49.284 | 0.748 | 0.025 | 0.630 | 0.106 | 0.432 | 0.006 |
| cpu / sum_1x127 | 0.045 | 25.191 | 0.004 | 0.023 | 0.010 | 0.500 | 0.001 | 0.006 |
| cpu / sum_17x257 | 0.042 | 25.926 | 0.011 | 0.019 | 0.105 | 0.007 | 0.002 | 0.005 |
| cpu / sum_128x1024 | 0.045 | 26.732 | 0.048 | 0.021 | 0.134 | 0.101 | 0.002 | 0.005 |
| cpu / sum_64x4096 | 0.051 | 28.675 | 0.083 | 0.023 | 0.200 | 0.053 | 0.001 | 0.006 |
| cpu / softmax_1x127 | 0.153 | 41.579 | 0.004 | 0.036 | 0.020 | 0.631 | 0.002 | 0.005 |
| cpu / softmax_17x257 | 0.061 | 32.276 | 0.008 | 0.016 | 0.205 | 0.057 | 0.007 | 0.006 |
| cpu / softmax_128x1024 | 0.072 | 31.663 | 0.050 | 0.026 | 0.256 | 0.108 | 0.056 | 0.006 |
| cpu / softmax_64x4096 | 0.058 | 31.551 | 0.140 | 0.023 | 0.663 | 0.147 | 0.104 | 0.007 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
