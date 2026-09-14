# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T20:56:11.694175+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `True`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | add_1x127 | 1×256×1 / 2 | 0 | 0.093 | 0.572 | 0.093 | 0.586 | 0.16× | 0.084 | 0.709 |
| cpu | add_17x257 | 1×256×1 / 2 | 0 | 2.840 | 0.958 | 2.841 | 1.005 | 2.96× | 2.875 | 1.084 |
| cpu | add_128x1024 | 1×256×1 / 2 | 0 | 8.112 | 38.329 | 9.628 | 42.938 | 0.21× | 4.958 | 43.625 |
| cpu | add_4096x256 | 1×256×1 / 2 | 0 | 25.516 | 83.344 | 28.301 | 84.547 | 0.31× | 21.792 | 81.709 |
| cpu | sum_1x127 | 1×127×1 / 2 | 0 | 1.839 | 0.810 | 1.886 | 0.825 | 2.27× | 1.834 | 0.959 |
| cpu | sum_17x257 | 1×257×1 / 2 | 0 | 62.567 | 1.088 | 63.061 | 1.103 | 57.51× | 62.666 | 1.250 |
| cpu | sum_128x1024 | 1×1024×1 / 2 | 0 | 775.437 | 37.824 | 996.267 | 38.767 | 20.50× | 989.208 | 41.833 |
| cpu | sum_64x4096 | 1×4096×1 / 2 | 0 | 1843.490 | 40.548 | 2068.019 | 41.667 | 45.46× | 1897.667 | 48.959 |
| cpu | softmax_1x127 | 1×127×1 / 2 | 0 | 5.699 | 0.632 | 5.849 | 0.637 | 9.02× | 5.667 | 0.750 |
| cpu | softmax_17x257 | 1×257×1 / 2 | 0 | 194.440 | 33.964 | 194.458 | 35.914 | 5.72× | 194.458 | 31.041 |
| cpu | softmax_128x1024 | 1×1024×1 / 2 | 0 | 1509.167 | 85.743 | 1636.099 | 89.095 | 17.60× | 1527.250 | 90.583 |
| cpu | softmax_64x4096 | 1×4096×1 / 2 | 0 | 4072.403 | 128.670 | 4467.508 | 132.079 | 31.65× | 4561.500 | 113.708 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / add_1x127 | 0.037 | 33.751 | 0.003 | 0.063 | 0.009 | 0.048 | 0.003 | 0.052 |
| cpu / add_17x257 | 0.035 | 37.560 | 0.007 | 0.025 | 0.013 | 0.006 | 0.021 | 0.009 |
| cpu / add_128x1024 | 0.040 | 29.008 | 0.089 | 0.026 | 0.186 | 0.100 | 0.056 | 0.005 |
| cpu / add_4096x256 | 0.034 | 26.360 | 0.736 | 0.023 | 0.575 | 0.101 | 0.437 | 0.006 |
| cpu / sum_1x127 | 0.043 | 22.969 | 0.004 | 0.019 | 0.028 | 0.288 | 0.001 | 0.019 |
| cpu / sum_17x257 | 0.039 | 23.919 | 0.006 | 0.024 | 0.084 | 0.010 | 0.003 | 0.006 |
| cpu / sum_128x1024 | 0.040 | 24.005 | 0.043 | 0.017 | 0.586 | 0.104 | 0.003 | 0.003 |
| cpu / sum_64x4096 | 0.043 | 24.379 | 0.085 | 0.015 | 1.165 | 0.048 | 0.001 | 0.005 |
| cpu / softmax_1x127 | 0.058 | 30.125 | 0.003 | 0.019 | 0.019 | 0.375 | 0.002 | 0.009 |
| cpu / softmax_17x257 | 0.050 | 29.444 | 0.004 | 0.022 | 0.208 | 0.078 | 0.011 | 0.006 |
| cpu / softmax_128x1024 | 0.050 | 29.882 | 0.044 | 0.018 | 1.674 | 0.102 | 0.052 | 0.006 |
| cpu / softmax_64x4096 | 0.048 | 29.764 | 0.110 | 0.014 | 3.400 | 0.119 | 0.105 | 0.005 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
