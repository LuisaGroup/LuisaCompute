# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T21:24:50.196220+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `True`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only a versioned compiler-owned shared FP32 exp materialization; the DSL and execution hierarchy remain target-independent. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | add_1x127 | 1×256×1 / 2 | 0 | 0.072 | 0.550 | 0.072 | 0.563 | 0.13× | 0.084 | 0.625 |
| cpu | add_17x257 | 1×256×1 / 2 | 0 | 0.419 | 0.921 | 0.427 | 0.937 | 0.45× | 0.417 | 1.042 |
| cpu | add_128x1024 | 1×256×1 / 2 | 0 | 4.430 | 37.405 | 6.476 | 39.511 | 0.12× | 5.250 | 36.375 |
| cpu | add_4096x256 | 1×256×1 / 2 | 0 | 31.627 | 95.858 | 43.954 | 98.173 | 0.33× | 28.375 | 78.958 |
| cpu | sum_1x127 | 1×127×1 / 2 | 0 | 0.064 | 0.757 | 0.065 | 0.779 | 0.09× | 0.125 | 0.916 |
| cpu | sum_17x257 | 1×257×1 / 2 | 0 | 2.187 | 1.038 | 2.249 | 1.044 | 2.11× | 2.209 | 1.125 |
| cpu | sum_128x1024 | 1×1024×1 / 2 | 0 | 16.498 | 37.638 | 16.993 | 39.467 | 0.44× | 16.250 | 38.417 |
| cpu | sum_64x4096 | 1×4096×1 / 2 | 0 | 33.630 | 40.709 | 46.947 | 42.287 | 0.83× | 33.583 | 40.417 |
| cpu | softmax_1x127 | 1×127×1 / 2 | 0 | 0.550 | 0.609 | 0.551 | 0.621 | 0.90× | 0.542 | 0.667 |
| cpu | softmax_17x257 | 1×257×1 / 2 | 0 | 5.545 | 33.091 | 6.397 | 34.761 | 0.17× | 5.708 | 36.667 |
| cpu | softmax_128x1024 | 1×1024×1 / 2 | 0 | 79.296 | 89.132 | 84.414 | 91.276 | 0.89× | 79.000 | 95.166 |
| cpu | softmax_64x4096 | 1×4096×1 / 2 | 0 | 157.774 | 128.701 | 237.435 | 128.981 | 1.23× | 155.750 | 131.083 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / add_1x127 | 0.038 | 34.944 | 0.003 | 0.051 | 0.013 | 0.012 | 0.002 | 0.017 |
| cpu / add_17x257 | 0.036 | 40.438 | 0.004 | 0.013 | 0.013 | 0.004 | 0.009 | 0.018 |
| cpu / add_128x1024 | 0.040 | 26.936 | 0.089 | 0.020 | 0.187 | 0.053 | 0.066 | 0.005 |
| cpu / add_4096x256 | 0.038 | 26.584 | 0.742 | 0.022 | 0.467 | 0.138 | 0.547 | 0.005 |
| cpu / sum_1x127 | 0.045 | 21.531 | 0.003 | 0.021 | 0.010 | 0.063 | 0.001 | 0.005 |
| cpu / sum_17x257 | 0.042 | 35.646 | 0.006 | 0.014 | 0.012 | 0.005 | 0.002 | 0.009 |
| cpu / sum_128x1024 | 0.040 | 23.349 | 0.050 | 0.016 | 0.184 | 0.102 | 0.001 | 0.005 |
| cpu / sum_64x4096 | 0.041 | 23.355 | 0.097 | 0.028 | 0.200 | 0.048 | 0.002 | 0.006 |
| cpu / softmax_1x127 | 0.053 | 31.626 | 0.003 | 0.019 | 0.011 | 0.044 | 0.002 | 0.006 |
| cpu / softmax_17x257 | 0.053 | 36.209 | 0.005 | 0.014 | 0.162 | 0.047 | 0.009 | 0.005 |
| cpu / softmax_128x1024 | 0.054 | 28.116 | 0.077 | 0.025 | 0.269 | 0.103 | 0.057 | 0.005 |
| cpu / softmax_64x4096 | 0.126 | 28.188 | 0.088 | 0.015 | 0.335 | 0.133 | 0.107 | 0.004 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
