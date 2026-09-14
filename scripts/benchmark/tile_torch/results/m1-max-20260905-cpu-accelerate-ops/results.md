# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T21:19:39.593879+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `True`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `accelerate`. Accelerate consumes only a versioned compiler-owned shared FP32 exp materialization; the DSL and execution hierarchy remain target-independent. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | add_1x127 | 1×256×1 / 2 | 0 | 0.093 | 0.555 | 0.093 | 0.568 | 0.17× | 0.125 | 0.667 |
| cpu | add_17x257 | 1×256×1 / 2 | 0 | 2.842 | 0.927 | 2.850 | 0.938 | 3.07× | 2.833 | 1.042 |
| cpu | add_128x1024 | 1×256×1 / 2 | 0 | 4.490 | 39.706 | 4.871 | 40.700 | 0.11× | 4.250 | 41.125 |
| cpu | add_4096x256 | 1×256×1 / 2 | 0 | 27.829 | 83.278 | 38.662 | 85.066 | 0.33× | 20.250 | 85.334 |
| cpu | sum_1x127 | 1×127×1 / 2 | 0 | 0.024 | 0.773 | 0.024 | 0.782 | 0.03× | 0.042 | 0.875 |
| cpu | sum_17x257 | 1×257×1 / 2 | 0 | 2.157 | 1.075 | 2.283 | 1.101 | 2.01× | 2.125 | 1.250 |
| cpu | sum_128x1024 | 1×1024×1 / 2 | 0 | 3.382 | 38.614 | 3.456 | 40.755 | 0.09× | 3.416 | 37.625 |
| cpu | sum_64x4096 | 1×4096×1 / 2 | 0 | 5.462 | 40.466 | 8.339 | 40.659 | 0.13× | 12.042 | 40.959 |
| cpu | softmax_1x127 | 1×127×1 / 2 | 0 | 0.126 | 0.604 | 0.127 | 0.604 | 0.21× | 0.125 | 0.708 |
| cpu | softmax_17x257 | 1×257×1 / 2 | 0 | 2.578 | 37.109 | 3.073 | 40.946 | 0.07× | 2.667 | 42.708 |
| cpu | softmax_128x1024 | 1×1024×1 / 2 | 0 | 14.493 | 88.725 | 26.515 | 90.846 | 0.16× | 14.500 | 93.250 |
| cpu | softmax_64x4096 | 1×4096×1 / 2 | 0 | 47.248 | 129.229 | 47.398 | 129.529 | 0.37× | 36.000 | 128.584 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / add_1x127 | 0.035 | 30.903 | 0.003 | 0.049 | 0.013 | 0.012 | 0.002 | 0.017 |
| cpu / add_17x257 | 0.036 | 35.886 | 0.007 | 0.013 | 0.018 | 0.004 | 0.006 | 0.012 |
| cpu / add_128x1024 | 0.036 | 26.383 | 0.088 | 0.028 | 0.172 | 0.051 | 0.051 | 0.004 |
| cpu / add_4096x256 | 0.037 | 26.427 | 0.850 | 0.021 | 0.511 | 0.140 | 0.437 | 0.006 |
| cpu / sum_1x127 | 0.042 | 20.695 | 0.003 | 0.019 | 0.011 | 0.046 | 0.004 | 0.003 |
| cpu / sum_17x257 | 0.043 | 32.176 | 0.007 | 0.009 | 0.159 | 0.005 | 0.001 | 0.008 |
| cpu / sum_128x1024 | 0.045 | 23.395 | 0.052 | 0.018 | 0.133 | 0.106 | 0.002 | 0.004 |
| cpu / sum_64x4096 | 0.039 | 22.980 | 0.115 | 0.020 | 0.174 | 0.052 | 0.002 | 0.005 |
| cpu / softmax_1x127 | 0.051 | 28.540 | 0.005 | 0.017 | 0.012 | 0.050 | 0.003 | 0.004 |
| cpu / softmax_17x257 | 0.053 | 46.393 | 0.029 | 0.012 | 0.131 | 0.070 | 0.014 | 0.006 |
| cpu / softmax_128x1024 | 0.058 | 26.367 | 0.058 | 0.026 | 0.184 | 0.093 | 0.068 | 0.004 |
| cpu / softmax_64x4096 | 0.049 | 28.143 | 0.121 | 0.015 | 0.342 | 0.127 | 0.118 | 0.005 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
