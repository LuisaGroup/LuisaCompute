# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T21:22:07.124515+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `True`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `accelerate`. Accelerate consumes only a versioned compiler-owned shared FP32 exp materialization; the DSL and execution hierarchy remain target-independent. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | add_1x127 | 1×256×1 / 2 | 0 | 0.068 | 0.568 | 0.069 | 0.581 | 0.12× | 0.083 | 0.667 |
| cpu | add_17x257 | 1×256×1 / 2 | 0 | 0.417 | 0.944 | 0.425 | 0.969 | 0.44× | 0.417 | 1.042 |
| cpu | add_128x1024 | 1×256×1 / 2 | 0 | 4.701 | 38.321 | 5.757 | 41.897 | 0.12× | 4.584 | 38.250 |
| cpu | add_4096x256 | 1×256×1 / 2 | 0 | 32.188 | 82.662 | 43.876 | 83.194 | 0.39× | 32.042 | 83.375 |
| cpu | sum_1x127 | 1×127×1 / 2 | 0 | 0.024 | 0.840 | 0.024 | 0.843 | 0.03× | 0.042 | 0.958 |
| cpu | sum_17x257 | 1×257×1 / 2 | 0 | 0.371 | 1.131 | 0.371 | 1.139 | 0.33× | 0.375 | 1.291 |
| cpu | sum_128x1024 | 1×1024×1 / 2 | 0 | 3.685 | 37.474 | 5.885 | 40.131 | 0.10× | 3.625 | 41.834 |
| cpu | sum_64x4096 | 1×4096×1 / 2 | 0 | 5.800 | 40.658 | 10.054 | 41.168 | 0.14× | 5.541 | 38.875 |
| cpu | softmax_1x127 | 1×127×1 / 2 | 0 | 0.126 | 0.638 | 0.127 | 0.659 | 0.20× | 0.125 | 0.791 |
| cpu | softmax_17x257 | 1×257×1 / 2 | 0 | 2.748 | 33.529 | 3.367 | 34.369 | 0.08× | 2.959 | 27.500 |
| cpu | softmax_128x1024 | 1×1024×1 / 2 | 0 | 14.511 | 89.416 | 18.130 | 91.193 | 0.16× | 45.791 | 85.959 |
| cpu | softmax_64x4096 | 1×4096×1 / 2 | 0 | 41.289 | 129.042 | 45.975 | 131.149 | 0.32× | 95.458 | 126.667 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / add_1x127 | 0.038 | 35.078 | 0.003 | 0.057 | 0.010 | 0.013 | 0.001 | 0.020 |
| cpu / add_17x257 | 0.036 | 40.368 | 0.008 | 0.012 | 0.012 | 0.004 | 0.004 | 0.008 |
| cpu / add_128x1024 | 0.038 | 27.089 | 0.111 | 0.025 | 0.184 | 0.089 | 0.127 | 0.005 |
| cpu / add_4096x256 | 0.039 | 26.707 | 0.732 | 0.019 | 0.446 | 0.130 | 0.560 | 0.004 |
| cpu / sum_1x127 | 0.042 | 20.960 | 0.003 | 0.036 | 0.011 | 0.080 | 0.002 | 0.021 |
| cpu / sum_17x257 | 0.038 | 24.495 | 0.004 | 0.029 | 0.011 | 0.006 | 0.015 | 0.005 |
| cpu / sum_128x1024 | 0.039 | 23.140 | 0.046 | 0.018 | 0.186 | 0.109 | 0.005 | 0.005 |
| cpu / sum_64x4096 | 0.039 | 23.099 | 0.096 | 0.017 | 0.155 | 0.047 | 0.001 | 0.004 |
| cpu / softmax_1x127 | 0.061 | 28.737 | 0.003 | 0.018 | 0.012 | 0.046 | 0.002 | 0.006 |
| cpu / softmax_17x257 | 0.056 | 46.257 | 0.005 | 0.024 | 0.129 | 0.054 | 0.008 | 0.006 |
| cpu / softmax_128x1024 | 0.058 | 26.258 | 0.044 | 0.025 | 0.182 | 0.111 | 0.061 | 0.005 |
| cpu / softmax_64x4096 | 0.054 | 27.338 | 0.088 | 0.018 | 0.224 | 0.129 | 0.202 | 0.004 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
