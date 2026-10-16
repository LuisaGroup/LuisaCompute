# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T21:14:38.688009+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `True`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `accelerate`. Accelerate consumes only a versioned compiler-owned shared FP32 exp materialization; the DSL and execution hierarchy remain target-independent. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | softmax_1x127 | 1×127×1 / 2 | 0 | 0.372 | 0.656 | 0.378 | 0.671 | 0.57× | 0.375 | 0.750 |
| cpu | softmax_17x257 | 1×257×1 / 2 | 0 | 8.736 | 35.187 | 11.787 | 37.416 | 0.25× | 6.875 | 26.250 |
| cpu | softmax_128x1024 | 1×1024×1 / 2 | 0 | 109.024 | 86.312 | 129.677 | 87.238 | 1.26× | 99.750 | 84.167 |
| cpu | softmax_64x4096 | 1×4096×1 / 2 | 0 | 191.111 | 121.125 | 225.754 | 129.880 | 1.58× | 201.125 | 121.875 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / softmax_1x127 | 0.055 | 32.212 | 0.003 | 0.041 | 0.015 | 0.018 | 0.007 | 0.053 |
| cpu / softmax_17x257 | 0.049 | 43.274 | 0.009 | 0.043 | 0.179 | 0.081 | 0.007 | 0.006 |
| cpu / softmax_128x1024 | 0.052 | 28.442 | 0.052 | 0.023 | 0.314 | 0.096 | 0.066 | 0.005 |
| cpu / softmax_64x4096 | 0.048 | 29.759 | 0.098 | 0.015 | 0.354 | 0.122 | 0.112 | 0.004 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
