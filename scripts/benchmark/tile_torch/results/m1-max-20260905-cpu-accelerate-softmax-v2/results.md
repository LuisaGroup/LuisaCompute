# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T21:18:40.261061+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `True`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `accelerate`. Accelerate consumes only a versioned compiler-owned shared FP32 exp materialization; the DSL and execution hierarchy remain target-independent. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | softmax_1x127 | 1×127×1 / 2 | 0 | 0.127 | 0.606 | 0.127 | 0.622 | 0.21× | 0.125 | 0.708 |
| cpu | softmax_17x257 | 1×257×1 / 2 | 0 | 2.612 | 33.412 | 3.115 | 34.151 | 0.08× | 2.375 | 41.375 |
| cpu | softmax_128x1024 | 1×1024×1 / 2 | 0 | 14.538 | 88.919 | 21.673 | 90.512 | 0.16× | 14.209 | 87.292 |
| cpu | softmax_64x4096 | 1×4096×1 / 2 | 0 | 45.875 | 128.990 | 67.614 | 129.318 | 0.36× | 44.250 | 131.250 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / softmax_1x127 | 0.050 | 28.861 | 0.003 | 0.043 | 0.015 | 0.021 | 0.005 | 0.018 |
| cpu / softmax_17x257 | 0.053 | 48.633 | 0.004 | 0.024 | 0.178 | 0.044 | 0.014 | 0.003 |
| cpu / softmax_128x1024 | 0.055 | 25.938 | 0.074 | 0.027 | 0.190 | 0.116 | 0.055 | 0.005 |
| cpu / softmax_64x4096 | 0.052 | 27.202 | 0.106 | 0.017 | 0.226 | 0.146 | 0.123 | 0.005 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
