# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T21:28:28.631340+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `cblas`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only a versioned compiler-owned shared FP32 exp materialization; the DSL and execution hierarchy remain target-independent. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 8×8×16 / 2 | 1 | 0.494 | 0.921 | 0.527 | 0.935 | 0.54× | 0.417 | 1.000 |
| cpu | gemm_128x128x128 | 8×8×16 / 2 | 1 | 4.613 | 5.000 | 4.707 | 5.116 | 0.92× | 4.417 | 5.000 |
| cpu | gemm_512x512x512 | 8×8×16 / 2 | 1 | 130.757 | 131.210 | 132.395 | 136.719 | 1.00× | 133.833 | 133.667 |
| cpu | gemm_1024x1024x1024 | 8×8×16 / 2 | 1 | 1003.701 | 909.245 | 1102.856 | 928.245 | 1.10× | 897.792 | 930.041 |
| cpu | gemm_256x1024x128 | 8×8×16 / 2 | 1 | 65.671 | 65.976 | 66.851 | 66.287 | 1.00× | 65.209 | 67.875 |
| cpu | gemm_1024x128x256 | 8×8×16 / 2 | 1 | 62.720 | 63.068 | 63.383 | 64.061 | 0.99× | 62.458 | 63.000 |
| cpu | gemm_127x193x61 | 8×8×16 / 2 | 1 | 6.263 | 6.823 | 6.294 | 6.836 | 0.92× | 6.208 | 6.583 |
| cpu | gemm_513x257x129 | 8×8×16 / 2 | 1 | 43.181 | 44.962 | 44.456 | 45.347 | 0.96× | 43.041 | 43.583 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.045 | 25.731 | 0.004 | 0.051 | 0.049 | 0.056 | 0.004 | 0.026 |
| cpu / gemm_128x128x128 | 0.048 | 26.741 | 0.006 | 0.028 | 0.065 | 0.049 | 0.015 | 0.006 |
| cpu / gemm_512x512x512 | 0.042 | 25.709 | 0.182 | 0.016 | 0.425 | 0.177 | 0.183 | 0.006 |
| cpu / gemm_1024x1024x1024 | 0.041 | 25.799 | 0.838 | 0.039 | 1.572 | 1.472 | 0.420 | 0.020 |
| cpu / gemm_256x1024x128 | 0.045 | 26.119 | 0.072 | 0.020 | 0.221 | 0.134 | 0.251 | 0.005 |
| cpu / gemm_1024x128x256 | 0.041 | 25.829 | 0.111 | 0.011 | 0.164 | 0.087 | 0.129 | 0.010 |
| cpu / gemm_127x193x61 | 0.042 | 26.044 | 0.009 | 0.017 | 0.069 | 0.042 | 0.009 | 0.003 |
| cpu / gemm_513x257x129 | 0.045 | 25.921 | 0.040 | 0.023 | 0.176 | 0.110 | 0.137 | 0.005 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| cpu / gemm_32x32x32 | accelerate_cblas_sgemm | 0.352 | 1.403× | 0.500 |
| cpu / gemm_128x128x128 | accelerate_cblas_sgemm | 4.071 | 1.133× | 4.333 |
| cpu / gemm_512x512x512 | accelerate_cblas_sgemm | 129.983 | 1.006× | 122.958 |
| cpu / gemm_1024x1024x1024 | accelerate_cblas_sgemm | 1001.778 | 1.002× | 1169.792 |
| cpu / gemm_256x1024x128 | accelerate_cblas_sgemm | 64.268 | 1.022× | 65.333 |
| cpu / gemm_1024x128x256 | accelerate_cblas_sgemm | 61.386 | 1.022× | 62.458 |
| cpu / gemm_127x193x61 | accelerate_cblas_sgemm | 5.768 | 1.086× | 6.334 |
| cpu / gemm_513x257x129 | accelerate_cblas_sgemm | 42.143 | 1.025× | 42.875 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
