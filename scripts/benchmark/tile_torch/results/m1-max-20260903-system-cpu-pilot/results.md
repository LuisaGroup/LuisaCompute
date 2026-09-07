# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T07:07:09.319303+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `True`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 4×16×32 / 2 | 0 | 4.576 | 0.877 | 6.063 | 0.891 | 5.22× | 2.500 | 1.000 |
| cpu | gemm_128x128x128 | 4×16×32 / 2 | 0 | 42.134 | 4.894 | 50.931 | 4.949 | 8.61× | 27.000 | 4.833 |
| cpu | gemm_512x512x512 | 4×16×32 / 2 | 0 | 1466.140 | 137.767 | 1842.312 | 139.822 | 10.64× | 1242.708 | 138.083 |
| cpu | gemm_1024x1024x1024 | 4×16×32 / 2 | 0 | 10079.688 | 999.848 | 11455.592 | 1007.446 | 10.08× | 9528.209 | 979.292 |
| cpu | gemm_256x1024x128 | 4×16×32 / 2 | 0 | 597.094 | 67.457 | 711.446 | 67.976 | 8.85× | 502.833 | 65.833 |
| cpu | gemm_1024x128x256 | 4×16×32 / 2 | 0 | 422.103 | 64.338 | 649.822 | 64.863 | 6.56× | 370.875 | 62.958 |
| cpu | gemm_127x193x61 | 4×16×32 / 2 | 0 | 71.201 | 6.538 | 78.567 | 6.604 | 10.89× | 67.042 | 6.583 |
| cpu | gemm_513x257x129 | 4×16×32 / 2 | 0 | 633.803 | 44.627 | 793.778 | 44.935 | 14.20× | 685.584 | 43.375 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.069 | 115.461 | 0.004 | 0.094 | 0.164 | 0.038 | 0.002 | 0.029 |
| cpu / gemm_128x128x128 | 0.058 | 169.194 | 0.014 | 0.025 | 0.154 | 0.021 | 0.004 | 0.006 |
| cpu / gemm_512x512x512 | 0.056 | 171.232 | 0.167 | 0.026 | 1.896 | 0.188 | 0.049 | 0.005 |
| cpu / gemm_1024x1024x1024 | 0.046 | 170.936 | 0.806 | 0.030 | 10.267 | 1.716 | 0.576 | 0.007 |
| cpu / gemm_256x1024x128 | 0.068 | 168.509 | 0.044 | 0.034 | 0.568 | 0.133 | 0.066 | 0.005 |
| cpu / gemm_1024x128x256 | 0.064 | 168.044 | 0.034 | 0.016 | 0.554 | 0.087 | 0.016 | 0.005 |
| cpu / gemm_127x193x61 | 0.058 | 268.100 | 0.013 | 0.018 | 0.170 | 0.031 | 0.013 | 0.006 |
| cpu / gemm_513x257x129 | 0.046 | 234.255 | 0.017 | 0.022 | 0.856 | 0.128 | 0.017 | 0.005 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and all six rotating implementation orders are recorded in JSON.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| cpu / gemm_32x32x32 | accelerate_cblas_sgemm | 0.496 | 9.230× | 0.667 |
| cpu / gemm_128x128x128 | accelerate_cblas_sgemm | 4.195 | 10.044× | 4.417 |
| cpu / gemm_512x512x512 | accelerate_cblas_sgemm | 139.557 | 10.506× | 149.291 |
| cpu / gemm_1024x1024x1024 | accelerate_cblas_sgemm | 998.383 | 10.096× | 919.458 |
| cpu / gemm_256x1024x128 | accelerate_cblas_sgemm | 66.267 | 9.010× | 67.750 |
| cpu / gemm_1024x128x256 | accelerate_cblas_sgemm | 62.190 | 6.787× | 62.500 |
| cpu / gemm_127x193x61 | accelerate_cblas_sgemm | 5.808 | 12.258× | 5.792 |
| cpu / gemm_513x257x129 | accelerate_cblas_sgemm | 43.394 | 14.606× | 46.791 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
