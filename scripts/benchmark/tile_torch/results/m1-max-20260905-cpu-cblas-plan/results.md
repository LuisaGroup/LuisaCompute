# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T20:51:19.162172+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 4×16×32 / 2 | 0 | 0.527 | 0.901 | 0.534 | 0.941 | 0.58× | 0.583 | 0.875 |
| cpu | gemm_128x128x128 | 4×16×32 / 2 | 0 | 4.465 | 4.792 | 4.567 | 4.804 | 0.93× | 4.542 | 4.875 |
| cpu | gemm_512x512x512 | 4×16×32 / 2 | 0 | 138.243 | 140.695 | 140.875 | 141.610 | 0.98× | 134.125 | 142.166 |
| cpu | gemm_1024x1024x1024 | 4×16×32 / 2 | 0 | 1030.158 | 974.263 | 1076.863 | 985.191 | 1.06× | 976.250 | 976.791 |
| cpu | gemm_256x1024x128 | 4×16×32 / 2 | 0 | 65.990 | 66.362 | 67.561 | 66.395 | 0.99× | 65.875 | 66.292 |
| cpu | gemm_1024x128x256 | 4×16×32 / 2 | 0 | 62.964 | 63.403 | 63.504 | 64.427 | 0.99× | 62.709 | 63.500 |
| cpu | gemm_127x193x61 | 4×16×32 / 2 | 0 | 6.067 | 6.795 | 6.096 | 6.834 | 0.89× | 6.209 | 6.833 |
| cpu | gemm_513x257x129 | 4×16×32 / 2 | 0 | 43.397 | 43.930 | 44.104 | 44.010 | 0.99× | 46.000 | 43.958 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.046 | 26.505 | 0.007 | 0.040 | 0.059 | 0.033 | 0.003 | 0.071 |
| cpu / gemm_128x128x128 | 0.040 | 26.430 | 0.006 | 0.038 | 0.058 | 0.054 | 0.035 | 0.005 |
| cpu / gemm_512x512x512 | 0.041 | 27.479 | 0.183 | 0.033 | 0.371 | 0.212 | 0.101 | 0.004 |
| cpu / gemm_1024x1024x1024 | 0.038 | 27.418 | 0.789 | 0.026 | 1.558 | 1.364 | 0.534 | 0.008 |
| cpu / gemm_256x1024x128 | 0.041 | 26.660 | 0.066 | 0.024 | 0.236 | 0.137 | 0.097 | 0.006 |
| cpu / gemm_1024x128x256 | 0.042 | 27.450 | 0.106 | 0.013 | 0.171 | 0.072 | 0.153 | 0.011 |
| cpu / gemm_127x193x61 | 0.042 | 26.536 | 0.008 | 0.020 | 0.060 | 0.038 | 0.007 | 0.005 |
| cpu / gemm_513x257x129 | 0.041 | 26.719 | 0.039 | 0.021 | 0.140 | 0.102 | 0.112 | 0.005 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| cpu / gemm_32x32x32 | accelerate_cblas_sgemm | 0.360 | 1.463× | 0.541 |
| cpu / gemm_128x128x128 | accelerate_cblas_sgemm | 4.062 | 1.099× | 4.209 |
| cpu / gemm_512x512x512 | accelerate_cblas_sgemm | 134.322 | 1.029× | 131.000 |
| cpu / gemm_1024x1024x1024 | accelerate_cblas_sgemm | 1144.248 | 0.900× | 985.375 |
| cpu / gemm_256x1024x128 | accelerate_cblas_sgemm | 64.835 | 1.018× | 65.792 |
| cpu / gemm_1024x128x256 | accelerate_cblas_sgemm | 61.203 | 1.029× | 62.750 |
| cpu / gemm_127x193x61 | accelerate_cblas_sgemm | 5.889 | 1.030× | 6.000 |
| cpu / gemm_513x257x129 | accelerate_cblas_sgemm | 42.035 | 1.032× | 46.000 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
