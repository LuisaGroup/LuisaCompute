# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T09:57:30.184273+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `True`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 4×16×32 / 2 | 0 | 6.795 | 0.917 | 24.138 | 0.922 | 7.41× | 6.333 | 1.041 |
| cpu | gemm_128x128x128 | 4×16×32 / 2 | 0 | 12.740 | 4.990 | 13.013 | 5.023 | 2.55× | 7.958 | 5.083 |
| cpu | gemm_512x512x512 | 4×16×32 / 2 | 0 | 582.084 | 136.526 | 851.339 | 138.183 | 4.26× | 437.625 | 141.375 |
| cpu | gemm_1024x1024x1024 | 4×16×32 / 2 | 0 | 5654.222 | 992.419 | 5738.097 | 1008.083 | 5.70× | 5391.000 | 1021.000 |
| cpu | gemm_256x1024x128 | 4×16×32 / 2 | 0 | 210.887 | 69.514 | 229.251 | 70.134 | 3.03× | 229.958 | 65.833 |
| cpu | gemm_1024x128x256 | 4×16×32 / 2 | 0 | 146.902 | 65.600 | 189.444 | 66.010 | 2.24× | 98.292 | 63.208 |
| cpu | gemm_127x193x61 | 4×16×32 / 2 | 0 | 182.324 | 6.602 | 193.339 | 6.750 | 27.61× | 186.250 | 6.333 |
| cpu | gemm_513x257x129 | 4×16×32 / 2 | 0 | 1252.341 | 45.108 | 1829.925 | 45.796 | 27.76× | 1484.250 | 45.125 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.076 | 36.977 | 0.004 | 0.050 | 0.108 | 0.040 | 0.002 | 0.004 |
| cpu / gemm_128x128x128 | 0.048 | 40.390 | 0.006 | 0.007 | 0.144 | 0.007 | 0.011 | 0.007 |
| cpu / gemm_512x512x512 | 0.046 | 37.726 | 0.208 | 0.033 | 0.791 | 0.310 | 0.100 | 0.006 |
| cpu / gemm_1024x1024x1024 | 0.056 | 37.209 | 0.873 | 0.028 | 5.218 | 1.226 | 0.513 | 0.006 |
| cpu / gemm_256x1024x128 | 0.051 | 40.440 | 0.074 | 0.028 | 0.381 | 0.162 | 0.087 | 0.004 |
| cpu / gemm_1024x128x256 | 0.051 | 36.638 | 0.101 | 0.017 | 0.523 | 0.096 | 0.113 | 0.006 |
| cpu / gemm_127x193x61 | 0.046 | 102.742 | 0.009 | 0.020 | 0.212 | 0.047 | 0.012 | 0.006 |
| cpu / gemm_513x257x129 | 0.048 | 169.592 | 0.013 | 0.015 | 1.424 | 0.123 | 0.022 | 0.006 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
