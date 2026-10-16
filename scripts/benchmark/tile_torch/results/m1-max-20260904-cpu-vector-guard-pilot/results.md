# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T10:07:27.168500+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `True`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 4×16×32 / 2 | 0 | 6.658 | 0.909 | 8.051 | 0.916 | 7.32× | 5.541 | 1.208 |
| cpu | gemm_128x128x128 | 4×16×32 / 2 | 0 | 12.924 | 4.995 | 13.378 | 5.091 | 2.59× | 15.084 | 5.041 |
| cpu | gemm_512x512x512 | 4×16×32 / 2 | 0 | 568.582 | 137.256 | 592.110 | 138.868 | 4.14× | 427.709 | 143.917 |
| cpu | gemm_1024x1024x1024 | 4×16×32 / 2 | 0 | 5372.333 | 1050.704 | 5736.445 | 1057.911 | 5.11× | 4887.458 | 1036.166 |
| cpu | gemm_256x1024x128 | 4×16×32 / 2 | 0 | 180.406 | 68.239 | 196.746 | 68.455 | 2.64× | 304.458 | 65.792 |
| cpu | gemm_1024x128x256 | 4×16×32 / 2 | 0 | 152.444 | 65.214 | 181.079 | 65.597 | 2.34× | 127.833 | 64.459 |
| cpu | gemm_127x193x61 | 4×16×32 / 2 | 0 | 30.070 | 6.678 | 40.962 | 6.797 | 4.50× | 91.542 | 6.667 |
| cpu | gemm_513x257x129 | 4×16×32 / 2 | 0 | 193.871 | 45.158 | 204.067 | 45.754 | 4.29× | 151.250 | 44.875 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.057 | 35.356 | 0.004 | 0.049 | 0.209 | 0.034 | 0.003 | 0.016 |
| cpu / gemm_128x128x128 | 0.058 | 41.517 | 0.007 | 0.015 | 0.182 | 0.011 | 0.012 | 0.006 |
| cpu / gemm_512x512x512 | 0.043 | 37.609 | 0.194 | 0.031 | 1.038 | 0.477 | 0.110 | 0.011 |
| cpu / gemm_1024x1024x1024 | 0.053 | 38.849 | 0.763 | 0.035 | 4.980 | 1.587 | 0.550 | 0.007 |
| cpu / gemm_256x1024x128 | 0.049 | 40.936 | 0.113 | 0.031 | 0.278 | 0.202 | 0.115 | 0.005 |
| cpu / gemm_1024x128x256 | 0.052 | 36.310 | 0.112 | 0.018 | 0.521 | 0.090 | 0.085 | 0.006 |
| cpu / gemm_127x193x61 | 0.060 | 133.917 | 0.010 | 0.022 | 0.202 | 0.040 | 0.010 | 0.006 |
| cpu / gemm_513x257x129 | 0.048 | 183.622 | 0.029 | 0.016 | 0.284 | 0.087 | 0.024 | 0.005 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
