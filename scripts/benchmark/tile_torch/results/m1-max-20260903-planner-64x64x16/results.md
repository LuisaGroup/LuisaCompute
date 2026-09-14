# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T04:38:27.782787+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 64×64×16 / 1 | 8 | 7.860 | 26.227 | 8.289 | 27.272 | 0.30× | 222.416 | 267.625 |
| metal | gemm_128x128x128 | 64×64×16 / 1 | 8 | 15.967 | 26.643 | 16.903 | 27.107 | 0.60× | 248.959 | 272.709 |
| metal | gemm_512x512x512 | 64×64×16 / 1 | 8 | 63.573 | 48.370 | 64.366 | 48.824 | 1.31× | 283.500 | 300.792 |
| metal | gemm_1024x1024x1024 | 64×64×16 / 1 | 8 | 466.379 | 288.956 | 471.514 | 295.344 | 1.61× | 658.292 | 539.000 |
| metal | gemm_256x1024x128 | 64×64×16 / 1 | 8 | 21.611 | 30.144 | 21.768 | 30.759 | 0.72× | 250.375 | 256.916 |
| metal | gemm_1024x128x256 | 64×64×16 / 1 | 8 | 31.338 | 29.131 | 31.693 | 30.054 | 1.08× | 265.083 | 246.084 |
| metal | gemm_127x193x61 | 64×64×16 / 1 | 8 | 18.327 | 26.991 | 18.608 | 27.827 | 0.68× | 234.625 | 254.708 |
| metal | gemm_513x257x129 | 64×64×16 / 1 | 8 | 35.085 | 35.054 | 35.365 | 35.362 | 1.00× | 263.458 | 289.042 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.076 | 46.054 | 1.308 | 2.924 | 76.207 | 38.104 | 0.267 | 0.307 |
| metal / gemm_128x128x128 | 0.054 | 43.832 | 1.409 | 0.686 | 75.529 | 3.939 | 0.335 | 0.261 |
| metal / gemm_512x512x512 | 0.060 | 43.795 | 1.838 | 1.586 | 76.030 | 3.688 | 0.446 | 0.371 |
| metal / gemm_1024x1024x1024 | 0.056 | 44.018 | 2.925 | 1.956 | 77.240 | 4.850 | 0.851 | 0.436 |
| metal / gemm_256x1024x128 | 0.057 | 45.059 | 1.748 | 1.108 | 81.041 | 4.181 | 0.506 | 1.187 |
| metal / gemm_1024x128x256 | 0.053 | 43.919 | 1.708 | 1.082 | 74.704 | 3.739 | 0.386 | 0.293 |
| metal / gemm_127x193x61 | 0.060 | 49.565 | 1.661 | 1.133 | 79.127 | 6.248 | 0.293 | 0.306 |
| metal / gemm_513x257x129 | 0.050 | 49.913 | 1.718 | 0.940 | 84.150 | 3.440 | 0.483 | 0.307 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
