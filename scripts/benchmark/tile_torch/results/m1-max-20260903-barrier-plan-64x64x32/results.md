# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T06:05:49.562881+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 64×64×32 / 1 | 8 | 5.472 | 26.798 | 5.603 | 27.198 | 0.20× | 206.875 | 285.584 |
| metal | gemm_128x128x128 | 64×64×32 / 1 | 8 | 12.044 | 26.922 | 12.114 | 27.382 | 0.45× | 247.625 | 277.042 |
| metal | gemm_512x512x512 | 64×64×32 / 1 | 8 | 54.091 | 48.050 | 54.870 | 48.274 | 1.13× | 291.250 | 301.375 |
| metal | gemm_1024x1024x1024 | 64×64×32 / 1 | 8 | 320.113 | 286.692 | 322.339 | 290.302 | 1.12× | 524.584 | 530.042 |
| metal | gemm_256x1024x128 | 64×64×32 / 1 | 8 | 18.059 | 29.620 | 18.122 | 30.107 | 0.61× | 240.500 | 249.875 |
| metal | gemm_1024x128x256 | 64×64×32 / 1 | 8 | 22.141 | 29.876 | 22.366 | 30.449 | 0.74× | 269.084 | 255.584 |
| metal | gemm_127x193x61 | 64×64×32 / 1 | 8 | 11.192 | 27.026 | 11.275 | 28.236 | 0.41× | 243.708 | 268.666 |
| metal | gemm_513x257x129 | 64×64×32 / 1 | 8 | 26.970 | 34.722 | 27.594 | 35.120 | 0.78× | 302.625 | 266.541 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.070 | 49.056 | 1.310 | 3.087 | 78.290 | 43.225 | 0.248 | 0.335 |
| metal / gemm_128x128x128 | 0.062 | 45.920 | 1.595 | 0.611 | 72.110 | 3.869 | 0.329 | 0.361 |
| metal / gemm_512x512x512 | 0.059 | 46.690 | 2.405 | 1.237 | 74.657 | 3.714 | 0.467 | 0.363 |
| metal / gemm_1024x1024x1024 | 0.054 | 45.257 | 2.754 | 1.984 | 73.309 | 3.601 | 0.839 | 0.397 |
| metal / gemm_256x1024x128 | 0.059 | 45.866 | 1.654 | 1.113 | 72.348 | 8.503 | 0.534 | 0.333 |
| metal / gemm_1024x128x256 | 0.060 | 46.266 | 1.686 | 0.826 | 71.746 | 4.184 | 0.430 | 0.397 |
| metal / gemm_127x193x61 | 0.056 | 57.196 | 1.517 | 1.151 | 87.523 | 4.848 | 0.337 | 0.294 |
| metal / gemm_513x257x129 | 0.056 | 66.952 | 1.473 | 0.723 | 95.937 | 3.474 | 0.461 | 0.325 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
