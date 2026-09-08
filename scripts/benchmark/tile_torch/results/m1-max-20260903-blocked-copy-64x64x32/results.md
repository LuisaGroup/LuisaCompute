# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T06:30:13.451502+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 64×64×32 / 1 | 8 | 5.978 | 26.320 | 6.262 | 26.706 | 0.23× | 242.292 | 264.917 |
| metal | gemm_128x128x128 | 64×64×32 / 1 | 8 | 13.289 | 26.413 | 13.378 | 27.150 | 0.50× | 234.417 | 276.875 |
| metal | gemm_512x512x512 | 64×64×32 / 1 | 8 | 54.866 | 48.378 | 55.537 | 48.806 | 1.13× | 273.125 | 306.958 |
| metal | gemm_1024x1024x1024 | 64×64×32 / 1 | 8 | 319.041 | 289.227 | 323.239 | 290.898 | 1.10× | 515.042 | 524.292 |
| metal | gemm_256x1024x128 | 64×64×32 / 1 | 8 | 17.862 | 29.616 | 17.936 | 30.368 | 0.60× | 225.292 | 276.500 |
| metal | gemm_1024x128x256 | 64×64×32 / 1 | 8 | 22.151 | 29.977 | 22.561 | 30.650 | 0.74× | 277.084 | 250.209 |
| metal | gemm_127x193x61 | 64×64×32 / 1 | 8 | 10.976 | 27.167 | 11.159 | 27.982 | 0.40× | 238.834 | 279.166 |
| metal | gemm_513x257x129 | 64×64×32 / 1 | 8 | 26.755 | 34.892 | 27.262 | 35.278 | 0.77× | 279.375 | 307.916 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.069 | 48.842 | 2.427 | 3.667 | 76.181 | 40.202 | 0.284 | 0.333 |
| metal / gemm_128x128x128 | 0.059 | 46.406 | 1.424 | 0.592 | 71.447 | 4.728 | 0.308 | 0.324 |
| metal / gemm_512x512x512 | 0.051 | 45.858 | 1.850 | 1.267 | 69.962 | 8.382 | 0.456 | 0.337 |
| metal / gemm_1024x1024x1024 | 0.053 | 45.176 | 3.425 | 2.106 | 70.081 | 3.663 | 0.772 | 0.394 |
| metal / gemm_256x1024x128 | 0.052 | 45.033 | 1.350 | 1.111 | 69.330 | 4.196 | 0.559 | 0.342 |
| metal / gemm_1024x128x256 | 0.057 | 46.364 | 1.661 | 1.185 | 70.789 | 4.548 | 0.412 | 0.324 |
| metal / gemm_127x193x61 | 0.058 | 64.604 | 1.637 | 1.267 | 88.159 | 4.593 | 0.333 | 0.280 |
| metal / gemm_513x257x129 | 0.060 | 65.961 | 1.545 | 0.635 | 88.157 | 3.864 | 0.490 | 0.341 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
