# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T04:26:03.878411+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×64×32 / 1 | 8 | 7.797 | 27.407 | 8.169 | 28.548 | 0.28× | 218.041 | 260.000 |
| metal | gemm_128x128x128 | 32×64×32 / 1 | 8 | 11.956 | 26.814 | 13.234 | 27.731 | 0.45× | 217.875 | 259.083 |
| metal | gemm_512x512x512 | 32×64×32 / 1 | 8 | 78.119 | 48.137 | 78.287 | 48.597 | 1.62× | 306.208 | 329.917 |
| metal | gemm_1024x1024x1024 | 32×64×32 / 1 | 8 | 474.457 | 289.087 | 476.317 | 289.984 | 1.64× | 669.959 | 520.167 |
| metal | gemm_256x1024x128 | 32×64×32 / 1 | 8 | 25.456 | 29.925 | 25.869 | 30.537 | 0.85× | 252.291 | 247.666 |
| metal | gemm_1024x128x256 | 32×64×32 / 1 | 8 | 22.842 | 29.131 | 24.049 | 30.181 | 0.78× | 266.041 | 255.042 |
| metal | gemm_127x193x61 | 32×64×32 / 1 | 8 | 21.654 | 27.915 | 21.915 | 29.286 | 0.78× | 245.041 | 284.500 |
| metal | gemm_513x257x129 | 32×64×32 / 1 | 8 | 41.790 | 34.751 | 42.362 | 35.070 | 1.20× | 276.417 | 272.167 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.053 | 42.891 | 1.092 | 3.342 | 0.803 | 43.939 | 0.303 | 0.339 |
| metal / gemm_128x128x128 | 0.053 | 42.781 | 1.174 | 0.618 | 0.998 | 5.239 | 0.291 | 0.293 |
| metal / gemm_512x512x512 | 0.057 | 43.516 | 1.915 | 1.048 | 1.351 | 3.755 | 0.535 | 0.406 |
| metal / gemm_1024x1024x1024 | 0.051 | 42.671 | 3.231 | 2.300 | 1.536 | 4.244 | 0.833 | 0.405 |
| metal / gemm_256x1024x128 | 0.052 | 42.819 | 1.394 | 1.688 | 1.027 | 4.194 | 0.643 | 0.310 |
| metal / gemm_1024x128x256 | 0.054 | 42.778 | 1.659 | 0.643 | 1.067 | 4.562 | 0.367 | 0.354 |
| metal / gemm_127x193x61 | 0.065 | 50.127 | 1.643 | 1.262 | 1.041 | 4.507 | 0.313 | 0.319 |
| metal / gemm_513x257x129 | 0.059 | 48.566 | 1.626 | 1.050 | 1.345 | 5.551 | 0.405 | 0.292 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
