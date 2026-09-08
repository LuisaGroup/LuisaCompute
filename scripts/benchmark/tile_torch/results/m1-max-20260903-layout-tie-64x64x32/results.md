# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T06:24:32.637029+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 64×64×32 / 1 | 8 | 6.343 | 26.462 | 6.455 | 27.135 | 0.24× | 204.959 | 272.583 |
| metal | gemm_128x128x128 | 64×64×32 / 1 | 8 | 13.301 | 26.510 | 13.510 | 27.165 | 0.50× | 250.292 | 283.083 |
| metal | gemm_512x512x512 | 64×64×32 / 1 | 8 | 57.216 | 48.293 | 57.650 | 48.540 | 1.18× | 297.375 | 300.541 |
| metal | gemm_1024x1024x1024 | 64×64×32 / 1 | 8 | 333.451 | 289.222 | 339.081 | 294.242 | 1.15× | 530.708 | 527.083 |
| metal | gemm_256x1024x128 | 64×64×32 / 1 | 8 | 19.845 | 29.820 | 20.106 | 30.646 | 0.67× | 251.458 | 257.666 |
| metal | gemm_1024x128x256 | 64×64×32 / 1 | 8 | 24.571 | 29.106 | 31.331 | 29.701 | 0.84× | 242.208 | 245.167 |
| metal | gemm_127x193x61 | 64×64×32 / 1 | 8 | 12.165 | 26.938 | 12.485 | 28.627 | 0.45× | 237.125 | 258.500 |
| metal | gemm_513x257x129 | 64×64×32 / 1 | 8 | 27.879 | 35.215 | 28.698 | 35.565 | 0.79× | 251.500 | 270.000 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.074 | 49.165 | 1.407 | 3.164 | 78.815 | 41.672 | 0.303 | 0.445 |
| metal / gemm_128x128x128 | 0.052 | 44.334 | 1.399 | 0.628 | 71.650 | 4.268 | 0.312 | 0.294 |
| metal / gemm_512x512x512 | 0.061 | 47.899 | 1.818 | 1.145 | 74.099 | 6.045 | 0.455 | 0.327 |
| metal / gemm_1024x1024x1024 | 0.053 | 44.751 | 2.931 | 2.041 | 73.313 | 4.830 | 0.991 | 0.414 |
| metal / gemm_256x1024x128 | 0.060 | 44.725 | 1.650 | 1.046 | 71.294 | 4.551 | 0.512 | 0.321 |
| metal / gemm_1024x128x256 | 0.063 | 44.989 | 1.434 | 0.697 | 75.023 | 3.889 | 0.409 | 0.305 |
| metal / gemm_127x193x61 | 0.077 | 55.822 | 1.231 | 1.254 | 84.182 | 6.191 | 0.291 | 0.307 |
| metal / gemm_513x257x129 | 0.058 | 62.639 | 1.606 | 1.873 | 88.276 | 4.456 | 0.411 | 0.299 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
