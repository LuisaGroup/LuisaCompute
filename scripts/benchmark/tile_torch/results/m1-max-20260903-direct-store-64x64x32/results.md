# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T05:40:45.951695+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 64×64×32 / 1 | 8 | 5.879 | 26.337 | 6.298 | 26.617 | 0.22× | 227.958 | 280.500 |
| metal | gemm_128x128x128 | 64×64×32 / 1 | 8 | 13.654 | 26.280 | 14.018 | 27.283 | 0.52× | 240.000 | 253.375 |
| metal | gemm_512x512x512 | 64×64×32 / 1 | 8 | 57.133 | 48.362 | 57.808 | 49.447 | 1.18× | 282.667 | 304.583 |
| metal | gemm_1024x1024x1024 | 64×64×32 / 1 | 8 | 327.682 | 288.618 | 333.483 | 292.395 | 1.14× | 519.625 | 587.500 |
| metal | gemm_256x1024x128 | 64×64×32 / 1 | 8 | 20.057 | 29.216 | 20.180 | 31.000 | 0.69× | 255.167 | 260.917 |
| metal | gemm_1024x128x256 | 64×64×32 / 1 | 8 | 24.990 | 29.537 | 25.674 | 29.998 | 0.85× | 264.708 | 275.917 |
| metal | gemm_127x193x61 | 64×64×32 / 1 | 8 | 12.368 | 27.516 | 12.563 | 29.264 | 0.45× | 265.208 | 276.791 |
| metal | gemm_513x257x129 | 64×64×32 / 1 | 8 | 28.187 | 35.056 | 28.463 | 35.168 | 0.80× | 259.875 | 295.500 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.077 | 48.438 | 1.187 | 3.321 | 0.922 | 45.400 | 0.291 | 0.331 |
| metal / gemm_128x128x128 | 0.057 | 44.187 | 1.242 | 1.613 | 73.153 | 4.167 | 0.291 | 0.327 |
| metal / gemm_512x512x512 | 0.054 | 44.434 | 2.933 | 1.145 | 70.859 | 3.676 | 0.455 | 0.338 |
| metal / gemm_1024x1024x1024 | 0.060 | 46.775 | 3.002 | 2.265 | 75.677 | 4.649 | 1.074 | 0.483 |
| metal / gemm_256x1024x128 | 0.060 | 45.527 | 1.331 | 1.166 | 71.624 | 4.457 | 0.546 | 0.350 |
| metal / gemm_1024x128x256 | 0.054 | 44.628 | 1.575 | 0.692 | 72.356 | 3.574 | 0.411 | 0.360 |
| metal / gemm_127x193x61 | 0.062 | 55.831 | 1.632 | 1.135 | 1.286 | 4.447 | 0.344 | 0.294 |
| metal / gemm_513x257x129 | 0.056 | 64.144 | 1.356 | 0.890 | 1.319 | 3.713 | 0.418 | 0.330 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
