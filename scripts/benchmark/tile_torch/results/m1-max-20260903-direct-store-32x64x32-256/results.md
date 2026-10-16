# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T05:43:54.166109+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×64×32 / 1 | 4 | 5.301 | 26.701 | 5.581 | 27.311 | 0.20× | 223.125 | 278.750 |
| metal | gemm_128x128x128 | 32×64×32 / 1 | 4 | 11.929 | 26.467 | 12.121 | 28.605 | 0.45× | 250.792 | 277.167 |
| metal | gemm_512x512x512 | 32×64×32 / 1 | 4 | 57.454 | 48.324 | 58.208 | 48.916 | 1.19× | 304.834 | 304.375 |
| metal | gemm_1024x1024x1024 | 32×64×32 / 1 | 4 | 394.937 | 288.364 | 402.916 | 291.823 | 1.37× | 619.667 | 534.833 |
| metal | gemm_256x1024x128 | 32×64×32 / 1 | 4 | 19.706 | 30.399 | 20.220 | 33.213 | 0.65× | 271.417 | 257.583 |
| metal | gemm_1024x128x256 | 32×64×32 / 1 | 4 | 22.697 | 30.549 | 23.089 | 31.361 | 0.74× | 276.792 | 258.917 |
| metal | gemm_127x193x61 | 32×64×32 / 1 | 4 | 9.433 | 26.679 | 9.588 | 27.704 | 0.35× | 238.084 | 272.250 |
| metal | gemm_513x257x129 | 32×64×32 / 1 | 4 | 23.362 | 34.963 | 28.088 | 35.120 | 0.67× | 274.042 | 275.584 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.056 | 43.762 | 1.305 | 3.174 | 64.638 | 40.112 | 0.288 | 1.053 |
| metal / gemm_128x128x128 | 0.055 | 44.406 | 1.355 | 0.650 | 63.385 | 3.907 | 1.354 | 0.368 |
| metal / gemm_512x512x512 | 0.059 | 44.584 | 1.875 | 1.066 | 64.972 | 3.620 | 0.527 | 0.364 |
| metal / gemm_1024x1024x1024 | 0.059 | 43.121 | 2.776 | 1.961 | 66.668 | 3.578 | 0.856 | 0.399 |
| metal / gemm_256x1024x128 | 0.059 | 44.205 | 1.727 | 1.193 | 73.887 | 4.415 | 0.526 | 0.344 |
| metal / gemm_1024x128x256 | 0.058 | 44.818 | 1.908 | 1.088 | 67.839 | 3.419 | 0.385 | 0.272 |
| metal / gemm_127x193x61 | 0.053 | 55.065 | 1.553 | 1.166 | 74.808 | 4.543 | 0.306 | 0.326 |
| metal / gemm_513x257x129 | 0.054 | 59.005 | 1.329 | 1.076 | 74.487 | 4.056 | 0.475 | 0.352 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
