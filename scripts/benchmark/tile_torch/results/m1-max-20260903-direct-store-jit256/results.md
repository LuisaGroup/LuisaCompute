# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T05:45:15.532140+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×64×32 / 1 | 4 | 5.485 | 27.336 | 5.509 | 27.626 | 0.20× | 229.416 | 271.500 |
| metal | gemm_128x128x128 | 32×64×32 / 1 | 4 | 11.781 | 26.523 | 11.931 | 28.422 | 0.44× | 242.458 | 281.625 |
| metal | gemm_512x512x512 | 64×64×64 / 1 | 8 | 54.502 | 48.370 | 54.772 | 48.585 | 1.13× | 292.333 | 310.542 |
| metal | gemm_1024x1024x1024 | 64×64×32 / 1 | 8 | 328.169 | 288.333 | 331.371 | 289.917 | 1.14× | 520.208 | 538.208 |
| metal | gemm_256x1024x128 | 32×64×32 / 1 | 4 | 20.060 | 29.002 | 20.151 | 29.554 | 0.69× | 259.917 | 263.834 |
| metal | gemm_1024x128x256 | 64×64×64 / 1 | 8 | 22.862 | 29.632 | 23.015 | 30.991 | 0.77× | 265.166 | 265.375 |
| metal | gemm_127x193x61 | 32×64×32 / 1 | 4 | 9.472 | 27.704 | 9.581 | 28.687 | 0.34× | 239.542 | 270.167 |
| metal | gemm_513x257x129 | 32×64×32 / 1 | 4 | 23.578 | 34.860 | 23.780 | 35.272 | 0.68× | 287.000 | 298.500 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.062 | 44.592 | 1.363 | 1.133 | 64.927 | 0.392 | 0.279 | 0.266 |
| metal / gemm_128x128x128 | 0.055 | 43.591 | 1.208 | 1.327 | 1.160 | 0.405 | 0.347 | 0.436 |
| metal / gemm_512x512x512 | 0.063 | 44.671 | 1.611 | 1.100 | 1.264 | 0.854 | 0.544 | 0.387 |
| metal / gemm_1024x1024x1024 | 0.064 | 47.129 | 2.823 | 1.435 | 1.673 | 0.770 | 0.874 | 1.129 |
| metal / gemm_256x1024x128 | 0.058 | 42.839 | 1.740 | 1.077 | 1.236 | 0.797 | 0.517 | 0.369 |
| metal / gemm_1024x128x256 | 0.060 | 45.866 | 1.661 | 0.750 | 1.238 | 0.436 | 0.415 | 0.351 |
| metal / gemm_127x193x61 | 0.059 | 57.269 | 1.737 | 0.868 | 1.376 | 0.955 | 0.321 | 0.275 |
| metal / gemm_513x257x129 | 0.058 | 59.376 | 1.620 | 1.186 | 1.389 | 0.481 | 0.460 | 0.332 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

| Device / case | Valid / attempted candidates | Selection wall ms |
|---|---:|---:|
| metal / gemm_32x32x32 | 2 / 4 | 3308.302 |
| metal / gemm_128x128x128 | 4 / 4 | 5338.658 |
| metal / gemm_512x512x512 | 4 / 4 | 5369.717 |
| metal / gemm_1024x1024x1024 | 4 / 4 | 5730.965 |
| metal / gemm_256x1024x128 | 4 / 4 | 5101.400 |
| metal / gemm_1024x128x256 | 4 / 4 | 5375.100 |
| metal / gemm_127x193x61 | 2 / 4 | 3332.653 |
| metal / gemm_513x257x129 | 2 / 4 | 3465.761 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
