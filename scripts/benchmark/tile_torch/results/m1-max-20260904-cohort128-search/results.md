# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T04:03:22.344853+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 64×64×32 / 1 | 16 | 8.619 | 27.662 | 8.785 | 28.383 | 0.31× | 241.625 | 251.125 |
| metal | gemm_128x128x128 | 32×32×128 / 1 | 4 | 8.198 | 28.986 | 8.274 | 30.015 | 0.28× | 250.333 | 266.416 |
| metal | gemm_512x512x512 | 64×64×32 / 1 | 16 | 57.694 | 49.705 | 59.554 | 49.975 | 1.16× | 283.416 | 311.875 |
| metal | gemm_1024x1024x1024 | 64×64×32 / 1 | 16 | 342.764 | 294.551 | 342.807 | 296.145 | 1.16× | 539.333 | 510.000 |
| metal | gemm_256x1024x128 | 64×64×32 / 1 | 16 | 19.651 | 30.203 | 19.810 | 30.635 | 0.65× | 220.208 | 271.667 |
| metal | gemm_1024x128x256 | 32×64×64 / 1 | 8 | 22.148 | 29.653 | 22.846 | 30.070 | 0.75× | 250.583 | 253.083 |
| metal | gemm_127x193x61 | 32×64×64 / 1 | 8 | 12.380 | 27.196 | 12.581 | 29.000 | 0.46× | 240.375 | 269.083 |
| metal | gemm_513x257x129 | 64×64×32 / 1 | 16 | 35.976 | 35.066 | 36.866 | 35.648 | 1.03× | 276.541 | 269.417 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.056 | 49.641 | 1.460 | 1.033 | 0.877 | 0.457 | 0.236 | 0.247 |
| metal / gemm_128x128x128 | 0.059 | 41.653 | 1.408 | 1.058 | 1.103 | 0.448 | 0.269 | 0.364 |
| metal / gemm_512x512x512 | 0.059 | 47.389 | 1.471 | 1.204 | 1.132 | 0.488 | 0.432 | 0.384 |
| metal / gemm_1024x1024x1024 | 0.053 | 46.580 | 2.821 | 0.984 | 1.569 | 0.717 | 0.817 | 0.379 |
| metal / gemm_256x1024x128 | 0.058 | 46.949 | 1.650 | 1.095 | 1.171 | 0.430 | 0.500 | 0.347 |
| metal / gemm_1024x128x256 | 0.056 | 43.944 | 1.770 | 1.099 | 1.050 | 0.422 | 0.395 | 0.334 |
| metal / gemm_127x193x61 | 0.055 | 53.475 | 1.177 | 1.004 | 1.005 | 0.411 | 0.305 | 0.264 |
| metal / gemm_513x257x129 | 0.062 | 64.665 | 1.244 | 1.075 | 1.241 | 0.465 | 0.453 | 0.338 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

| Device / case | Valid / attempted candidates | Selection wall ms |
|---|---:|---:|
| metal / gemm_32x32x32 | 3 / 6 | 3784.964 |
| metal / gemm_128x128x128 | 6 / 6 | 4345.088 |
| metal / gemm_512x512x512 | 6 / 6 | 4464.222 |
| metal / gemm_1024x1024x1024 | 6 / 6 | 4730.657 |
| metal / gemm_256x1024x128 | 6 / 6 | 4303.623 |
| metal / gemm_1024x128x256 | 6 / 6 | 4350.833 |
| metal / gemm_127x193x61 | 2 / 6 | 2298.909 |
| metal / gemm_513x257x129 | 2 / 6 | 2333.971 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
