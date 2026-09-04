# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T04:11:14.352300+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×64×32 / 1 | 4 | 4.921 | 27.203 | 5.034 | 29.432 | 0.18× | 218.625 | 258.917 |
| metal | gemm_128x128x128 | 32×32×128 / 1 | 2 | 7.216 | 27.020 | 7.451 | 28.884 | 0.27× | 220.000 | 256.875 |
| metal | gemm_512x512x512 | 64×64×64 / 1 | 8 | 52.906 | 49.494 | 53.953 | 49.977 | 1.07× | 297.417 | 292.709 |
| metal | gemm_1024x1024x1024 | 64×64×32 / 1 | 8 | 335.604 | 302.421 | 340.076 | 303.140 | 1.11× | 548.208 | 645.625 |
| metal | gemm_256x1024x128 | 64×64×32 / 1 | 8 | 18.954 | 28.932 | 19.805 | 29.792 | 0.66× | 242.917 | 248.000 |
| metal | gemm_1024x128x256 | 32×64×32 / 1 | 4 | 22.393 | 29.301 | 27.957 | 30.326 | 0.76× | 216.833 | 266.125 |
| metal | gemm_127x193x61 | 32×64×32 / 1 | 4 | 8.826 | 27.816 | 9.110 | 28.472 | 0.32× | 241.459 | 262.000 |
| metal | gemm_513x257x129 | 32×64×32 / 1 | 4 | 22.472 | 35.590 | 22.705 | 45.371 | 0.63× | 236.958 | 297.583 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.052 | 44.673 | 1.481 | 1.135 | 0.744 | 0.466 | 0.263 | 0.247 |
| metal / gemm_128x128x128 | 0.054 | 43.256 | 1.386 | 1.485 | 1.192 | 0.453 | 0.298 | 0.260 |
| metal / gemm_512x512x512 | 0.054 | 48.023 | 2.643 | 0.927 | 1.311 | 0.493 | 0.423 | 0.362 |
| metal / gemm_1024x1024x1024 | 0.062 | 47.696 | 2.909 | 0.972 | 1.713 | 0.746 | 0.773 | 0.434 |
| metal / gemm_256x1024x128 | 0.056 | 45.346 | 1.388 | 2.156 | 1.211 | 0.456 | 0.425 | 0.330 |
| metal / gemm_1024x128x256 | 0.056 | 42.985 | 1.487 | 1.043 | 1.296 | 0.412 | 0.280 | 0.745 |
| metal / gemm_127x193x61 | 0.059 | 55.560 | 1.540 | 1.076 | 1.345 | 0.439 | 0.292 | 0.806 |
| metal / gemm_513x257x129 | 0.061 | 63.581 | 1.581 | 1.079 | 1.315 | 0.453 | 0.339 | 0.293 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

| Device / case | Valid / attempted candidates | Selection wall ms |
|---|---:|---:|
| metal / gemm_32x32x32 | 12 / 16 | 9298.759 |
| metal / gemm_128x128x128 | 16 / 16 | 11201.431 |
| metal / gemm_512x512x512 | 16 / 16 | 11496.071 |
| metal / gemm_1024x1024x1024 | 16 / 16 | 12243.367 |
| metal / gemm_256x1024x128 | 16 / 16 | 11150.611 |
| metal / gemm_1024x128x256 | 16 / 16 | 11223.028 |
| metal / gemm_127x193x61 | 8 / 16 | 7310.585 |
| metal / gemm_513x257x129 | 8 / 16 | 7504.226 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
