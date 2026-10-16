# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T01:51:15.900581+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 16×32×32 / 1 | 1 | 4.216 | 29.091 | 6.187 | 30.541 | 0.14× | 222.625 | 291.750 |
| metal | gemm_128x128x128 | 16×32×64 / 2 | 2 | 7.747 | 30.604 | 8.077 | 31.385 | 0.25× | 251.542 | 286.125 |
| metal | gemm_512x512x512 | 16×32×64 / 1 | 1 | 135.212 | 52.849 | 136.762 | 54.329 | 2.56× | 361.917 | 316.542 |
| metal | gemm_1024x1024x1024 | 16×32×32 / 1 | 1 | 1002.684 | 315.625 | 1044.958 | 317.393 | 3.18× | 1170.416 | 847.833 |
| metal | gemm_256x1024x128 | 16×32×64 / 1 | 1 | 39.944 | 32.142 | 40.934 | 33.735 | 1.24× | 243.292 | 266.583 |
| metal | gemm_1024x128x256 | 16×32×64 / 1 | 1 | 38.090 | 33.272 | 39.487 | 34.185 | 1.14× | 282.708 | 286.625 |
| metal | gemm_127x193x61 | 16×32×64 / 1 | 1 | 8.213 | 30.222 | 9.402 | 32.451 | 0.27× | 259.834 | 422.458 |
| metal | gemm_513x257x129 | 16×32×32 / 1 | 1 | 37.403 | 36.876 | 38.606 | 38.254 | 1.01× | 287.542 | 302.959 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.056 | 46.921 | 1.273 | 1.061 | 0.996 | 0.412 | 0.246 | 0.298 |
| metal / gemm_128x128x128 | 0.069 | 52.481 | 2.208 | 0.822 | 1.826 | 0.535 | 0.380 | 0.312 |
| metal / gemm_512x512x512 | 0.061 | 49.132 | 1.646 | 1.188 | 1.329 | 0.520 | 0.557 | 0.329 |
| metal / gemm_1024x1024x1024 | 0.057 | 48.783 | 3.114 | 0.987 | 3.252 | 0.744 | 1.009 | 0.660 |
| metal / gemm_256x1024x128 | 0.065 | 48.413 | 1.611 | 1.080 | 1.235 | 0.481 | 0.694 | 0.390 |
| metal / gemm_1024x128x256 | 0.067 | 47.454 | 1.569 | 0.691 | 1.129 | 0.403 | 0.482 | 0.312 |
| metal / gemm_127x193x61 | 0.055 | 51.832 | 2.593 | 0.891 | 1.305 | 0.418 | 0.292 | 0.288 |
| metal / gemm_513x257x129 | 0.073 | 53.406 | 1.595 | 0.754 | 1.942 | 0.508 | 0.594 | 0.347 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

| Device / case | Valid / attempted candidates | Selection wall ms |
|---|---:|---:|
| metal / gemm_32x32x32 | 8 / 8 | 6775.531 |
| metal / gemm_128x128x128 | 8 / 8 | 6837.167 |
| metal / gemm_512x512x512 | 8 / 8 | 7017.554 |
| metal / gemm_1024x1024x1024 | 8 / 8 | 7374.497 |
| metal / gemm_256x1024x128 | 8 / 8 | 6731.825 |
| metal / gemm_1024x128x256 | 8 / 8 | 6733.435 |
| metal / gemm_127x193x61 | 8 / 8 | 7033.397 |
| metal / gemm_513x257x129 | 8 / 8 | 6913.316 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
