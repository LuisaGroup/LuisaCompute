# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T02:43:28.090709+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×32×32 / 2 | 1 | 5.222 | 28.775 | 6.022 | 33.215 | 0.18× | 217.917 | 373.416 |
| metal | gemm_128x128x128 | 16×32×64 / 2 | 2 | 7.650 | 29.271 | 8.526 | 30.810 | 0.26× | 247.292 | 286.958 |
| metal | gemm_512x512x512 | 16×32×64 / 1 | 1 | 210.386 | 103.476 | 233.644 | 112.698 | 2.03× | 479.625 | 457.417 |
| metal | gemm_1024x1024x1024 | 16×32×32 / 1 | 1 | 2169.481 | 835.920 | 2270.091 | 1011.779 | 2.60× | 2375.916 | 1194.042 |
| metal | gemm_256x1024x128 | 16×32×32 / 1 | 1 | 48.686 | 35.124 | 54.313 | 38.399 | 1.39× | 305.541 | 355.792 |
| metal | gemm_1024x128x256 | 16×32×64 / 1 | 1 | 45.833 | 38.611 | 51.646 | 42.906 | 1.19× | 271.875 | 274.958 |
| metal | gemm_127x193x61 | 16×32×64 / 1 | 1 | 8.414 | 29.737 | 8.519 | 34.737 | 0.28× | 311.916 | 271.666 |
| metal | gemm_513x257x129 | 16×32×32 / 1 | 1 | 40.211 | 44.996 | 46.882 | 48.644 | 0.89× | 416.500 | 349.000 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.057 | 45.281 | 1.272 | 3.713 | 1.194 | 0.965 | 0.579 | 0.338 |
| metal / gemm_128x128x128 | 0.064 | 50.990 | 1.504 | 0.917 | 3.061 | 0.478 | 2.614 | 0.360 |
| metal / gemm_512x512x512 | 0.061 | 47.665 | 2.110 | 0.969 | 1.733 | 0.496 | 0.505 | 0.383 |
| metal / gemm_1024x1024x1024 | 0.061 | 47.601 | 5.737 | 3.603 | 4.046 | 2.499 | 2.162 | 0.686 |
| metal / gemm_256x1024x128 | 0.292 | 53.331 | 3.877 | 1.164 | 1.881 | 1.069 | 0.820 | 0.374 |
| metal / gemm_1024x128x256 | 0.061 | 46.234 | 1.766 | 1.198 | 1.443 | 0.591 | 0.503 | 2.691 |
| metal / gemm_127x193x61 | 0.059 | 52.519 | 1.621 | 0.881 | 1.586 | 0.467 | 0.380 | 0.307 |
| metal / gemm_513x257x129 | 0.055 | 54.183 | 1.339 | 2.017 | 1.487 | 0.511 | 0.430 | 0.291 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

| Device / case | Valid / attempted candidates | Selection wall ms |
|---|---:|---:|
| metal / gemm_32x32x32 | 10 / 10 | 8283.346 |
| metal / gemm_128x128x128 | 10 / 10 | 8251.513 |
| metal / gemm_512x512x512 | 10 / 10 | 10589.351 |
| metal / gemm_1024x1024x1024 | 10 / 10 | 10138.966 |
| metal / gemm_256x1024x128 | 10 / 10 | 9187.120 |
| metal / gemm_1024x128x256 | 10 / 10 | 9360.129 |
| metal / gemm_127x193x61 | 10 / 10 | 8654.518 |
| metal / gemm_513x257x129 | 10 / 10 | 9215.031 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
