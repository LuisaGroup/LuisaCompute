# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T06:51:17.335003+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×32×32 / 1 | 1 | 3.079 | 27.375 | 3.206 | 27.413 | 0.11× | 230.167 | 229.666 |
| metal | gemm_128x128x128 | 32×32×128 / 1 | 1 | 6.002 | 27.086 | 6.234 | 27.507 | 0.22× | 233.583 | 262.500 |
| metal | gemm_512x512x512 | 32×64×32 / 1 | 1 | 42.853 | 48.538 | 45.149 | 50.924 | 0.88× | 271.291 | 297.250 |
| metal | gemm_1024x1024x1024 | 64×64×1024 / 1 | 1 | 292.165 | 293.871 | 292.406 | 311.285 | 0.99× | 719.667 | 521.208 |
| metal | gemm_256x1024x128 | 32×64×32 / 1 | 1 | 16.053 | 29.369 | 17.085 | 29.789 | 0.55× | 219.333 | 275.334 |
| metal | gemm_1024x128x256 | 32×32×32 / 1 | 1 | 17.016 | 32.933 | 18.107 | 33.887 | 0.52× | 328.291 | 281.708 |
| metal | gemm_127x193x61 | 32×32×32 / 1 | 1 | 14.337 | 27.527 | 14.490 | 28.499 | 0.52× | 262.125 | 268.125 |
| metal | gemm_513x257x129 | 32×32×32 / 1 | 1 | 30.435 | 35.469 | 30.582 | 36.950 | 0.86× | 310.250 | 300.583 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.051 | 29.507 | 1.406 | 1.004 | 0.828 | 0.485 | 0.283 | 0.415 |
| metal / gemm_128x128x128 | 0.053 | 27.743 | 1.394 | 1.090 | 1.216 | 0.469 | 0.282 | 0.304 |
| metal / gemm_512x512x512 | 0.067 | 28.547 | 2.497 | 1.069 | 2.467 | 0.485 | 0.429 | 0.350 |
| metal / gemm_1024x1024x1024 | 0.052 | 32.121 | 3.071 | 1.035 | 2.237 | 2.313 | 0.877 | 0.449 |
| metal / gemm_256x1024x128 | 0.055 | 28.300 | 1.403 | 1.125 | 2.235 | 0.429 | 0.433 | 0.276 |
| metal / gemm_1024x128x256 | 0.057 | 30.759 | 1.480 | 1.090 | 1.481 | 0.469 | 0.413 | 0.305 |
| metal / gemm_127x193x61 | 0.051 | 32.223 | 1.546 | 1.014 | 1.121 | 0.424 | 0.299 | 0.281 |
| metal / gemm_513x257x129 | 0.053 | 32.150 | 1.341 | 0.985 | 1.094 | 0.437 | 0.379 | 0.269 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

| Device / case | Valid / attempted candidates | Selection wall ms |
|---|---:|---:|
| metal / gemm_32x32x32 | 8 / 33 | 8621.275 |
| metal / gemm_128x128x128 | 9 / 33 | 8544.502 |
| metal / gemm_512x512x512 | 13 / 33 | 10513.332 |
| metal / gemm_1024x1024x1024 | 22 / 33 | 15611.737 |
| metal / gemm_256x1024x128 | 9 / 33 | 8577.273 |
| metal / gemm_1024x128x256 | 9 / 33 | 8565.468 |
| metal / gemm_127x193x61 | 5 / 33 | 6849.200 |
| metal / gemm_513x257x129 | 5 / 33 | 6978.819 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
