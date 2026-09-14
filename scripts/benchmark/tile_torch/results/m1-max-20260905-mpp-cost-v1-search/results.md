# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T19:44:56.669223+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×32×32 / 1 | 1 | 2.811 | 26.516 | 3.096 | 26.579 | 0.11× | 225.375 | 259.666 |
| metal | gemm_128x128x128 | 32×32×128 / 1 | 1 | 5.745 | 27.217 | 5.807 | 27.226 | 0.21× | 229.750 | 269.417 |
| metal | gemm_512x512x512 | 128×32×128 / 1 | 1 | 42.917 | 49.245 | 42.933 | 49.291 | 0.87× | 279.917 | 319.333 |
| metal | gemm_1024x1024x1024 | 128×32×1024 / 1 | 1 | 277.202 | 293.289 | 277.473 | 293.597 | 0.95× | 544.542 | 534.042 |
| metal | gemm_256x1024x128 | 32×32×32 / 1 | 1 | 17.056 | 28.823 | 17.141 | 28.873 | 0.59× | 239.459 | 247.750 |
| metal | gemm_1024x128x256 | 32×32×128 / 1 | 1 | 18.066 | 28.729 | 18.095 | 28.847 | 0.63× | 241.000 | 270.833 |
| metal | gemm_127x193x61 | 32×32×32 / 1 | 1 | 9.585 | 27.200 | 9.595 | 27.558 | 0.35× | 256.792 | 252.667 |
| metal | gemm_513x257x129 | 32×32×32 / 1 | 1 | 22.805 | 35.428 | 22.984 | 35.429 | 0.64× | 246.209 | 297.709 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.051 | 27.365 | 1.449 | 0.716 | 0.786 | 0.411 | 0.279 | 0.229 |
| metal / gemm_128x128x128 | 0.050 | 27.596 | 1.382 | 1.023 | 1.647 | 0.423 | 0.267 | 0.247 |
| metal / gemm_512x512x512 | 0.055 | 27.728 | 1.761 | 1.151 | 4.189 | 0.463 | 0.451 | 0.315 |
| metal / gemm_1024x1024x1024 | 0.055 | 28.830 | 3.541 | 1.312 | 2.177 | 0.707 | 0.956 | 0.449 |
| metal / gemm_256x1024x128 | 0.053 | 27.624 | 1.569 | 1.090 | 1.418 | 0.442 | 0.411 | 0.267 |
| metal / gemm_1024x128x256 | 0.058 | 28.912 | 1.724 | 1.097 | 1.517 | 0.497 | 0.355 | 0.270 |
| metal / gemm_127x193x61 | 0.053 | 31.991 | 1.553 | 1.016 | 1.118 | 0.419 | 0.283 | 0.280 |
| metal / gemm_513x257x129 | 0.053 | 31.967 | 1.377 | 1.167 | 1.043 | 0.475 | 0.467 | 0.265 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

The model column is diagnostic only: it compares the lowest reported normalized kernel cost with the measured winner inside the same finite candidate set. Model regret is measured(model pick) / measured(best) - 1; it is not a hardware-optimality claim.

| Device / case | Valid / attempted candidates | Model pick / measured pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_32x32x32 | 11 / 45 | 32×32×32 / 32×32×32 | 42.05% | 8489.616 |
| metal / gemm_128x128x128 | 17 / 45 | 32×32×128 / 32×32×128 | 44.05% | 10615.660 |
| metal / gemm_512x512x512 | 28 / 45 | 64×64×512 / 128×32×128 | 6.28% | 14236.239 |
| metal / gemm_1024x1024x1024 | 34 / 45 | 128×32×1024 / 128×32×1024 | 0.00% | 17525.871 |
| metal / gemm_256x1024x128 | 17 / 45 | 64×64×128 / 32×32×32 | 3.45% | 10674.740 |
| metal / gemm_1024x128x256 | 22 / 45 | 32×64×32 / 32×32×128 | 84.10% | 12255.452 |
| metal / gemm_127x193x61 | 8 / 45 | 32×32×32 / 32×32×32 | 173.93% | 7495.861 |
| metal / gemm_513x257x129 | 8 / 45 | 32×64×32 / 32×32×32 | 239.58% | 7617.165 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
