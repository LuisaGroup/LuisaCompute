# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T19:58:02.528393+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×32×32 / 1 | 1 | 3.262 | 26.374 | 3.279 | 26.781 | 0.12× | 239.042 | 256.917 |
| metal | gemm_128x128x128 | 32×32×128 / 1 | 1 | 8.183 | 26.963 | 8.445 | 27.037 | 0.30× | 253.583 | 256.542 |
| metal | gemm_512x512x512 | 32×64×32 / 1 | 1 | 42.958 | 49.044 | 42.991 | 49.307 | 0.88× | 275.042 | 307.625 |
| metal | gemm_1024x1024x1024 | 128×32×1024 / 1 | 1 | 277.412 | 292.638 | 278.139 | 292.895 | 0.95× | 491.958 | 541.750 |
| metal | gemm_256x1024x128 | 64×64×128 / 1 | 1 | 17.379 | 29.929 | 24.787 | 34.930 | 0.58× | 268.125 | 262.042 |
| metal | gemm_1024x128x256 | 32×32×32 / 1 | 1 | 16.940 | 30.626 | 16.969 | 30.870 | 0.55× | 244.375 | 255.000 |
| metal | gemm_127x193x61 | 32×32×32 / 1 | 1 | 9.688 | 27.189 | 9.717 | 28.200 | 0.36× | 264.792 | 259.917 |
| metal | gemm_513x257x129 | 32×32×32 / 1 | 1 | 22.868 | 35.087 | 22.941 | 35.190 | 0.65× | 271.708 | 269.250 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.064 | 28.286 | 1.544 | 0.731 | 0.745 | 0.419 | 0.274 | 0.216 |
| metal / gemm_128x128x128 | 0.054 | 27.613 | 1.235 | 1.012 | 1.677 | 0.424 | 0.266 | 0.295 |
| metal / gemm_512x512x512 | 0.050 | 27.603 | 1.375 | 1.114 | 2.076 | 0.440 | 0.419 | 0.358 |
| metal / gemm_1024x1024x1024 | 0.051 | 27.355 | 2.593 | 0.965 | 2.045 | 0.755 | 0.719 | 0.438 |
| metal / gemm_256x1024x128 | 0.054 | 27.705 | 1.621 | 2.078 | 4.102 | 0.454 | 0.467 | 0.370 |
| metal / gemm_1024x128x256 | 0.054 | 28.042 | 1.693 | 1.074 | 1.748 | 0.419 | 0.316 | 0.310 |
| metal / gemm_127x193x61 | 0.049 | 31.936 | 1.474 | 1.075 | 1.155 | 0.408 | 0.257 | 0.266 |
| metal / gemm_513x257x129 | 0.049 | 31.808 | 1.471 | 0.666 | 1.120 | 0.432 | 0.328 | 0.267 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

The model column is diagnostic only: it compares the lowest reported normalized kernel cost with the measured winner inside the same finite candidate set. Model regret is measured(model pick) / measured(best) - 1; it is not a hardware-optimality claim.

| Device / case | Valid / attempted candidates | Model pick / measured pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_32x32x32 | 11 / 45 | 32×32×32 @ 256t / 32×32×32 @ 128t | 5.18% | 7298.411 |
| metal / gemm_128x128x128 | 17 / 45 | 32×32×128 @ 256t / 32×32×128 @ 256t | 0.00% | 7910.981 |
| metal / gemm_512x512x512 | 28 / 45 | 64×64×512 @ 256t / 32×64×32 @ 128t | 13.81% | 9502.292 |
| metal / gemm_1024x1024x1024 | 34 / 45 | 128×32×1024 @ 128t / 128×32×1024 @ 128t | 0.00% | 11868.097 |
| metal / gemm_256x1024x128 | 17 / 45 | 64×64×128 @ 256t / 64×64×128 @ 256t | 0.00% | 8027.199 |
| metal / gemm_1024x128x256 | 22 / 45 | 32×32×256 @ 128t / 32×32×32 @ 128t | 17.17% | 8935.062 |
| metal / gemm_127x193x61 | 8 / 45 | 32×32×32 @ 256t / 32×32×32 @ 256t | 0.00% | 6410.399 |
| metal / gemm_513x257x129 | 8 / 45 | 32×32×32 @ 128t / 32×32×32 @ 256t | 34.37% | 6432.690 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
