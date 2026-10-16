# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T04:47:21.233088+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 64×64×32 / 1 | 8 | 5.472 | 27.029 | 5.693 | 27.173 | 0.20× | 228.584 | 288.125 |
| metal | gemm_128x128x128 | 64×64×32 / 1 | 8 | 13.644 | 26.926 | 13.794 | 28.637 | 0.51× | 239.458 | 290.666 |
| metal | gemm_512x512x512 | 64×64×32 / 1 | 8 | 56.703 | 48.016 | 56.929 | 49.059 | 1.18× | 292.375 | 292.458 |
| metal | gemm_1024x1024x1024 | 64×64×32 / 1 | 8 | 407.044 | 287.716 | 410.118 | 292.594 | 1.41× | 610.416 | 541.375 |
| metal | gemm_256x1024x128 | 64×64×32 / 1 | 8 | 19.469 | 28.978 | 19.642 | 29.386 | 0.67× | 251.375 | 258.042 |
| metal | gemm_1024x128x256 | 64×64×32 / 1 | 8 | 24.983 | 29.873 | 25.202 | 30.451 | 0.84× | 260.250 | 262.250 |
| metal | gemm_127x193x61 | 64×64×32 / 1 | 8 | 11.501 | 26.905 | 11.546 | 28.637 | 0.43× | 220.542 | 282.250 |
| metal | gemm_513x257x129 | 64×64×32 / 1 | 8 | 27.415 | 34.553 | 27.472 | 35.599 | 0.79× | 262.459 | 286.334 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.082 | 52.047 | 1.094 | 3.133 | 79.499 | 39.531 | 0.248 | 0.319 |
| metal / gemm_128x128x128 | 0.053 | 47.090 | 1.320 | 0.567 | 80.597 | 3.925 | 0.303 | 0.316 |
| metal / gemm_512x512x512 | 0.062 | 47.175 | 1.895 | 1.048 | 80.710 | 8.211 | 0.505 | 0.344 |
| metal / gemm_1024x1024x1024 | 0.060 | 48.093 | 2.608 | 1.908 | 87.417 | 3.230 | 0.915 | 0.438 |
| metal / gemm_256x1024x128 | 0.058 | 47.183 | 2.446 | 1.097 | 82.280 | 4.208 | 0.534 | 0.326 |
| metal / gemm_1024x128x256 | 0.059 | 48.505 | 1.763 | 0.780 | 82.328 | 3.601 | 0.371 | 0.316 |
| metal / gemm_127x193x61 | 0.052 | 57.019 | 1.558 | 1.194 | 86.505 | 4.506 | 0.325 | 0.332 |
| metal / gemm_513x257x129 | 0.055 | 62.373 | 1.205 | 0.987 | 90.594 | 3.540 | 0.445 | 0.323 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
