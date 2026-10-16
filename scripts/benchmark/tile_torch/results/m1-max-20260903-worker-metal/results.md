# TileIR/TVMx vs PyTorch

Generated: 2026-09-02T19:56:10.527821+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR, but the current TVMx schedule lowers it to loops: no tensor-core claim. Pipeline stage markers currently run serially. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 8×8×16 | 95.252 | 29.136 | 97.472 | 31.880 | 3.27× | 397.417 | 411.666 |
| metal | gemm_128x128x128 | 8×8×16 | 442.688 | 34.106 | 455.766 | 35.734 | 12.98× | 658.083 | 262.250 |
| metal | gemm_512x512x512 | 8×8×16 | 2828.055 | 58.813 | 2895.258 | 61.657 | 48.09× | 3024.583 | 312.792 |
| metal | gemm_1024x1024x1024 | 8×8×16 | 14778.771 | 355.565 | 14963.629 | 371.991 | 41.56× | 15132.500 | 589.709 |
| metal | gemm_256x1024x128 | 8×8×16 | 764.303 | 33.364 | 787.372 | 35.127 | 22.91× | 919.500 | 291.625 |
| metal | gemm_1024x128x256 | 8×8×16 | 1438.913 | 32.961 | 1457.913 | 33.717 | 43.66× | 1631.334 | 281.541 |
| metal | gemm_127x193x61 | 8×8×16 | 335.047 | 30.945 | 355.348 | 35.869 | 10.83× | 583.875 | 376.166 |
| metal | gemm_513x257x129 | 8×8×16 | 975.508 | 45.955 | 1001.809 | 47.701 | 21.23× | 1185.875 | 323.166 |
| metal | add_1x127 | 1×256×1 | 100.209 | 3.739 | 106.096 | 4.265 | 26.80× | 356.625 | 298.833 |
| metal | add_17x257 | 1×256×1 | 230.414 | 4.664 | 239.481 | 4.857 | 49.40× | 533.666 | 234.167 |
| metal | add_128x1024 | 1×256×1 | 52.185 | 7.163 | 54.509 | 7.748 | 7.28× | 291.792 | 304.584 |
| metal | add_4096x256 | 1×256×1 | 96.917 | 27.642 | 97.353 | 28.629 | 3.51× | 380.042 | 295.417 |
| metal | sum_1x127 | 1×127×1 | 57.637 | 7.402 | 57.970 | 8.301 | 7.79× | 278.000 | 313.459 |
| metal | sum_17x257 | 1×257×1 | 171.660 | 5.048 | 179.024 | 6.100 | 34.01× | 449.250 | 266.542 |
| metal | sum_128x1024 | 1×1024×1 | 63.033 | 6.189 | 63.666 | 6.682 | 10.18× | 308.375 | 252.791 |
| metal | sum_64x4096 | 1×4096×1 | 238.247 | 19.621 | 245.167 | 20.281 | 12.14× | 490.583 | 276.750 |
| metal | softmax_1x127 | 1×127×1 | 88.161 | 29.697 | 94.859 | 32.286 | 2.97× | 352.542 | 327.666 |
| metal | softmax_17x257 | 1×257×1 | 249.959 | 33.609 | 258.983 | 34.042 | 7.44× | 476.708 | 367.958 |
| metal | softmax_128x1024 | 1×1024×1 | 246.081 | 39.456 | 248.371 | 41.253 | 6.24× | 442.584 | 343.417 |
| metal | softmax_64x4096 | 1×4096×1 | 788.545 | 37.590 | 835.304 | 38.209 | 20.98× | 1012.167 | 332.959 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.055 | 45.493 | 3.063 | 3.151 | 87.167 | 45.056 | 0.283 | 0.434 |
| metal / gemm_128x128x128 | 0.056 | 45.748 | 1.347 | 1.002 | 90.897 | 5.362 | 1.752 | 0.258 |
| metal / gemm_512x512x512 | 0.052 | 44.736 | 2.016 | 2.695 | 97.664 | 3.738 | 0.492 | 0.337 |
| metal / gemm_1024x1024x1024 | 0.054 | 44.555 | 2.835 | 2.685 | 117.226 | 3.609 | 1.639 | 0.388 |
| metal / gemm_256x1024x128 | 0.051 | 45.769 | 1.602 | 2.806 | 92.974 | 4.385 | 0.503 | 0.306 |
| metal / gemm_1024x128x256 | 0.061 | 46.433 | 1.523 | 0.751 | 94.813 | 6.756 | 0.408 | 0.363 |
| metal / gemm_127x193x61 | 0.054 | 51.416 | 1.496 | 1.273 | 92.463 | 6.324 | 0.318 | 0.304 |
| metal / gemm_513x257x129 | 0.074 | 51.607 | 1.337 | 0.896 | 98.592 | 3.652 | 0.882 | 0.371 |
| metal / add_1x127 | 0.041 | 40.424 | 1.320 | 0.997 | 74.151 | 62.706 | 0.402 | 0.378 |
| metal / add_17x257 | 0.048 | 43.981 | 3.179 | 0.739 | 77.574 | 0.273 | 0.425 | 1.506 |
| metal / add_128x1024 | 0.041 | 42.529 | 3.093 | 0.991 | 70.833 | 7.023 | 0.341 | 0.311 |
| metal / add_4096x256 | 0.052 | 41.460 | 2.768 | 1.233 | 13.642 | 7.534 | 0.834 | 0.419 |
| metal / sum_1x127 | 0.047 | 34.733 | 2.428 | 0.697 | 54.676 | 0.551 | 0.311 | 0.314 |
| metal / sum_17x257 | 0.051 | 34.830 | 1.344 | 0.367 | 58.856 | 0.349 | 0.337 | 0.320 |
| metal / sum_128x1024 | 0.056 | 34.435 | 2.440 | 0.840 | 71.880 | 0.258 | 1.974 | 0.321 |
| metal / sum_64x4096 | 0.053 | 34.893 | 1.468 | 0.853 | 227.154 | 0.561 | 0.343 | 0.787 |
| metal / softmax_1x127 | 0.066 | 36.932 | 0.935 | 0.992 | 58.208 | 10.192 | 0.361 | 1.973 |
| metal / softmax_17x257 | 0.068 | 37.839 | 1.226 | 0.480 | 64.905 | 3.496 | 0.296 | 1.332 |
| metal / softmax_128x1024 | 0.066 | 37.257 | 1.075 | 0.832 | 76.624 | 11.062 | 0.374 | 1.688 |
| metal / softmax_64x4096 | 0.060 | 36.931 | 1.319 | 0.816 | 180.216 | 3.909 | 0.782 | 0.610 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
