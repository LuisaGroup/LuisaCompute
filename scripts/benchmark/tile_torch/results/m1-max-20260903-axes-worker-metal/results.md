# TileIR/TVMx vs PyTorch

Generated: 2026-09-02T20:09:19.613144+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR, but the current TVMx schedule lowers it to loops: no tensor-core claim. Pipeline stage markers currently run serially. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 8×8×16 | 89.506 | 28.891 | 90.818 | 30.334 | 3.10× | 325.959 | 369.875 |
| metal | gemm_128x128x128 | 8×8×16 | 425.349 | 42.718 | 437.408 | 44.015 | 9.96× | 732.709 | 301.417 |
| metal | gemm_512x512x512 | 8×8×16 | 2786.298 | 57.601 | 2917.678 | 61.390 | 48.37× | 3007.125 | 326.625 |
| metal | gemm_1024x1024x1024 | 8×8×16 | 14725.208 | 359.588 | 15074.488 | 376.505 | 40.95× | 15197.125 | 572.375 |
| metal | gemm_256x1024x128 | 8×8×16 | 766.850 | 34.612 | 776.797 | 35.411 | 22.16× | 1006.750 | 267.291 |
| metal | gemm_1024x128x256 | 8×8×16 | 1412.935 | 32.543 | 1489.519 | 35.526 | 43.42× | 1587.291 | 295.750 |
| metal | gemm_127x193x61 | 8×8×16 | 324.215 | 30.300 | 331.694 | 33.911 | 10.70× | 550.875 | 288.541 |
| metal | gemm_513x257x129 | 8×8×16 | 973.129 | 42.333 | 1009.746 | 43.618 | 22.99× | 1208.166 | 367.583 |
| metal | add_1x127 | 1×256×1 | 102.402 | 4.231 | 104.218 | 4.334 | 24.20× | 418.875 | 264.333 |
| metal | add_17x257 | 1×256×1 | 223.624 | 4.761 | 227.148 | 5.077 | 46.97× | 470.958 | 212.875 |
| metal | add_128x1024 | 1×256×1 | 51.603 | 8.109 | 53.977 | 8.828 | 6.36× | 297.917 | 240.959 |
| metal | add_4096x256 | 1×256×1 | 94.407 | 28.782 | 97.371 | 30.450 | 3.28× | 344.583 | 299.666 |
| metal | sum_1x127 | 1×127×1 | 56.858 | 8.242 | 60.659 | 8.850 | 6.90× | 316.542 | 227.625 |
| metal | sum_17x257 | 1×257×1 | 169.255 | 5.233 | 177.376 | 5.762 | 32.34× | 416.042 | 285.250 |
| metal | sum_128x1024 | 1×1024×1 | 63.287 | 6.093 | 63.610 | 7.041 | 10.39× | 315.417 | 239.834 |
| metal | sum_64x4096 | 1×4096×1 | 228.875 | 20.083 | 235.709 | 20.275 | 11.40× | 442.292 | 317.166 |
| metal | softmax_1x127 | 1×127×1 | 88.971 | 28.990 | 95.979 | 33.559 | 3.07× | 352.958 | 356.875 |
| metal | softmax_17x257 | 1×257×1 | 249.953 | 35.905 | 255.719 | 63.219 | 6.96× | 482.250 | 433.709 |
| metal | softmax_128x1024 | 1×1024×1 | 245.325 | 39.663 | 254.623 | 41.372 | 6.19× | 528.000 | 350.584 |
| metal | softmax_64x4096 | 1×4096×1 | 787.807 | 36.056 | 805.633 | 37.493 | 21.85× | 1058.458 | 412.791 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.049 | 48.246 | 1.386 | 3.008 | 2.947 | 49.590 | 0.311 | 0.319 |
| metal / gemm_128x128x128 | 0.058 | 48.476 | 2.203 | 0.604 | 6.931 | 4.509 | 0.317 | 0.308 |
| metal / gemm_512x512x512 | 0.054 | 48.181 | 1.811 | 2.386 | 9.691 | 7.853 | 0.561 | 0.575 |
| metal / gemm_1024x1024x1024 | 0.062 | 50.401 | 3.084 | 3.095 | 23.166 | 3.537 | 1.209 | 1.375 |
| metal / gemm_256x1024x128 | 0.056 | 49.125 | 2.516 | 0.980 | 6.710 | 5.714 | 0.554 | 0.349 |
| metal / gemm_1024x128x256 | 0.052 | 49.909 | 3.399 | 1.203 | 7.893 | 4.050 | 0.450 | 0.428 |
| metal / gemm_127x193x61 | 0.069 | 57.290 | 2.693 | 1.150 | 3.987 | 5.220 | 0.365 | 0.442 |
| metal / gemm_513x257x129 | 0.077 | 52.011 | 1.625 | 0.737 | 6.986 | 4.570 | 0.428 | 0.329 |
| metal / add_1x127 | 0.046 | 40.089 | 1.620 | 1.150 | 16.395 | 62.507 | 0.277 | 0.353 |
| metal / add_17x257 | 0.058 | 43.592 | 1.388 | 0.746 | 16.174 | 0.289 | 0.331 | 0.317 |
| metal / add_128x1024 | 0.041 | 41.754 | 1.750 | 2.555 | 11.993 | 9.145 | 0.383 | 0.397 |
| metal / add_4096x256 | 0.050 | 42.417 | 3.090 | 1.630 | 16.821 | 7.432 | 2.763 | 0.532 |
| metal / sum_1x127 | 0.050 | 34.407 | 2.227 | 1.383 | 4.672 | 0.589 | 0.288 | 0.321 |
| metal / sum_17x257 | 0.056 | 34.278 | 1.027 | 0.389 | 10.180 | 0.566 | 0.291 | 0.334 |
| metal / sum_128x1024 | 0.047 | 34.865 | 1.243 | 0.547 | 16.793 | 0.411 | 0.349 | 0.292 |
| metal / sum_64x4096 | 0.050 | 35.076 | 1.247 | 0.722 | 158.339 | 0.506 | 0.310 | 1.549 |
| metal / softmax_1x127 | 0.083 | 37.885 | 1.270 | 0.977 | 57.639 | 13.504 | 0.308 | 0.317 |
| metal / softmax_17x257 | 0.076 | 35.560 | 1.177 | 0.443 | 61.090 | 7.244 | 1.543 | 0.506 |
| metal / softmax_128x1024 | 0.060 | 37.482 | 1.295 | 0.814 | 75.742 | 4.675 | 0.400 | 0.321 |
| metal / softmax_64x4096 | 0.059 | 38.355 | 1.549 | 0.536 | 176.671 | 2.295 | 0.544 | 0.340 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
