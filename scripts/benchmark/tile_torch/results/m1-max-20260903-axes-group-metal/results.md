# TileIR/TVMx vs PyTorch

Generated: 2026-09-02T20:08:22.702226+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR, but the current TVMx schedule lowers it to loops: no tensor-core claim. Pipeline stage markers currently run serially. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 8×8×16 | 6.779 | 30.415 | 7.607 | 32.324 | 0.22× | 263.583 | 290.708 |
| metal | gemm_128x128x128 | 8×8×16 | 19.678 | 31.215 | 20.070 | 32.819 | 0.63× | 238.083 | 404.125 |
| metal | gemm_512x512x512 | 8×8×16 | 573.736 | 57.839 | 579.671 | 64.155 | 9.92× | 706.834 | 318.083 |
| metal | gemm_1024x1024x1024 | 8×8×16 | 4274.933 | 357.187 | 4500.087 | 366.846 | 11.97× | 4460.292 | 586.167 |
| metal | gemm_256x1024x128 | 8×8×16 | 142.921 | 31.476 | 157.749 | 33.756 | 4.54× | 373.834 | 329.625 |
| metal | gemm_1024x128x256 | 8×8×16 | 146.748 | 32.061 | 162.104 | 33.297 | 4.58× | 429.042 | 290.209 |
| metal | gemm_127x193x61 | 8×8×16 | 13.702 | 30.096 | 13.844 | 33.984 | 0.46× | 235.167 | 288.084 |
| metal | gemm_513x257x129 | 8×8×16 | 107.106 | 42.222 | 117.533 | 48.057 | 2.54× | 325.084 | 351.291 |
| metal | add_1x127 | 1×256×1 | 3.817 | 3.940 | 4.031 | 4.925 | 0.97× | 229.333 | 217.958 |
| metal | add_17x257 | 1×256×1 | 3.517 | 4.661 | 3.942 | 5.163 | 0.75× | 262.666 | 282.417 |
| metal | add_128x1024 | 1×256×1 | 6.560 | 8.122 | 7.307 | 8.166 | 0.81× | 205.250 | 240.459 |
| metal | add_4096x256 | 1×256×1 | 24.938 | 29.529 | 26.353 | 30.617 | 0.84× | 261.041 | 312.000 |
| metal | sum_1x127 | 1×127×1 | 13.787 | 7.300 | 14.373 | 7.384 | 1.89× | 251.625 | 300.334 |
| metal | sum_17x257 | 1×257×1 | 25.533 | 5.442 | 26.170 | 6.034 | 4.69× | 290.042 | 198.208 |
| metal | sum_128x1024 | 1×1024×1 | 13.150 | 5.991 | 13.241 | 6.316 | 2.19× | 266.917 | 257.375 |
| metal | sum_64x4096 | 1×4096×1 | 39.456 | 20.323 | 39.677 | 23.527 | 1.94× | 283.375 | 333.958 |
| metal | softmax_1x127 | 1×127×1 | 25.739 | 30.698 | 26.764 | 35.253 | 0.84× | 280.208 | 293.125 |
| metal | softmax_17x257 | 1×257×1 | 49.700 | 32.788 | 50.108 | 34.274 | 1.52× | 331.625 | 376.750 |
| metal | softmax_128x1024 | 1×1024×1 | 27.496 | 39.728 | 28.144 | 41.558 | 0.69× | 297.333 | 343.542 |
| metal | softmax_64x4096 | 1×4096×1 | 86.553 | 36.290 | 90.232 | 38.257 | 2.39× | 307.250 | 362.375 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.051 | 42.564 | 1.534 | 3.267 | 57.939 | 45.746 | 0.253 | 0.509 |
| metal / gemm_128x128x128 | 0.076 | 49.912 | 2.689 | 0.684 | 58.232 | 6.692 | 0.459 | 0.280 |
| metal / gemm_512x512x512 | 0.049 | 43.337 | 2.272 | 1.011 | 59.892 | 3.519 | 0.910 | 0.383 |
| metal / gemm_1024x1024x1024 | 0.064 | 42.608 | 2.830 | 3.290 | 67.922 | 3.443 | 0.861 | 0.437 |
| metal / gemm_256x1024x128 | 0.051 | 41.402 | 1.412 | 1.129 | 57.486 | 4.765 | 0.545 | 0.361 |
| metal / gemm_1024x128x256 | 0.055 | 43.467 | 1.661 | 2.409 | 58.628 | 3.937 | 0.896 | 0.431 |
| metal / gemm_127x193x61 | 0.086 | 47.305 | 1.524 | 1.028 | 60.656 | 6.748 | 0.341 | 0.348 |
| metal / gemm_513x257x129 | 0.052 | 48.482 | 2.453 | 0.862 | 61.726 | 4.804 | 0.404 | 0.337 |
| metal / add_1x127 | 0.045 | 39.973 | 1.445 | 1.180 | 51.239 | 64.531 | 0.419 | 0.341 |
| metal / add_17x257 | 0.044 | 42.330 | 3.111 | 0.922 | 52.697 | 0.283 | 0.299 | 0.273 |
| metal / add_128x1024 | 0.051 | 39.514 | 3.556 | 1.340 | 51.639 | 7.787 | 0.399 | 0.267 |
| metal / add_4096x256 | 0.047 | 37.719 | 3.704 | 2.937 | 1.598 | 8.043 | 0.840 | 0.569 |
| metal / sum_1x127 | 0.057 | 33.453 | 1.097 | 0.764 | 53.924 | 0.564 | 0.278 | 0.305 |
| metal / sum_17x257 | 0.053 | 36.512 | 1.044 | 0.404 | 55.412 | 0.377 | 0.273 | 0.309 |
| metal / sum_128x1024 | 0.050 | 35.751 | 1.714 | 1.010 | 56.305 | 0.353 | 0.361 | 0.285 |
| metal / sum_64x4096 | 0.048 | 36.268 | 1.923 | 0.800 | 57.184 | 0.531 | 0.335 | 0.483 |
| metal / softmax_1x127 | 0.079 | 36.713 | 1.332 | 0.929 | 52.803 | 11.213 | 0.353 | 0.310 |
| metal / softmax_17x257 | 0.061 | 38.491 | 1.221 | 0.310 | 56.817 | 3.078 | 0.414 | 0.319 |
| metal / softmax_128x1024 | 0.069 | 37.964 | 0.994 | 0.465 | 59.265 | 5.599 | 0.448 | 0.344 |
| metal / softmax_64x4096 | 0.063 | 36.568 | 2.705 | 0.664 | 62.631 | 2.266 | 0.623 | 0.320 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
