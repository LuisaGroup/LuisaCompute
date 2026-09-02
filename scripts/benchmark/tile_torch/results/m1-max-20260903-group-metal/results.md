# TileIR/TVMx vs PyTorch

Generated: 2026-09-02T19:55:40.804022+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR, but the current TVMx schedule lowers it to loops: no tensor-core claim. Pipeline stage markers currently run serially. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 8×8×16 | 7.161 | 29.470 | 7.557 | 31.358 | 0.24× | 255.708 | 443.625 |
| metal | gemm_128x128x128 | 8×8×16 | 19.236 | 41.709 | 21.492 | 42.893 | 0.46× | 276.333 | 283.000 |
| metal | gemm_512x512x512 | 8×8×16 | 571.042 | 59.752 | 619.634 | 61.132 | 9.56× | 756.209 | 329.250 |
| metal | gemm_1024x1024x1024 | 8×8×16 | 4300.333 | 357.437 | 4364.533 | 368.250 | 12.03× | 4593.750 | 617.292 |
| metal | gemm_256x1024x128 | 8×8×16 | 142.846 | 32.781 | 155.976 | 35.636 | 4.36× | 371.917 | 282.000 |
| metal | gemm_1024x128x256 | 8×8×16 | 145.726 | 33.026 | 158.563 | 33.444 | 4.41× | 386.875 | 297.042 |
| metal | gemm_127x193x61 | 8×8×16 | 12.859 | 30.012 | 14.547 | 32.597 | 0.43× | 247.875 | 299.333 |
| metal | gemm_513x257x129 | 8×8×16 | 109.131 | 43.727 | 109.964 | 47.501 | 2.50× | 306.875 | 349.792 |
| metal | add_1x127 | 1×256×1 | 3.895 | 3.980 | 4.182 | 4.049 | 0.98× | 215.000 | 214.791 |
| metal | add_17x257 | 1×256×1 | 3.595 | 4.363 | 3.860 | 4.869 | 0.82× | 221.333 | 221.792 |
| metal | add_128x1024 | 1×256×1 | 6.725 | 7.609 | 6.825 | 7.885 | 0.88× | 283.667 | 248.041 |
| metal | add_4096x256 | 1×256×1 | 25.627 | 27.512 | 26.238 | 28.400 | 0.93× | 277.709 | 280.708 |
| metal | sum_1x127 | 1×127×1 | 13.913 | 7.652 | 14.576 | 7.926 | 1.82× | 274.792 | 233.750 |
| metal | sum_17x257 | 1×257×1 | 25.582 | 4.854 | 25.763 | 5.348 | 5.27× | 275.083 | 238.708 |
| metal | sum_128x1024 | 1×1024×1 | 13.574 | 5.999 | 13.654 | 6.591 | 2.26× | 323.041 | 301.250 |
| metal | sum_64x4096 | 1×4096×1 | 39.863 | 17.827 | 40.468 | 18.869 | 2.24× | 288.458 | 299.417 |
| metal | softmax_1x127 | 1×127×1 | 26.029 | 29.797 | 26.138 | 32.115 | 0.87× | 263.333 | 332.250 |
| metal | softmax_17x257 | 1×257×1 | 47.563 | 36.124 | 49.173 | 36.762 | 1.32× | 248.417 | 366.958 |
| metal | softmax_128x1024 | 1×1024×1 | 27.840 | 38.696 | 28.148 | 39.285 | 0.72× | 287.250 | 326.292 |
| metal | softmax_64x4096 | 1×4096×1 | 86.775 | 35.813 | 88.822 | 36.330 | 2.42× | 398.041 | 424.042 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.064 | 45.068 | 1.252 | 4.614 | 62.482 | 55.216 | 0.287 | 0.308 |
| metal / gemm_128x128x128 | 0.055 | 43.464 | 1.437 | 0.691 | 60.880 | 4.310 | 0.386 | 0.285 |
| metal / gemm_512x512x512 | 0.053 | 42.172 | 1.872 | 3.130 | 58.935 | 7.716 | 0.500 | 0.612 |
| metal / gemm_1024x1024x1024 | 0.058 | 43.507 | 2.967 | 2.809 | 66.555 | 3.439 | 1.114 | 0.415 |
| metal / gemm_256x1024x128 | 0.050 | 43.683 | 1.635 | 1.340 | 61.624 | 7.725 | 1.034 | 0.326 |
| metal / gemm_1024x128x256 | 0.060 | 43.796 | 1.580 | 0.779 | 59.123 | 6.725 | 0.367 | 0.582 |
| metal / gemm_127x193x61 | 0.052 | 48.195 | 2.798 | 0.865 | 60.695 | 5.889 | 0.348 | 0.413 |
| metal / gemm_513x257x129 | 0.069 | 46.738 | 1.535 | 0.693 | 61.600 | 4.811 | 0.376 | 0.321 |
| metal / add_1x127 | 0.056 | 40.389 | 1.243 | 0.937 | 51.067 | 61.426 | 0.309 | 0.333 |
| metal / add_17x257 | 0.042 | 41.121 | 1.713 | 2.330 | 52.783 | 0.326 | 0.310 | 0.312 |
| metal / add_128x1024 | 0.053 | 37.725 | 1.612 | 1.278 | 52.478 | 7.446 | 0.399 | 0.303 |
| metal / add_4096x256 | 0.051 | 39.652 | 3.834 | 1.298 | 1.506 | 7.934 | 1.034 | 0.423 |
| metal / sum_1x127 | 0.050 | 35.067 | 1.689 | 2.427 | 53.732 | 0.535 | 0.254 | 0.378 |
| metal / sum_17x257 | 0.051 | 35.438 | 2.888 | 0.416 | 53.851 | 0.362 | 0.337 | 0.296 |
| metal / sum_128x1024 | 0.051 | 36.860 | 1.281 | 1.195 | 55.104 | 6.170 | 0.284 | 0.287 |
| metal / sum_64x4096 | 0.060 | 36.113 | 1.295 | 0.750 | 57.615 | 0.481 | 0.309 | 0.278 |
| metal / softmax_1x127 | 0.066 | 35.875 | 1.233 | 1.375 | 54.247 | 10.851 | 0.390 | 1.957 |
| metal / softmax_17x257 | 0.058 | 38.240 | 1.852 | 0.341 | 58.534 | 3.521 | 0.286 | 0.274 |
| metal / softmax_128x1024 | 0.094 | 36.307 | 1.097 | 0.679 | 58.777 | 4.806 | 0.440 | 0.392 |
| metal / softmax_64x4096 | 0.074 | 39.076 | 1.841 | 1.167 | 63.404 | 2.770 | 0.539 | 0.388 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
