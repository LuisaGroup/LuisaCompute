# TileIR/TVMx vs PyTorch

Generated: 2026-09-02T23:04:55.716859+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR, but the current TVMx schedule lowers it to loops: no tensor-core claim. Pipeline window: `1`; 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 8×8×16 | 6.702 | 0.903 | 7.123 | 0.923 | 7.42× | 3.459 | 1.000 |
| cpu | gemm_128x128x128 | 8×8×16 | 122.500 | 4.946 | 131.303 | 4.988 | 24.77× | 92.042 | 4.709 |
| cpu | gemm_512x512x512 | 8×8×16 | 5835.639 | 141.295 | 6067.081 | 151.355 | 41.30× | 5097.875 | 141.875 |
| cpu | gemm_1024x1024x1024 | 8×8×16 | 37870.208 | 1057.956 | 41168.866 | 1155.267 | 35.80× | 37178.042 | 1040.333 |
| cpu | gemm_256x1024x128 | 8×8×16 | 1299.836 | 67.750 | 1560.179 | 68.495 | 19.19× | 1311.083 | 65.791 |
| cpu | gemm_1024x128x256 | 8×8×16 | 1302.502 | 64.670 | 1471.353 | 65.087 | 20.14× | 1355.542 | 63.833 |
| cpu | gemm_127x193x61 | 8×8×16 | 103.397 | 6.638 | 154.245 | 6.683 | 15.58× | 104.542 | 6.250 |
| cpu | gemm_513x257x129 | 8×8×16 | 977.721 | 45.047 | 1014.529 | 45.832 | 21.70× | 866.958 | 43.334 |
| metal | gemm_32x32x32 | 8×8×16 | 89.360 | 29.710 | 90.844 | 32.720 | 3.01× | 307.666 | 270.041 |
| metal | gemm_128x128x128 | 8×8×16 | 418.207 | 32.545 | 451.791 | 37.550 | 12.85× | 632.250 | 298.583 |
| metal | gemm_512x512x512 | 8×8×16 | 2811.571 | 60.770 | 2910.287 | 61.834 | 46.27× | 3237.333 | 330.666 |
| metal | gemm_1024x1024x1024 | 8×8×16 | 15195.666 | 356.165 | 15452.433 | 381.637 | 42.66× | 14889.791 | 650.167 |
| metal | gemm_256x1024x128 | 8×8×16 | 762.543 | 32.201 | 775.990 | 34.623 | 23.68× | 967.167 | 286.709 |
| metal | gemm_1024x128x256 | 8×8×16 | 1416.985 | 32.363 | 1488.696 | 35.786 | 43.78× | 1629.667 | 282.958 |
| metal | gemm_127x193x61 | 8×8×16 | 320.880 | 29.908 | 334.987 | 33.634 | 10.73× | 603.167 | 307.333 |
| metal | gemm_513x257x129 | 8×8×16 | 978.167 | 43.527 | 1009.372 | 44.378 | 22.47× | 1201.375 | 344.417 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.061 | 163.004 | 0.004 | 0.433 | 0.150 | 0.367 | 0.001 | 0.016 |
| cpu / gemm_128x128x128 | 0.048 | 159.517 | 0.006 | 0.018 | 0.189 | 0.016 | 0.008 | 0.007 |
| cpu / gemm_512x512x512 | 0.056 | 160.177 | 0.125 | 0.031 | 5.005 | 0.382 | 0.096 | 0.007 |
| cpu / gemm_1024x1024x1024 | 0.050 | 162.103 | 0.874 | 0.058 | 35.559 | 1.858 | 0.475 | 0.007 |
| cpu / gemm_256x1024x128 | 0.061 | 160.795 | 0.020 | 0.028 | 1.463 | 0.133 | 0.087 | 0.006 |
| cpu / gemm_1024x128x256 | 0.047 | 160.208 | 0.035 | 0.021 | 1.224 | 0.092 | 0.017 | 0.005 |
| cpu / gemm_127x193x61 | 0.067 | 144.262 | 0.008 | 0.031 | 0.214 | 0.042 | 0.006 | 0.006 |
| cpu / gemm_513x257x129 | 0.044 | 143.167 | 0.012 | 0.021 | 1.255 | 0.094 | 0.019 | 0.006 |
| metal / gemm_32x32x32 | 0.060 | 50.003 | 2.977 | 4.308 | 88.372 | 48.670 | 1.394 | 0.358 |
| metal / gemm_128x128x128 | 0.053 | 47.988 | 3.296 | 0.696 | 91.270 | 4.251 | 0.306 | 0.291 |
| metal / gemm_512x512x512 | 0.124 | 76.297 | 1.618 | 1.267 | 100.210 | 3.771 | 0.506 | 0.332 |
| metal / gemm_1024x1024x1024 | 0.053 | 50.810 | 3.526 | 1.904 | 121.222 | 3.653 | 4.878 | 0.524 |
| metal / gemm_256x1024x128 | 0.047 | 50.193 | 2.654 | 1.107 | 95.365 | 4.301 | 0.513 | 0.433 |
| metal / gemm_1024x128x256 | 0.055 | 50.029 | 3.043 | 1.022 | 94.202 | 3.707 | 0.540 | 0.428 |
| metal / gemm_127x193x61 | 0.059 | 56.192 | 1.633 | 1.280 | 96.764 | 9.134 | 0.385 | 0.289 |
| metal / gemm_513x257x129 | 0.054 | 55.101 | 3.050 | 0.670 | 98.816 | 4.115 | 0.417 | 0.352 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
