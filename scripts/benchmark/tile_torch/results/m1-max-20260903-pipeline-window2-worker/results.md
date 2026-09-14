# TileIR/TVMx vs PyTorch

Generated: 2026-09-02T23:05:46.313465+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR, but the current TVMx schedule lowers it to loops: no tensor-core claim. Pipeline window: `2`; 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 8×8×16 | 6.110 | 0.875 | 6.946 | 0.880 | 6.98× | 6.000 | 0.959 |
| cpu | gemm_128x128x128 | 8×8×16 | 57.974 | 4.892 | 102.236 | 4.920 | 11.85× | 35.625 | 4.917 |
| cpu | gemm_512x512x512 | 8×8×16 | 1631.628 | 148.157 | 1942.278 | 153.692 | 11.01× | 1328.375 | 135.083 |
| cpu | gemm_1024x1024x1024 | 8×8×16 | 12576.604 | 1010.839 | 14052.358 | 1052.851 | 12.44× | 11187.833 | 1013.667 |
| cpu | gemm_256x1024x128 | 8×8×16 | 573.947 | 69.572 | 617.190 | 71.097 | 8.25× | 560.334 | 68.750 |
| cpu | gemm_1024x128x256 | 8×8×16 | 484.454 | 65.185 | 502.922 | 65.389 | 7.43× | 519.542 | 62.959 |
| cpu | gemm_127x193x61 | 8×8×16 | 57.878 | 6.536 | 66.316 | 6.580 | 8.85× | 36.416 | 6.375 |
| cpu | gemm_513x257x129 | 8×8×16 | 502.464 | 44.898 | 596.920 | 45.257 | 11.19× | 543.083 | 43.333 |
| metal | gemm_32x32x32 | 8×8×16 | 96.720 | 29.425 | 101.623 | 32.033 | 3.29× | 390.708 | 282.542 |
| metal | gemm_128x128x128 | 8×8×16 | 458.503 | 33.619 | 462.860 | 37.815 | 13.64× | 649.333 | 304.541 |
| metal | gemm_512x512x512 | 8×8×16 | 2613.012 | 57.713 | 2684.320 | 62.385 | 45.28× | 2856.292 | 320.666 |
| metal | gemm_1024x1024x1024 | 8×8×16 | 14571.875 | 355.905 | 14733.413 | 382.278 | 40.94× | 14990.542 | 646.792 |
| metal | gemm_256x1024x128 | 8×8×16 | 732.673 | 34.895 | 764.815 | 35.516 | 21.00× | 1019.000 | 302.666 |
| metal | gemm_1024x128x256 | 8×8×16 | 1339.015 | 33.229 | 1403.248 | 35.325 | 40.30× | 1498.125 | 280.917 |
| metal | gemm_127x193x61 | 8×8×16 | 346.016 | 30.936 | 355.833 | 34.396 | 11.19× | 583.166 | 303.792 |
| metal | gemm_513x257x129 | 8×8×16 | 948.492 | 42.403 | 988.933 | 43.642 | 22.37× | 1140.375 | 354.833 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.043 | 291.441 | 0.006 | 0.074 | 0.108 | 0.037 | 0.002 | 0.015 |
| cpu / gemm_128x128x128 | 0.049 | 232.292 | 0.006 | 0.013 | 0.201 | 0.010 | 0.011 | 0.006 |
| cpu / gemm_512x512x512 | 0.047 | 235.184 | 0.180 | 0.029 | 2.221 | 0.304 | 0.033 | 0.006 |
| cpu / gemm_1024x1024x1024 | 0.043 | 238.430 | 0.928 | 0.037 | 11.964 | 1.438 | 0.439 | 0.009 |
| cpu / gemm_256x1024x128 | 0.043 | 236.411 | 0.022 | 0.038 | 0.922 | 0.224 | 0.086 | 0.006 |
| cpu / gemm_1024x128x256 | 0.051 | 233.830 | 0.063 | 0.020 | 1.092 | 0.078 | 0.070 | 0.005 |
| cpu / gemm_127x193x61 | 0.047 | 209.771 | 0.009 | 0.022 | 0.197 | 0.035 | 0.005 | 0.006 |
| cpu / gemm_513x257x129 | 0.048 | 210.638 | 0.023 | 0.013 | 0.877 | 0.058 | 0.016 | 0.006 |
| metal / gemm_32x32x32 | 0.055 | 56.433 | 1.750 | 5.930 | 105.598 | 60.947 | 0.336 | 1.464 |
| metal / gemm_128x128x128 | 0.059 | 57.641 | 1.468 | 1.098 | 116.567 | 6.812 | 0.309 | 0.342 |
| metal / gemm_512x512x512 | 0.052 | 58.409 | 1.885 | 1.758 | 125.576 | 4.870 | 0.695 | 0.356 |
| metal / gemm_1024x1024x1024 | 0.052 | 59.182 | 2.934 | 1.997 | 146.236 | 3.490 | 1.247 | 0.505 |
| metal / gemm_256x1024x128 | 0.060 | 57.973 | 1.571 | 1.304 | 119.638 | 5.961 | 1.205 | 0.334 |
| metal / gemm_1024x128x256 | 0.053 | 58.947 | 2.920 | 0.727 | 121.110 | 3.738 | 0.473 | 1.824 |
| metal / gemm_127x193x61 | 0.059 | 67.156 | 1.319 | 2.212 | 125.839 | 6.562 | 0.386 | 0.319 |
| metal / gemm_513x257x129 | 0.081 | 66.534 | 1.570 | 0.975 | 127.583 | 3.829 | 0.827 | 0.334 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
