# TileIR/TVMx vs PyTorch

Generated: 2026-09-02T23:09:51.620100+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR, but the current TVMx schedule lowers it to loops: no tensor-core claim. Pipeline window: `1`; 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 8×8×16 | 7.168 | 29.170 | 7.531 | 30.889 | 0.25× | 268.458 | 266.959 |
| metal | gemm_128x128x128 | 8×8×16 | 18.914 | 32.240 | 20.099 | 33.665 | 0.59× | 245.583 | 361.875 |
| metal | gemm_512x512x512 | 8×8×16 | 593.536 | 60.299 | 598.358 | 61.102 | 9.84× | 720.042 | 364.833 |
| metal | gemm_1024x1024x1024 | 8×8×16 | 4333.927 | 353.448 | 4416.462 | 385.358 | 12.26× | 4418.375 | 608.708 |
| metal | gemm_256x1024x128 | 8×8×16 | 145.301 | 34.541 | 153.539 | 35.765 | 4.21× | 471.125 | 277.167 |
| metal | gemm_1024x128x256 | 8×8×16 | 145.296 | 33.203 | 159.173 | 34.494 | 4.38× | 364.625 | 276.166 |
| metal | gemm_127x193x61 | 8×8×16 | 12.892 | 31.769 | 14.377 | 36.920 | 0.41× | 262.833 | 405.625 |
| metal | gemm_513x257x129 | 8×8×16 | 105.535 | 44.032 | 115.844 | 44.753 | 2.40× | 338.042 | 296.042 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.049 | 42.073 | 2.590 | 3.623 | 58.590 | 46.512 | 0.280 | 1.177 |
| metal / gemm_128x128x128 | 0.048 | 42.509 | 3.684 | 0.999 | 57.637 | 7.373 | 0.345 | 0.405 |
| metal / gemm_512x512x512 | 0.065 | 41.240 | 1.932 | 1.994 | 59.840 | 7.486 | 0.512 | 0.338 |
| metal / gemm_1024x1024x1024 | 0.090 | 42.667 | 4.507 | 2.112 | 66.311 | 7.392 | 1.931 | 0.473 |
| metal / gemm_256x1024x128 | 0.046 | 42.500 | 3.212 | 3.038 | 59.307 | 9.917 | 0.562 | 0.311 |
| metal / gemm_1024x128x256 | 0.056 | 42.142 | 2.498 | 1.320 | 59.135 | 6.808 | 0.355 | 0.318 |
| metal / gemm_127x193x61 | 0.050 | 46.928 | 1.650 | 1.359 | 59.454 | 4.620 | 0.363 | 0.341 |
| metal / gemm_513x257x129 | 0.047 | 48.049 | 1.530 | 0.982 | 58.932 | 4.020 | 0.343 | 1.689 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
