# TileIR/TVMx vs PyTorch

Generated: 2026-09-02T23:10:21.905169+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR, but the current TVMx schedule lowers it to loops: no tensor-core claim. Pipeline window: `2`; 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 8×8×16 | 6.959 | 29.803 | 8.333 | 32.072 | 0.23× | 692.458 | 289.041 |
| metal | gemm_128x128x128 | 8×8×16 | 20.260 | 32.696 | 20.672 | 34.086 | 0.62× | 229.709 | 376.209 |
| metal | gemm_512x512x512 | 8×8×16 | 588.039 | 60.171 | 631.982 | 61.912 | 9.77× | 726.083 | 346.416 |
| metal | gemm_1024x1024x1024 | 8×8×16 | 4438.979 | 347.806 | 4750.290 | 373.515 | 12.76× | 4496.250 | 576.833 |
| metal | gemm_256x1024x128 | 8×8×16 | 145.678 | 34.233 | 160.204 | 36.424 | 4.26× | 408.209 | 300.000 |
| metal | gemm_1024x128x256 | 8×8×16 | 151.408 | 31.830 | 161.566 | 32.246 | 4.76× | 335.583 | 272.209 |
| metal | gemm_127x193x61 | 8×8×16 | 13.267 | 29.149 | 13.743 | 34.342 | 0.46× | 247.541 | 267.166 |
| metal | gemm_513x257x129 | 8×8×16 | 106.319 | 43.000 | 115.519 | 43.899 | 2.47× | 329.250 | 339.542 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.051 | 44.347 | 1.238 | 5.132 | 63.243 | 53.298 | 0.426 | 0.343 |
| metal / gemm_128x128x128 | 0.052 | 46.357 | 1.564 | 2.186 | 65.544 | 4.142 | 0.320 | 0.329 |
| metal / gemm_512x512x512 | 0.050 | 45.340 | 2.238 | 1.031 | 64.419 | 5.218 | 0.570 | 0.308 |
| metal / gemm_1024x1024x1024 | 0.055 | 46.384 | 2.696 | 2.113 | 72.847 | 4.546 | 0.865 | 0.496 |
| metal / gemm_256x1024x128 | 0.056 | 45.632 | 2.778 | 3.312 | 61.851 | 6.885 | 0.613 | 0.591 |
| metal / gemm_1024x128x256 | 0.053 | 46.099 | 1.569 | 2.801 | 64.666 | 3.667 | 0.466 | 0.424 |
| metal / gemm_127x193x61 | 0.053 | 52.551 | 1.546 | 1.060 | 65.044 | 6.413 | 0.337 | 1.720 |
| metal / gemm_513x257x129 | 0.062 | 51.783 | 1.583 | 0.679 | 66.826 | 5.617 | 0.397 | 0.312 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
