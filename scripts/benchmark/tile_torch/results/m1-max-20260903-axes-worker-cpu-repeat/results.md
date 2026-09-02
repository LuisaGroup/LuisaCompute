# TileIR/TVMx vs PyTorch

Generated: 2026-09-02T20:07:26.554072+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR, but the current TVMx schedule lowers it to loops: no tensor-core claim. Pipeline stage markers currently run serially. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 8×8×16 | 6.013 | 0.893 | 7.766 | 0.910 | 6.73× | 8.875 | 1.083 |
| cpu | gemm_128x128x128 | 8×8×16 | 116.958 | 4.983 | 120.566 | 5.027 | 23.47× | 99.417 | 5.000 |
| cpu | gemm_512x512x512 | 8×8×16 | 4451.625 | 140.899 | 4711.925 | 142.870 | 31.59× | 5060.667 | 138.792 |
| cpu | gemm_1024x1024x1024 | 8×8×16 | 38508.292 | 1042.217 | 39875.042 | 1123.004 | 36.95× | 36309.667 | 975.083 |
| cpu | gemm_256x1024x128 | 8×8×16 | 1445.833 | 68.263 | 1525.840 | 69.181 | 21.18× | 1356.542 | 65.833 |
| cpu | gemm_1024x128x256 | 8×8×16 | 1454.734 | 65.140 | 1513.402 | 65.732 | 22.33× | 1203.250 | 63.000 |
| cpu | gemm_127x193x61 | 8×8×16 | 123.412 | 6.676 | 124.573 | 6.758 | 18.49× | 92.916 | 6.583 |
| cpu | gemm_513x257x129 | 8×8×16 | 834.086 | 45.048 | 948.107 | 45.472 | 18.52× | 981.250 | 43.500 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.050 | 142.911 | 0.003 | 0.066 | 0.134 | 0.040 | 0.002 | 0.034 |
| cpu / gemm_128x128x128 | 0.053 | 143.844 | 0.007 | 0.016 | 0.329 | 0.008 | 0.006 | 0.006 |
| cpu / gemm_512x512x512 | 0.054 | 143.792 | 0.272 | 0.042 | 5.356 | 0.395 | 0.102 | 0.010 |
| cpu / gemm_1024x1024x1024 | 0.043 | 141.681 | 0.869 | 0.040 | 34.979 | 1.348 | 0.465 | 0.010 |
| cpu / gemm_256x1024x128 | 0.042 | 142.235 | 0.020 | 0.030 | 1.648 | 0.141 | 0.037 | 0.006 |
| cpu / gemm_1024x128x256 | 0.059 | 144.634 | 0.037 | 0.027 | 1.289 | 0.112 | 0.023 | 0.006 |
| cpu / gemm_127x193x61 | 0.058 | 141.333 | 0.005 | 0.028 | 0.377 | 0.048 | 0.009 | 0.006 |
| cpu / gemm_513x257x129 | 0.043 | 136.522 | 0.026 | 0.022 | 1.174 | 0.122 | 0.017 | 0.006 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
