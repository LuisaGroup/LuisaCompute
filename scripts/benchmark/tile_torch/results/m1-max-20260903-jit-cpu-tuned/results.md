# TileIR/TVMx vs PyTorch

Generated: 2026-09-03T01:41:31.232254+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `worker`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`. When enabled, CPU independent-element domains are packed into SIMD without changing inner serial/reduction order. Disabling this does not disable LLVM's own optimizations.

Both sides use device-resident inputs and preallocated outputs. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices; other cases retain contraction loops. The matrix-calls column counts static call sites in generated Metal source, not dynamic instruction executions. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | gemm_32x32x32 | 8×32×32 / 1 | 0 | 3.783 | 0.902 | 3.935 | 0.933 | 4.19× | 2.417 | 0.917 |
| cpu | gemm_128x128x128 | 4×16×32 / 2 | 0 | 47.915 | 4.915 | 93.703 | 4.949 | 9.75× | 52.250 | 4.792 |
| cpu | gemm_512x512x512 | 4×16×32 / 2 | 0 | 1457.244 | 155.586 | 2178.349 | 173.910 | 9.37× | 1542.958 | 148.000 |
| cpu | gemm_1024x1024x1024 | 4×16×32 / 2 | 0 | 10939.083 | 1089.843 | 11479.217 | 1136.611 | 10.04× | 10352.250 | 1114.750 |
| cpu | gemm_256x1024x128 | 8×32×32 / 2 | 0 | 493.663 | 74.513 | 921.812 | 79.052 | 6.63× | 666.584 | 70.084 |
| cpu | gemm_1024x128x256 | 8×32×32 / 2 | 0 | 506.476 | 67.965 | 533.931 | 70.725 | 7.45× | 467.250 | 63.000 |
| cpu | gemm_127x193x61 | 8×32×32 / 1 | 0 | 66.710 | 7.086 | 77.391 | 7.267 | 9.41× | 68.084 | 6.750 |
| cpu | gemm_513x257x129 | 8×32×32 / 2 | 0 | 581.972 | 49.933 | 615.590 | 50.619 | 11.66× | 550.958 | 49.458 |
| cpu | add_1x127 | 1×256×1 / 2 | 0 | 0.271 | 0.581 | 0.277 | 0.584 | 0.47× | 0.250 | 0.667 |
| cpu | add_17x257 | 1×256×1 / 2 | 0 | 7.533 | 0.999 | 8.080 | 1.006 | 7.54× | 7.458 | 1.125 |
| cpu | add_128x1024 | 1×256×1 / 2 | 0 | 22.390 | 44.776 | 25.239 | 45.423 | 0.50× | 26.917 | 41.000 |
| cpu | add_4096x256 | 1×256×1 / 2 | 0 | 113.383 | 98.944 | 259.771 | 257.795 | 1.15× | 122.834 | 75.375 |
| cpu | sum_1x127 | 1×127×1 / 2 | 0 | 2.147 | 0.823 | 2.161 | 0.828 | 2.61× | 2.084 | 0.958 |
| cpu | sum_17x257 | 1×257×1 / 2 | 0 | 35.791 | 1.176 | 81.413 | 1.203 | 30.42× | 28.292 | 1.250 |
| cpu | sum_128x1024 | 1×1024×1 / 2 | 0 | 929.831 | 51.855 | 1051.154 | 56.578 | 17.93× | 1026.459 | 65.167 |
| cpu | sum_64x4096 | 1×4096×1 / 2 | 0 | 1635.324 | 53.077 | 1705.690 | 55.218 | 30.81× | 1724.417 | 52.833 |
| cpu | softmax_1x127 | 1×127×1 / 2 | 0 | 6.077 | 0.659 | 6.097 | 0.666 | 9.22× | 6.167 | 0.750 |
| cpu | softmax_17x257 | 1×257×1 / 2 | 0 | 56.425 | 40.021 | 74.694 | 42.461 | 1.41× | 43.125 | 37.833 |
| cpu | softmax_128x1024 | 1×1024×1 / 2 | 0 | 1534.510 | 94.364 | 1639.060 | 99.725 | 16.26× | 1447.666 | 92.958 |
| cpu | softmax_64x4096 | 1×4096×1 / 2 | 0 | 3029.528 | 148.849 | 4498.922 | 151.534 | 20.35× | 3027.583 | 161.208 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 0.045 | 127.920 | 0.003 | 0.020 | 0.102 | 0.020 | 0.002 | 0.006 |
| cpu / gemm_128x128x128 | 0.054 | 191.975 | 0.012 | 0.018 | 0.164 | 0.015 | 0.007 | 0.006 |
| cpu / gemm_512x512x512 | 0.046 | 195.193 | 0.252 | 0.037 | 1.116 | 0.282 | 0.038 | 0.007 |
| cpu / gemm_1024x1024x1024 | 0.047 | 196.279 | 0.770 | 0.043 | 10.951 | 1.505 | 0.478 | 0.006 |
| cpu / gemm_256x1024x128 | 0.059 | 216.652 | 0.033 | 0.027 | 0.860 | 0.144 | 0.059 | 0.006 |
| cpu / gemm_1024x128x256 | 0.055 | 219.204 | 0.180 | 0.032 | 0.515 | 0.107 | 0.017 | 0.008 |
| cpu / gemm_127x193x61 | 0.044 | 172.266 | 0.006 | 0.032 | 0.224 | 0.046 | 0.007 | 0.006 |
| cpu / gemm_513x257x129 | 0.044 | 286.974 | 0.016 | 0.021 | 0.467 | 0.083 | 0.021 | 0.008 |
| cpu / add_1x127 | 0.048 | 62.968 | 0.004 | 0.023 | 0.012 | 0.058 | 0.002 | 0.006 |
| cpu / add_17x257 | 0.044 | 81.410 | 0.006 | 0.013 | 0.129 | 0.004 | 0.008 | 0.006 |
| cpu / add_128x1024 | 0.040 | 85.447 | 0.040 | 0.032 | 0.610 | 0.054 | 0.065 | 0.007 |
| cpu / add_4096x256 | 0.038 | 84.115 | 0.919 | 0.033 | 0.715 | 0.158 | 0.457 | 0.007 |
| cpu / sum_1x127 | 0.053 | 34.292 | 0.003 | 0.028 | 0.018 | 0.063 | 0.001 | 0.006 |
| cpu / sum_17x257 | 0.045 | 50.493 | 0.013 | 0.015 | 0.132 | 0.006 | 0.002 | 0.005 |
| cpu / sum_128x1024 | 0.060 | 42.564 | 0.051 | 0.021 | 0.952 | 0.359 | 0.001 | 0.007 |
| cpu / sum_64x4096 | 0.043 | 40.197 | 0.088 | 0.028 | 1.296 | 0.073 | 0.001 | 0.006 |
| cpu / softmax_1x127 | 0.055 | 50.350 | 0.003 | 0.022 | 0.017 | 0.062 | 0.002 | 0.006 |
| cpu / softmax_17x257 | 0.062 | 55.150 | 0.005 | 0.017 | 0.506 | 0.080 | 0.007 | 0.007 |
| cpu / softmax_128x1024 | 0.053 | 45.702 | 0.049 | 0.026 | 1.076 | 0.106 | 0.050 | 0.006 |
| cpu / softmax_64x4096 | 0.058 | 46.382 | 0.103 | 0.020 | 2.829 | 0.145 | 0.079 | 0.007 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

| Device / case | Valid / attempted candidates | Selection wall ms |
|---|---:|---:|
| cpu / gemm_32x32x32 | 6 / 6 | 6301.453 |
| cpu / gemm_128x128x128 | 6 / 6 | 6459.318 |
| cpu / gemm_512x512x512 | 6 / 6 | 6257.703 |
| cpu / gemm_1024x1024x1024 | 6 / 6 | 7743.043 |
| cpu / gemm_256x1024x128 | 6 / 6 | 6255.005 |
| cpu / gemm_1024x128x256 | 6 / 6 | 6353.803 |
| cpu / gemm_127x193x61 | 6 / 6 | 6153.017 |
| cpu / gemm_513x257x129 | 6 / 6 | 6448.816 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
