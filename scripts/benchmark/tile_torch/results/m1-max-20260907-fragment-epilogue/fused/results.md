# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T17:26:22.812551+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_128x128x128 | 64×64×4096 / 1 | 1 | 11.425 | 30.676 | 11.933 | 31.947 | 0.37× | 222.500 | 296.292 |
| metal | gemm_127x193x61 | FAILED | | | | | | | | |
| metal | gemm_1024x1024x1024 | 64×64×4096 / 1 | 1 | 339.407 | 364.561 | 340.809 | 368.670 | 0.93× | 724.416 | 689.334 |
| metal | gemm_4096x4096x4096 | 64×64×4096 / 1 | 1 | 21948.875 | 19994.125 | 23612.767 | 21876.917 | 1.10× | 20974.667 | 20005.083 |
| metal | gemm_128x2048x512 | 64×64×4096 / 1 | 1 | 59.519 | 60.487 | 61.265 | 61.966 | 0.98× | 330.167 | 414.208 |
| metal | gemm_2048x128x512 | 64×64×4096 / 1 | 1 | 59.821 | 62.515 | 60.676 | 63.694 | 0.96× | 625.000 | 349.125 |
| metal | gemm_relu_128x128x128 | 64×64×4096 / 1 | 1 | 10.782 | 40.517 | 10.830 | 45.776 | 0.27× | 231.833 | 488.667 |
| metal | gemm_relu_127x193x61 | FAILED | | | | | | | | |
| metal | gemm_relu_1024x1024x1024 | 64×64×4096 / 1 | 1 | 396.346 | 447.582 | 429.537 | 464.591 | 0.89× | 591.041 | 670.917 |
| metal | gemm_relu_4096x4096x4096 | 64×64×4096 / 1 | 1 | 24394.250 | 21486.291 | 24844.600 | 22953.917 | 1.14× | 24317.667 | 21616.125 |
| metal | gemm_relu_128x2048x512 | 64×64×4096 / 1 | 1 | 57.770 | 87.118 | 64.007 | 89.031 | 0.66× | 285.584 | 536.000 |
| metal | gemm_relu_2048x128x512 | 64×64×4096 / 1 | 1 | 59.992 | 89.688 | 63.468 | 93.077 | 0.67× | 308.375 | 436.667 |
| metal | gemm_gelu_128x128x128 | 64×64×4096 / 1 | 1 | 11.992 | 43.190 | 12.262 | 44.055 | 0.28× | 232.542 | 310.042 |
| metal | gemm_gelu_127x193x61 | FAILED | | | | | | | | |
| metal | gemm_gelu_1024x1024x1024 | 64×64×4096 / 1 | 1 | 399.236 | 450.076 | 400.945 | 454.594 | 0.89× | 621.042 | 696.875 |
| metal | gemm_gelu_4096x4096x4096 | 64×64×4096 / 1 | 1 | 24235.708 | 21392.958 | 25036.350 | 22209.550 | 1.13× | 24339.792 | 21252.750 |
| metal | gemm_gelu_128x2048x512 | 64×64×4096 / 1 | 1 | 69.775 | 100.531 | 70.220 | 103.314 | 0.69× | 366.500 | 391.708 |
| metal | gemm_gelu_2048x128x512 | 64×64×4096 / 1 | 1 | 70.322 | 97.143 | 72.061 | 99.162 | 0.72× | 345.167 | 573.834 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_128x128x128 | 0.117 | 36.710 | 1.272 | 4.737 | 164.332 | 49.849 | 0.315 | 0.672 |
| metal / gemm_1024x1024x1024 | 0.060 | 35.782 | 6.693 | 2.074 | 3.968 | 4.391 | 0.733 | 0.473 |
| metal / gemm_4096x4096x4096 | 0.060 | 32.054 | 33.991 | 46.690 | 35.105 | 33.585 | 10.639 | 5.017 |
| metal / gemm_128x2048x512 | 0.069 | 35.735 | 2.721 | 2.114 | 167.630 | 5.271 | 0.420 | 0.345 |
| metal / gemm_2048x128x512 | 0.054 | 34.785 | 2.402 | 1.019 | 171.455 | 4.023 | 0.616 | 0.405 |
| metal / gemm_relu_128x128x128 | 0.071 | 36.189 | 1.417 | 0.961 | 216.611 | 66.514 | 0.259 | 0.296 |
| metal / gemm_relu_1024x1024x1024 | 0.067 | 37.255 | 5.118 | 1.416 | 256.780 | 7.468 | 1.087 | 0.384 |
| metal / gemm_relu_4096x4096x4096 | 0.059 | 33.483 | 30.536 | 17.883 | 303.113 | 24.094 | 9.793 | 2.533 |
| metal / gemm_relu_128x2048x512 | 0.065 | 36.903 | 2.906 | 1.595 | 242.218 | 4.321 | 0.538 | 0.271 |
| metal / gemm_relu_2048x128x512 | 0.061 | 35.244 | 4.381 | 0.775 | 237.752 | 0.511 | 0.505 | 0.329 |
| metal / gemm_gelu_128x128x128 | 0.067 | 36.659 | 1.467 | 1.713 | 219.246 | 9.701 | 0.755 | 0.347 |
| metal / gemm_gelu_1024x1024x1024 | 0.070 | 37.330 | 3.981 | 1.537 | 263.718 | 12.096 | 0.969 | 0.461 |
| metal / gemm_gelu_4096x4096x4096 | 0.071 | 33.796 | 34.222 | 19.074 | 288.645 | 33.522 | 9.241 | 1.953 |
| metal / gemm_gelu_128x2048x512 | 0.074 | 37.576 | 4.550 | 3.335 | 243.771 | 8.326 | 0.506 | 0.308 |
| metal / gemm_gelu_2048x128x512 | 0.067 | 36.506 | 2.377 | 0.940 | 242.731 | 0.556 | 0.480 | 0.317 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_128x128x128 / native | 8.771 | 10.167 | 11.425 | 222.500 | 0.920× |
| gemm_128x128x128 / torch | 14.910 | 37.792 | 30.676 | 296.292 | 2.082× |
| gemm_1024x1024x1024 / native | 325.901 | 538.875 | 339.407 | 724.416 | 1.002× |
| gemm_1024x1024x1024 / torch | 337.701 | 285.250 | 364.561 | 689.334 | 1.048× |
| gemm_4096x4096x4096 / native | 20465.458 | 20305.333 | 21948.875 | 20974.667 | 1.010× |
| gemm_4096x4096x4096 / torch | 18664.500 | 18463.917 | 19994.125 | 20005.083 | 0.962× |
| gemm_128x2048x512 / native | 52.228 | 114.375 | 59.519 | 330.167 | 0.982× |
| gemm_128x2048x512 / torch | 43.988 | 46.542 | 60.487 | 414.208 | 1.427× |
| gemm_2048x128x512 / native | 52.702 | 48.458 | 59.821 | 625.000 | 0.982× |
| gemm_2048x128x512 / torch | 43.786 | 46.542 | 62.515 | 349.125 | 1.417× |
| gemm_relu_128x128x128 / native | 8.947 | 10.250 | 10.782 | 231.833 | 1.155× |
| gemm_relu_128x128x128 / torch | 19.846 | 179.250 | 40.517 | 488.667 | 2.397× |
| gemm_relu_1024x1024x1024 / native | 372.776 | 331.958 | 396.346 | 591.041 | 1.000× |
| gemm_relu_1024x1024x1024 / torch | 398.060 | 330.167 | 447.582 | 670.917 | 1.099× |
| gemm_relu_4096x4096x4096 / native | 22725.958 | 22980.500 | 24394.250 | 24317.667 | 0.995× |
| gemm_relu_4096x4096x4096 / torch | 20914.250 | 20964.875 | 21486.291 | 21616.125 | 0.986× |
| gemm_relu_128x2048x512 / native | 52.576 | 49.792 | 57.770 | 285.584 | 1.009× |
| gemm_relu_128x2048x512 / torch | 67.224 | 102.250 | 87.118 | 536.000 | 1.326× |
| gemm_relu_2048x128x512 / native | 52.876 | 48.917 | 59.992 | 308.375 | 1.004× |
| gemm_relu_2048x128x512 / torch | 62.962 | 74.750 | 89.688 | 436.667 | 1.645× |
| gemm_gelu_128x128x128 / native | 9.189 | 10.875 | 11.992 | 232.542 | 1.140× |
| gemm_gelu_128x128x128 / torch | 25.532 | 31.417 | 43.190 | 310.042 | 1.938× |
| gemm_gelu_1024x1024x1024 / native | 359.573 | 411.792 | 399.236 | 621.042 | 1.000× |
| gemm_gelu_1024x1024x1024 / torch | 421.833 | 349.917 | 450.076 | 696.875 | 1.094× |
| gemm_gelu_4096x4096x4096 / native | 22931.167 | 22945.250 | 24235.708 | 24339.792 | 1.019× |
| gemm_gelu_4096x4096x4096 / torch | 20958.708 | 20802.125 | 21392.958 | 21252.750 | 1.010× |
| gemm_gelu_128x2048x512 / native | 60.650 | 78.625 | 69.775 | 366.500 | 0.965× |
| gemm_gelu_128x2048x512 / torch | 67.771 | 82.250 | 100.531 | 391.708 | 1.559× |
| gemm_gelu_2048x128x512 / native | 59.385 | 74.958 | 70.322 | 345.167 | 0.994× |
| gemm_gelu_2048x128x512 / torch | 67.438 | 83.625 | 97.143 | 573.834 | 1.769× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_128x128x128 | 8.420 | 23.649 | 10.959 | 19.417 | 222.500 | 296.292 |
| gemm_1024x1024x1024 | 321.968 | 336.656 | 271.334 | 278.417 | 724.416 | 689.334 |
| gemm_4096x4096x4096 | 20572.292 | 17954.875 | 20702.542 | 18603.500 | 20974.667 | 20005.083 |
| gemm_128x2048x512 | 51.264 | 48.239 | 48.333 | 46.875 | 330.167 | 414.208 |
| gemm_2048x128x512 | 51.212 | 48.283 | 49.125 | 46.625 | 625.000 | 349.125 |
| gemm_relu_128x128x128 | 9.604 | 28.742 | 10.375 | 34.166 | 231.833 | 488.667 |
| gemm_relu_1024x1024x1024 | 379.101 | 403.380 | 437.208 | 324.501 | 591.041 | 670.917 |
| gemm_relu_4096x4096x4096 | 22590.083 | 20733.043 | 22849.000 | 20965.625 | 24317.667 | 21616.125 |
| gemm_relu_128x2048x512 | 53.071 | 80.544 | 48.125 | 70.582 | 285.584 | 536.000 |
| gemm_relu_2048x128x512 | 52.508 | 91.325 | 49.583 | 72.459 | 308.375 | 436.667 |
| gemm_gelu_128x128x128 | 9.960 | 33.101 | 10.833 | 26.584 | 232.542 | 310.042 |
| gemm_gelu_1024x1024x1024 | 362.523 | 423.143 | 333.000 | 341.333 | 621.042 | 696.875 |
| gemm_gelu_4096x4096x4096 | 23358.584 | 21060.750 | 22997.209 | 20991.291 | 24339.792 | 21252.750 |
| gemm_gelu_128x2048x512 | 58.380 | 84.194 | 80.667 | 75.792 | 366.500 | 391.708 |
| gemm_gelu_2048x128x512 | 58.963 | 99.642 | 79.083 | 75.791 | 345.167 | 573.834 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
