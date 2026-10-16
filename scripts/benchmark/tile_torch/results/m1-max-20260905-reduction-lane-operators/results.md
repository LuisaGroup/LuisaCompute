# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T07:13:13.152956+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sum_1x127 | 1×127×1 / 2 | 0 | 2.636 | 11.588 | 3.079 | 15.320 | 0.23× | 197.875 | 231.750 |
| metal | sum_17x257 | 1×257×1 / 2 | 0 | 2.894 | 7.760 | 3.345 | 8.146 | 0.37× | 202.292 | 222.041 |
| metal | sum_64x4096 | 1×4096×1 / 2 | 0 | 4.282 | 15.347 | 4.332 | 15.487 | 0.28× | 232.458 | 343.792 |
| metal | sum_1024x4096 | 1×4096×1 / 2 | 0 | 26.003 | 29.041 | 26.291 | 29.907 | 0.90× | 269.375 | 305.208 |
| metal | softmax_1x127 | 1×127×1 / 2 | 0 | 2.741 | 30.060 | 2.834 | 30.665 | 0.09× | 218.666 | 333.333 |
| metal | softmax_17x257 | 1×257×1 / 2 | 0 | 3.298 | 35.086 | 3.397 | 35.622 | 0.09× | 195.292 | 400.875 |
| metal | softmax_64x4096 | 1×4096×1 / 2 | 0 | 8.800 | 35.401 | 8.900 | 57.533 | 0.25× | 238.250 | 318.333 |
| metal | softmax_1024x4096 | 1×4096×1 / 2 | 0 | 73.757 | 145.713 | 76.297 | 146.880 | 0.51× | 330.083 | 561.208 |
| metal | rmsnorm_1x127 | 1×127×1 / 2 | 0 | 3.157 | 7.528 | 3.449 | 7.747 | 0.42× | 191.625 | 235.750 |
| metal | rmsnorm_17x257 | 1×257×1 / 2 | 0 | 3.936 | 6.231 | 4.782 | 6.406 | 0.63× | 194.083 | 218.500 |
| metal | rmsnorm_64x4096 | 1×4096×1 / 2 | 0 | 9.082 | 12.422 | 9.385 | 12.816 | 0.73× | 233.166 | 250.625 |
| metal | rmsnorm_1024x4096 | 1×4096×1 / 2 | 0 | 75.786 | 76.220 | 76.909 | 80.184 | 0.99× | 384.791 | 324.042 |
| metal | layernorm_1x127 | 1×127×1 / 2 | 0 | 4.126 | 14.572 | 4.719 | 15.826 | 0.28× | 184.042 | 257.708 |
| metal | layernorm_17x257 | 1×257×1 / 2 | 0 | 3.719 | 20.438 | 3.734 | 24.348 | 0.18× | 240.500 | 243.833 |
| metal | layernorm_64x4096 | 1×4096×1 / 2 | 0 | 9.560 | 24.478 | 9.791 | 24.756 | 0.39× | 248.667 | 255.125 |
| metal | layernorm_1024x4096 | 1×4096×1 / 2 | 0 | 82.107 | 222.004 | 85.451 | 228.478 | 0.37× | 333.833 | 479.500 |
| metal | residual_layernorm_1x127 | 1×127×1 / 2 | 0 | 2.629 | 11.133 | 2.662 | 11.691 | 0.24× | 187.417 | 228.250 |
| metal | residual_layernorm_17x257 | 1×257×1 / 2 | 0 | 3.123 | 12.106 | 3.793 | 12.156 | 0.26× | 192.709 | 233.917 |
| metal | residual_layernorm_64x4096 | 1×4096×1 / 2 | 0 | 9.555 | 36.793 | 9.693 | 46.917 | 0.26× | 246.083 | 338.750 |
| metal | residual_layernorm_1024x4096 | 1×4096×1 / 2 | 0 | 107.759 | 276.974 | 107.874 | 287.373 | 0.39× | 335.291 | 562.250 |
| metal | cross_entropy_1x127 | 1×127×1 / 2 | 0 | 3.168 | 129.461 | 3.209 | 129.685 | 0.02× | 245.417 | 575.042 |
| metal | cross_entropy_17x257 | 1×257×1 / 2 | 0 | 3.717 | 134.264 | 4.230 | 136.657 | 0.03× | 208.791 | 452.083 |
| metal | cross_entropy_64x4096 | 1×4096×1 / 2 | 0 | 5.602 | 130.018 | 5.765 | 130.364 | 0.04× | 230.209 | 521.417 |
| metal | cross_entropy_1024x4096 | 1×4096×1 / 2 | 0 | 44.480 | 164.224 | 46.707 | 168.785 | 0.27× | 265.042 | 508.584 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sum_1x127 | 0.056 | 38.065 | 1.042 | 4.007 | 51.357 | 65.879 | 0.257 | 0.335 |
| metal / sum_17x257 | 0.055 | 43.653 | 0.987 | 0.456 | 54.671 | 1.013 | 0.429 | 0.370 |
| metal / sum_64x4096 | 0.058 | 42.045 | 1.469 | 0.451 | 57.434 | 4.245 | 0.252 | 0.256 |
| metal / sum_1024x4096 | 0.059 | 41.930 | 6.025 | 42.766 | 3.557 | 0.487 | 0.304 | 0.297 |
| metal / softmax_1x127 | 0.066 | 43.476 | 1.102 | 0.484 | 53.703 | 20.161 | 0.263 | 0.329 |
| metal / softmax_17x257 | 0.078 | 56.854 | 1.221 | 0.371 | 59.880 | 4.410 | 0.245 | 0.298 |
| metal / softmax_64x4096 | 0.066 | 49.209 | 1.163 | 0.561 | 66.303 | 3.332 | 0.470 | 0.355 |
| metal / softmax_1024x4096 | 0.064 | 50.791 | 6.100 | 17.235 | 6.205 | 3.162 | 3.136 | 0.918 |
| metal / rmsnorm_1x127 | 0.061 | 45.433 | 1.251 | 0.532 | 39.141 | 0.496 | 0.262 | 0.286 |
| metal / rmsnorm_17x257 | 0.062 | 54.880 | 1.363 | 0.727 | 42.294 | 0.233 | 0.329 | 0.276 |
| metal / rmsnorm_64x4096 | 0.073 | 50.028 | 1.537 | 0.673 | 43.062 | 7.455 | 0.587 | 0.361 |
| metal / rmsnorm_1024x4096 | 0.077 | 52.261 | 6.618 | 19.293 | 4.274 | 5.638 | 4.209 | 1.049 |
| metal / layernorm_1x127 | 0.070 | 51.032 | 1.246 | 1.169 | 55.536 | 1.758 | 0.260 | 0.341 |
| metal / layernorm_17x257 | 0.072 | 66.529 | 1.328 | 0.628 | 61.299 | 0.311 | 0.279 | 0.755 |
| metal / layernorm_64x4096 | 0.076 | 58.927 | 2.986 | 1.107 | 70.037 | 0.321 | 0.548 | 0.382 |
| metal / layernorm_1024x4096 | 0.107 | 59.318 | 6.158 | 17.956 | 5.680 | 0.642 | 3.637 | 0.617 |
| metal / residual_layernorm_1x127 | 0.075 | 52.673 | 1.282 | 0.693 | 54.688 | 0.610 | 0.231 | 0.308 |
| metal / residual_layernorm_17x257 | 0.067 | 58.554 | 1.419 | 0.743 | 55.581 | 0.304 | 0.269 | 0.287 |
| metal / residual_layernorm_64x4096 | 0.070 | 59.425 | 1.765 | 0.764 | 65.291 | 0.678 | 0.483 | 0.632 |
| metal / residual_layernorm_1024x4096 | 0.070 | 58.784 | 9.615 | 19.028 | 4.725 | 0.869 | 2.639 | 0.740 |
| metal / cross_entropy_1x127 | 0.071 | 49.301 | 1.271 | 0.629 | 54.507 | 18.545 | 0.718 | 0.568 |
| metal / cross_entropy_17x257 | 0.079 | 65.043 | 1.816 | 0.707 | 57.130 | 15.309 | 0.271 | 0.605 |
| metal / cross_entropy_64x4096 | 0.089 | 57.334 | 1.587 | 1.241 | 64.512 | 9.252 | 0.257 | 0.608 |
| metal / cross_entropy_1024x4096 | 0.080 | 58.470 | 6.433 | 17.482 | 5.208 | 10.220 | 0.301 | 0.833 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| sum_1x127 / native | 1.967 | 5.583 | 2.636 | 197.875 | 1.001× |
| sum_1x127 / torch | 13.993 | 24.042 | 11.588 | 231.750 | 1.023× |
| sum_17x257 / native | 2.848 | 4.667 | 2.894 | 202.292 | 0.992× |
| sum_17x257 / torch | 6.729 | 12.125 | 7.760 | 222.041 | 1.024× |
| sum_64x4096 / native | 4.294 | 6.542 | 4.282 | 232.458 | 1.062× |
| sum_64x4096 / torch | 12.963 | 15.667 | 15.347 | 343.792 | 1.044× |
| sum_1024x4096 / native | 25.133 | 31.417 | 26.003 | 269.375 | 0.984× |
| sum_1024x4096 / torch | 25.855 | 33.250 | 29.041 | 305.208 | 0.995× |
| softmax_1x127 / native | 2.837 | 4.625 | 2.741 | 218.666 | 0.992× |
| softmax_1x127 / torch | 15.906 | 38.000 | 30.060 | 333.333 | 5.426× |
| softmax_17x257 / native | 3.210 | 5.208 | 3.298 | 195.292 | 0.933× |
| softmax_17x257 / torch | 15.011 | 60.625 | 35.086 | 400.875 | 5.853× |
| softmax_64x4096 / native | 8.065 | 10.958 | 8.800 | 238.250 | 0.916× |
| softmax_64x4096 / torch | 22.534 | 61.708 | 35.401 | 318.333 | 3.365× |
| softmax_1024x4096 / native | 69.855 | 71.375 | 73.757 | 330.083 | 0.998× |
| softmax_1024x4096 / torch | 125.322 | 128.250 | 145.713 | 561.208 | 1.206× |
| rmsnorm_1x127 / native | 2.839 | 4.875 | 3.157 | 191.625 | 1.041× |
| rmsnorm_1x127 / torch | 4.516 | 8.958 | 7.528 | 235.750 | 0.989× |
| rmsnorm_17x257 / native | 3.724 | 8.125 | 3.936 | 194.083 | 1.019× |
| rmsnorm_17x257 / torch | 3.279 | 7.125 | 6.231 | 218.500 | 1.027× |
| rmsnorm_64x4096 / native | 8.853 | 11.750 | 9.082 | 233.166 | 0.969× |
| rmsnorm_64x4096 / torch | 9.756 | 12.500 | 12.422 | 250.625 | 1.048× |
| rmsnorm_1024x4096 / native | 71.400 | 72.542 | 75.786 | 384.791 | 1.002× |
| rmsnorm_1024x4096 / torch | 69.101 | 71.375 | 76.220 | 324.042 | 0.999× |
| layernorm_1x127 / native | 3.541 | 5.833 | 4.126 | 184.042 | 0.930× |
| layernorm_1x127 / torch | 6.361 | 11.792 | 14.572 | 257.708 | 1.000× |
| layernorm_17x257 / native | 3.570 | 5.667 | 3.719 | 240.500 | 1.066× |
| layernorm_17x257 / torch | 11.916 | 18.500 | 20.438 | 243.833 | 1.007× |
| layernorm_64x4096 / native | 8.771 | 12.250 | 9.560 | 248.667 | 1.003× |
| layernorm_64x4096 / torch | 20.329 | 21.625 | 24.478 | 255.125 | 0.913× |
| layernorm_1024x4096 / native | 76.810 | 89.708 | 82.107 | 333.833 | 1.005× |
| layernorm_1024x4096 / torch | 209.836 | 205.333 | 222.004 | 479.500 | 1.000× |
| residual_layernorm_1x127 / native | 2.452 | 4.542 | 2.629 | 187.417 | 1.096× |
| residual_layernorm_1x127 / torch | 5.826 | 9.667 | 11.133 | 228.250 | 0.988× |
| residual_layernorm_17x257 / native | 2.947 | 6.042 | 3.123 | 192.709 | 1.014× |
| residual_layernorm_17x257 / torch | 6.528 | 10.333 | 12.106 | 233.917 | 1.014× |
| residual_layernorm_64x4096 / native | 8.721 | 11.958 | 9.555 | 246.083 | 0.990× |
| residual_layernorm_64x4096 / torch | 28.840 | 45.375 | 36.793 | 338.750 | 1.008× |
| residual_layernorm_1024x4096 / native | 98.924 | 120.917 | 107.759 | 335.291 | 0.999× |
| residual_layernorm_1024x4096 / torch | 256.843 | 267.625 | 276.974 | 562.250 | 1.013× |
| cross_entropy_1x127 / native | 2.962 | 5.125 | 3.168 | 245.417 | 1.028× |
| cross_entropy_1x127 / torch | 36.377 | 67.750 | 129.461 | 575.042 | 2.296× |
| cross_entropy_17x257 / native | 3.316 | 6.125 | 3.717 | 208.791 | 0.916× |
| cross_entropy_17x257 / torch | 34.797 | 60.875 | 134.264 | 452.083 | 2.401× |
| cross_entropy_64x4096 / native | 5.648 | 7.958 | 5.602 | 230.209 | 0.938× |
| cross_entropy_64x4096 / torch | 53.030 | 89.333 | 130.018 | 521.417 | 1.814× |
| cross_entropy_1024x4096 / native | 40.210 | 46.375 | 44.480 | 265.042 | 1.003× |
| cross_entropy_1024x4096 / torch | 131.488 | 137.542 | 164.224 | 508.584 | 1.310× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| sum_1x127 | 1.969 | 13.946 | 4.833 | 23.334 | 197.875 | 231.750 |
| sum_17x257 | 2.803 | 6.192 | 4.500 | 12.291 | 202.292 | 222.041 |
| sum_64x4096 | 4.236 | 13.540 | 6.541 | 15.583 | 232.458 | 343.792 |
| sum_1024x4096 | 24.697 | 25.701 | 31.166 | 34.459 | 269.375 | 305.208 |
| softmax_1x127 | 2.823 | 67.719 | 4.291 | 46.250 | 218.666 | 333.333 |
| softmax_17x257 | 3.112 | 56.837 | 5.166 | 63.666 | 195.292 | 400.875 |
| softmax_64x4096 | 7.505 | 62.007 | 11.417 | 60.875 | 238.250 | 318.333 |
| softmax_1024x4096 | 69.695 | 132.489 | 71.250 | 132.875 | 330.083 | 561.208 |
| rmsnorm_1x127 | 2.840 | 4.615 | 4.958 | 7.583 | 191.625 | 235.750 |
| rmsnorm_17x257 | 3.974 | 3.367 | 7.083 | 7.500 | 194.083 | 218.500 |
| rmsnorm_64x4096 | 8.998 | 10.126 | 11.583 | 12.583 | 233.166 | 250.625 |
| rmsnorm_1024x4096 | 71.536 | 69.135 | 83.375 | 71.917 | 384.791 | 324.042 |
| layernorm_1x127 | 3.267 | 6.292 | 6.083 | 12.000 | 184.042 | 257.708 |
| layernorm_17x257 | 3.883 | 11.512 | 5.625 | 18.667 | 240.500 | 243.833 |
| layernorm_64x4096 | 8.989 | 18.836 | 12.334 | 21.333 | 248.667 | 255.125 |
| layernorm_1024x4096 | 76.612 | 209.849 | 78.416 | 204.792 | 333.833 | 479.500 |
| residual_layernorm_1x127 | 2.729 | 5.631 | 4.542 | 9.584 | 187.417 | 228.250 |
| residual_layernorm_17x257 | 3.183 | 6.156 | 6.125 | 10.042 | 192.709 | 233.917 |
| residual_layernorm_64x4096 | 8.234 | 28.072 | 11.875 | 44.250 | 246.083 | 338.750 |
| residual_layernorm_1024x4096 | 98.710 | 258.662 | 103.667 | 264.667 | 335.291 | 562.250 |
| cross_entropy_1x127 | 3.025 | 58.658 | 5.000 | 71.334 | 245.417 | 575.042 |
| cross_entropy_17x257 | 3.018 | 57.067 | 6.083 | 66.125 | 208.791 | 452.083 |
| cross_entropy_64x4096 | 5.285 | 60.091 | 9.042 | 104.291 | 230.209 | 521.417 |
| cross_entropy_1024x4096 | 39.811 | 142.821 | 50.583 | 139.291 | 265.042 | 508.584 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
