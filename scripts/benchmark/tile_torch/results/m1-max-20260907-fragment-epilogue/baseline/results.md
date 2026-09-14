# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T17:11:12.596214+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_128x128x128 | 64×64×4096 / 1 | 1 | 10.523 | 29.352 | 11.198 | 30.889 | 0.36× | 339.500 | 280.292 |
| metal | gemm_127x193x61 | FAILED | | | | | | | | |
| metal | gemm_1024x1024x1024 | 64×64×4096 / 1 | 1 | 339.316 | 379.923 | 358.605 | 394.361 | 0.89× | 562.250 | 560.375 |
| metal | gemm_4096x4096x4096 | 64×64×4096 / 1 | 1 | 21384.708 | 20331.833 | 22945.525 | 20936.450 | 1.05× | 22046.833 | 20464.417 |
| metal | gemm_128x2048x512 | 64×64×4096 / 1 | 1 | 59.322 | 63.228 | 59.990 | 65.399 | 0.94× | 288.250 | 342.291 |
| metal | gemm_2048x128x512 | 64×64×4096 / 1 | 1 | 59.659 | 61.325 | 59.849 | 63.829 | 0.97× | 379.709 | 390.250 |
| metal | gemm_relu_128x128x128 | 64×64×4096 / 1 | 1 | 11.841 | 39.783 | 12.060 | 50.699 | 0.30× | 402.458 | 419.958 |
| metal | gemm_relu_127x193x61 | FAILED | | | | | | | | |
| metal | gemm_relu_1024x1024x1024 | 64×64×4096 / 1 | 1 | 356.256 | 432.242 | 364.433 | 482.747 | 0.82× | 506.292 | 743.834 |
| metal | gemm_relu_4096x4096x4096 | 64×64×4096 / 1 | 1 | 22519.125 | 21132.000 | 23493.350 | 23575.792 | 1.07× | 21859.292 | 22234.458 |
| metal | gemm_relu_128x2048x512 | 64×64×4096 / 1 | 1 | 61.823 | 90.880 | 62.709 | 98.788 | 0.68× | 275.541 | 384.458 |
| metal | gemm_relu_2048x128x512 | 64×64×4096 / 1 | 1 | 60.416 | 90.347 | 62.329 | 95.160 | 0.67× | 286.666 | 414.458 |
| metal | gemm_gelu_128x128x128 | 64×64×4096 / 1 | 1 | 11.770 | 43.513 | 12.890 | 46.677 | 0.27× | 239.416 | 718.084 |
| metal | gemm_gelu_127x193x61 | FAILED | | | | | | | | |
| metal | gemm_gelu_1024x1024x1024 | 64×64×4096 / 1 | 1 | 414.974 | 448.322 | 457.758 | 505.531 | 0.93× | 528.708 | 669.542 |
| metal | gemm_gelu_4096x4096x4096 | 64×64×4096 / 1 | 1 | 25196.916 | 21951.250 | 25494.250 | 23206.308 | 1.15× | 24983.917 | 21431.458 |
| metal | gemm_gelu_128x2048x512 | 64×64×4096 / 1 | 1 | 60.522 | 99.957 | 66.527 | 104.904 | 0.61× | 289.834 | 584.042 |
| metal | gemm_gelu_2048x128x512 | 64×64×4096 / 1 | 1 | 66.252 | 98.572 | 66.381 | 101.653 | 0.67× | 279.375 | 456.000 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_128x128x128 | 0.211 | 39.117 | 2.043 | 5.989 | 209.810 | 51.844 | 0.269 | 0.618 |
| metal / gemm_1024x1024x1024 | 0.065 | 37.640 | 5.871 | 2.924 | 167.977 | 8.614 | 0.805 | 0.389 |
| metal / gemm_4096x4096x4096 | 0.067 | 33.047 | 31.903 | 46.697 | 180.757 | 37.060 | 9.022 | 3.383 |
| metal / gemm_128x2048x512 | 0.062 | 36.164 | 4.380 | 3.579 | 232.144 | 9.005 | 0.502 | 0.361 |
| metal / gemm_2048x128x512 | 0.060 | 35.489 | 3.426 | 0.972 | 234.679 | 5.318 | 0.420 | 0.382 |
| metal / gemm_relu_128x128x128 | 0.065 | 36.735 | 2.045 | 0.963 | 225.601 | 109.484 | 0.324 | 0.545 |
| metal / gemm_relu_1024x1024x1024 | 0.072 | 38.847 | 3.537 | 1.760 | 278.761 | 7.162 | 1.034 | 0.459 |
| metal / gemm_relu_4096x4096x4096 | 0.063 | 34.764 | 30.793 | 20.768 | 306.562 | 28.520 | 9.108 | 1.647 |
| metal / gemm_relu_128x2048x512 | 0.061 | 38.984 | 4.543 | 3.553 | 251.187 | 3.747 | 0.546 | 0.303 |
| metal / gemm_relu_2048x128x512 | 0.061 | 37.471 | 2.359 | 0.787 | 251.988 | 0.529 | 0.452 | 0.346 |
| metal / gemm_gelu_128x128x128 | 0.075 | 36.374 | 1.557 | 1.375 | 228.910 | 9.990 | 0.370 | 0.280 |
| metal / gemm_gelu_1024x1024x1024 | 0.068 | 39.157 | 3.665 | 1.901 | 285.144 | 11.576 | 0.904 | 0.420 |
| metal / gemm_gelu_4096x4096x4096 | 0.063 | 32.437 | 31.612 | 19.467 | 310.223 | 26.592 | 10.079 | 1.618 |
| metal / gemm_gelu_128x2048x512 | 0.080 | 37.490 | 4.859 | 1.868 | 258.458 | 7.383 | 0.449 | 0.316 |
| metal / gemm_gelu_2048x128x512 | 0.064 | 35.667 | 2.372 | 1.109 | 253.906 | 0.544 | 0.422 | 0.321 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_128x128x128 / native | 12.232 | 10.667 | 10.523 | 339.500 | 1.031× |
| gemm_128x128x128 / torch | 15.014 | 35.708 | 29.352 | 280.292 | 2.003× |
| gemm_1024x1024x1024 / native | 327.239 | 274.500 | 339.316 | 562.250 | 1.018× |
| gemm_1024x1024x1024 / torch | 336.651 | 309.667 | 379.923 | 560.375 | 1.041× |
| gemm_4096x4096x4096 / native | 20589.375 | 20586.333 | 21384.708 | 22046.833 | 0.988× |
| gemm_4096x4096x4096 / torch | 19251.875 | 18438.375 | 20331.833 | 20464.417 | 0.986× |
| gemm_128x2048x512 / native | 51.878 | 49.792 | 59.322 | 288.250 | 1.023× |
| gemm_128x2048x512 / torch | 44.635 | 61.125 | 63.228 | 342.291 | 1.342× |
| gemm_2048x128x512 / native | 51.289 | 49.792 | 59.659 | 379.709 | 1.031× |
| gemm_2048x128x512 / torch | 44.131 | 53.917 | 61.325 | 390.250 | 1.354× |
| gemm_relu_128x128x128 / native | 9.474 | 12.000 | 11.841 | 402.458 | 0.900× |
| gemm_relu_128x128x128 / torch | 18.903 | 25.708 | 39.783 | 419.958 | 3.750× |
| gemm_relu_1024x1024x1024 / native | 335.363 | 274.333 | 356.256 | 506.292 | 1.003× |
| gemm_relu_1024x1024x1024 / torch | 404.281 | 321.750 | 432.242 | 743.834 | 1.099× |
| gemm_relu_4096x4096x4096 / native | 20679.625 | 20952.292 | 22519.125 | 21859.292 | 0.969× |
| gemm_relu_4096x4096x4096 / torch | 20778.125 | 20765.208 | 21132.000 | 22234.458 | 1.008× |
| gemm_relu_128x2048x512 / native | 52.766 | 49.875 | 61.823 | 275.541 | 1.003× |
| gemm_relu_128x2048x512 / torch | 63.193 | 155.750 | 90.880 | 384.458 | 1.427× |
| gemm_relu_2048x128x512 / native | 53.199 | 49.417 | 60.416 | 286.666 | 0.995× |
| gemm_relu_2048x128x512 / torch | 62.622 | 78.792 | 90.347 | 414.458 | 1.443× |
| gemm_gelu_128x128x128 / native | 9.566 | 11.625 | 11.770 | 239.416 | 0.960× |
| gemm_gelu_128x128x128 / torch | 24.986 | 49.083 | 43.513 | 718.084 | 3.037× |
| gemm_gelu_1024x1024x1024 / native | 398.309 | 322.042 | 414.974 | 528.708 | 0.998× |
| gemm_gelu_1024x1024x1024 / torch | 423.644 | 357.750 | 448.322 | 669.542 | 1.088× |
| gemm_gelu_4096x4096x4096 / native | 23889.333 | 23926.750 | 25196.916 | 24983.917 | 1.002× |
| gemm_gelu_4096x4096x4096 / torch | 20957.042 | 20920.042 | 21951.250 | 21431.458 | 0.995× |
| gemm_gelu_128x2048x512 / native | 53.912 | 50.250 | 60.522 | 289.834 | 1.007× |
| gemm_gelu_128x2048x512 / torch | 78.115 | 79.500 | 99.957 | 584.042 | 1.557× |
| gemm_gelu_2048x128x512 / native | 55.247 | 51.458 | 66.252 | 279.375 | 0.974× |
| gemm_gelu_2048x128x512 / torch | 94.460 | 77.125 | 98.572 | 456.000 | 1.279× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_128x128x128 | 9.009 | 19.269 | 10.125 | 24.125 | 339.500 | 280.292 |
| gemm_1024x1024x1024 | 331.578 | 336.909 | 277.708 | 287.375 | 562.250 | 560.375 |
| gemm_4096x4096x4096 | 20352.333 | 18562.083 | 20669.375 | 18634.250 | 22046.833 | 20464.417 |
| gemm_128x2048x512 | 51.533 | 47.410 | 50.208 | 46.292 | 288.250 | 342.291 |
| gemm_2048x128x512 | 52.801 | 47.085 | 48.792 | 46.291 | 379.709 | 390.250 |
| gemm_relu_128x128x128 | 8.992 | 38.338 | 10.750 | 210.709 | 402.458 | 419.958 |
| gemm_relu_1024x1024x1024 | 335.293 | 408.493 | 300.083 | 339.416 | 506.292 | 743.834 |
| gemm_relu_4096x4096x4096 | 20028.708 | 20854.041 | 20073.458 | 20816.334 | 21859.292 | 22234.458 |
| gemm_relu_128x2048x512 | 51.768 | 69.016 | 48.958 | 70.375 | 275.541 | 384.458 |
| gemm_relu_2048x128x512 | 52.941 | 68.635 | 50.166 | 69.416 | 286.666 | 414.458 |
| gemm_gelu_128x128x128 | 9.294 | 37.588 | 11.459 | 36.583 | 239.416 | 718.084 |
| gemm_gelu_1024x1024x1024 | 398.866 | 425.524 | 322.500 | 354.333 | 528.708 | 669.542 |
| gemm_gelu_4096x4096x4096 | 23936.667 | 20694.375 | 24086.667 | 20913.126 | 24983.917 | 21431.458 |
| gemm_gelu_128x2048x512 | 53.882 | 100.456 | 56.250 | 79.625 | 289.834 | 584.042 |
| gemm_gelu_2048x128x512 | 52.139 | 99.593 | 51.083 | 80.251 | 279.375 | 456.000 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
