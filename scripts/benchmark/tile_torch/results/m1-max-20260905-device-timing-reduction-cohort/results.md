# TileIR/TVMx vs PyTorch

> Historical ABI-1 diagnostic capture: no no-counter control, and subsequent
> testing found substantial probe perturbation. These GPU ratios are **not**
> an uninstrumented performance ranking. See the
> [observer audit](../m1-max-20260905-device-timing-counter-control/notes.md).

Generated: 2026-09-05T06:23:36.080872+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sum_1x127 | 1×127×1 / 2 | 0 | 2.868 | 6.567 | 2.958 | 6.633 | 0.44× | 392.292 | 233.042 |
| metal | sum_17x257 | 1×257×1 / 2 | 0 | 2.455 | 4.582 | 2.588 | 5.097 | 0.54× | 236.833 | 193.333 |
| metal | sum_128x1024 | 1×1024×1 / 2 | 0 | 3.265 | 5.670 | 3.598 | 5.718 | 0.58× | 435.958 | 220.709 |
| metal | sum_64x4096 | 1×4096×1 / 2 | 0 | 4.134 | 14.940 | 4.439 | 15.514 | 0.28× | 262.958 | 249.417 |
| metal | softmax_1x127 | 1×127×1 / 2 | 0 | 3.133 | 29.457 | 3.285 | 30.647 | 0.11× | 216.625 | 332.458 |
| metal | softmax_17x257 | 1×257×1 / 2 | 0 | 2.968 | 30.350 | 2.986 | 30.885 | 0.10× | 215.208 | 379.667 |
| metal | softmax_128x1024 | 1×1024×1 / 2 | 0 | 5.246 | 35.672 | 5.250 | 36.406 | 0.15× | 226.875 | 340.750 |
| metal | softmax_64x4096 | 1×4096×1 / 2 | 0 | 8.129 | 33.924 | 8.177 | 34.306 | 0.24× | 218.750 | 336.333 |
| metal | rmsnorm_1x127 | 1×127×1 / 2 | 0 | 3.440 | 7.542 | 3.449 | 7.851 | 0.46× | 193.375 | 239.750 |
| metal | rmsnorm_17x257 | 1×257×1 / 2 | 0 | 4.960 | 6.412 | 5.175 | 7.155 | 0.77× | 235.083 | 213.500 |
| metal | rmsnorm_128x1024 | 1×1024×1 / 2 | 0 | 6.333 | 9.276 | 6.720 | 9.722 | 0.68× | 273.708 | 259.375 |
| metal | rmsnorm_64x4096 | 1×4096×1 / 2 | 0 | 10.594 | 12.753 | 11.022 | 13.295 | 0.83× | 231.625 | 241.542 |
| metal | layernorm_1x127 | 1×127×1 / 2 | 0 | 3.693 | 8.739 | 3.917 | 8.988 | 0.42× | 248.750 | 237.959 |
| metal | layernorm_17x257 | 1×257×1 / 2 | 0 | 4.856 | 9.748 | 4.923 | 10.252 | 0.50× | 229.709 | 314.333 |
| metal | layernorm_128x1024 | 1×1024×1 / 2 | 0 | 7.052 | 13.777 | 7.208 | 16.444 | 0.51× | 291.291 | 223.458 |
| metal | layernorm_64x4096 | 1×4096×1 / 2 | 0 | 11.949 | 23.710 | 12.157 | 23.990 | 0.50× | 230.500 | 303.541 |
| metal | residual_layernorm_1x127 | 1×127×1 / 2 | 0 | 3.015 | 11.439 | 3.263 | 12.514 | 0.26× | 204.041 | 204.625 |
| metal | residual_layernorm_17x257 | 1×257×1 / 2 | 0 | 3.219 | 11.463 | 3.412 | 11.931 | 0.28× | 235.458 | 223.083 |
| metal | residual_layernorm_128x1024 | 1×1024×1 / 2 | 0 | 6.001 | 19.088 | 6.100 | 19.553 | 0.31× | 248.042 | 268.875 |
| metal | residual_layernorm_64x4096 | 1×4096×1 / 2 | 0 | 9.293 | 26.118 | 9.333 | 27.123 | 0.36× | 229.291 | 276.625 |
| metal | cross_entropy_1x127 | 1×127×1 / 2 | 0 | 3.726 | 122.158 | 4.020 | 124.599 | 0.03× | 212.625 | 447.250 |
| metal | cross_entropy_17x257 | 1×257×1 / 2 | 0 | 3.000 | 125.820 | 3.202 | 127.644 | 0.02× | 201.334 | 465.625 |
| metal | cross_entropy_128x1024 | 1×1024×1 / 2 | 0 | 3.906 | 169.069 | 4.023 | 274.833 | 0.02× | 260.042 | 658.791 |
| metal | cross_entropy_64x4096 | 1×4096×1 / 2 | 0 | 5.594 | 129.462 | 5.633 | 131.976 | 0.04× | 248.875 | 497.458 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sum_1x127 | 0.065 | 39.450 | 1.401 | 3.721 | 0.758 | 61.102 | 0.266 | 0.276 |
| metal / sum_17x257 | 0.058 | 44.407 | 2.121 | 0.302 | 0.997 | 0.516 | 0.275 | 0.291 |
| metal / sum_128x1024 | 0.056 | 44.089 | 1.159 | 0.855 | 2.022 | 1.000 | 0.283 | 0.336 |
| metal / sum_64x4096 | 0.067 | 41.646 | 1.224 | 0.491 | 3.278 | 3.871 | 0.228 | 0.314 |
| metal / softmax_1x127 | 0.067 | 52.270 | 1.148 | 0.996 | 0.898 | 15.808 | 0.244 | 0.294 |
| metal / softmax_17x257 | 0.067 | 55.221 | 1.030 | 0.365 | 0.834 | 5.769 | 0.321 | 0.264 |
| metal / softmax_128x1024 | 0.066 | 48.706 | 1.119 | 0.505 | 2.035 | 3.937 | 0.379 | 0.314 |
| metal / softmax_64x4096 | 0.065 | 52.365 | 1.444 | 0.506 | 4.819 | 3.832 | 0.499 | 0.310 |
| metal / rmsnorm_1x127 | 0.059 | 53.427 | 1.464 | 0.592 | 0.760 | 0.492 | 0.300 | 0.350 |
| metal / rmsnorm_17x257 | 0.073 | 112.618 | 1.618 | 0.810 | 1.020 | 0.213 | 0.253 | 0.290 |
| metal / rmsnorm_128x1024 | 0.068 | 54.662 | 1.420 | 0.747 | 2.182 | 0.751 | 0.407 | 0.308 |
| metal / rmsnorm_64x4096 | 0.062 | 58.408 | 1.527 | 0.591 | 3.865 | 7.123 | 0.645 | 0.340 |
| metal / layernorm_1x127 | 0.076 | 60.957 | 1.393 | 1.210 | 58.384 | 6.188 | 0.238 | 0.387 |
| metal / layernorm_17x257 | 0.083 | 67.961 | 1.750 | 0.698 | 60.526 | 0.266 | 0.477 | 0.334 |
| metal / layernorm_128x1024 | 0.080 | 64.342 | 1.379 | 0.997 | 59.590 | 4.899 | 0.637 | 0.395 |
| metal / layernorm_64x4096 | 0.072 | 65.123 | 1.730 | 0.740 | 67.631 | 0.282 | 0.418 | 0.330 |
| metal / residual_layernorm_1x127 | 0.072 | 60.366 | 1.230 | 1.010 | 43.087 | 4.937 | 0.843 | 0.285 |
| metal / residual_layernorm_17x257 | 0.069 | 63.086 | 1.473 | 0.614 | 42.646 | 0.372 | 0.260 | 0.373 |
| metal / residual_layernorm_128x1024 | 0.068 | 61.218 | 1.478 | 1.124 | 51.425 | 5.190 | 0.359 | 0.303 |
| metal / residual_layernorm_64x4096 | 0.072 | 62.897 | 1.758 | 0.895 | 44.037 | 0.800 | 0.402 | 0.437 |
| metal / cross_entropy_1x127 | 0.080 | 51.073 | 1.364 | 1.681 | 0.737 | 16.405 | 0.252 | 0.540 |
| metal / cross_entropy_17x257 | 0.073 | 66.853 | 1.270 | 0.587 | 0.919 | 11.159 | 0.252 | 0.555 |
| metal / cross_entropy_128x1024 | 0.088 | 56.705 | 1.349 | 0.820 | 2.322 | 9.229 | 0.316 | 0.537 |
| metal / cross_entropy_64x4096 | 0.072 | 55.363 | 1.643 | 0.573 | 5.808 | 7.608 | 0.345 | 0.596 |

## GPU execution versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

| Case | Native GPU batch µs/op | Torch GPU batch µs/op | Native / Torch GPU | Native GPU single µs | Torch GPU single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| sum_1x127 | 2.985 | 5.844 | 0.511× | 5.084 | 9.792 | 392.292 | 233.042 |
| sum_17x257 | 2.182 | 2.708 | 0.806× | 4.500 | 5.125 | 236.833 | 193.333 |
| sum_128x1024 | 2.877 | 4.208 | 0.684× | 5.584 | 6.583 | 435.958 | 220.709 |
| sum_64x4096 | 3.753 | 12.053 | 0.311× | 6.291 | 15.291 | 262.958 | 249.417 |
| softmax_1x127 | 2.900 | 69.011 | 0.042× | 5.292 | 61.292 | 216.625 | 332.458 |
| softmax_17x257 | 2.641 | 56.922 | 0.046× | 5.125 | 49.500 | 215.208 | 379.667 |
| softmax_128x1024 | 5.214 | 61.900 | 0.084× | 7.584 | 56.875 | 226.875 | 340.750 |
| softmax_64x4096 | 7.541 | 60.646 | 0.124× | 10.583 | 57.500 | 218.750 | 336.333 |
| rmsnorm_1x127 | 3.064 | 4.210 | 0.728× | 5.958 | 7.792 | 193.375 | 239.750 |
| rmsnorm_17x257 | 4.587 | 3.691 | 1.243× | 6.708 | 7.208 | 235.083 | 213.500 |
| rmsnorm_128x1024 | 5.811 | 5.945 | 0.977× | 9.334 | 8.709 | 273.708 | 259.375 |
| rmsnorm_64x4096 | 10.820 | 8.751 | 1.236× | 12.583 | 14.917 | 231.625 | 241.542 |
| layernorm_1x127 | 3.014 | 4.121 | 0.731× | 5.917 | 7.958 | 248.750 | 237.959 |
| layernorm_17x257 | 4.653 | 4.373 | 1.064× | 7.291 | 7.875 | 229.709 | 314.333 |
| layernorm_128x1024 | 6.843 | 9.012 | 0.759× | 9.625 | 12.417 | 291.291 | 223.458 |
| layernorm_64x4096 | 11.810 | 19.062 | 0.620× | 15.416 | 21.292 | 230.500 | 303.541 |
| residual_layernorm_1x127 | 2.715 | 5.958 | 0.456× | 5.500 | 9.583 | 204.041 | 204.625 |
| residual_layernorm_17x257 | 3.101 | 6.431 | 0.482× | 5.708 | 10.084 | 235.458 | 223.083 |
| residual_layernorm_128x1024 | 5.680 | 12.857 | 0.442× | 8.583 | 16.500 | 248.042 | 268.875 |
| residual_layernorm_64x4096 | 8.677 | 19.906 | 0.436× | 11.708 | 23.958 | 229.291 | 276.625 |
| cross_entropy_1x127 | 3.508 | 49.456 | 0.071× | 6.084 | 47.291 | 212.625 | 447.250 |
| cross_entropy_17x257 | 2.719 | 81.365 | 0.033× | 5.042 | 37.334 | 201.334 | 465.625 |
| cross_entropy_128x1024 | 3.773 | 76.462 | 0.049× | 6.375 | 61.374 | 260.042 | 658.791 |
| cross_entropy_64x4096 | 5.277 | 56.516 | 0.093× | 9.333 | 60.625 | 248.875 | 497.458 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
