# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T08:36:36.267534+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sum_17x257 | 1×257×1 / 2 | 0 | 4.149 | 6.505 | 4.414 | 7.667 | 0.64× | 204.334 | 228.750 |
| metal | sum_7x1537 | 1×1537×1 / 2 | 0 | 3.769 | 7.708 | 4.299 | 7.929 | 0.49× | 223.667 | 231.917 |
| metal | sum_64x4096 | 1×4096×1 / 2 | 0 | 4.371 | 15.048 | 4.415 | 15.892 | 0.29× | 230.458 | 254.583 |
| metal | sum_1024x4096 | 1×4096×1 / 2 | 0 | 28.207 | 29.027 | 28.599 | 30.288 | 0.97× | 272.666 | 255.167 |
| metal | sum_128x8192 | 1×8192×1 / 2 | 0 | 9.706 | 31.126 | 10.766 | 38.931 | 0.31× | 223.500 | 269.500 |
| metal | softmax_17x257 | 1×257×1 / 2 | 0 | 3.801 | 32.132 | 4.930 | 34.096 | 0.12× | 210.000 | 324.208 |
| metal | softmax_7x1537 | 1×1537×1 / 2 | 0 | 4.192 | 33.344 | 4.358 | 34.587 | 0.13× | 224.792 | 427.583 |
| metal | softmax_64x4096 | 1×4096×1 / 2 | 0 | 8.624 | 46.310 | 9.236 | 58.853 | 0.19× | 250.125 | 387.917 |
| metal | softmax_1024x4096 | 1×4096×1 / 2 | 0 | 77.917 | 135.322 | 80.451 | 139.823 | 0.58× | 308.416 | 412.000 |
| metal | softmax_128x8192 | 1×8192×1 / 2 | 0 | 25.826 | 47.520 | 26.665 | 50.770 | 0.54× | 280.958 | 394.375 |
| metal | rmsnorm_17x257 | 1×257×1 / 2 | 0 | 3.255 | 7.332 | 3.304 | 11.961 | 0.44× | 193.000 | 218.458 |
| metal | rmsnorm_7x1537 | 1×1537×1 / 2 | 0 | 4.455 | 10.207 | 4.751 | 10.933 | 0.44× | 200.291 | 213.958 |
| metal | rmsnorm_64x4096 | 1×4096×1 / 2 | 0 | 8.675 | 14.164 | 9.181 | 16.295 | 0.61× | 213.458 | 334.000 |
| metal | rmsnorm_1024x4096 | 1×4096×1 / 2 | 0 | 75.110 | 81.190 | 76.752 | 82.137 | 0.93× | 306.792 | 322.834 |
| metal | rmsnorm_128x8192 | 1×8192×1 / 2 | 0 | 22.722 | 29.345 | 23.856 | 41.472 | 0.77× | 252.166 | 290.375 |
| metal | layernorm_17x257 | 1×257×1 / 2 | 0 | 3.837 | 8.827 | 4.246 | 9.142 | 0.43× | 241.917 | 219.083 |
| metal | layernorm_7x1537 | 1×1537×1 / 2 | 0 | 5.238 | 12.338 | 5.705 | 13.249 | 0.42× | 224.292 | 210.916 |
| metal | layernorm_64x4096 | 1×4096×1 / 2 | 0 | 10.325 | 32.354 | 10.915 | 34.931 | 0.32× | 246.209 | 310.958 |
| metal | layernorm_1024x4096 | 1×4096×1 / 2 | 0 | 83.469 | 229.560 | 85.572 | 231.980 | 0.36× | 326.042 | 599.083 |
| metal | layernorm_128x8192 | 1×8192×1 / 2 | 0 | 31.032 | 78.868 | 33.645 | 81.523 | 0.39× | 265.750 | 348.625 |
| metal | residual_layernorm_17x257 | 1×257×1 / 2 | 0 | 4.269 | 13.465 | 4.652 | 15.289 | 0.32× | 238.333 | 252.792 |
| metal | residual_layernorm_7x1537 | 1×1537×1 / 2 | 0 | 4.712 | 13.872 | 4.784 | 14.655 | 0.34× | 221.625 | 243.666 |
| metal | residual_layernorm_64x4096 | 1×4096×1 / 2 | 0 | 9.448 | 26.199 | 9.820 | 26.385 | 0.36× | 282.291 | 247.500 |
| metal | residual_layernorm_1024x4096 | 1×4096×1 / 2 | 0 | 106.731 | 270.834 | 112.615 | 275.943 | 0.39× | 347.417 | 529.000 |
| metal | residual_layernorm_128x8192 | 1×8192×1 / 2 | 0 | 28.092 | 74.681 | 28.229 | 89.378 | 0.38× | 280.625 | 354.625 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sum_17x257 | 0.073 | 41.551 | 1.090 | 4.182 | 44.243 | 65.029 | 0.866 | 0.279 |
| metal / sum_7x1537 | 0.065 | 39.710 | 1.367 | 0.322 | 1.035 | 0.242 | 0.244 | 0.260 |
| metal / sum_64x4096 | 0.051 | 39.926 | 1.381 | 0.524 | 5.982 | 4.409 | 0.277 | 0.261 |
| metal / sum_1024x4096 | 0.052 | 41.165 | 5.680 | 41.983 | 8.948 | 0.394 | 0.232 | 0.253 |
| metal / sum_128x8192 | 0.051 | 41.349 | 1.991 | 0.572 | 6.883 | 0.300 | 0.290 | 0.281 |
| metal / softmax_17x257 | 0.070 | 49.036 | 1.054 | 0.392 | 1.367 | 22.274 | 0.253 | 0.294 |
| metal / softmax_7x1537 | 0.067 | 47.534 | 1.057 | 0.568 | 1.102 | 6.204 | 0.313 | 0.303 |
| metal / softmax_64x4096 | 0.067 | 48.491 | 1.369 | 0.464 | 5.824 | 2.897 | 0.674 | 0.383 |
| metal / softmax_1024x4096 | 0.162 | 55.641 | 5.767 | 16.638 | 8.731 | 3.315 | 3.354 | 0.577 |
| metal / softmax_128x8192 | 0.083 | 50.347 | 2.383 | 0.655 | 9.779 | 2.982 | 1.075 | 0.387 |
| metal / rmsnorm_17x257 | 0.066 | 48.304 | 1.270 | 0.735 | 1.394 | 0.998 | 0.283 | 0.300 |
| metal / rmsnorm_7x1537 | 0.061 | 47.126 | 1.281 | 0.553 | 1.616 | 0.232 | 0.272 | 0.303 |
| metal / rmsnorm_64x4096 | 0.065 | 51.574 | 1.912 | 1.018 | 6.281 | 7.126 | 0.581 | 0.352 |
| metal / rmsnorm_1024x4096 | 0.060 | 49.926 | 6.041 | 17.687 | 10.367 | 5.822 | 3.374 | 0.762 |
| metal / rmsnorm_128x8192 | 0.061 | 51.332 | 2.368 | 0.892 | 6.407 | 0.508 | 1.299 | 0.510 |
| metal / layernorm_17x257 | 0.079 | 57.295 | 2.138 | 1.095 | 60.903 | 0.507 | 0.283 | 0.310 |
| metal / layernorm_7x1537 | 0.091 | 57.514 | 2.098 | 0.701 | 55.787 | 0.293 | 0.278 | 0.501 |
| metal / layernorm_64x4096 | 0.095 | 61.465 | 1.651 | 1.415 | 70.719 | 0.276 | 0.509 | 0.394 |
| metal / layernorm_1024x4096 | 0.078 | 58.345 | 7.131 | 17.900 | 7.170 | 1.021 | 3.467 | 0.707 |
| metal / layernorm_128x8192 | 0.085 | 57.939 | 2.310 | 0.931 | 74.395 | 57.712 | 0.996 | 1.179 |
| metal / residual_layernorm_17x257 | 0.087 | 55.217 | 1.773 | 1.107 | 57.284 | 0.521 | 0.266 | 0.502 |
| metal / residual_layernorm_7x1537 | 0.075 | 58.380 | 1.483 | 0.576 | 57.156 | 0.244 | 0.290 | 0.357 |
| metal / residual_layernorm_64x4096 | 0.071 | 59.272 | 2.358 | 1.073 | 63.749 | 0.590 | 0.418 | 0.340 |
| metal / residual_layernorm_1024x4096 | 0.071 | 58.779 | 6.852 | 17.163 | 6.103 | 0.770 | 2.502 | 0.739 |
| metal / residual_layernorm_128x8192 | 0.064 | 60.513 | 2.948 | 1.096 | 68.850 | 0.524 | 0.877 | 0.474 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| sum_17x257 / native | 3.092 | 6.000 | 4.149 | 204.334 | 1.094× |
| sum_17x257 / torch | 6.631 | 12.250 | 6.505 | 228.750 | 1.012× |
| sum_7x1537 / native | 3.859 | 7.625 | 3.769 | 223.667 | 1.177× |
| sum_7x1537 / torch | 5.818 | 11.125 | 7.708 | 231.917 | 1.178× |
| sum_64x4096 / native | 4.483 | 7.167 | 4.371 | 230.458 | 0.980× |
| sum_64x4096 / torch | 13.335 | 15.750 | 15.048 | 254.583 | 0.944× |
| sum_1024x4096 / native | 27.319 | 34.417 | 28.207 | 272.666 | 0.972× |
| sum_1024x4096 / torch | 26.514 | 32.750 | 29.027 | 255.167 | 0.982× |
| sum_128x8192 / native | 9.691 | 13.292 | 9.706 | 223.500 | 0.973× |
| sum_128x8192 / torch | 24.588 | 29.625 | 31.126 | 269.500 | 0.937× |
| softmax_17x257 / native | 3.778 | 8.542 | 3.801 | 210.000 | 1.057× |
| softmax_17x257 / torch | 13.270 | 66.333 | 32.132 | 324.208 | 5.455× |
| softmax_7x1537 / native | 4.673 | 6.958 | 4.192 | 224.792 | 0.948× |
| softmax_7x1537 / torch | 16.614 | 80.458 | 33.344 | 427.583 | 4.602× |
| softmax_64x4096 / native | 8.534 | 12.417 | 8.624 | 250.125 | 0.938× |
| softmax_64x4096 / torch | 27.745 | 76.625 | 46.310 | 387.917 | 3.146× |
| softmax_1024x4096 / native | 73.954 | 75.208 | 77.917 | 308.416 | 1.000× |
| softmax_1024x4096 / torch | 121.184 | 130.250 | 135.322 | 412.000 | 1.185× |
| softmax_128x8192 / native | 24.980 | 31.208 | 25.826 | 280.958 | 0.995× |
| softmax_128x8192 / torch | 38.691 | 70.083 | 47.520 | 394.375 | 2.101× |
| rmsnorm_17x257 / native | 3.384 | 5.625 | 3.255 | 193.000 | 0.966× |
| rmsnorm_17x257 / torch | 5.115 | 12.375 | 7.332 | 218.458 | 0.893× |
| rmsnorm_7x1537 / native | 4.505 | 6.958 | 4.455 | 200.291 | 0.970× |
| rmsnorm_7x1537 / torch | 5.838 | 9.458 | 10.207 | 213.958 | 0.950× |
| rmsnorm_64x4096 / native | 8.766 | 11.500 | 8.675 | 213.458 | 0.997× |
| rmsnorm_64x4096 / torch | 9.477 | 13.000 | 14.164 | 334.000 | 1.050× |
| rmsnorm_1024x4096 / native | 70.094 | 71.250 | 75.110 | 306.792 | 1.000× |
| rmsnorm_1024x4096 / torch | 70.901 | 71.750 | 81.190 | 322.834 | 0.993× |
| rmsnorm_128x8192 / native | 22.288 | 27.458 | 22.722 | 252.166 | 0.988× |
| rmsnorm_128x8192 / torch | 22.615 | 28.625 | 29.345 | 290.375 | 0.999× |
| layernorm_17x257 / native | 3.525 | 7.917 | 3.837 | 241.917 | 1.049× |
| layernorm_17x257 / torch | 4.353 | 8.042 | 8.827 | 219.083 | 1.165× |
| layernorm_7x1537 / native | 5.633 | 7.417 | 5.238 | 224.292 | 0.921× |
| layernorm_7x1537 / torch | 10.413 | 16.208 | 12.338 | 210.916 | 1.074× |
| layernorm_64x4096 / native | 10.163 | 12.750 | 10.325 | 246.209 | 1.011× |
| layernorm_64x4096 / torch | 29.469 | 37.875 | 32.354 | 310.958 | 0.925× |
| layernorm_1024x4096 / native | 79.161 | 80.083 | 83.469 | 326.042 | 0.997× |
| layernorm_1024x4096 / torch | 210.145 | 205.250 | 229.560 | 599.083 | 0.994× |
| layernorm_128x8192 / native | 29.281 | 41.333 | 31.032 | 265.750 | 0.963× |
| layernorm_128x8192 / torch | 66.693 | 72.625 | 78.868 | 348.625 | 0.997× |
| residual_layernorm_17x257 / native | 3.505 | 6.875 | 4.269 | 238.333 | 1.054× |
| residual_layernorm_17x257 / torch | 6.564 | 10.125 | 13.465 | 252.792 | 1.074× |
| residual_layernorm_7x1537 / native | 4.345 | 7.000 | 4.712 | 221.625 | 1.000× |
| residual_layernorm_7x1537 / torch | 7.329 | 11.250 | 13.872 | 243.666 | 1.076× |
| residual_layernorm_64x4096 / native | 9.443 | 12.708 | 9.448 | 282.291 | 1.036× |
| residual_layernorm_64x4096 / torch | 19.908 | 24.583 | 26.199 | 247.500 | 1.028× |
| residual_layernorm_1024x4096 / native | 106.511 | 144.542 | 106.731 | 347.417 | 0.975× |
| residual_layernorm_1024x4096 / torch | 251.043 | 281.125 | 270.834 | 529.000 | 1.005× |
| residual_layernorm_128x8192 / native | 25.852 | 32.375 | 28.092 | 280.625 | 0.965× |
| residual_layernorm_128x8192 / torch | 66.921 | 76.708 | 74.681 | 354.625 | 0.995× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| sum_17x257 | 3.346 | 7.570 | 6.125 | 12.125 | 204.334 | 228.750 |
| sum_7x1537 | 4.374 | 6.417 | 8.458 | 10.375 | 223.667 | 231.917 |
| sum_64x4096 | 4.292 | 12.376 | 6.958 | 15.875 | 230.458 | 254.583 |
| sum_1024x4096 | 26.802 | 26.607 | 34.750 | 33.166 | 272.666 | 255.167 |
| sum_128x8192 | 9.422 | 21.221 | 13.334 | 29.958 | 223.500 | 269.500 |
| softmax_17x257 | 3.993 | 51.239 | 7.292 | 63.625 | 210.000 | 324.208 |
| softmax_7x1537 | 4.439 | 62.018 | 6.792 | 66.875 | 224.792 | 427.583 |
| softmax_64x4096 | 7.643 | 68.839 | 10.959 | 77.292 | 250.125 | 387.917 |
| softmax_1024x4096 | 73.979 | 125.014 | 77.292 | 130.625 | 308.416 | 412.000 |
| softmax_128x8192 | 25.340 | 66.898 | 30.667 | 71.625 | 280.958 | 394.375 |
| rmsnorm_17x257 | 3.266 | 4.914 | 5.750 | 11.209 | 193.000 | 218.458 |
| rmsnorm_7x1537 | 4.418 | 5.870 | 6.375 | 9.417 | 200.291 | 213.958 |
| rmsnorm_64x4096 | 8.418 | 9.659 | 11.417 | 13.875 | 213.458 | 334.000 |
| rmsnorm_1024x4096 | 69.876 | 70.421 | 71.792 | 72.042 | 306.792 | 322.834 |
| rmsnorm_128x8192 | 22.205 | 22.426 | 26.750 | 28.166 | 252.166 | 290.375 |
| layernorm_17x257 | 4.199 | 5.163 | 6.750 | 8.500 | 241.917 | 219.083 |
| layernorm_7x1537 | 5.280 | 11.242 | 7.458 | 14.708 | 224.292 | 210.916 |
| layernorm_64x4096 | 9.456 | 28.450 | 12.542 | 37.083 | 246.209 | 310.958 |
| layernorm_1024x4096 | 78.792 | 208.754 | 80.709 | 203.708 | 326.042 | 599.083 |
| layernorm_128x8192 | 29.027 | 67.919 | 36.667 | 70.917 | 265.750 | 348.625 |
| residual_layernorm_17x257 | 3.597 | 7.078 | 6.959 | 10.000 | 238.333 | 252.792 |
| residual_layernorm_7x1537 | 4.323 | 8.441 | 7.000 | 12.708 | 221.625 | 243.666 |
| residual_layernorm_64x4096 | 10.339 | 20.522 | 12.500 | 24.750 | 282.291 | 247.500 |
| residual_layernorm_1024x4096 | 103.869 | 251.749 | 123.458 | 263.375 | 347.417 | 529.000 |
| residual_layernorm_128x8192 | 25.040 | 66.698 | 32.208 | 76.583 | 280.625 | 354.625 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
