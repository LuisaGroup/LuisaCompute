# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T08:37:22.147383+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sum_17x257 | 1×257×1 / 2 | 0 | 3.579 | 5.466 | 4.281 | 5.946 | 0.65× | 190.459 | 210.833 |
| metal | sum_7x1537 | 1×1537×1 / 2 | 0 | 3.369 | 7.407 | 3.987 | 10.840 | 0.45× | 229.917 | 201.041 |
| metal | sum_64x4096 | 1×4096×1 / 2 | 0 | 4.858 | 15.326 | 5.037 | 15.999 | 0.32× | 222.208 | 280.875 |
| metal | sum_1024x4096 | 1×4096×1 / 2 | 0 | 28.456 | 30.407 | 30.483 | 30.892 | 0.94× | 266.791 | 304.375 |
| metal | sum_128x8192 | 1×8192×1 / 2 | 0 | 9.266 | 24.431 | 9.386 | 25.071 | 0.38× | 247.042 | 255.167 |
| metal | softmax_17x257 | 1×257×1 / 2 | 0 | 3.603 | 31.952 | 3.958 | 32.285 | 0.11× | 199.667 | 351.000 |
| metal | softmax_7x1537 | 1×1537×1 / 2 | 0 | 3.795 | 32.260 | 3.931 | 37.074 | 0.12× | 213.083 | 426.333 |
| metal | softmax_64x4096 | 1×4096×1 / 2 | 0 | 8.999 | 35.796 | 13.491 | 37.644 | 0.25× | 233.833 | 367.209 |
| metal | softmax_1024x4096 | 1×4096×1 / 2 | 0 | 55.606 | 135.464 | 57.905 | 137.273 | 0.41× | 303.000 | 413.208 |
| metal | softmax_128x8192 | 1×8192×1 / 2 | 0 | 21.478 | 48.117 | 23.643 | 56.972 | 0.45× | 254.000 | 355.417 |
| metal | rmsnorm_17x257 | 1×257×1 / 2 | 0 | 2.993 | 6.174 | 3.023 | 6.252 | 0.48× | 202.958 | 240.708 |
| metal | rmsnorm_7x1537 | 1×1537×1 / 2 | 0 | 4.199 | 7.295 | 4.547 | 7.600 | 0.58× | 243.292 | 245.250 |
| metal | rmsnorm_64x4096 | 1×4096×1 / 2 | 0 | 8.453 | 15.316 | 8.692 | 15.420 | 0.55× | 249.583 | 240.625 |
| metal | rmsnorm_1024x4096 | 1×4096×1 / 2 | 0 | 57.862 | 76.870 | 58.466 | 79.210 | 0.75× | 472.792 | 337.916 |
| metal | rmsnorm_128x8192 | 1×8192×1 / 2 | 0 | 22.252 | 26.656 | 23.275 | 28.691 | 0.83× | 286.750 | 260.000 |
| metal | layernorm_17x257 | 1×257×1 / 2 | 0 | 3.329 | 12.256 | 3.559 | 13.700 | 0.27× | 205.625 | 229.917 |
| metal | layernorm_7x1537 | 1×1537×1 / 2 | 0 | 4.576 | 11.211 | 4.817 | 11.467 | 0.41× | 212.000 | 224.250 |
| metal | layernorm_64x4096 | 1×4096×1 / 2 | 0 | 9.184 | 29.864 | 9.857 | 38.228 | 0.31× | 240.417 | 240.375 |
| metal | layernorm_1024x4096 | 1×4096×1 / 2 | 0 | 67.991 | 223.762 | 70.550 | 225.136 | 0.30× | 301.000 | 536.333 |
| metal | layernorm_128x8192 | 1×8192×1 / 2 | 0 | 25.033 | 76.547 | 27.327 | 79.721 | 0.33× | 241.333 | 346.709 |
| metal | residual_layernorm_17x257 | 1×257×1 / 2 | 0 | 3.323 | 11.746 | 3.407 | 12.315 | 0.28× | 234.084 | 261.666 |
| metal | residual_layernorm_7x1537 | 1×1537×1 / 2 | 0 | 4.701 | 14.427 | 4.815 | 15.302 | 0.33× | 215.667 | 277.875 |
| metal | residual_layernorm_64x4096 | 1×4096×1 / 2 | 0 | 9.656 | 27.170 | 9.850 | 27.548 | 0.36× | 250.750 | 266.792 |
| metal | residual_layernorm_1024x4096 | 1×4096×1 / 2 | 0 | 106.357 | 268.173 | 106.996 | 268.425 | 0.40× | 332.375 | 537.250 |
| metal | residual_layernorm_128x8192 | 1×8192×1 / 2 | 0 | 27.308 | 73.504 | 27.573 | 78.209 | 0.37× | 240.875 | 332.417 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sum_17x257 | 0.053 | 40.149 | 1.018 | 4.109 | 1.483 | 63.805 | 0.257 | 0.368 |
| metal / sum_7x1537 | 0.052 | 40.905 | 1.161 | 0.413 | 1.051 | 0.325 | 0.280 | 0.294 |
| metal / sum_64x4096 | 0.053 | 40.260 | 1.282 | 0.481 | 6.341 | 3.694 | 0.305 | 0.236 |
| metal / sum_1024x4096 | 0.053 | 40.674 | 6.313 | 43.684 | 7.922 | 0.356 | 0.284 | 0.288 |
| metal / sum_128x8192 | 0.049 | 41.230 | 2.228 | 0.940 | 6.840 | 0.937 | 0.247 | 0.231 |
| metal / softmax_17x257 | 0.067 | 49.156 | 1.064 | 0.680 | 59.260 | 17.576 | 0.265 | 0.321 |
| metal / softmax_7x1537 | 0.082 | 53.148 | 1.031 | 0.467 | 58.846 | 4.905 | 0.290 | 0.288 |
| metal / softmax_64x4096 | 0.062 | 54.381 | 1.206 | 0.763 | 65.996 | 3.643 | 0.594 | 0.399 |
| metal / softmax_1024x4096 | 0.071 | 52.400 | 6.153 | 16.621 | 8.215 | 3.002 | 4.043 | 0.597 |
| metal / softmax_128x8192 | 0.061 | 52.692 | 2.247 | 0.557 | 73.100 | 2.289 | 1.079 | 0.521 |
| metal / rmsnorm_17x257 | 0.058 | 54.134 | 1.490 | 1.555 | 57.265 | 0.803 | 0.265 | 0.284 |
| metal / rmsnorm_7x1537 | 0.071 | 52.381 | 1.713 | 0.837 | 55.902 | 0.238 | 0.273 | 0.274 |
| metal / rmsnorm_64x4096 | 0.072 | 54.203 | 1.523 | 1.038 | 63.437 | 7.561 | 1.274 | 0.490 |
| metal / rmsnorm_1024x4096 | 0.060 | 54.761 | 6.425 | 17.358 | 8.036 | 7.595 | 3.693 | 0.617 |
| metal / rmsnorm_128x8192 | 0.069 | 57.084 | 2.580 | 1.584 | 67.345 | 0.660 | 1.079 | 0.512 |
| metal / layernorm_17x257 | 0.080 | 62.009 | 1.609 | 1.621 | 59.801 | 0.446 | 0.453 | 0.323 |
| metal / layernorm_7x1537 | 0.068 | 60.660 | 1.338 | 0.662 | 57.797 | 0.312 | 0.245 | 0.317 |
| metal / layernorm_64x4096 | 0.074 | 64.476 | 1.638 | 0.685 | 66.768 | 0.270 | 0.480 | 0.342 |
| metal / layernorm_1024x4096 | 0.078 | 62.583 | 6.597 | 17.346 | 6.231 | 0.899 | 3.460 | 1.383 |
| metal / layernorm_128x8192 | 0.084 | 62.319 | 2.679 | 1.556 | 76.178 | 0.605 | 0.994 | 0.519 |
| metal / residual_layernorm_17x257 | 0.075 | 56.801 | 1.319 | 0.741 | 1.409 | 0.945 | 0.243 | 0.308 |
| metal / residual_layernorm_7x1537 | 0.075 | 58.151 | 1.909 | 9.549 | 1.041 | 0.287 | 0.402 | 0.313 |
| metal / residual_layernorm_64x4096 | 0.075 | 59.922 | 2.006 | 0.899 | 6.160 | 0.626 | 0.604 | 0.338 |
| metal / residual_layernorm_1024x4096 | 0.070 | 58.249 | 8.736 | 18.387 | 5.641 | 0.814 | 2.371 | 0.663 |
| metal / residual_layernorm_128x8192 | 0.081 | 58.483 | 2.887 | 1.133 | 6.950 | 0.471 | 0.962 | 0.446 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| sum_17x257 / native | 3.143 | 6.042 | 3.579 | 190.459 | 1.043× |
| sum_17x257 / torch | 4.946 | 16.375 | 5.466 | 210.833 | 1.013× |
| sum_7x1537 / native | 3.359 | 6.000 | 3.369 | 229.917 | 0.986× |
| sum_7x1537 / torch | 5.473 | 10.875 | 7.407 | 201.041 | 1.099× |
| sum_64x4096 / native | 4.810 | 7.208 | 4.858 | 222.208 | 0.923× |
| sum_64x4096 / torch | 13.125 | 15.750 | 15.326 | 280.875 | 1.002× |
| sum_1024x4096 / native | 27.663 | 34.333 | 28.456 | 266.791 | 0.977× |
| sum_1024x4096 / torch | 26.961 | 34.000 | 30.407 | 304.375 | 0.992× |
| sum_128x8192 / native | 9.122 | 12.542 | 9.266 | 247.042 | 0.950× |
| sum_128x8192 / torch | 20.574 | 25.083 | 24.431 | 255.167 | 1.036× |
| softmax_17x257 / native | 3.785 | 6.917 | 3.603 | 199.667 | 0.943× |
| softmax_17x257 / torch | 19.171 | 95.542 | 31.952 | 351.000 | 5.082× |
| softmax_7x1537 / native | 4.169 | 6.500 | 3.795 | 213.083 | 0.977× |
| softmax_7x1537 / torch | 18.021 | 87.625 | 32.260 | 426.333 | 4.335× |
| softmax_64x4096 / native | 7.486 | 11.708 | 8.999 | 233.833 | 0.956× |
| softmax_64x4096 / torch | 22.424 | 63.833 | 35.796 | 367.209 | 3.231× |
| softmax_1024x4096 / native | 53.826 | 54.417 | 55.606 | 303.000 | 0.983× |
| softmax_1024x4096 / torch | 121.811 | 131.167 | 135.464 | 413.208 | 1.200× |
| softmax_128x8192 / native | 20.049 | 24.667 | 21.478 | 254.000 | 1.019× |
| softmax_128x8192 / torch | 38.801 | 73.167 | 48.117 | 355.417 | 2.090× |
| rmsnorm_17x257 / native | 3.021 | 5.875 | 2.993 | 202.958 | 0.988× |
| rmsnorm_17x257 / torch | 3.288 | 7.083 | 6.174 | 240.708 | 1.000× |
| rmsnorm_7x1537 / native | 4.245 | 5.917 | 4.199 | 243.292 | 0.968× |
| rmsnorm_7x1537 / torch | 4.533 | 8.125 | 7.295 | 245.250 | 0.870× |
| rmsnorm_64x4096 / native | 8.247 | 11.042 | 8.453 | 249.583 | 1.050× |
| rmsnorm_64x4096 / torch | 12.162 | 18.125 | 15.316 | 240.625 | 1.095× |
| rmsnorm_1024x4096 / native | 56.928 | 103.000 | 57.862 | 472.792 | 1.003× |
| rmsnorm_1024x4096 / torch | 70.344 | 72.125 | 76.870 | 337.916 | 1.008× |
| rmsnorm_128x8192 / native | 20.939 | 24.750 | 22.252 | 286.750 | 0.967× |
| rmsnorm_128x8192 / torch | 22.846 | 28.500 | 26.656 | 260.000 | 0.978× |
| layernorm_17x257 / native | 3.319 | 5.625 | 3.329 | 205.625 | 1.014× |
| layernorm_17x257 / torch | 6.182 | 10.625 | 12.256 | 229.917 | 1.051× |
| layernorm_7x1537 / native | 4.746 | 6.792 | 4.576 | 212.000 | 0.987× |
| layernorm_7x1537 / torch | 6.795 | 9.417 | 11.211 | 224.250 | 1.042× |
| layernorm_64x4096 / native | 9.486 | 11.875 | 9.184 | 240.417 | 1.013× |
| layernorm_64x4096 / torch | 22.180 | 26.167 | 29.864 | 240.375 | 0.989× |
| layernorm_1024x4096 / native | 65.624 | 80.125 | 67.991 | 301.000 | 1.005× |
| layernorm_1024x4096 / torch | 209.550 | 203.750 | 223.762 | 536.333 | 0.997× |
| layernorm_128x8192 / native | 24.044 | 28.333 | 25.033 | 241.333 | 1.023× |
| layernorm_128x8192 / torch | 67.859 | 72.917 | 76.547 | 346.709 | 1.011× |
| residual_layernorm_17x257 / native | 3.435 | 5.833 | 3.323 | 234.084 | 0.931× |
| residual_layernorm_17x257 / torch | 6.845 | 11.875 | 11.746 | 261.666 | 0.901× |
| residual_layernorm_7x1537 / native | 4.363 | 6.833 | 4.701 | 215.667 | 1.000× |
| residual_layernorm_7x1537 / torch | 8.538 | 12.833 | 14.427 | 277.875 | 0.998× |
| residual_layernorm_64x4096 / native | 10.014 | 12.750 | 9.656 | 250.750 | 0.964× |
| residual_layernorm_64x4096 / torch | 20.594 | 28.083 | 27.170 | 266.792 | 0.995× |
| residual_layernorm_1024x4096 / native | 99.396 | 107.167 | 106.357 | 332.375 | 0.990× |
| residual_layernorm_1024x4096 / torch | 250.470 | 250.250 | 268.173 | 537.250 | 1.005× |
| residual_layernorm_128x8192 / native | 26.127 | 32.417 | 27.308 | 240.875 | 1.031× |
| residual_layernorm_128x8192 / torch | 62.887 | 75.750 | 73.504 | 332.417 | 1.001× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| sum_17x257 | 3.121 | 5.013 | 5.917 | 12.959 | 190.459 | 210.833 |
| sum_7x1537 | 3.291 | 6.531 | 6.042 | 10.209 | 229.917 | 201.041 |
| sum_64x4096 | 4.395 | 12.770 | 7.375 | 15.667 | 222.208 | 280.875 |
| sum_1024x4096 | 27.019 | 26.734 | 34.125 | 33.791 | 266.791 | 304.375 |
| sum_128x8192 | 8.963 | 21.125 | 12.584 | 23.917 | 247.042 | 255.167 |
| softmax_17x257 | 3.578 | 80.445 | 7.958 | 96.041 | 199.667 | 351.000 |
| softmax_7x1537 | 4.108 | 65.001 | 6.000 | 80.166 | 213.083 | 426.333 |
| softmax_64x4096 | 7.670 | 60.813 | 11.500 | 57.667 | 233.833 | 367.209 |
| softmax_1024x4096 | 52.878 | 127.320 | 57.417 | 136.167 | 303.000 | 413.208 |
| softmax_128x8192 | 20.070 | 66.432 | 25.041 | 68.125 | 254.000 | 355.417 |
| rmsnorm_17x257 | 3.016 | 3.126 | 4.958 | 8.375 | 202.958 | 240.708 |
| rmsnorm_7x1537 | 4.127 | 4.013 | 6.084 | 7.333 | 243.292 | 245.250 |
| rmsnorm_64x4096 | 8.662 | 12.082 | 11.042 | 18.333 | 249.583 | 240.625 |
| rmsnorm_1024x4096 | 55.991 | 70.637 | 77.792 | 71.625 | 472.792 | 337.916 |
| rmsnorm_128x8192 | 20.214 | 22.770 | 25.833 | 28.125 | 286.750 | 260.000 |
| layernorm_17x257 | 3.372 | 6.457 | 5.791 | 11.875 | 205.625 | 229.917 |
| layernorm_7x1537 | 4.738 | 6.867 | 7.667 | 11.042 | 212.000 | 224.250 |
| layernorm_64x4096 | 9.592 | 21.471 | 11.833 | 29.792 | 240.417 | 240.375 |
| layernorm_1024x4096 | 65.771 | 209.185 | 64.625 | 202.375 | 301.000 | 536.333 |
| layernorm_128x8192 | 23.890 | 68.545 | 28.500 | 72.667 | 241.333 | 346.709 |
| residual_layernorm_17x257 | 3.201 | 6.130 | 5.750 | 10.167 | 234.084 | 261.666 |
| residual_layernorm_7x1537 | 4.592 | 8.413 | 6.625 | 11.291 | 215.667 | 277.875 |
| residual_layernorm_64x4096 | 9.167 | 20.173 | 12.625 | 24.834 | 250.750 | 266.792 |
| residual_layernorm_1024x4096 | 98.798 | 251.697 | 105.792 | 265.834 | 332.375 | 537.250 |
| residual_layernorm_128x8192 | 26.012 | 62.942 | 32.208 | 70.292 | 240.875 | 332.417 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
