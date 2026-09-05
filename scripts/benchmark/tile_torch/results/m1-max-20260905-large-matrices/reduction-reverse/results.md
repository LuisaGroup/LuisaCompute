# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T16:16:47.877747+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_16384x4096 | 1×4096×1 / 2 | 0 | 1620.400 | 1848.708 | 1637.436 | 1872.609 | 0.88× | 1801.500 | 2105.334 |
| metal | softmax_8192x4096 | 1×4096×1 / 2 | 0 | 811.046 | 953.910 | 822.334 | 967.734 | 0.85× | 1042.209 | 1308.792 |
| metal | softmax_4096x8192 | 1×8192×1 / 2 | 0 | 813.569 | 1064.470 | 820.463 | 1072.234 | 0.76× | 1003.958 | 1326.917 |
| metal | softmax_1024x16384 | 1×16384×1 / 2 | 0 | 400.674 | 605.212 | 404.108 | 613.333 | 0.66× | 644.208 | 1098.459 |
| metal | softmax_1024x8192 | 1×8192×1 / 2 | 0 | 156.132 | 269.728 | 156.971 | 273.778 | 0.58× | 415.500 | 586.625 |
| metal | softmax_37x8191 | 1×8191×1 / 2 | 0 | 12.429 | 35.401 | 12.616 | 35.668 | 0.35× | 232.333 | 345.875 |
| metal | rmsnorm_16384x4096 | 1×4096×1 / 2 | 0 | 1612.747 | 1655.492 | 1621.516 | 1664.597 | 0.97× | 1764.625 | 1894.000 |
| metal | rmsnorm_8192x4096 | 1×4096×1 / 2 | 0 | 818.186 | 835.513 | 821.603 | 838.546 | 0.98× | 980.833 | 1042.750 |
| metal | rmsnorm_4096x8192 | 1×8192×1 / 2 | 0 | 820.112 | 829.610 | 822.595 | 837.991 | 0.99× | 1074.667 | 1093.000 |
| metal | rmsnorm_1024x16384 | 1×16384×1 / 2 | 0 | 410.066 | 426.315 | 416.583 | 431.802 | 0.96× | 617.875 | 699.958 |
| metal | rmsnorm_1024x8192 | 1×8192×1 / 2 | 0 | 159.996 | 170.140 | 160.925 | 171.948 | 0.94× | 394.667 | 532.375 |
| metal | rmsnorm_37x8191 | 1×8191×1 / 2 | 0 | 12.264 | 17.263 | 12.431 | 18.054 | 0.71× | 299.958 | 273.542 |
| metal | layernorm_16384x4096 | 1×4096×1 / 2 | 0 | 1614.695 | 3301.366 | 1629.015 | 3404.017 | 0.49× | 1779.125 | 3559.333 |
| metal | layernorm_8192x4096 | 1×4096×1 / 2 | 0 | 808.017 | 1631.546 | 827.006 | 1646.679 | 0.50× | 1071.667 | 1880.750 |
| metal | layernorm_4096x8192 | 1×8192×1 / 2 | 0 | 813.069 | 2072.771 | 827.550 | 2080.570 | 0.39× | 1062.541 | 2319.417 |
| metal | layernorm_1024x16384 | 1×16384×1 / 2 | 0 | 425.276 | 1348.301 | 428.515 | 1362.908 | 0.32× | 668.875 | 1560.084 |
| metal | layernorm_1024x8192 | 1×8192×1 / 2 | 0 | 165.391 | 527.129 | 166.718 | 529.503 | 0.31× | 430.958 | 713.291 |
| metal | layernorm_37x8191 | 1×8191×1 / 2 | 0 | 14.093 | 42.031 | 14.279 | 42.418 | 0.34× | 249.917 | 296.666 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_16384x4096 | 0.077 | 58.917 | 78.784 | 49.845 | 87.629 | 52.855 | 56.776 | 6.304 |
| metal / softmax_8192x4096 | 0.061 | 56.518 | 38.692 | 8.248 | 12.367 | 4.082 | 29.847 | 4.302 |
| metal / softmax_4096x8192 | 0.066 | 52.911 | 37.492 | 20.636 | 15.300 | 7.239 | 31.004 | 4.788 |
| metal / softmax_1024x16384 | 0.068 | 58.461 | 18.766 | 3.030 | 12.965 | 2.996 | 11.624 | 2.756 |
| metal / softmax_1024x8192 | 0.059 | 53.618 | 11.129 | 15.295 | 10.719 | 3.870 | 8.626 | 1.011 |
| metal / softmax_37x8191 | 0.067 | 61.032 | 1.312 | 0.774 | 72.020 | 5.054 | 0.713 | 0.377 |
| metal / rmsnorm_16384x4096 | 0.071 | 54.123 | 69.912 | 23.737 | 17.687 | 78.442 | 47.553 | 6.538 |
| metal / rmsnorm_8192x4096 | 0.070 | 56.937 | 37.049 | 9.264 | 12.981 | 1.783 | 29.543 | 5.554 |
| metal / rmsnorm_4096x8192 | 0.063 | 55.092 | 36.072 | 17.312 | 13.754 | 8.457 | 24.524 | 5.364 |
| metal / rmsnorm_1024x16384 | 0.076 | 58.616 | 18.655 | 2.520 | 14.425 | 0.727 | 12.152 | 2.444 |
| metal / rmsnorm_1024x8192 | 0.064 | 54.521 | 10.805 | 19.876 | 6.165 | 9.716 | 6.952 | 1.182 |
| metal / rmsnorm_37x8191 | 0.121 | 61.704 | 1.746 | 1.749 | 6.005 | 0.247 | 0.556 | 0.331 |
| metal / layernorm_16384x4096 | 0.079 | 67.168 | 76.950 | 24.129 | 17.491 | 20.588 | 60.241 | 6.097 |
| metal / layernorm_8192x4096 | 0.085 | 68.213 | 40.727 | 8.864 | 10.466 | 6.301 | 26.917 | 7.191 |
| metal / layernorm_4096x8192 | 0.093 | 63.612 | 37.767 | 17.905 | 16.112 | 16.984 | 23.648 | 4.716 |
| metal / layernorm_1024x16384 | 0.090 | 69.042 | 18.097 | 2.899 | 13.713 | 2.134 | 13.285 | 3.431 |
| metal / layernorm_1024x8192 | 0.073 | 64.667 | 12.249 | 17.489 | 10.575 | 11.869 | 10.641 | 1.310 |
| metal / layernorm_37x8191 | 0.085 | 71.272 | 1.584 | 0.854 | 7.015 | 0.339 | 0.470 | 0.401 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| softmax_16384x4096 / native | 1579.250 | 1550.417 | 1620.400 | 1801.500 | 1.004× |
| softmax_16384x4096 / torch | 1746.515 | 1616.167 | 1848.708 | 2105.334 | 0.997× |
| softmax_8192x4096 / native | 791.883 | 741.083 | 811.046 | 1042.209 | 0.993× |
| softmax_8192x4096 / torch | 892.630 | 823.417 | 953.910 | 1308.792 | 1.015× |
| softmax_4096x8192 / native | 790.384 | 756.167 | 813.569 | 1003.958 | 1.003× |
| softmax_4096x8192 / torch | 1005.225 | 953.333 | 1064.470 | 1326.917 | 0.998× |
| softmax_1024x16384 / native | 398.671 | 372.583 | 400.674 | 644.208 | 1.001× |
| softmax_1024x16384 / torch | 574.360 | 577.000 | 605.212 | 1098.459 | 1.030× |
| softmax_1024x8192 / native | 151.426 | 153.042 | 156.132 | 415.500 | 1.011× |
| softmax_1024x8192 / torch | 254.767 | 244.875 | 269.728 | 586.625 | 1.110× |
| softmax_37x8191 / native | 11.077 | 14.958 | 12.429 | 232.333 | 0.999× |
| softmax_37x8191 / torch | 27.438 | 66.917 | 35.401 | 345.875 | 2.863× |
| rmsnorm_16384x4096 / native | 1599.977 | 1485.375 | 1612.747 | 1764.625 | 0.995× |
| rmsnorm_16384x4096 / torch | 1609.897 | 1496.667 | 1655.492 | 1894.000 | 1.001× |
| rmsnorm_8192x4096 / native | 793.517 | 743.833 | 818.186 | 980.833 | 0.998× |
| rmsnorm_8192x4096 / torch | 802.545 | 750.250 | 835.513 | 1042.750 | 1.003× |
| rmsnorm_4096x8192 / native | 793.130 | 740.125 | 820.112 | 1074.667 | 1.004× |
| rmsnorm_4096x8192 / torch | 798.693 | 755.417 | 829.610 | 1093.000 | 1.021× |
| rmsnorm_1024x16384 / native | 406.567 | 374.417 | 410.066 | 617.875 | 0.995× |
| rmsnorm_1024x16384 / torch | 415.215 | 387.083 | 426.315 | 699.958 | 1.004× |
| rmsnorm_1024x8192 / native | 154.504 | 152.875 | 159.996 | 394.667 | 1.004× |
| rmsnorm_1024x8192 / torch | 162.822 | 180.875 | 170.140 | 532.375 | 0.997× |
| rmsnorm_37x8191 / native | 11.876 | 14.750 | 12.264 | 299.958 | 0.982× |
| rmsnorm_37x8191 / torch | 13.613 | 16.375 | 17.263 | 273.542 | 0.929× |
| layernorm_16384x4096 / native | 1597.094 | 1486.208 | 1614.695 | 1779.125 | 0.998× |
| layernorm_16384x4096 / torch | 3168.398 | 3676.875 | 3301.366 | 3559.333 | 0.997× |
| layernorm_8192x4096 / native | 797.558 | 746.875 | 808.017 | 1071.667 | 0.996× |
| layernorm_8192x4096 / torch | 1588.868 | 1484.083 | 1631.546 | 1880.750 | 1.004× |
| layernorm_4096x8192 / native | 802.535 | 755.208 | 813.069 | 1062.541 | 1.002× |
| layernorm_4096x8192 / torch | 2031.924 | 1880.750 | 2072.771 | 2319.417 | 1.002× |
| layernorm_1024x16384 / native | 419.659 | 406.750 | 425.276 | 668.875 | 0.996× |
| layernorm_1024x16384 / torch | 1332.201 | 1269.375 | 1348.301 | 1560.084 | 0.999× |
| layernorm_1024x8192 / native | 161.676 | 175.500 | 165.391 | 430.958 | 0.999× |
| layernorm_1024x8192 / torch | 509.042 | 483.458 | 527.129 | 713.291 | 1.000× |
| layernorm_37x8191 / native | 12.482 | 16.292 | 14.093 | 249.917 | 1.002× |
| layernorm_37x8191 / torch | 34.523 | 38.125 | 42.031 | 296.666 | 1.018× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| softmax_16384x4096 | 1590.512 | 1741.604 | 1553.166 | 1620.833 | 1801.500 | 2105.334 |
| softmax_8192x4096 | 790.544 | 904.513 | 733.542 | 835.709 | 1042.209 | 1308.792 |
| softmax_4096x8192 | 792.758 | 996.862 | 755.000 | 960.167 | 1003.958 | 1326.917 |
| softmax_1024x16384 | 400.034 | 576.821 | 373.917 | 538.958 | 644.208 | 1098.459 |
| softmax_1024x8192 | 152.950 | 261.739 | 154.208 | 254.375 | 415.500 | 586.625 |
| softmax_37x8191 | 11.022 | 63.787 | 14.959 | 63.833 | 232.333 | 345.875 |
| rmsnorm_16384x4096 | 1594.357 | 1610.722 | 1491.917 | 1500.708 | 1764.625 | 1894.000 |
| rmsnorm_8192x4096 | 794.872 | 804.669 | 747.583 | 741.875 | 980.833 | 1042.750 |
| rmsnorm_4096x8192 | 795.635 | 815.287 | 736.542 | 763.750 | 1074.667 | 1093.000 |
| rmsnorm_1024x16384 | 405.575 | 417.419 | 379.750 | 389.125 | 617.875 | 699.958 |
| rmsnorm_1024x8192 | 155.465 | 163.302 | 163.584 | 169.042 | 394.667 | 532.375 |
| rmsnorm_37x8191 | 11.910 | 12.355 | 14.666 | 16.167 | 299.958 | 273.542 |
| layernorm_16384x4096 | 1590.984 | 3180.690 | 1487.375 | 2955.292 | 1779.125 | 3559.333 |
| layernorm_8192x4096 | 794.106 | 1590.963 | 753.125 | 1484.125 | 1071.667 | 1880.750 |
| layernorm_4096x8192 | 804.301 | 2032.389 | 750.042 | 1880.000 | 1062.541 | 2319.417 |
| layernorm_1024x16384 | 418.425 | 1332.282 | 395.208 | 1263.750 | 668.875 | 1560.084 |
| layernorm_1024x8192 | 161.288 | 509.576 | 171.125 | 478.583 | 430.958 | 713.291 |
| layernorm_37x8191 | 12.493 | 35.003 | 16.208 | 38.042 | 249.917 | 296.666 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
