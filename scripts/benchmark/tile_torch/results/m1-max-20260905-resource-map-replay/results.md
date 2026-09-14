# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / softmax_23x769 | 4 | 3.255 | 3.108 | 1.050× [1.023, 1.063] | 31.743 |
| metal / softmax_128x2048 | 4 | 7.801 | 7.357 | 1.068× [1.052, 1.126] | 35.144 |
| metal / softmax_1024x4096 | 4 | 60.162 | 50.156 | 1.199× [1.193, 1.231] | 131.108 |
| metal / softmax_128x8193 | 4 | 22.821 | 19.788 | 1.145× [1.123, 1.219] | 44.802 |
| metal / rmsnorm_23x769 | 4 | 3.524 | 3.188 | 1.105× [1.098, 1.109] | 6.804 |
| metal / rmsnorm_128x2048 | 4 | 8.432 | 7.986 | 1.060× [1.019, 1.110] | 11.847 |
| metal / rmsnorm_1024x4096 | 4 | 65.918 | 53.938 | 1.221× [1.211, 1.222] | 74.218 |
| metal / rmsnorm_128x8193 | 4 | 22.029 | 21.062 | 1.042× [1.018, 1.063] | 26.017 |
| metal / layernorm_23x769 | 4 | 4.286 | 4.256 | 1.010× [0.985, 1.047] | 9.422 |
| metal / layernorm_128x2048 | 4 | 9.199 | 8.223 | 1.117× [1.100, 1.169] | 21.059 |
| metal / layernorm_1024x4096 | 4 | 77.521 | 62.284 | 1.248× [1.204, 1.289] | 213.604 |
| metal / layernorm_128x8193 | 4 | 26.919 | 24.604 | 1.097× [1.088, 1.114] | 75.285 |

## Separately sampled GPU execution

Completed command-buffer GPU timestamps, with no encoder hooks or counter attachments. This includes GPU work/gaps inside each command buffer, not CPU encoding or completion notification, and is not individual-kernel time. Each GPU phase uses its own repetition count. Values are medians of per-round p50 times. Speedups are medians of paired round ratios, with min–max ranges, not confidence intervals. Incomplete GPU control pairs withhold statistics; instrumented compute-pass samples remain diagnostics in JSON.

| Backend / case | GPU pairs | Reference GPU µs/op | Candidate GPU µs/op | Paired GPU speedup [range] | Candidate-run Torch GPU µs/op |
|---|---:|---:|---:|---:|---:|
| metal / softmax_23x769 | 4/4 | 3.297 | 3.147 | 1.048× [0.992, 1.105] | 18.256 |
| metal / softmax_128x2048 | 4/4 | 8.479 | 7.484 | 1.141× [0.980, 1.169] | 27.419 |
| metal / softmax_1024x4096 | 4/4 | 59.162 | 49.179 | 1.200× [1.195, 1.210] | 122.653 |
| metal / softmax_128x8193 | 4/4 | 22.560 | 19.436 | 1.152× [1.056, 1.244] | 39.158 |
| metal / rmsnorm_23x769 | 4/4 | 3.787 | 3.138 | 1.211× [0.967, 1.246] | 3.757 |
| metal / rmsnorm_128x2048 | 4/4 | 8.484 | 8.139 | 1.035× [1.000, 1.084] | 8.998 |
| metal / rmsnorm_1024x4096 | 4/4 | 64.211 | 52.826 | 1.214× [1.198, 1.221] | 69.742 |
| metal / rmsnorm_128x8193 | 4/4 | 21.708 | 20.725 | 1.046× [1.037, 1.057] | 23.031 |
| metal / layernorm_23x769 | 4/4 | 4.111 | 4.328 | 0.924× [0.909, 1.023] | 5.374 |
| metal / layernorm_128x2048 | 4/4 | 9.220 | 8.185 | 1.136× [1.058, 1.178] | 16.535 |
| metal / layernorm_1024x4096 | 4/4 | 75.660 | 61.316 | 1.234× [1.210, 1.240] | 205.799 |
| metal / layernorm_128x8193 | 4/4 | 26.480 | 24.276 | 1.095× [1.069, 1.098] | 69.610 |

## Single-call GPU versus end-to-end dispatch

These are separate phases; host samples are uninstrumented. Do not subtract their medians to estimate CPU cost. Torch remains the recorded eager operator sequence, not a compiled fused graph.

| Backend / case | Candidate GPU µs | Candidate E2E µs | Candidate-run Torch GPU µs | Candidate-run Torch E2E µs |
|---|---:|---:|---:|---:|
| metal / softmax_23x769 | 5.896 | 199.042 | 63.104 | 341.625 |
| metal / softmax_128x2048 | 10.125 | 233.250 | 60.479 | 400.188 |
| metal / softmax_1024x4096 | 55.729 | 293.729 | 130.271 | 416.478 |
| metal / softmax_128x8193 | 23.854 | 250.250 | 70.771 | 350.792 |
| metal / rmsnorm_23x769 | 5.562 | 215.084 | 7.313 | 227.834 |
| metal / rmsnorm_128x2048 | 10.813 | 229.375 | 12.521 | 265.978 |
| metal / rmsnorm_1024x4096 | 71.708 | 303.354 | 79.979 | 323.521 |
| metal / rmsnorm_128x8193 | 25.167 | 247.938 | 28.792 | 275.355 |
| metal / layernorm_23x769 | 7.187 | 214.062 | 8.625 | 239.521 |
| metal / layernorm_128x2048 | 11.062 | 227.834 | 18.750 | 273.396 |
| metal / layernorm_1024x4096 | 73.396 | 297.042 | 203.646 | 468.771 |
| metal / layernorm_128x8193 | 28.771 | 249.666 | 80.292 | 313.062 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
