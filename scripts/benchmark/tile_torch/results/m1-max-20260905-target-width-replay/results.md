# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / sum_17x257 | 4 | 2.580 | 2.606 | 0.990× [0.927, 1.009] | 4.480 |
| metal / sum_64x4096 | 4 | 3.900 | 3.894 | 1.013× [0.917, 1.088] | 15.457 |
| metal / sum_1024x4096 | 4 | 24.756 | 23.866 | 1.045× [1.011, 1.057] | 27.607 |
| metal / sum_7x1537 | 4 | 2.790 | 2.785 | 1.002× [0.996, 1.011] | 6.149 |
| metal / sum_128x8192 | 4 | 8.131 | 8.649 | 0.942× [0.914, 0.979] | 23.193 |
| metal / softmax_17x257 | 4 | 2.711 | 3.244 | 0.839× [0.828, 0.858] | 32.509 |
| metal / softmax_64x4096 | 4 | 8.035 | 7.787 | 1.015× [1.012, 1.068] | 34.446 |
| metal / softmax_1024x4096 | 4 | 69.703 | 60.261 | 1.156× [1.144, 1.159] | 128.715 |
| metal / softmax_7x1537 | 4 | 3.819 | 3.837 | 0.995× [0.981, 1.065] | 32.726 |
| metal / softmax_128x8192 | 4 | 22.197 | 22.374 | 0.990× [0.972, 1.053] | 44.357 |
| metal / rmsnorm_17x257 | 4 | 3.286 | 3.141 | 1.048× [1.039, 1.092] | 6.259 |
| metal / rmsnorm_64x4096 | 4 | 8.699 | 8.668 | 1.011× [0.974, 1.039] | 12.187 |
| metal / rmsnorm_1024x4096 | 4 | 72.401 | 66.326 | 1.092× [1.073, 1.116] | 73.113 |
| metal / rmsnorm_7x1537 | 4 | 4.267 | 4.257 | 1.003× [1.000, 1.010] | 7.106 |
| metal / rmsnorm_128x8192 | 4 | 21.718 | 21.546 | 1.018× [0.976, 1.028] | 25.389 |

## Separately sampled GPU execution

Completed command-buffer GPU timestamps, with no encoder hooks or counter attachments. This includes GPU work/gaps inside each command buffer, not CPU encoding or completion notification, and is not individual-kernel time. Each GPU phase uses its own repetition count. Values are medians of per-round p50 times. Speedups are medians of paired round ratios, with min–max ranges, not confidence intervals. Incomplete GPU control pairs withhold statistics; instrumented compute-pass samples remain diagnostics in JSON.

| Backend / case | GPU pairs | Reference GPU µs/op | Candidate GPU µs/op | Paired GPU speedup [range] | Candidate-run Torch GPU µs/op |
|---|---:|---:|---:|---:|---:|
| metal / sum_17x257 | 4/4 | 2.569 | 2.712 | 0.919× [0.848, 1.131] | 2.731 |
| metal / sum_64x4096 | 4/4 | 4.156 | 4.111 | 1.024× [0.975, 1.042] | 13.144 |
| metal / sum_1024x4096 | 4/4 | 24.837 | 23.625 | 1.051× [1.031, 1.080] | 26.511 |
| metal / sum_7x1537 | 4/4 | 2.917 | 2.875 | 0.995× [0.896, 1.068] | 4.754 |
| metal / sum_128x8192 | 4/4 | 8.462 | 9.063 | 0.936× [0.902, 0.987] | 20.562 |
| metal / softmax_17x257 | 4/4 | 2.695 | 3.363 | 0.796× [0.722, 0.838] | 14.106 |
| metal / softmax_64x4096 | 4/4 | 8.307 | 7.972 | 1.048× [0.991, 1.100] | 22.383 |
| metal / softmax_1024x4096 | 4/4 | 67.543 | 59.174 | 1.141× [1.111, 1.157] | 121.056 |
| metal / softmax_7x1537 | 4/4 | 4.122 | 4.143 | 0.990× [0.986, 1.011] | 18.202 |
| metal / softmax_128x8192 | 4/4 | 22.168 | 22.347 | 0.999× [0.986, 1.024] | 38.803 |
| metal / rmsnorm_17x257 | 4/4 | 3.203 | 3.196 | 0.989× [0.969, 1.076] | 3.484 |
| metal / rmsnorm_64x4096 | 4/4 | 8.841 | 8.795 | 1.006× [0.957, 1.032] | 9.601 |
| metal / rmsnorm_1024x4096 | 4/4 | 70.910 | 64.210 | 1.101× [1.092, 1.132] | 68.802 |
| metal / rmsnorm_7x1537 | 4/4 | 4.330 | 4.034 | 1.037× [1.018, 1.099] | 4.036 |
| metal / rmsnorm_128x8192 | 4/4 | 21.342 | 21.324 | 0.999× [0.990, 1.002] | 23.324 |

## Single-call GPU versus end-to-end dispatch

These are separate phases; host samples are uninstrumented. Do not subtract their medians to estimate CPU cost. Torch remains the recorded eager operator sequence, not a compiled fused graph.

| Backend / case | Candidate GPU µs | Candidate E2E µs | Candidate-run Torch GPU µs | Candidate-run Torch E2E µs |
|---|---:|---:|---:|---:|
| metal / sum_17x257 | 4.750 | 213.438 | 5.187 | 224.458 |
| metal / sum_64x4096 | 6.625 | 220.770 | 15.979 | 265.562 |
| metal / sum_1024x4096 | 30.333 | 263.583 | 33.521 | 271.125 |
| metal / sum_7x1537 | 4.958 | 214.812 | 6.813 | 207.458 |
| metal / sum_128x8192 | 12.104 | 243.062 | 23.604 | 263.708 |
| metal / softmax_17x257 | 6.354 | 207.459 | 77.437 | 363.917 |
| metal / softmax_64x4096 | 10.521 | 228.792 | 60.646 | 339.271 |
| metal / softmax_1024x4096 | 91.708 | 320.208 | 129.625 | 411.458 |
| metal / softmax_7x1537 | 6.583 | 200.250 | 100.771 | 338.000 |
| metal / softmax_128x8192 | 27.229 | 241.521 | 69.646 | 353.250 |
| metal / rmsnorm_17x257 | 6.083 | 203.458 | 6.979 | 215.812 |
| metal / rmsnorm_64x4096 | 11.521 | 233.959 | 12.312 | 237.938 |
| metal / rmsnorm_1024x4096 | 93.417 | 316.479 | 92.458 | 318.855 |
| metal / rmsnorm_7x1537 | 6.625 | 200.792 | 7.500 | 221.000 |
| metal / rmsnorm_128x8192 | 26.500 | 276.917 | 31.875 | 270.688 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
