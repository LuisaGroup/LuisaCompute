# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 4 | 5.578 | 4.958 | 1.134× [1.053, 1.250] | 32.196 |
| metal / softmax_256x3072 | 4 | 17.306 | 17.054 | 1.016× [0.995, 1.105] | 38.971 |
| metal / softmax_768x6144 | 4 | 56.212 | 56.259 | 1.001× [0.999, 1.041] | 131.443 |
| metal / softmax_64x12289 | 4 | 17.504 | 18.006 | 1.008× [0.939, 1.022] | 35.669 |
| metal / rmsnorm_37x1537 | 4 | 4.711 | 3.899 | 1.207× [1.188, 1.229] | 7.177 |
| metal / rmsnorm_256x3072 | 4 | 17.955 | 16.094 | 1.041× [0.941, 1.269] | 21.404 |
| metal / rmsnorm_768x6144 | 4 | 74.630 | 72.511 | 1.000× [0.994, 1.173] | 98.969 |
| metal / rmsnorm_64x12289 | 4 | 21.829 | 21.544 | 1.066× [0.971, 1.084] | 28.084 |
| metal / layernorm_37x1537 | 4 | 6.733 | 5.495 | 1.210× [1.142, 1.252] | 13.368 |
| metal / layernorm_256x3072 | 4 | 19.928 | 20.010 | 0.999× [0.870, 1.007] | 52.589 |
| metal / layernorm_768x6144 | 4 | 92.286 | 92.776 | 0.995× [0.973, 0.998] | 326.895 |
| metal / layernorm_64x12289 | 4 | 24.723 | 23.159 | 1.067× [1.050, 1.087] | 69.659 |

## Separately sampled GPU execution

Completed command-buffer GPU timestamps, with no encoder hooks or counter attachments. This includes GPU work/gaps inside each command buffer, not CPU encoding or completion notification, and is not individual-kernel time. Each GPU phase uses its own repetition count. Values are medians of per-round p50 times. Speedups are medians of paired round ratios, with min–max ranges, not confidence intervals. Incomplete GPU control pairs withhold statistics; instrumented compute-pass samples remain diagnostics in JSON.

| Backend / case | GPU pairs | Reference GPU µs/op | Candidate GPU µs/op | Paired GPU speedup [range] | Candidate-run Torch GPU µs/op |
|---|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 4/4 | 4.745 | 4.066 | 1.157× [1.118, 1.224] | 18.167 |
| metal / softmax_256x3072 | 4/4 | 16.503 | 13.915 | 1.006× [0.791, 1.780] | 29.322 |
| metal / softmax_768x6144 | 4/4 | 55.539 | 55.559 | 1.000× [0.819, 1.004] | 123.949 |
| metal / softmax_64x12289 | 4/4 | 17.155 | 18.343 | 1.006× [0.874, 1.034] | 30.664 |
| metal / rmsnorm_37x1537 | 4/4 | 4.648 | 3.747 | 1.222× [0.578, 1.260] | 4.736 |
| metal / rmsnorm_256x3072 | 4/4 | 14.437 | 17.211 | 0.921× [0.487, 1.006] | 17.009 |
| metal / rmsnorm_768x6144 | 4/4 | 59.887 | 64.576 | 1.001× [0.866, 1.072] | 84.754 |
| metal / rmsnorm_64x12289 | 4/4 | 18.204 | 18.946 | 1.060× [0.892, 1.454] | 20.164 |
| metal / layernorm_37x1537 | 4/4 | 5.456 | 4.674 | 1.175× [0.508, 1.216] | 7.789 |
| metal / layernorm_256x3072 | 4/4 | 16.398 | 18.052 | 0.991× [0.847, 1.020] | 41.525 |
| metal / layernorm_768x6144 | 4/4 | 82.561 | 77.742 | 1.060× [0.992, 1.407] | 308.701 |
| metal / layernorm_64x12289 | 4/4 | 22.662 | 21.516 | 1.065× [0.916, 1.251] | 55.913 |

## Single-call GPU versus end-to-end dispatch

These are separate phases; host samples are uninstrumented. Do not subtract their medians to estimate CPU cost. Torch remains the recorded eager operator sequence, not a compiled fused graph.

| Backend / case | Candidate GPU µs | Candidate E2E µs | Candidate-run Torch GPU µs | Candidate-run Torch E2E µs |
|---|---:|---:|---:|---:|
| metal / softmax_37x1537 | 6.583 | 227.291 | 65.625 | 342.625 |
| metal / softmax_256x3072 | 18.167 | 239.958 | 48.708 | 308.562 |
| metal / softmax_768x6144 | 58.792 | 271.271 | 137.438 | 410.688 |
| metal / softmax_64x12289 | 21.896 | 241.187 | 64.062 | 343.479 |
| metal / rmsnorm_37x1537 | 6.542 | 206.729 | 8.292 | 212.000 |
| metal / rmsnorm_256x3072 | 18.437 | 254.584 | 21.417 | 262.938 |
| metal / rmsnorm_768x6144 | 79.229 | 276.271 | 96.542 | 310.688 |
| metal / rmsnorm_64x12289 | 20.417 | 249.646 | 24.708 | 260.480 |
| metal / layernorm_37x1537 | 7.187 | 199.062 | 10.375 | 224.541 |
| metal / layernorm_256x3072 | 20.292 | 244.500 | 44.750 | 277.479 |
| metal / layernorm_768x6144 | 80.812 | 283.021 | 279.437 | 518.146 |
| metal / layernorm_64x12289 | 22.854 | 319.812 | 62.625 | 308.438 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
