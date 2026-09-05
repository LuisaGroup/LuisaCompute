# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 4 | 4.405 | 5.598 | 0.852× [0.688, 0.949] | 30.727 |
| metal / softmax_256x3072 | 4 | 17.052 | 17.824 | 0.962× [0.880, 0.968] | 39.328 |
| metal / softmax_768x6144 | 4 | 72.403 | 77.972 | 0.917× [0.823, 0.965] | 160.538 |
| metal / softmax_1024x4096 | 4 | 63.987 | 65.017 | 0.972× [0.958, 1.010] | 148.773 |
| metal / rmsnorm_37x1537 | 4 | 4.919 | 5.516 | 0.911× [0.883, 0.928] | 8.723 |
| metal / rmsnorm_256x3072 | 4 | 18.316 | 18.808 | 0.976× [0.973, 0.990] | 23.117 |
| metal / rmsnorm_768x6144 | 4 | 77.906 | 81.389 | 0.935× [0.908, 0.982] | 100.653 |
| metal / rmsnorm_1024x4096 | 4 | 64.318 | 79.571 | 0.813× [0.801, 0.822] | 88.939 |
| metal / layernorm_37x1537 | 4 | 5.434 | 6.091 | 0.900× [0.862, 0.956] | 13.594 |
| metal / layernorm_256x3072 | 4 | 20.360 | 20.171 | 1.009× [1.003, 1.044] | 53.791 |
| metal / layernorm_768x6144 | 4 | 91.792 | 89.702 | 1.031× [0.995, 1.067] | 347.452 |
| metal / layernorm_1024x4096 | 4 | 76.544 | 94.142 | 0.815× [0.809, 0.827] | 251.094 |

## Separately sampled GPU execution

Completed command-buffer GPU timestamps, with no encoder hooks or counter attachments. This includes GPU work/gaps inside each command buffer, not CPU encoding or completion notification, and is not individual-kernel time. Each GPU phase uses its own repetition count. Values are medians of per-round p50 times. Speedups are medians of paired round ratios, with min–max ranges, not confidence intervals. Incomplete GPU control pairs withhold statistics; instrumented compute-pass samples remain diagnostics in JSON.

| Backend / case | GPU pairs | Reference GPU µs/op | Candidate GPU µs/op | Paired GPU speedup [range] | Candidate-run Torch GPU µs/op |
|---|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 4/4 | 4.091 | 4.621 | 0.884× [0.824, 0.898] | 16.583 |
| metal / softmax_256x3072 | 4/4 | 14.454 | 15.126 | 0.935× [0.796, 1.182] | 29.387 |
| metal / softmax_768x6144 | 4/4 | 60.714 | 66.225 | 0.918× [0.895, 0.928] | 148.510 |
| metal / softmax_1024x4096 | 4/4 | 54.409 | 57.434 | 0.953× [0.937, 0.964] | 137.706 |
| metal / rmsnorm_37x1537 | 4/4 | 4.076 | 4.570 | 0.853× [0.480, 0.976] | 5.015 |
| metal / rmsnorm_256x3072 | 4/4 | 16.640 | 18.009 | 0.981× [0.878, 1.027] | 16.992 |
| metal / rmsnorm_768x6144 | 4/4 | 62.332 | 69.435 | 0.920× [0.876, 0.938] | 80.507 |
| metal / rmsnorm_1024x4096 | 4/4 | 55.876 | 67.402 | 0.837× [0.812, 0.885] | 71.933 |
| metal / layernorm_37x1537 | 4/4 | 4.531 | 5.352 | 0.814× [0.597, 0.925] | 8.375 |
| metal / layernorm_256x3072 | 4/4 | 18.593 | 17.867 | 1.058× [1.019, 1.210] | 43.407 |
| metal / layernorm_768x6144 | 4/4 | 78.633 | 76.078 | 1.044× [1.026, 1.346] | 317.643 |
| metal / layernorm_1024x4096 | 4/4 | 64.790 | 79.309 | 0.818× [0.811, 0.838] | 238.213 |

## Single-call GPU versus end-to-end dispatch

These are separate phases; host samples are uninstrumented. Do not subtract their medians to estimate CPU cost. Torch remains the recorded eager operator sequence, not a compiled fused graph.

| Backend / case | Candidate GPU µs | Candidate E2E µs | Candidate-run Torch GPU µs | Candidate-run Torch E2E µs |
|---|---:|---:|---:|---:|
| metal / softmax_37x1537 | 7.375 | 207.834 | 70.187 | 330.354 |
| metal / softmax_256x3072 | 18.667 | 235.250 | 50.792 | 337.770 |
| metal / softmax_768x6144 | 85.542 | 313.312 | 150.208 | 498.480 |
| metal / softmax_1024x4096 | 57.021 | 338.875 | 136.854 | 429.854 |
| metal / rmsnorm_37x1537 | 7.167 | 218.459 | 8.125 | 224.542 |
| metal / rmsnorm_256x3072 | 19.229 | 274.812 | 21.208 | 270.395 |
| metal / rmsnorm_768x6144 | 73.188 | 322.146 | 99.479 | 353.209 |
| metal / rmsnorm_1024x4096 | 68.021 | 324.813 | 108.188 | 368.437 |
| metal / layernorm_37x1537 | 8.479 | 215.667 | 10.417 | 236.459 |
| metal / layernorm_256x3072 | 20.188 | 245.855 | 44.813 | 300.980 |
| metal / layernorm_768x6144 | 98.354 | 324.395 | 276.250 | 553.375 |
| metal / layernorm_1024x4096 | 92.271 | 342.479 | 200.625 | 451.271 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
