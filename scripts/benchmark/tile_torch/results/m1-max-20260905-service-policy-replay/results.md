# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 4 | 4.486 | 4.889 | 0.920× [0.908, 0.945] | 31.600 |
| metal / softmax_256x3072 | 4 | 15.630 | 14.605 | 1.072× [1.053, 1.081] | 34.775 |
| metal / softmax_768x6144 | 4 | 80.042 | 58.321 | 1.372× [1.351, 1.377] | 136.194 |
| metal / softmax_64x12289 | 4 | 21.746 | 18.757 | 1.160× [1.122, 1.206] | 36.498 |
| metal / rmsnorm_37x1537 | 4 | 4.925 | 4.888 | 1.017× [1.002, 1.023] | 7.680 |
| metal / rmsnorm_256x3072 | 4 | 16.570 | 15.160 | 1.092× [1.050, 1.097] | 20.280 |
| metal / rmsnorm_768x6144 | 4 | 79.924 | 62.432 | 1.280× [1.275, 1.292] | 83.321 |
| metal / rmsnorm_64x12289 | 4 | 19.663 | 18.032 | 1.093× [1.083, 1.098] | 21.836 |
| metal / layernorm_37x1537 | 4 | 4.949 | 5.808 | 0.847× [0.821, 0.879] | 12.151 |
| metal / layernorm_256x3072 | 4 | 19.563 | 17.076 | 1.147× [1.134, 1.164] | 46.737 |
| metal / layernorm_768x6144 | 4 | 94.791 | 76.873 | 1.233× [1.221, 1.238] | 290.837 |
| metal / layernorm_64x12289 | 4 | 25.105 | 20.554 | 1.223× [1.200, 1.232] | 61.091 |

## Separately sampled GPU execution

Completed command-buffer GPU timestamps, with no encoder hooks or counter attachments. This includes GPU work/gaps inside each command buffer, not CPU encoding or completion notification, and is not individual-kernel time. Each GPU phase uses its own repetition count. Values are medians of per-round p50 times. Speedups are medians of paired round ratios, with min–max ranges, not confidence intervals. Incomplete GPU control pairs withhold statistics; instrumented compute-pass samples remain diagnostics in JSON.

| Backend / case | GPU pairs | Reference GPU µs/op | Candidate GPU µs/op | Paired GPU speedup [range] | Candidate-run Torch GPU µs/op |
|---|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 4/4 | 4.644 | 5.169 | 0.904× [0.846, 0.932] | 19.102 |
| metal / softmax_256x3072 | 4/4 | 15.790 | 14.683 | 1.075× [1.037, 1.112] | 28.972 |
| metal / softmax_768x6144 | 4/4 | 78.017 | 57.414 | 1.360× [1.353, 1.364] | 127.085 |
| metal / softmax_64x12289 | 4/4 | 21.407 | 18.494 | 1.152× [1.149, 1.172] | 31.138 |
| metal / rmsnorm_37x1537 | 4/4 | 5.090 | 5.157 | 0.987× [0.894, 1.043] | 5.021 |
| metal / rmsnorm_256x3072 | 4/4 | 16.511 | 15.048 | 1.089× [1.059, 1.146] | 17.480 |
| metal / rmsnorm_768x6144 | 4/4 | 78.097 | 60.753 | 1.287× [1.279, 1.289] | 78.642 |
| metal / rmsnorm_64x12289 | 4/4 | 19.608 | 17.894 | 1.089× [1.074, 1.122] | 19.481 |
| metal / layernorm_37x1537 | 4/4 | 5.175 | 5.904 | 0.876× [0.818, 0.928] | 8.461 |
| metal / layernorm_256x3072 | 4/4 | 19.355 | 16.859 | 1.147× [1.118, 1.163] | 40.743 |
| metal / layernorm_768x6144 | 4/4 | 92.931 | 75.251 | 1.231× [1.218, 1.243] | 281.360 |
| metal / layernorm_64x12289 | 4/4 | 24.693 | 20.231 | 1.224× [1.187, 1.243] | 56.369 |

## Single-call GPU versus end-to-end dispatch

These are separate phases; host samples are uninstrumented. Do not subtract their medians to estimate CPU cost. Torch remains the recorded eager operator sequence, not a compiled fused graph.

| Backend / case | Candidate GPU µs | Candidate E2E µs | Candidate-run Torch GPU µs | Candidate-run Torch E2E µs |
|---|---:|---:|---:|---:|
| metal / softmax_37x1537 | 7.188 | 213.584 | 77.958 | 325.604 |
| metal / softmax_256x3072 | 18.333 | 246.041 | 48.938 | 324.771 |
| metal / softmax_768x6144 | 80.625 | 285.562 | 138.125 | 439.895 |
| metal / softmax_64x12289 | 22.000 | 266.021 | 67.208 | 344.062 |
| metal / rmsnorm_37x1537 | 7.354 | 207.833 | 9.104 | 234.041 |
| metal / rmsnorm_256x3072 | 18.646 | 254.584 | 23.021 | 257.312 |
| metal / rmsnorm_768x6144 | 80.479 | 321.375 | 96.271 | 338.583 |
| metal / rmsnorm_64x12289 | 21.229 | 257.770 | 27.521 | 257.791 |
| metal / layernorm_37x1537 | 8.271 | 213.229 | 10.667 | 236.917 |
| metal / layernorm_256x3072 | 20.292 | 267.855 | 44.958 | 290.979 |
| metal / layernorm_768x6144 | 96.271 | 340.416 | 289.229 | 532.854 |
| metal / layernorm_64x12289 | 23.792 | 251.312 | 64.833 | 322.500 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
