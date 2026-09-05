# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / sum_17x257 | 4 | 2.868 | 2.865 | 1.000× [0.997, 1.002] | 4.349 |
| metal / sum_7x1537 | 4 | 3.004 | 2.986 | 0.998× [0.978, 1.032] | 5.922 |
| metal / sum_64x4096 | 4 | 4.169 | 4.235 | 0.983× [0.974, 1.060] | 14.721 |
| metal / sum_1024x4096 | 4 | 27.564 | 27.619 | 0.992× [0.981, 1.032] | 27.552 |
| metal / sum_128x8192 | 4 | 9.175 | 9.135 | 0.999× [0.970, 1.064] | 23.280 |
| metal / softmax_17x257 | 4 | 3.396 | 2.930 | 1.163× [1.148, 1.177] | 31.581 |
| metal / softmax_7x1537 | 4 | 4.141 | 3.565 | 1.160× [1.035, 1.183] | 32.222 |
| metal / softmax_64x4096 | 4 | 8.161 | 7.609 | 1.072× [1.041, 1.106] | 33.962 |
| metal / softmax_1024x4096 | 4 | 75.391 | 54.727 | 1.381× [1.375, 1.384] | 130.437 |
| metal / softmax_128x8192 | 4 | 24.786 | 20.050 | 1.251× [1.226, 1.262] | 45.032 |
| metal / rmsnorm_17x257 | 4 | 3.158 | 2.934 | 1.077× [0.983, 1.100] | 6.234 |
| metal / rmsnorm_7x1537 | 4 | 4.299 | 3.818 | 1.114× [1.104, 1.141] | 7.150 |
| metal / rmsnorm_64x4096 | 4 | 8.736 | 7.924 | 1.086× [1.034, 1.121] | 12.078 |
| metal / rmsnorm_1024x4096 | 4 | 72.111 | 56.342 | 1.279× [1.260, 1.319] | 73.506 |
| metal / rmsnorm_128x8192 | 4 | 23.020 | 20.627 | 1.102× [1.046, 1.126] | 25.644 |
| metal / layernorm_17x257 | 4 | 3.839 | 3.213 | 1.192× [0.991, 1.239] | 9.019 |
| metal / layernorm_7x1537 | 4 | 5.018 | 4.428 | 1.136× [1.123, 1.143] | 10.836 |
| metal / layernorm_64x4096 | 4 | 9.683 | 8.959 | 1.082× [1.037, 1.130] | 24.248 |
| metal / layernorm_1024x4096 | 4 | 81.771 | 66.384 | 1.229× [1.212, 1.254] | 216.860 |
| metal / layernorm_128x8192 | 4 | 28.453 | 23.722 | 1.203× [1.150, 1.217] | 72.180 |
| metal / residual_layernorm_17x257 | 4 | 3.249 | 3.249 | 1.000× [0.990, 1.026] | 11.696 |
| metal / residual_layernorm_7x1537 | 4 | 4.649 | 4.703 | 0.992× [0.969, 0.996] | 13.304 |
| metal / residual_layernorm_64x4096 | 4 | 9.240 | 9.683 | 0.957× [0.915, 0.976] | 27.199 |
| metal / residual_layernorm_1024x4096 | 4 | 103.414 | 103.016 | 1.001× [0.951, 1.065] | 263.945 |
| metal / residual_layernorm_128x8192 | 4 | 25.710 | 25.560 | 1.013× [0.948, 1.046] | 67.651 |

## Separately sampled GPU execution

Completed command-buffer GPU timestamps, with no encoder hooks or counter attachments. This includes GPU work/gaps inside each command buffer, not CPU encoding or completion notification, and is not individual-kernel time. Each GPU phase uses its own repetition count. Values are medians of per-round p50 times. Speedups are medians of paired round ratios, with min–max ranges, not confidence intervals. Incomplete GPU control pairs withhold statistics; instrumented compute-pass samples remain diagnostics in JSON.

| Backend / case | GPU pairs | Reference GPU µs/op | Candidate GPU µs/op | Paired GPU speedup [range] | Candidate-run Torch GPU µs/op |
|---|---:|---:|---:|---:|---:|
| metal / sum_17x257 | 4/4 | 2.749 | 3.035 | 0.919× [0.855, 1.091] | 2.831 |
| metal / sum_7x1537 | 4/4 | 3.056 | 3.045 | 1.010× [0.908, 1.083] | 4.600 |
| metal / sum_64x4096 | 4/4 | 4.396 | 4.379 | 0.987× [0.960, 1.096] | 13.582 |
| metal / sum_1024x4096 | 4/4 | 27.329 | 27.616 | 0.993× [0.959, 1.021] | 27.046 |
| metal / sum_128x8192 | 4/4 | 9.245 | 9.080 | 1.010× [0.965, 1.188] | 20.305 |
| metal / softmax_17x257 | 4/4 | 3.154 | 2.937 | 1.082× [1.037, 1.143] | 14.989 |
| metal / softmax_7x1537 | 4/4 | 4.172 | 3.963 | 1.058× [1.010, 1.111] | 22.771 |
| metal / softmax_64x4096 | 4/4 | 8.543 | 7.852 | 1.099× [1.053, 1.125] | 22.388 |
| metal / softmax_1024x4096 | 4/4 | 74.198 | 53.949 | 1.378× [1.373, 1.395] | 121.715 |
| metal / softmax_128x8192 | 4/4 | 24.757 | 19.806 | 1.246× [1.223, 1.255] | 40.201 |
| metal / rmsnorm_17x257 | 4/4 | 3.225 | 3.016 | 1.092× [0.999, 1.123] | 3.610 |
| metal / rmsnorm_7x1537 | 4/4 | 4.420 | 3.860 | 1.138× [1.066, 1.277] | 4.121 |
| metal / rmsnorm_64x4096 | 4/4 | 9.086 | 8.505 | 1.074× [0.972, 1.100] | 9.612 |
| metal / rmsnorm_1024x4096 | 4/4 | 70.668 | 55.863 | 1.265× [1.246, 1.345] | 69.108 |
| metal / rmsnorm_128x8192 | 4/4 | 22.449 | 20.667 | 1.090× [1.017, 1.133] | 22.902 |
| metal / layernorm_17x257 | 4/4 | 3.963 | 3.189 | 1.253× [0.977, 1.380] | 4.955 |
| metal / layernorm_7x1537 | 4/4 | 5.319 | 4.553 | 1.168× [1.070, 1.313] | 6.477 |
| metal / layernorm_64x4096 | 4/4 | 9.903 | 9.338 | 1.054× [1.033, 1.149] | 19.364 |
| metal / layernorm_1024x4096 | 4/4 | 79.150 | 64.704 | 1.221× [1.213, 1.251] | 206.598 |
| metal / layernorm_128x8192 | 4/4 | 27.914 | 23.749 | 1.176× [1.173, 1.182] | 66.350 |
| metal / residual_layernorm_17x257 | 4/4 | 3.314 | 3.254 | 1.037× [0.870, 1.042] | 6.399 |
| metal / residual_layernorm_7x1537 | 4/4 | 4.410 | 4.640 | 0.980× [0.915, 1.008] | 7.921 |
| metal / residual_layernorm_64x4096 | 4/4 | 9.243 | 9.758 | 0.953× [0.857, 1.004] | 20.253 |
| metal / residual_layernorm_1024x4096 | 4/4 | 97.753 | 98.869 | 0.988× [0.909, 1.086] | 250.889 |
| metal / residual_layernorm_128x8192 | 4/4 | 25.473 | 25.299 | 1.006× [0.990, 1.037] | 61.175 |

## Single-call GPU versus end-to-end dispatch

These are separate phases; host samples are uninstrumented. Do not subtract their medians to estimate CPU cost. Torch remains the recorded eager operator sequence, not a compiled fused graph.

| Backend / case | Candidate GPU µs | Candidate E2E µs | Candidate-run Torch GPU µs | Candidate-run Torch E2E µs |
|---|---:|---:|---:|---:|
| metal / sum_17x257 | 5.188 | 203.750 | 5.083 | 216.292 |
| metal / sum_7x1537 | 5.187 | 227.209 | 6.896 | 215.312 |
| metal / sum_64x4096 | 6.896 | 220.875 | 15.687 | 264.500 |
| metal / sum_1024x4096 | 34.375 | 320.145 | 34.583 | 269.584 |
| metal / sum_128x8192 | 12.646 | 250.084 | 23.437 | 280.063 |
| metal / softmax_17x257 | 5.146 | 222.105 | 70.437 | 388.708 |
| metal / softmax_7x1537 | 5.917 | 211.855 | 92.208 | 342.896 |
| metal / softmax_64x4096 | 10.583 | 238.625 | 61.083 | 323.583 |
| metal / softmax_1024x4096 | 69.354 | 295.792 | 130.812 | 479.355 |
| metal / softmax_128x8192 | 24.771 | 261.146 | 71.063 | 366.166 |
| metal / rmsnorm_17x257 | 5.271 | 214.541 | 7.083 | 211.792 |
| metal / rmsnorm_7x1537 | 6.125 | 207.959 | 7.563 | 224.459 |
| metal / rmsnorm_64x4096 | 10.833 | 240.188 | 12.500 | 238.855 |
| metal / rmsnorm_1024x4096 | 60.646 | 324.417 | 87.604 | 341.459 |
| metal / rmsnorm_128x8192 | 24.562 | 246.417 | 28.208 | 320.625 |
| metal / layernorm_17x257 | 5.792 | 200.812 | 7.958 | 224.250 |
| metal / layernorm_7x1537 | 6.708 | 209.875 | 10.042 | 237.084 |
| metal / layernorm_64x4096 | 11.854 | 227.584 | 21.625 | 275.166 |
| metal / layernorm_1024x4096 | 82.375 | 325.395 | 204.708 | 459.770 |
| metal / layernorm_128x8192 | 28.208 | 250.938 | 74.292 | 313.270 |
| metal / residual_layernorm_17x257 | 5.854 | 224.979 | 11.521 | 231.084 |
| metal / residual_layernorm_7x1537 | 6.938 | 212.375 | 11.833 | 239.854 |
| metal / residual_layernorm_64x4096 | 12.792 | 231.708 | 25.354 | 279.521 |
| metal / residual_layernorm_1024x4096 | 116.062 | 370.354 | 266.396 | 627.250 |
| metal / residual_layernorm_128x8192 | 31.437 | 270.188 | 79.104 | 326.000 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
