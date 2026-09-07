# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / sigmoid_pair_1x127 | 6 | 112.062 | 2.965 | 37.344× [35.819, 38.687] | 14.396 |
| metal / sigmoid_pair_37x1537 | 6 | 256.790 | 5.105 | 50.217× [48.207, 52.432] | 18.902 |
| metal / sigmoid_pair_1024x4096 | 6 | 368.048 | 129.785 | 2.852× [2.716, 3.181] | 307.374 |
| metal / sigmoid_pair_4096x4096 | 6 | 1638.985 | 696.195 | 2.356× [2.300, 2.426] | 1797.053 |
| metal / gelu_pair_1x127 | 6 | 126.604 | 4.238 | 29.701× [28.451, 29.948] | 11.070 |
| metal / gelu_pair_37x1537 | 6 | 286.035 | 9.233 | 30.998× [29.849, 31.282] | 14.460 |
| metal / gelu_pair_1024x4096 | 6 | 465.522 | 126.374 | 3.678× [3.304, 3.761] | 226.334 |
| metal / gelu_pair_4096x4096 | 6 | 2117.143 | 700.926 | 3.007× [2.781, 3.185] | 1281.771 |

## Separately sampled GPU execution

Completed command-buffer GPU timestamps, with no encoder hooks or counter attachments. This includes GPU work/gaps inside each command buffer, not CPU encoding or completion notification, and is not individual-kernel time. Each GPU phase uses its own repetition count. Values are medians of per-round p50 times. Speedups are medians of paired round ratios, with min–max ranges, not confidence intervals. Incomplete GPU control pairs withhold statistics; instrumented compute-pass samples remain diagnostics in JSON.

| Backend / case | GPU pairs | Reference GPU µs/op | Candidate GPU µs/op | Paired GPU speedup [range] | Candidate-run Torch GPU µs/op |
|---|---:|---:|---:|---:|---:|
| metal / sigmoid_pair_1x127 | 6/6 | 96.588 | 2.018 | 48.138× [38.869, 56.445] | 6.523 |
| metal / sigmoid_pair_37x1537 | 6/6 | 252.647 | 3.546 | 71.257× [69.692, 72.015] | 9.803 |
| metal / sigmoid_pair_1024x4096 | 6/6 | 366.747 | 105.319 | 3.503× [2.760, 3.726] | 288.584 |
| metal / sigmoid_pair_4096x4096 | 6/6 | 1566.623 | 679.060 | 2.298× [2.200, 2.388] | 1712.971 |
| metal / gelu_pair_1x127 | 6/6 | 124.775 | 3.016 | 41.279× [20.283, 43.101] | 4.414 |
| metal / gelu_pair_37x1537 | 6/6 | 279.678 | 6.468 | 43.277× [43.044, 43.741] | 6.793 |
| metal / gelu_pair_1024x4096 | 6/6 | 451.546 | 105.544 | 4.232× [3.855, 4.391] | 197.518 |
| metal / gelu_pair_4096x4096 | 6/6 | 2045.484 | 677.876 | 3.025× [2.921, 3.133] | 1227.067 |

## Single-call GPU versus end-to-end dispatch

These are separate phases; host samples are uninstrumented. Do not subtract their medians to estimate CPU cost. Torch remains the recorded eager operator sequence, not a compiled fused graph.

| Backend / case | Candidate GPU µs | Candidate E2E µs | Candidate-run Torch GPU µs | Candidate-run Torch E2E µs |
|---|---:|---:|---:|---:|
| metal / sigmoid_pair_1x127 | 4.167 | 217.250 | 9.187 | 251.105 |
| metal / sigmoid_pair_37x1537 | 7.188 | 258.271 | 13.250 | 258.791 |
| metal / sigmoid_pair_1024x4096 | 108.250 | 399.229 | 250.708 | 543.958 |
| metal / sigmoid_pair_4096x4096 | 582.625 | 876.958 | 1552.646 | 1772.000 |
| metal / gelu_pair_1x127 | 5.958 | 254.645 | 7.208 | 303.541 |
| metal / gelu_pair_37x1537 | 9.708 | 264.459 | 9.688 | 257.584 |
| metal / gelu_pair_1024x4096 | 133.979 | 407.230 | 200.271 | 472.834 |
| metal / gelu_pair_4096x4096 | 597.000 | 855.958 | 1029.521 | 1416.812 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
