# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / sum_1x4096 | 4 | 3.157 | 3.149 | 1.010× [0.905, 1.122] | 8.096 |
| metal / sum_64x4096 | 4 | 4.743 | 4.453 | 1.062× [0.873, 1.072] | 15.627 |
| metal / sum_1024x4096 | 4 | 25.760 | 23.235 | 1.107× [1.100, 1.188] | 27.876 |
| metal / sum_17x257 | 4 | 3.095 | 2.954 | 1.057× [0.990, 1.206] | 4.526 |
| metal / sum_1024x257 | 4 | 4.428 | 4.380 | 1.011× [0.907, 1.268] | 6.203 |
| metal / softmax_1x4096 | 4 | 4.549 | 4.548 | 1.002× [0.999, 1.003] | 29.344 |
| metal / softmax_64x4096 | 4 | 8.903 | 8.920 | 1.004× [0.946, 1.033] | 31.951 |
| metal / softmax_1024x4096 | 4 | 67.218 | 63.376 | 1.062× [1.053, 1.079] | 128.156 |
| metal / softmax_17x257 | 4 | 3.336 | 3.335 | 0.985× [0.965, 1.058] | 29.504 |
| metal / softmax_1024x257 | 4 | 8.386 | 7.953 | 1.051× [1.041, 1.072] | 33.350 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
