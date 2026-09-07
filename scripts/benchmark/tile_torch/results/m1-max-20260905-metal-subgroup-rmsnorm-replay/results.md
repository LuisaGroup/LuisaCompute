# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / rmsnorm_1x127 | 4 | 103.180 | 3.792 | 27.216× [24.924, 28.020] | 7.164 |
| metal / rmsnorm_17x257 | 4 | 268.202 | 5.366 | 49.871× [49.180, 54.574] | 6.141 |
| metal / rmsnorm_128x1024 | 4 | 144.082 | 6.805 | 21.192× [20.989, 21.207] | 8.802 |
| metal / rmsnorm_64x4096 | 4 | 524.444 | 11.160 | 47.096× [46.344, 50.864] | 12.474 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
