# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 4 | 4.666 | 4.931 | 1.010× [0.754, 1.217] | 0.922 |
| cpu / gemm_128x128x128 | 4 | 24.099 | 14.992 | 1.636× [1.267, 2.094] | 4.845 |
| cpu / gemm_512x512x512 | 4 | 988.463 | 617.067 | 1.507× [1.461, 1.735] | 143.446 |
| cpu / gemm_1024x1024x1024 | 4 | 8215.518 | 5605.283 | 1.453× [1.386, 1.530] | 1043.329 |
| cpu / gemm_256x1024x128 | 4 | 331.239 | 191.638 | 1.759× [1.589, 1.838] | 68.921 |
| cpu / gemm_1024x128x256 | 4 | 265.420 | 190.182 | 1.485× [1.225, 1.778] | 63.332 |
| cpu / gemm_127x193x61 | 4 | 35.034 | 33.504 | 0.981× [0.918, 1.401] | 6.505 |
| cpu / gemm_513x257x129 | 4 | 377.179 | 392.023 | 0.937× [0.860, 1.111] | 43.469 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
