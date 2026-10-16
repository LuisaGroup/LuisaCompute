# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 4 | 5.679 | 5.988 | 0.971× [0.929, 1.003] | 26.315 |
| metal / gemm_128x128x128 | 4 | 12.905 | 13.148 | 0.978× [0.944, 1.005] | 27.037 |
| metal / gemm_512x512x512 | 4 | 55.363 | 54.843 | 1.011× [1.003, 1.015] | 48.290 |
| metal / gemm_1024x1024x1024 | 4 | 321.976 | 318.454 | 1.010× [1.009, 1.039] | 289.398 |
| metal / gemm_256x1024x128 | 4 | 19.208 | 18.857 | 1.021× [0.979, 1.040] | 29.707 |
| metal / gemm_1024x128x256 | 4 | 23.644 | 23.614 | 1.002× [0.961, 1.009] | 29.524 |
| metal / gemm_127x193x61 | 4 | 11.851 | 11.842 | 1.011× [0.952, 1.016] | 26.815 |
| metal / gemm_513x257x129 | 4 | 27.952 | 27.877 | 1.004× [0.957, 1.010] | 34.686 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
