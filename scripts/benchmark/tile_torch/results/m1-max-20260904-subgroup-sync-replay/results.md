# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 4 | 3.705 | 3.607 | 1.038× [0.967, 1.052] | 28.598 |
| metal / gemm_128x128x128 | 4 | 6.587 | 6.561 | 1.014× [0.935, 1.034] | 29.121 |
| metal / gemm_512x512x512 | 4 | 56.571 | 62.624 | 0.902× [0.862, 0.956] | 63.853 |
| metal / gemm_1024x1024x1024 | 4 | 383.786 | 379.720 | 1.011× [0.995, 1.072] | 382.319 |
| metal / gemm_256x1024x128 | 4 | 22.773 | 21.689 | 1.037× [0.988, 1.081] | 34.608 |
| metal / gemm_1024x128x256 | 4 | 22.985 | 22.392 | 1.028× [0.990, 1.053] | 33.288 |
| metal / gemm_127x193x61 | 4 | 7.890 | 8.045 | 0.985× [0.958, 1.020] | 30.574 |
| metal / gemm_513x257x129 | 4 | 27.135 | 26.728 | 1.015× [0.993, 1.058] | 45.427 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
