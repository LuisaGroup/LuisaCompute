# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 4 | 6.255 | 6.119 | 1.034× [1.004, 1.091] | 26.554 |
| metal / gemm_128x128x128 | 4 | 14.010 | 13.055 | 1.071× [1.063, 1.081] | 27.119 |
| metal / gemm_512x512x512 | 4 | 57.135 | 55.394 | 1.031× [1.026, 1.035] | 48.285 |
| metal / gemm_1024x1024x1024 | 4 | 327.454 | 321.218 | 1.019× [0.986, 1.020] | 289.100 |
| metal / gemm_256x1024x128 | 4 | 19.894 | 19.068 | 1.039× [1.021, 1.103] | 29.476 |
| metal / gemm_1024x128x256 | 4 | 25.524 | 23.507 | 1.089× [1.083, 1.102] | 29.878 |
| metal / gemm_127x193x61 | 4 | 12.323 | 11.919 | 1.035× [1.008, 1.075] | 26.980 |
| metal / gemm_513x257x129 | 4 | 28.111 | 27.531 | 1.026× [0.976, 1.043] | 34.559 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
