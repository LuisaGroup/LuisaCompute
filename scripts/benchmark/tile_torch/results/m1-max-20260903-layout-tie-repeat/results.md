# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 4 | 6.039 | 6.310 | 0.961× [0.932, 0.970] | 26.797 |
| metal / gemm_128x128x128 | 4 | 13.159 | 13.225 | 0.995× [0.893, 1.032] | 26.865 |
| metal / gemm_512x512x512 | 4 | 55.575 | 57.244 | 0.972× [0.964, 0.973] | 48.404 |
| metal / gemm_1024x1024x1024 | 4 | 321.488 | 333.388 | 0.965× [0.963, 0.967] | 289.348 |
| metal / gemm_256x1024x128 | 4 | 19.358 | 19.863 | 0.977× [0.970, 1.009] | 29.736 |
| metal / gemm_1024x128x256 | 4 | 23.573 | 24.157 | 0.981× [0.967, 0.983] | 29.791 |
| metal / gemm_127x193x61 | 4 | 11.999 | 12.229 | 0.977× [0.958, 1.076] | 27.276 |
| metal / gemm_513x257x129 | 4 | 27.868 | 28.414 | 0.985× [0.962, 0.991] | 34.438 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
