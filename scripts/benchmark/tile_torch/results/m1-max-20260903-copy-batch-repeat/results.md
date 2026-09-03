# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 4 | 6.490 | 5.683 | 1.149× [1.105, 1.162] | 27.080 |
| metal / gemm_128x128x128 | 4 | 13.663 | 13.723 | 0.995× [0.990, 1.005] | 27.062 |
| metal / gemm_512x512x512 | 4 | 56.736 | 56.676 | 1.002× [0.999, 1.007] | 47.976 |
| metal / gemm_1024x1024x1024 | 4 | 407.528 | 407.691 | 1.000× [0.996, 1.002] | 288.183 |
| metal / gemm_256x1024x128 | 4 | 19.490 | 19.469 | 1.005× [0.994, 1.039] | 29.915 |
| metal / gemm_1024x128x256 | 4 | 24.867 | 25.043 | 0.994× [0.988, 1.002] | 30.197 |
| metal / gemm_127x193x61 | 4 | 17.619 | 11.392 | 1.536× [1.505, 1.595] | 27.125 |
| metal / gemm_513x257x129 | 4 | 36.400 | 26.939 | 1.349× [1.336, 1.357] | 34.363 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
