# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 4 | 4.710 | 3.988 | 1.401× [0.854, 1.571] | 0.934 |
| cpu / gemm_128x128x128 | 4 | 34.952 | 31.737 | 1.059× [0.893, 1.432] | 4.947 |
| cpu / gemm_512x512x512 | 4 | 1427.075 | 1371.072 | 1.159× [0.883, 1.267] | 134.336 |
| cpu / gemm_1024x1024x1024 | 4 | 10409.542 | 10469.537 | 0.991× [0.835, 1.494] | 961.819 |
| cpu / gemm_256x1024x128 | 4 | 488.840 | 463.428 | 1.032× [0.845, 1.525] | 67.543 |
| cpu / gemm_1024x128x256 | 4 | 396.268 | 370.803 | 1.065× [0.881, 1.334] | 64.290 |
| cpu / gemm_127x193x61 | 4 | 44.649 | 37.518 | 1.125× [0.226, 1.256] | 6.534 |
| cpu / gemm_513x257x129 | 4 | 411.734 | 479.000 | 0.870× [0.746, 0.999] | 44.438 |
| cpu / add_1x127 | 4 | 0.427 | 0.268 | 1.593× [1.574, 1.683] | 0.589 |
| cpu / add_17x257 | 4 | 6.551 | 6.144 | 1.678× [0.489, 2.051] | 0.977 |
| cpu / add_128x1024 | 4 | 18.519 | 17.919 | 1.243× [0.379, 1.626] | 39.053 |
| cpu / add_4096x256 | 4 | 116.567 | 96.152 | 1.212× [1.121, 1.342] | 84.481 |
| cpu / sum_1x127 | 4 | 2.112 | 2.121 | 1.000× [0.989, 1.045] | 0.857 |
| cpu / sum_17x257 | 4 | 33.143 | 28.030 | 1.108× [0.962, 1.270] | 1.161 |
| cpu / sum_128x1024 | 4 | 662.111 | 661.901 | 0.870× [0.458, 1.535] | 39.319 |
| cpu / sum_64x4096 | 4 | 1386.546 | 1521.621 | 0.916× [0.645, 1.598] | 44.292 |
| cpu / softmax_1x127 | 4 | 5.874 | 6.017 | 0.983× [0.969, 0.989] | 0.654 |
| cpu / softmax_17x257 | 4 | 80.663 | 67.555 | 1.286× [0.622, 1.435] | 37.012 |
| cpu / softmax_128x1024 | 4 | 1491.452 | 1438.275 | 1.063× [0.771, 1.642] | 86.327 |
| cpu / softmax_64x4096 | 4 | 3332.421 | 3328.884 | 1.059× [0.591, 2.095] | 119.087 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
