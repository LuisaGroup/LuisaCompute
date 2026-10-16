# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 4 | 6.306 | 8.136 | 0.792× [0.696, 0.916] | 27.316 |
| metal / gemm_128x128x128 | 4 | 15.616 | 12.171 | 1.277× [1.177, 1.349] | 27.514 |
| metal / gemm_512x512x512 | 4 | 124.155 | 77.361 | 1.609× [1.579, 1.644] | 48.393 |
| metal / gemm_1024x1024x1024 | 4 | 998.477 | 474.079 | 2.106× [2.096, 2.129] | 293.380 |
| metal / gemm_256x1024x128 | 4 | 35.814 | 25.468 | 1.401× [1.367, 1.470] | 30.300 |
| metal / gemm_1024x128x256 | 4 | 34.228 | 23.891 | 1.432× [1.381, 1.510] | 30.876 |
| metal / gemm_127x193x61 | 4 | 14.238 | 20.937 | 0.687× [0.637, 0.720] | 27.308 |
| metal / gemm_513x257x129 | 4 | 53.952 | 41.769 | 1.296× [1.252, 1.333] | 34.380 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
