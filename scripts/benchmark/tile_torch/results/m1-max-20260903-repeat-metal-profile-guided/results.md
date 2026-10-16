# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 4 | 4.220 | 4.578 | 0.936× [0.904, 0.955] | 32.195 |
| metal / gemm_128x128x128 | 4 | 7.442 | 7.683 | 0.969× [0.955, 0.998] | 30.582 |
| metal / gemm_512x512x512 | 4 | 115.998 | 118.825 | 0.996× [0.806, 1.018] | 53.671 |
| metal / gemm_1024x1024x1024 | 4 | 904.828 | 970.736 | 0.991× [0.849, 1.048] | 313.056 |
| metal / gemm_256x1024x128 | 4 | 44.037 | 39.113 | 1.033× [0.935, 1.382] | 35.428 |
| metal / gemm_1024x128x256 | 4 | 33.648 | 33.712 | 1.005× [0.971, 1.292] | 38.037 |
| metal / gemm_127x193x61 | 4 | 8.358 | 8.050 | 0.984× [0.955, 1.162] | 32.425 |
| metal / gemm_513x257x129 | 4 | 35.703 | 35.438 | 1.007× [0.960, 1.027] | 38.617 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
