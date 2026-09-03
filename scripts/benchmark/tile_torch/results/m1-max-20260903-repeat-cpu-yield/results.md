# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 4 | 9.529 | 9.612 | 1.062× [0.896, 1.124] | 0.874 |
| cpu / gemm_128x128x128 | 4 | 81.960 | 78.038 | 1.093× [0.843, 1.148] | 4.921 |
| cpu / gemm_512x512x512 | 4 | 2448.468 | 2272.065 | 1.035× [0.814, 1.324] | 148.855 |
| cpu / gemm_1024x1024x1024 | 4 | 16143.299 | 15778.021 | 1.010× [0.803, 1.187] | 1234.321 |
| cpu / gemm_256x1024x128 | 4 | 950.729 | 942.488 | 1.004× [0.922, 1.105] | 68.489 |
| cpu / gemm_1024x128x256 | 4 | 745.132 | 778.351 | 0.979× [0.827, 1.202] | 65.088 |
| cpu / gemm_127x193x61 | 4 | 103.285 | 98.560 | 1.022× [0.910, 1.196] | 6.523 |
| cpu / gemm_513x257x129 | 4 | 788.088 | 785.867 | 0.992× [0.858, 1.071] | 45.023 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
