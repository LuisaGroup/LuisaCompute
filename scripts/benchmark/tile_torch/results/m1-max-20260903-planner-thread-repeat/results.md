# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 4 | 7.511 | 5.673 | 1.312× [1.291, 1.420] | 27.142 |
| metal / gemm_128x128x128 | 4 | 12.151 | 11.115 | 1.083× [0.999, 1.222] | 27.331 |
| metal / gemm_512x512x512 | 4 | 76.885 | 74.039 | 1.037× [1.022, 1.053] | 48.168 |
| metal / gemm_1024x1024x1024 | 4 | 474.298 | 465.241 | 1.018× [0.986, 1.022] | 287.830 |
| metal / gemm_256x1024x128 | 4 | 24.889 | 23.840 | 1.044× [1.029, 1.094] | 30.207 |
| metal / gemm_1024x128x256 | 4 | 23.032 | 22.031 | 1.042× [0.963, 1.072] | 30.390 |
| metal / gemm_127x193x61 | 4 | 20.382 | 12.114 | 1.670× [1.579, 1.847] | 27.689 |
| metal / gemm_513x257x129 | 4 | 41.175 | 28.400 | 1.453× [1.423, 1.477] | 34.329 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
