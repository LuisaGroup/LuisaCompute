# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 4 | 6.507 | 6.132 | 1.058× [1.012, 1.765] | 0.913 |
| cpu / gemm_128x128x128 | 4 | 69.788 | 43.722 | 1.598× [1.444, 2.124] | 4.931 |
| cpu / gemm_512x512x512 | 4 | 2141.448 | 1908.219 | 1.144× [0.935, 1.334] | 149.735 |
| cpu / gemm_1024x1024x1024 | 4 | 14076.709 | 12351.330 | 1.140× [1.020, 1.248] | 1044.902 |
| cpu / gemm_256x1024x128 | 4 | 740.390 | 492.523 | 1.495× [1.451, 2.017] | 70.255 |
| cpu / gemm_1024x128x256 | 4 | 638.819 | 515.912 | 1.229× [1.145, 1.770] | 63.477 |
| cpu / gemm_127x193x61 | 4 | 90.795 | 58.891 | 1.542× [1.072, 1.708] | 6.499 |
| cpu / gemm_513x257x129 | 4 | 739.177 | 529.624 | 1.390× [1.276, 1.851] | 46.795 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
