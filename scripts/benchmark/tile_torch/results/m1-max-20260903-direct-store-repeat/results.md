# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 4 | 6.201 | 6.176 | 1.004× [0.969, 1.039] | 27.006 |
| metal / gemm_128x128x128 | 4 | 14.681 | 14.054 | 1.046× [1.040, 1.050] | 26.786 |
| metal / gemm_512x512x512 | 4 | 57.605 | 57.121 | 1.008× [1.005, 1.011] | 48.301 |
| metal / gemm_1024x1024x1024 | 4 | 406.999 | 327.762 | 1.242× [1.241, 1.242] | 288.270 |
| metal / gemm_256x1024x128 | 4 | 20.508 | 20.005 | 1.026× [1.016, 1.049] | 29.587 |
| metal / gemm_1024x128x256 | 4 | 26.056 | 25.435 | 1.025× [0.996, 1.046] | 29.563 |
| metal / gemm_127x193x61 | 4 | 12.265 | 12.275 | 1.000× [0.980, 1.019] | 26.848 |
| metal / gemm_513x257x129 | 4 | 28.031 | 28.077 | 0.998× [0.989, 1.010] | 34.569 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
