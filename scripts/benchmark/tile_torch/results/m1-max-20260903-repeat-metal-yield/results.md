# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 4 | 4.598 | 4.507 | 1.001× [0.973, 1.025] | 30.304 |
| metal / gemm_128x128x128 | 4 | 8.077 | 7.684 | 1.053× [1.036, 1.077] | 30.721 |
| metal / gemm_512x512x512 | 4 | 148.006 | 116.137 | 1.271× [1.227, 1.344] | 55.986 |
| metal / gemm_1024x1024x1024 | 4 | 1095.253 | 980.069 | 1.120× [1.070, 1.159] | 339.288 |
| metal / gemm_256x1024x128 | 4 | 43.254 | 37.313 | 1.167× [1.125, 1.181] | 33.812 |
| metal / gemm_1024x128x256 | 4 | 42.072 | 33.530 | 1.256× [1.163, 1.321] | 35.126 |
| metal / gemm_127x193x61 | 4 | 8.600 | 8.766 | 0.995× [0.934, 1.014] | 31.210 |
| metal / gemm_513x257x129 | 4 | 40.937 | 38.459 | 1.072× [1.038, 1.102] | 40.442 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
