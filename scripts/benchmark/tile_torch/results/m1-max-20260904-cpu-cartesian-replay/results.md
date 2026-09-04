# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 4 | 4.588 | 4.584 | 0.940× [0.909, 1.194] | 0.874 |
| cpu / gemm_128x128x128 | 4 | 32.046 | 25.690 | 1.397× [1.028, 1.507] | 4.886 |
| cpu / gemm_512x512x512 | 4 | 1414.340 | 1053.471 | 1.365× [1.195, 1.611] | 143.846 |
| cpu / gemm_1024x1024x1024 | 4 | 11187.410 | 8083.594 | 1.331× [1.269, 1.459] | 1022.927 |
| cpu / gemm_256x1024x128 | 4 | 410.364 | 297.457 | 1.408× [1.255, 1.453] | 68.603 |
| cpu / gemm_1024x128x256 | 4 | 418.689 | 257.752 | 1.625× [1.227, 1.796] | 64.087 |
| cpu / gemm_127x193x61 | 4 | 44.327 | 38.126 | 1.116× [1.059, 2.105] | 6.483 |
| cpu / gemm_513x257x129 | 4 | 482.570 | 403.025 | 1.156× [0.829, 1.263] | 44.245 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
