# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 4 | 5.249 | 4.748 | 1.116× [0.925, 1.181] | 26.837 |
| metal / gemm_128x128x128 | 4 | 9.821 | 6.841 | 1.453× [1.413, 1.565] | 26.679 |
| metal / gemm_512x512x512 | 4 | 53.536 | 53.447 | 0.998× [0.979, 1.029] | 48.720 |
| metal / gemm_1024x1024x1024 | 4 | 319.760 | 319.902 | 0.999× [0.968, 1.004] | 291.547 |
| metal / gemm_256x1024x128 | 4 | 19.326 | 19.086 | 1.024× [0.995, 1.028] | 30.070 |
| metal / gemm_1024x128x256 | 4 | 21.265 | 19.793 | 1.083× [1.056, 1.098] | 30.201 |
| metal / gemm_127x193x61 | 4 | 9.174 | 8.683 | 1.054× [1.032, 1.218] | 27.058 |
| metal / gemm_513x257x129 | 4 | 23.197 | 22.591 | 1.032× [0.958, 1.044] | 34.655 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
