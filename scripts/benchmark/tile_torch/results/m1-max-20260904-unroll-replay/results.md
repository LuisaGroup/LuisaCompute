# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 4 | 5.196 | 5.286 | 0.950× [0.887, 1.074] | 27.090 |
| metal / gemm_128x128x128 | 4 | 10.861 | 10.344 | 1.052× [0.992, 1.080] | 27.056 |
| metal / gemm_512x512x512 | 4 | 53.522 | 51.123 | 1.050× [1.029, 1.057] | 48.795 |
| metal / gemm_1024x1024x1024 | 4 | 320.084 | 375.548 | 0.853× [0.851, 0.855] | 291.063 |
| metal / gemm_256x1024x128 | 4 | 19.352 | 18.720 | 1.033× [0.984, 1.053] | 29.140 |
| metal / gemm_1024x128x256 | 4 | 21.374 | 21.092 | 1.015× [0.972, 1.050] | 29.516 |
| metal / gemm_127x193x61 | 4 | 9.132 | 8.898 | 1.019× [1.002, 1.186] | 27.015 |
| metal / gemm_513x257x129 | 4 | 22.866 | 22.028 | 1.029× [1.023, 1.090] | 34.645 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
