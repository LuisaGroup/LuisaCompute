# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0 | — | — | — | — |
| metal / gemm_128x128x128 | 0 | — | — | — | — |
| metal / gemm_512x512x512 | 0 | — | — | — | — |
| metal / gemm_1024x1024x1024 | 0 | — | — | — | — |
| metal / gemm_256x1024x128 | 0 | — | — | — | — |
| metal / gemm_1024x128x256 | 0 | — | — | — | — |
| metal / gemm_127x193x61 | 0 | — | — | — | — |
| metal / gemm_513x257x129 | 0 | — | — | — | — |

Failed measurements: 27. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
