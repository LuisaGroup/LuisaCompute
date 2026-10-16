# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / residual_layernorm_1x127 | 4 | 3.692 | 3.506 | 1.057× [1.027, 1.137] | 10.992 |
| metal / residual_layernorm_17x257 | 4 | 3.648 | 3.632 | 1.008× [0.957, 1.039] | 11.896 |
| metal / residual_layernorm_128x1024 | 4 | 8.244 | 6.084 | 1.354× [1.313, 1.366] | 18.835 |
| metal / residual_layernorm_64x4096 | 4 | 13.548 | 9.591 | 1.421× [1.392, 1.471] | 26.776 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
