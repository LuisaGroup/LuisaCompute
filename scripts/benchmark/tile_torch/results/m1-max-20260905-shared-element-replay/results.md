# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gelu_add_1x127 | 4 | 84.540 | 2.585 | 32.701× [31.008, 33.326] | 9.260 |
| metal / gelu_add_17x257 | 4 | 141.631 | 2.637 | 52.636× [50.877, 57.412] | 9.805 |
| metal / gelu_add_128x1024 | 4 | 52.764 | 5.293 | 9.908× [9.644, 11.792] | 16.383 |
| metal / gelu_add_4096x256 | 4 | 75.641 | 19.929 | 3.834× [3.767, 3.886] | 61.022 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
