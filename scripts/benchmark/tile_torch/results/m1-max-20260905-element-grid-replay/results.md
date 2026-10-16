# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / add_1x127 | 4 | 88.431 | 2.552 | 34.488× [33.373, 42.175] | 3.604 |
| metal / add_17x257 | 4 | 215.877 | 2.735 | 79.173× [70.295, 93.541] | 3.897 |
| metal / add_128x1024 | 4 | 44.836 | 5.201 | 8.504× [8.148, 8.778] | 6.362 |
| metal / add_4096x256 | 4 | 78.358 | 18.622 | 4.202× [4.154, 4.296] | 22.397 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
