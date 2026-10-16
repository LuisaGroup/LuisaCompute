# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| cpu / add_1x127 | 6 | 0.068 | 0.068 | 1.001× [0.978, 1.067] | 0.548 |
| cpu / add_17x257 | 6 | 0.421 | 0.418 | 1.001× [0.998, 1.018] | 0.934 |
| cpu / add_128x1024 | 6 | 4.698 | 4.713 | 1.004× [0.885, 1.226] | 38.289 |
| cpu / add_4096x256 | 6 | 32.807 | 32.207 | 1.022× [0.958, 1.437] | 84.070 |
| cpu / sum_1x127 | 6 | 0.064 | 0.024 | 2.708× [2.587, 2.774] | 0.772 |
| cpu / sum_17x257 | 6 | 2.186 | 0.375 | 5.828× [5.564, 5.939] | 1.060 |
| cpu / sum_128x1024 | 6 | 16.640 | 3.703 | 4.581× [3.534, 4.978] | 37.578 |
| cpu / sum_64x4096 | 6 | 33.738 | 5.512 | 6.123× [5.267, 7.228] | 40.651 |
| cpu / softmax_1x127 | 6 | 0.551 | 0.126 | 4.357× [4.276, 4.370] | 0.619 |
| cpu / softmax_17x257 | 6 | 5.436 | 2.555 | 2.098× [1.980, 2.286] | 33.428 |
| cpu / softmax_128x1024 | 6 | 79.242 | 14.527 | 5.460× [5.159, 5.609] | 88.699 |
| cpu / softmax_64x4096 | 6 | 156.785 | 41.876 | 3.753× [3.524, 4.113] | 128.818 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
