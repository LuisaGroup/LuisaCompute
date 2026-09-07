# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / layernorm_1x127 | 4 | 131.413 | 4.577 | 28.675× [27.944, 29.063] | 8.239 |
| metal / layernorm_17x257 | 4 | 337.366 | 5.693 | 58.942× [57.105, 64.220] | 8.633 |
| metal / layernorm_128x1024 | 4 | 280.352 | 7.517 | 37.333× [36.900, 37.661] | 13.930 |
| metal / layernorm_64x4096 | 4 | 928.945 | 12.306 | 75.536× [74.338, 82.088] | 24.488 |
| metal / cross_entropy_1x127 | 4 | 62.412 | 4.446 | 14.042× [13.737, 14.854] | 108.512 |
| metal / cross_entropy_17x257 | 4 | 191.603 | 3.228 | 59.357× [53.681, 61.339] | 107.258 |
| metal / cross_entropy_128x1024 | 4 | 74.350 | 4.370 | 17.015× [16.097, 17.463] | 108.205 |
| metal / cross_entropy_64x4096 | 4 | 355.493 | 5.774 | 60.879× [59.618, 63.291] | 111.832 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
