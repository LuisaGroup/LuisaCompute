# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / softmax_37x769 | 4 | 7.375 | 6.399 | 1.138× [1.093, 1.192] | 33.223 |
| metal / softmax_1024x1024 | 4 | 20.308 | 23.522 | 0.873× [0.816, 0.881] | 62.479 |
| metal / softmax_16384x257 | 4 | 63.143 | 61.694 | 1.020× [0.986, 1.071] | 239.981 |
| metal / softmax_4096x1024 | 4 | 65.243 | 71.109 | 0.920× [0.800, 0.954] | 149.190 |
| metal / rmsnorm_37x769 | 4 | 8.020 | 6.775 | 1.165× [0.933, 1.191] | 8.369 |
| metal / rmsnorm_1024x1024 | 4 | 26.444 | 24.227 | 1.063× [1.014, 1.115] | 30.498 |
| metal / rmsnorm_16384x257 | 4 | 60.991 | 62.541 | 0.957× [0.826, 0.996] | 84.161 |
| metal / rmsnorm_4096x1024 | 4 | 73.008 | 70.777 | 1.055× [0.995, 1.068] | 96.196 |
| metal / layernorm_37x769 | 4 | 8.321 | 7.196 | 1.143× [1.107, 1.191] | 12.192 |
| metal / layernorm_1024x1024 | 4 | 25.022 | 26.323 | 0.969× [0.924, 1.010] | 60.721 |
| metal / layernorm_16384x257 | 4 | 67.583 | 68.523 | 0.996× [0.955, 1.023] | 159.046 |
| metal / layernorm_4096x1024 | 4 | 68.851 | 71.954 | 0.979× [0.941, 1.022] | 188.000 |

## Separately sampled GPU execution

Completed command-buffer GPU timestamps, with no encoder hooks or counter attachments. This includes GPU work/gaps inside each command buffer, not CPU encoding or completion notification, and is not individual-kernel time. Each GPU phase uses its own repetition count. Values are medians of per-round p50 times. Speedups are medians of paired round ratios, with min–max ranges, not confidence intervals. Incomplete GPU control pairs withhold statistics; instrumented compute-pass samples remain diagnostics in JSON.

| Backend / case | GPU pairs | Reference GPU µs/op | Candidate GPU µs/op | Paired GPU speedup [range] | Candidate-run Torch GPU µs/op |
|---|---:|---:|---:|---:|---:|
| metal / softmax_37x769 | 4/4 | 5.235 | 4.719 | 1.132× [1.051, 1.192] | 15.606 |
| metal / softmax_1024x1024 | 4/4 | 16.790 | 21.430 | 0.862× [0.722, 0.923] | 42.981 |
| metal / softmax_16384x257 | 4/4 | 52.321 | 51.147 | 0.997× [0.963, 1.085] | 205.231 |
| metal / softmax_4096x1024 | 4/4 | 52.727 | 55.668 | 0.929× [0.869, 0.970] | 131.804 |
| metal / rmsnorm_37x769 | 4/4 | 5.893 | 5.060 | 1.188× [1.078, 1.530] | 4.138 |
| metal / rmsnorm_1024x1024 | 4/4 | 19.479 | 19.185 | 1.026× [0.993, 1.081] | 20.641 |
| metal / rmsnorm_16384x257 | 4/4 | 52.831 | 49.288 | 1.030× [0.997, 1.149] | 60.849 |
| metal / rmsnorm_4096x1024 | 4/4 | 58.212 | 55.143 | 1.055× [1.018, 1.130] | 69.641 |
| metal / layernorm_37x769 | 4/4 | 6.186 | 5.408 | 1.161× [1.079, 1.200] | 5.661 |
| metal / layernorm_1024x1024 | 4/4 | 18.986 | 19.192 | 0.998× [0.792, 1.098] | 42.215 |
| metal / layernorm_16384x257 | 4/4 | 53.593 | 53.846 | 0.985× [0.931, 1.075] | 142.503 |
| metal / layernorm_4096x1024 | 4/4 | 56.376 | 53.818 | 1.048× [0.923, 1.129] | 167.925 |

## Single-call GPU versus end-to-end dispatch

These are separate phases; host samples are uninstrumented. Do not subtract their medians to estimate CPU cost. Torch remains the recorded eager operator sequence, not a compiled fused graph.

| Backend / case | Candidate GPU µs | Candidate E2E µs | Candidate-run Torch GPU µs | Candidate-run Torch E2E µs |
|---|---:|---:|---:|---:|
| metal / softmax_37x769 | 7.812 | 212.625 | 71.604 | 374.792 |
| metal / softmax_1024x1024 | 30.583 | 297.250 | 67.354 | 398.083 |
| metal / softmax_16384x257 | 91.667 | 317.000 | 193.312 | 554.792 |
| metal / softmax_4096x1024 | 99.646 | 365.292 | 128.813 | 486.979 |
| metal / rmsnorm_37x769 | 8.292 | 235.105 | 13.479 | 230.938 |
| metal / rmsnorm_1024x1024 | 27.542 | 252.625 | 28.583 | 292.812 |
| metal / rmsnorm_16384x257 | 90.688 | 317.104 | 89.938 | 370.000 |
| metal / rmsnorm_4096x1024 | 92.792 | 320.167 | 99.313 | 352.042 |
| metal / layernorm_37x769 | 9.104 | 246.584 | 10.500 | 339.166 |
| metal / layernorm_1024x1024 | 35.208 | 263.021 | 46.750 | 349.104 |
| metal / layernorm_16384x257 | 99.208 | 299.562 | 115.917 | 386.291 |
| metal / layernorm_4096x1024 | 91.479 | 325.229 | 136.563 | 457.771 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
