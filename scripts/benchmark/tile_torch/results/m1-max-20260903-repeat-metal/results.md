# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 4 | 5.289 | 4.214 | 1.289× [1.098, 1.401] | 27.286 |
| metal / gemm_128x128x128 | 4 | 14.057 | 7.389 | 1.895× [1.714, 2.093] | 27.286 |
| metal / gemm_512x512x512 | 4 | 348.420 | 125.049 | 2.787× [2.769, 2.795] | 48.283 |
| metal / gemm_1024x1024x1024 | 4 | 2483.813 | 931.848 | 2.667× [2.660, 2.672] | 288.684 |
| metal / gemm_256x1024x128 | 4 | 87.082 | 36.836 | 2.357× [2.294, 2.422] | 30.001 |
| metal / gemm_1024x128x256 | 4 | 94.889 | 36.247 | 2.627× [2.597, 2.659] | 30.134 |
| metal / gemm_127x193x61 | 4 | 9.812 | 8.004 | 1.228× [1.077, 1.414] | 27.107 |
| metal / gemm_513x257x129 | 4 | 63.612 | 35.773 | 1.766× [1.749, 1.859] | 34.556 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
