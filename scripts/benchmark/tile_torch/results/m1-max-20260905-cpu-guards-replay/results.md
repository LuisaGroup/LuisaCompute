# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| cpu / gemm_32x32x32 | 4 | 5.411 | 5.168 | 1.033× [0.966, 1.205] | 0.887 |
| cpu / gemm_128x128x128 | 4 | 16.461 | 15.209 | 0.974× [0.900, 1.150] | 4.937 |
| cpu / gemm_512x512x512 | 4 | 689.156 | 709.835 | 0.980× [0.928, 1.201] | 148.200 |
| cpu / gemm_1024x1024x1024 | 4 | 5973.946 | 5938.617 | 1.034× [0.899, 1.142] | 1072.482 |
| cpu / gemm_256x1024x128 | 4 | 215.764 | 231.670 | 0.932× [0.714, 1.311] | 69.514 |
| cpu / gemm_1024x128x256 | 4 | 168.559 | 187.285 | 0.911× [0.829, 1.185] | 64.919 |
| cpu / gemm_127x193x61 | 4 | 42.471 | 33.433 | 1.279× [1.160, 1.427] | 6.640 |
| cpu / gemm_513x257x129 | 4 | 516.224 | 282.626 | 1.818× [1.354, 2.601] | 44.910 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
