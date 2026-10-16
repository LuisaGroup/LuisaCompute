# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / rmsnorm_1x127 | 4 | 3.418 | 2.967 | 1.158× [1.116, 1.179] | 7.110 |
| metal / rmsnorm_17x257 | 4 | 4.759 | 3.198 | 1.493× [1.476, 1.501] | 6.399 |
| metal / rmsnorm_64x4096 | 4 | 10.663 | 9.060 | 1.169× [1.156, 1.201] | 12.539 |
| metal / rmsnorm_1024x4096 | 4 | 71.122 | 71.660 | 0.992× [0.987, 1.007] | 74.460 |

## Separately sampled GPU execution

Completed command-buffer GPU timestamps, with no encoder hooks or counter attachments. This includes GPU work/gaps inside each command buffer, not CPU encoding or completion notification, and is not individual-kernel time. Each GPU phase uses its own repetition count. Values are medians of per-round p50 times. Speedups are medians of paired round ratios, with min–max ranges, not confidence intervals. Incomplete GPU control pairs withhold statistics; instrumented compute-pass samples remain diagnostics in JSON.

| Backend / case | GPU pairs | Reference GPU µs/op | Candidate GPU µs/op | Paired GPU speedup [range] | Candidate-run Torch GPU µs/op |
|---|---:|---:|---:|---:|---:|
| metal / rmsnorm_1x127 | 4/4 | 3.421 | 2.829 | 1.208× [1.114, 1.231] | 4.396 |
| metal / rmsnorm_17x257 | 4/4 | 4.748 | 3.204 | 1.469× [1.404, 1.530] | 3.471 |
| metal / rmsnorm_64x4096 | 4/4 | 10.542 | 9.216 | 1.156× [1.104, 1.228] | 9.172 |
| metal / rmsnorm_1024x4096 | 4/4 | 68.942 | 68.148 | 1.015× [0.976, 1.031] | 68.891 |

## Single-call GPU versus end-to-end dispatch

These are separate phases; host samples are uninstrumented. Do not subtract their medians to estimate CPU cost. Torch remains the recorded eager operator sequence, not a compiled fused graph.

| Backend / case | Candidate GPU µs | Candidate E2E µs | Candidate-run Torch GPU µs | Candidate-run Torch E2E µs |
|---|---:|---:|---:|---:|
| metal / rmsnorm_1x127 | 4.896 | 210.021 | 7.771 | 223.500 |
| metal / rmsnorm_17x257 | 5.708 | 217.730 | 7.146 | 223.188 |
| metal / rmsnorm_64x4096 | 12.417 | 270.500 | 12.354 | 250.416 |
| metal / rmsnorm_1024x4096 | 77.625 | 348.812 | 97.229 | 333.729 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
