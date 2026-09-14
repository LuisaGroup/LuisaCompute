# Frozen-schedule repeat measurements

Every row is freshly captured/JIT-compiled and fully validated. Parameters are frozen; no search or minimum-of-rounds selection.

Times below are medians of per-round p50 synchronized host-wall measurements, including dispatch. The speedup range is min–max of paired round ratios, not a confidence interval.

| Backend / case | Valid pairs | Reference µs | Candidate µs | Paired speedup median [range] | Candidate-run PyTorch µs |
|---|---:|---:|---:|---:|---:|
| metal / softmax_37x769 | 4 | 3.478 | 5.044 | 0.683× [0.664, 0.701] | 31.944 |
| metal / softmax_1024x1024 | 4 | 17.456 | 19.424 | 0.890× [0.885, 0.911] | 49.839 |
| metal / softmax_16384x257 | 4 | 63.092 | 50.814 | 1.236× [1.233, 1.258] | 184.715 |
| metal / softmax_4096x1024 | 4 | 50.371 | 55.348 | 0.910× [0.905, 0.922] | 109.409 |
| metal / rmsnorm_37x769 | 4 | 3.489 | 5.391 | 0.648× [0.643, 0.650] | 6.846 |
| metal / rmsnorm_1024x1024 | 4 | 18.012 | 20.592 | 0.884× [0.864, 0.885] | 24.522 |
| metal / rmsnorm_16384x257 | 4 | 51.473 | 51.962 | 0.999× [0.964, 1.022] | 67.366 |
| metal / rmsnorm_4096x1024 | 4 | 52.666 | 55.478 | 0.949× [0.902, 0.963] | 74.904 |
| metal / layernorm_37x769 | 4 | 3.915 | 5.763 | 0.687× [0.649, 0.704] | 10.463 |
| metal / layernorm_1024x1024 | 4 | 19.428 | 19.928 | 0.975× [0.932, 0.981] | 47.688 |
| metal / layernorm_16384x257 | 4 | 62.756 | 52.837 | 1.187× [1.163, 1.208] | 121.138 |
| metal / layernorm_4096x1024 | 4 | 56.005 | 55.435 | 1.007× [0.977, 1.021] | 149.491 |

## Separately sampled GPU execution

Completed command-buffer GPU timestamps, with no encoder hooks or counter attachments. This includes GPU work/gaps inside each command buffer, not CPU encoding or completion notification, and is not individual-kernel time. Each GPU phase uses its own repetition count. Values are medians of per-round p50 times. Speedups are medians of paired round ratios, with min–max ranges, not confidence intervals. Incomplete GPU control pairs withhold statistics; instrumented compute-pass samples remain diagnostics in JSON.

| Backend / case | GPU pairs | Reference GPU µs/op | Candidate GPU µs/op | Paired GPU speedup [range] | Candidate-run Torch GPU µs/op |
|---|---:|---:|---:|---:|---:|
| metal / softmax_37x769 | 4/4 | 3.315 | 4.685 | 0.716× [0.666, 0.726] | 21.029 |
| metal / softmax_1024x1024 | 4/4 | 16.197 | 18.474 | 0.872× [0.852, 0.916] | 42.874 |
| metal / softmax_16384x257 | 4/4 | 59.436 | 48.560 | 1.220× [1.208, 1.277] | 172.975 |
| metal / softmax_4096x1024 | 4/4 | 48.151 | 53.331 | 0.903× [0.882, 0.933] | 100.859 |
| metal / rmsnorm_37x769 | 4/4 | 3.155 | 4.935 | 0.635× [0.584, 0.662] | 3.910 |
| metal / rmsnorm_1024x1024 | 4/4 | 17.173 | 19.281 | 0.896× [0.863, 0.906] | 20.807 |
| metal / rmsnorm_16384x257 | 4/4 | 48.783 | 49.462 | 0.992× [0.978, 1.016] | 59.843 |
| metal / rmsnorm_4096x1024 | 4/4 | 49.347 | 53.390 | 0.923× [0.917, 0.928] | 68.387 |
| metal / layernorm_37x769 | 4/4 | 3.514 | 5.274 | 0.667× [0.643, 0.723] | 5.751 |
| metal / layernorm_1024x1024 | 4/4 | 18.117 | 18.961 | 0.959× [0.918, 0.993] | 41.364 |
| metal / layernorm_16384x257 | 4/4 | 60.993 | 50.509 | 1.210× [1.182, 1.230] | 115.731 |
| metal / layernorm_4096x1024 | 4/4 | 53.987 | 54.390 | 0.996× [0.974, 1.036] | 137.041 |

## Single-call GPU versus end-to-end dispatch

These are separate phases; host samples are uninstrumented. Do not subtract their medians to estimate CPU cost. Torch remains the recorded eager operator sequence, not a compiled fused graph.

| Backend / case | Candidate GPU µs | Candidate E2E µs | Candidate-run Torch GPU µs | Candidate-run Torch E2E µs |
|---|---:|---:|---:|---:|
| metal / softmax_37x769 | 7.250 | 225.834 | 101.104 | 372.250 |
| metal / softmax_1024x1024 | 23.375 | 287.312 | 66.542 | 350.125 |
| metal / softmax_16384x257 | 81.417 | 291.792 | 191.708 | 515.125 |
| metal / softmax_4096x1024 | 77.875 | 316.291 | 119.208 | 484.562 |
| metal / rmsnorm_37x769 | 7.500 | 224.271 | 8.458 | 263.062 |
| metal / rmsnorm_1024x1024 | 27.188 | 273.647 | 25.667 | 291.125 |
| metal / rmsnorm_16384x257 | 76.125 | 314.604 | 82.458 | 314.021 |
| metal / rmsnorm_4096x1024 | 69.625 | 339.146 | 85.375 | 344.333 |
| metal / layernorm_37x769 | 8.375 | 214.938 | 10.562 | 259.625 |
| metal / layernorm_1024x1024 | 24.521 | 250.853 | 44.646 | 297.479 |
| metal / layernorm_16384x257 | 62.333 | 316.688 | 113.542 | 377.062 |
| metal / layernorm_4096x1024 | 76.937 | 305.562 | 136.979 | 411.895 |

Failed measurements: 0. Raw samples, frozen schedules, hashes, ordering, and errors are in `results.json`.
