# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T05:08:10.991801+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sigmoid_pair_1x127 | 1×256×1 / 2 | 0 | 2.978 | 19.443 | 3.119 | 22.488 | 0.15× | 399.500 | 336.459 |
| metal | sigmoid_pair_37x1537 | 1×256×1 / 2 | 0 | 5.836 | 20.925 | 6.546 | 22.744 | 0.28× | 254.500 | 261.625 |
| metal | sigmoid_pair_1024x4096 | 1×256×1 / 2 | 0 | 128.202 | 291.371 | 146.815 | 337.839 | 0.44× | 476.333 | 680.333 |
| metal | sigmoid_pair_4096x4096 | 1×256×1 / 2 | 0 | 759.077 | 1975.475 | 797.910 | 2067.895 | 0.38× | 1399.333 | 1749.833 |
| metal | gelu_pair_1x127 | 1×256×1 / 2 | 0 | 5.096 | 15.389 | 5.388 | 20.047 | 0.33× | 168.375 | 681.625 |
| metal | gelu_pair_37x1537 | 1×256×1 / 2 | 0 | 11.052 | 16.576 | 11.143 | 20.893 | 0.67× | 222.291 | 240.041 |
| metal | gelu_pair_1024x4096 | 1×256×1 / 2 | 0 | 152.026 | 211.079 | 156.288 | 244.779 | 0.72× | 385.583 | 697.292 |
| metal | gelu_pair_4096x4096 | 1×256×1 / 2 | 0 | 754.000 | 1333.679 | 870.400 | 1571.231 | 0.57× | 893.750 | 1267.125 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sigmoid_pair_1x127 | 0.097 | 31.002 | 1.518 | 5.864 | 50.360 | 69.504 | 0.554 | 0.387 |
| metal / sigmoid_pair_37x1537 | 0.062 | 31.798 | 1.228 | 1.839 | 53.885 | 5.004 | 0.687 | 0.368 |
| metal / sigmoid_pair_1024x4096 | 0.057 | 31.096 | 5.841 | 44.702 | 55.682 | 64.114 | 10.070 | 1.239 |
| metal / sigmoid_pair_4096x4096 | 0.049 | 35.409 | 26.467 | 3.096 | 7.553 | 2.535 | 31.622 | 4.959 |
| metal / gelu_pair_1x127 | 0.062 | 32.425 | 1.653 | 8.303 | 61.925 | 0.821 | 0.729 | 0.473 |
| metal / gelu_pair_37x1537 | 0.068 | 35.232 | 1.210 | 1.077 | 59.774 | 1.520 | 0.608 | 0.274 |
| metal / gelu_pair_1024x4096 | 0.069 | 32.971 | 6.595 | 19.071 | 55.054 | 13.743 | 9.357 | 0.959 |
| metal / gelu_pair_4096x4096 | 0.078 | 33.864 | 20.259 | 4.977 | 18.543 | 3.567 | 51.961 | 3.331 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| sigmoid_pair_1x127 / native | 2.156 | 4.542 | 2.978 | 399.500 | 1.971× |
| sigmoid_pair_1x127 / torch | 13.088 | 248.125 | 19.443 | 336.459 | 1.949× |
| sigmoid_pair_37x1537 / native | 7.667 | 5.917 | 5.836 | 254.500 | 0.944× |
| sigmoid_pair_37x1537 / torch | 9.872 | 22.667 | 20.925 | 261.625 | 1.000× |
| sigmoid_pair_1024x4096 / native | 130.508 | 379.667 | 128.202 | 476.333 | 1.004× |
| sigmoid_pair_1024x4096 / torch | 273.951 | 248.125 | 291.371 | 680.333 | 0.983× |
| sigmoid_pair_4096x4096 / native | 732.827 | 948.542 | 759.077 | 1399.333 | 0.787× |
| sigmoid_pair_4096x4096 / torch | 1343.442 | 1308.625 | 1975.475 | 1749.833 | 1.307× |
| gelu_pair_1x127 / native | 6.463 | 7.875 | 5.096 | 168.375 | 0.757× |
| gelu_pair_1x127 / torch | 5.191 | 9.083 | 15.389 | 681.625 | 1.139× |
| gelu_pair_37x1537 / native | 7.600 | 9.542 | 11.052 | 222.291 | 0.856× |
| gelu_pair_37x1537 / torch | 8.189 | 11.500 | 16.576 | 240.041 | 0.980× |
| gelu_pair_1024x4096 / native | 131.299 | 137.042 | 152.026 | 385.583 | 0.812× |
| gelu_pair_1024x4096 / torch | 199.819 | 175.250 | 211.079 | 697.292 | 1.005× |
| gelu_pair_4096x4096 / native | 600.607 | 833.500 | 754.000 | 893.750 | 1.208× |
| gelu_pair_4096x4096 / torch | 1026.321 | 948.417 | 1333.679 | 1267.125 | 1.246× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| sigmoid_pair_1x127 | 4.249 | 37.079 | 5.000 | 18.292 | 399.500 | 336.459 |
| sigmoid_pair_37x1537 | 3.880 | 9.673 | 7.250 | 15.042 | 254.500 | 261.625 |
| sigmoid_pair_1024x4096 | 132.703 | 269.319 | 128.250 | 518.875 | 476.333 | 680.333 |
| sigmoid_pair_4096x4096 | 571.734 | 1756.342 | 581.375 | 1338.458 | 1399.333 | 1749.833 |
| gelu_pair_1x127 | 3.553 | 5.911 | 289.333 | 51.750 | 168.375 | 681.625 |
| gelu_pair_37x1537 | 6.508 | 8.022 | 11.375 | 9.750 | 222.291 | 240.041 |
| gelu_pair_1024x4096 | 106.706 | 200.918 | 128.125 | 184.166 | 385.583 | 697.292 |
| gelu_pair_4096x4096 | 725.470 | 1255.446 | 575.583 | 981.167 | 893.750 | 1267.125 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
