# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T04:49:34.085418+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sigmoid_pair_1x127 | 1×256×1 / 2 | 0 | 113.769 | 19.097 | 126.442 | 19.729 | 5.96× | 354.625 | 257.375 |
| metal | sigmoid_pair_37x1537 | 1×256×1 / 2 | 0 | 265.261 | 18.388 | 270.137 | 22.787 | 14.43× | 533.583 | 252.291 |
| metal | gelu_pair_1x127 | 1×256×1 / 2 | 0 | 127.620 | 11.819 | 127.959 | 11.834 | 10.80× | 679.250 | 231.417 |
| metal | gelu_pair_37x1537 | 1×256×1 / 2 | 0 | 289.224 | 14.660 | 301.374 | 14.720 | 19.73× | 623.041 | 242.375 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sigmoid_pair_1x127 | 0.100 | 30.817 | 2.015 | 6.252 | 97.253 | 312.044 | 0.498 | 0.733 |
| metal / sigmoid_pair_37x1537 | 0.056 | 34.315 | 1.517 | 0.395 | 83.118 | 5.339 | 0.575 | 0.298 |
| metal / gelu_pair_1x127 | 0.067 | 32.147 | 1.780 | 2.622 | 82.039 | 7.616 | 0.604 | 0.310 |
| metal / gelu_pair_37x1537 | 0.071 | 35.142 | 1.155 | 1.049 | 82.854 | 5.091 | 0.591 | 0.301 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| sigmoid_pair_1x127 / native | 113.713 | 99.958 | 113.769 | 354.625 | 0.998× |
| sigmoid_pair_1x127 / torch | 14.973 | 13.375 | 19.097 | 257.375 | 0.626× |
| sigmoid_pair_37x1537 / native | 259.121 | 302.125 | 265.261 | 533.583 | 0.986× |
| sigmoid_pair_37x1537 / torch | 9.515 | 12.375 | 18.388 | 252.291 | 1.022× |
| gelu_pair_1x127 / native | 124.049 | 121.000 | 127.620 | 679.250 | 1.075× |
| gelu_pair_1x127 / torch | 7.416 | 7.083 | 11.819 | 231.417 | 0.692× |
| gelu_pair_37x1537 / native | 286.858 | 374.542 | 289.224 | 623.041 | 0.979× |
| gelu_pair_37x1537 / torch | 8.830 | 9.000 | 14.660 | 242.375 | 0.775× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| sigmoid_pair_1x127 | 114.424 | 9.715 | 101.791 | 286.875 | 354.625 | 257.375 |
| sigmoid_pair_37x1537 | 255.315 | 9.725 | 248.167 | 27.916 | 533.583 | 252.291 |
| gelu_pair_1x127 | 126.878 | 5.128 | 114.583 | 8.250 | 679.250 | 231.417 |
| gelu_pair_37x1537 | 282.143 | 7.105 | 283.167 | 11.209 | 623.041 | 242.375 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
