# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T04:50:49.104655+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sigmoid_pair_1x127 | 1×256×1 / 2 | 0 | 113.667 | 15.776 | 116.708 | 17.331 | 7.21× | 595.250 | 258.125 |
| metal | sigmoid_pair_37x1537 | 1×256×1 / 2 | 0 | 267.923 | 20.662 | 269.732 | 21.353 | 12.97× | 1264.250 | 344.708 |
| metal | gelu_pair_1x127 | 1×256×1 / 2 | 0 | 133.059 | 10.152 | 134.824 | 10.654 | 13.11× | 415.166 | 322.167 |
| metal | gelu_pair_37x1537 | 1×256×1 / 2 | 0 | 289.092 | 16.838 | 293.324 | 17.923 | 17.17× | 1072.083 | 259.250 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sigmoid_pair_1x127 | 0.112 | 29.479 | 1.471 | 4.064 | 10.829 | 67.566 | 0.690 | 0.385 |
| metal / sigmoid_pair_37x1537 | 0.059 | 32.710 | 1.202 | 0.375 | 12.972 | 5.665 | 0.553 | 0.355 |
| metal / gelu_pair_1x127 | 0.057 | 30.851 | 1.082 | 1.406 | 9.982 | 0.845 | 0.488 | 0.253 |
| metal / gelu_pair_37x1537 | 0.064 | 32.830 | 1.075 | 0.740 | 12.468 | 4.420 | 0.780 | 0.647 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| sigmoid_pair_1x127 / native | 98.105 | 189.792 | 113.667 | 595.250 | 0.993× |
| sigmoid_pair_1x127 / torch | 7.837 | 20.875 | 15.776 | 258.125 | 1.206× |
| sigmoid_pair_37x1537 / native | 254.410 | 301.333 | 267.923 | 1264.250 | 1.002× |
| sigmoid_pair_37x1537 / torch | 11.318 | 195.542 | 20.662 | 344.708 | 0.851× |
| gelu_pair_1x127 / native | 115.224 | 211.333 | 133.059 | 415.166 | 0.977× |
| gelu_pair_1x127 / torch | 4.283 | 31.125 | 10.152 | 322.167 | 2.671× |
| gelu_pair_37x1537 / native | 279.442 | 357.542 | 289.092 | 1072.083 | 1.002× |
| gelu_pair_37x1537 / torch | 11.255 | 11.250 | 16.838 | 259.250 | 0.683× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| sigmoid_pair_1x127 | 96.771 | 9.144 | 103.417 | 13.125 | 595.250 | 258.125 |
| sigmoid_pair_37x1537 | 254.855 | 11.868 | 236.167 | 14.208 | 1264.250 | 344.708 |
| gelu_pair_1x127 | 112.546 | 11.821 | 113.959 | 6.625 | 415.166 | 322.167 |
| gelu_pair_37x1537 | 280.030 | 8.146 | 278.541 | 13.959 | 1072.083 | 259.250 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
