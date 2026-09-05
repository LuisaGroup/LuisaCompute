# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T10:21:54.105436+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_37x1537 | 1×1537×1 / 1 | 0 | 4.589 | 31.248 | 4.625 | 38.342 | 0.15× | 223.542 | 345.583 |
| metal | softmax_256x3072 | 1×3072×1 / 1 | 0 | 16.100 | 42.139 | 16.634 | 48.202 | 0.38× | 257.625 | 315.458 |
| metal | softmax_768x6144 | 1×6144×1 / 1 | 0 | 82.228 | 142.560 | 84.673 | 150.616 | 0.58× | 330.541 | 510.459 |
| metal | softmax_64x12289 | 1×12289×1 / 1 | 0 | 24.416 | 41.390 | 27.932 | 42.043 | 0.59× | 237.000 | 375.000 |
| metal | rmsnorm_37x1537 | 1×1537×1 / 1 | 0 | 5.142 | 7.698 | 5.364 | 8.206 | 0.67× | 216.458 | 255.666 |
| metal | rmsnorm_256x3072 | 1×3072×1 / 1 | 0 | 17.404 | 20.995 | 17.804 | 22.123 | 0.83× | 255.417 | 275.792 |
| metal | rmsnorm_768x6144 | 1×6144×1 / 1 | 0 | 82.557 | 86.111 | 86.440 | 87.535 | 0.96× | 365.084 | 323.834 |
| metal | rmsnorm_64x12289 | 1×12289×1 / 1 | 0 | 20.138 | 22.813 | 20.585 | 23.124 | 0.88× | 256.500 | 266.000 |
| metal | layernorm_37x1537 | 1×1537×1 / 1 | 0 | 5.629 | 17.377 | 5.851 | 18.136 | 0.32× | 218.916 | 257.083 |
| metal | layernorm_256x3072 | 1×3072×1 / 1 | 0 | 19.945 | 50.894 | 22.522 | 70.224 | 0.39× | 776.042 | 276.125 |
| metal | layernorm_768x6144 | 1×6144×1 / 1 | 0 | 96.594 | 298.303 | 99.365 | 302.251 | 0.32× | 343.708 | 566.417 |
| metal | layernorm_64x12289 | 1×12289×1 / 1 | 0 | 25.744 | 63.171 | 28.527 | 64.103 | 0.41× | 261.291 | 320.708 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 0.092 | 51.871 | 1.152 | 5.224 | 65.904 | 49.608 | 0.422 | 0.720 |
| metal / softmax_256x3072 | 0.068 | 49.287 | 2.661 | 1.495 | 64.767 | 6.143 | 0.867 | 0.361 |
| metal / softmax_768x6144 | 0.078 | 49.290 | 7.433 | 1.279 | 66.594 | 44.942 | 3.811 | 1.006 |
| metal / softmax_64x12289 | 0.066 | 51.915 | 2.149 | 0.772 | 72.528 | 2.733 | 0.722 | 1.158 |
| metal / rmsnorm_37x1537 | 0.062 | 52.546 | 1.368 | 0.923 | 58.466 | 254.173 | 0.304 | 0.309 |
| metal / rmsnorm_256x3072 | 0.061 | 51.086 | 2.194 | 1.816 | 59.803 | 4.561 | 0.840 | 0.394 |
| metal / rmsnorm_768x6144 | 0.061 | 51.270 | 6.437 | 17.494 | 63.874 | 7.238 | 3.821 | 1.228 |
| metal / rmsnorm_64x12289 | 0.075 | 52.252 | 2.955 | 0.848 | 66.749 | 0.469 | 0.939 | 0.463 |
| metal / layernorm_37x1537 | 0.078 | 59.996 | 1.467 | 0.759 | 63.903 | 1.063 | 1.089 | 0.344 |
| metal / layernorm_256x3072 | 0.083 | 60.012 | 2.124 | 2.244 | 69.836 | 0.555 | 0.965 | 0.460 |
| metal / layernorm_768x6144 | 0.075 | 59.808 | 6.628 | 17.280 | 70.454 | 1.369 | 3.793 | 0.916 |
| metal / layernorm_64x12289 | 0.074 | 63.533 | 2.611 | 0.913 | 82.536 | 0.358 | 0.825 | 0.430 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| softmax_37x1537 / native | 4.369 | 8.208 | 4.589 | 223.542 | 0.993× |
| softmax_37x1537 / torch | 20.217 | 75.583 | 31.248 | 345.583 | 4.331× |
| softmax_256x3072 / native | 15.748 | 19.875 | 16.100 | 257.625 | 0.975× |
| softmax_256x3072 / torch | 29.108 | 49.708 | 42.139 | 315.458 | 2.049× |
| softmax_768x6144 / native | 78.202 | 79.083 | 82.228 | 330.541 | 0.997× |
| softmax_768x6144 / torch | 128.692 | 142.125 | 142.560 | 510.459 | 1.234× |
| softmax_64x12289 / native | 21.877 | 29.208 | 24.416 | 237.000 | 1.032× |
| softmax_64x12289 / torch | 33.146 | 64.875 | 41.390 | 375.000 | 2.345× |
| rmsnorm_37x1537 / native | 5.232 | 7.500 | 5.142 | 216.458 | 0.933× |
| rmsnorm_37x1537 / torch | 5.507 | 8.333 | 7.698 | 255.666 | 0.996× |
| rmsnorm_256x3072 / native | 16.910 | 20.417 | 17.404 | 255.417 | 0.985× |
| rmsnorm_256x3072 / torch | 27.652 | 44.000 | 20.995 | 275.792 | 0.945× |
| rmsnorm_768x6144 / native | 78.286 | 79.833 | 82.557 | 365.084 | 1.002× |
| rmsnorm_768x6144 / torch | 78.538 | 81.292 | 86.111 | 323.834 | 0.998× |
| rmsnorm_64x12289 / native | 20.236 | 23.292 | 20.138 | 256.500 | 0.976× |
| rmsnorm_64x12289 / torch | 19.184 | 28.125 | 22.813 | 266.000 | 0.979× |
| layernorm_37x1537 / native | 5.255 | 8.000 | 5.629 | 218.916 | 0.951× |
| layernorm_37x1537 / torch | 11.988 | 16.917 | 17.377 | 257.083 | 1.016× |
| layernorm_256x3072 / native | 19.467 | 30.583 | 19.945 | 776.042 | 1.119× |
| layernorm_256x3072 / torch | 40.689 | 43.875 | 50.894 | 276.125 | 1.027× |
| layernorm_768x6144 / native | 92.518 | 93.583 | 96.594 | 343.708 | 0.997× |
| layernorm_768x6144 / torch | 280.247 | 276.542 | 298.303 | 566.417 | 0.998× |
| layernorm_64x12289 / native | 24.854 | 29.167 | 25.744 | 261.291 | 1.001× |
| layernorm_64x12289 / torch | 55.643 | 63.375 | 63.171 | 320.708 | 0.995× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| softmax_37x1537 | 4.337 | 68.109 | 7.125 | 72.417 | 223.542 | 345.583 |
| softmax_256x3072 | 15.357 | 48.355 | 20.125 | 50.167 | 257.625 | 315.458 |
| softmax_768x6144 | 78.245 | 141.711 | 80.167 | 148.208 | 330.541 | 510.459 |
| softmax_64x12289 | 22.628 | 64.557 | 25.750 | 65.750 | 237.000 | 375.000 |
| rmsnorm_37x1537 | 4.836 | 5.258 | 7.500 | 8.500 | 216.458 | 255.666 |
| rmsnorm_256x3072 | 16.394 | 36.107 | 20.250 | 43.958 | 255.417 | 275.792 |
| rmsnorm_768x6144 | 78.144 | 78.595 | 79.625 | 80.750 | 365.084 | 323.834 |
| rmsnorm_64x12289 | 19.751 | 19.460 | 23.041 | 24.625 | 256.500 | 266.000 |
| layernorm_37x1537 | 4.999 | 11.698 | 7.667 | 15.875 | 218.916 | 257.083 |
| layernorm_256x3072 | 21.791 | 41.471 | 25.916 | 43.959 | 776.042 | 276.125 |
| layernorm_768x6144 | 92.536 | 280.779 | 93.291 | 277.791 | 343.708 | 566.417 |
| layernorm_64x12289 | 24.820 | 55.962 | 29.041 | 63.042 | 261.291 | 320.708 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
