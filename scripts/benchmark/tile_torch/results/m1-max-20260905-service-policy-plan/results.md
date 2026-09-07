# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T10:24:30.683040+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_37x1537 | 1×1537×1 / 1 | 0 | 5.006 | 30.403 | 5.189 | 32.472 | 0.16× | 215.250 | 326.125 |
| metal | softmax_256x3072 | 1×3072×1 / 1 | 0 | 15.400 | 38.955 | 16.892 | 45.672 | 0.40× | 248.667 | 321.083 |
| metal | softmax_768x6144 | 1×6144×1 / 1 | 0 | 59.404 | 146.019 | 59.813 | 152.203 | 0.41× | 299.250 | 445.333 |
| metal | softmax_64x12289 | 1×12289×1 / 1 | 0 | 23.370 | 39.062 | 31.664 | 55.171 | 0.60× | 303.709 | 402.833 |
| metal | rmsnorm_37x1537 | 1×1537×1 / 1 | 0 | 5.200 | 8.026 | 5.669 | 8.352 | 0.65× | 220.833 | 237.667 |
| metal | rmsnorm_256x3072 | 1×3072×1 / 1 | 0 | 16.132 | 20.381 | 16.491 | 20.746 | 0.79× | 250.916 | 269.583 |
| metal | rmsnorm_768x6144 | 1×6144×1 / 1 | 0 | 63.062 | 88.960 | 65.447 | 90.836 | 0.71× | 284.167 | 339.834 |
| metal | rmsnorm_64x12289 | 1×12289×1 / 1 | 0 | 23.307 | 27.813 | 30.932 | 38.279 | 0.84× | 251.459 | 283.666 |
| metal | layernorm_37x1537 | 1×1537×1 / 1 | 0 | 5.791 | 12.480 | 5.867 | 12.673 | 0.46× | 222.542 | 243.042 |
| metal | layernorm_256x3072 | 1×3072×1 / 1 | 0 | 17.692 | 47.265 | 18.100 | 48.550 | 0.37× | 239.917 | 313.208 |
| metal | layernorm_768x6144 | 1×6144×1 / 1 | 0 | 78.826 | 296.982 | 80.560 | 307.737 | 0.27× | 333.083 | 536.375 |
| metal | layernorm_64x12289 | 1×12289×1 / 1 | 0 | 20.982 | 66.671 | 21.543 | 71.790 | 0.31× | 230.833 | 289.417 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 0.067 | 51.504 | 1.313 | 0.554 | 2.009 | 0.530 | 0.405 | 0.267 |
| metal / softmax_256x3072 | 0.074 | 53.214 | 1.663 | 0.869 | 2.448 | 0.516 | 0.883 | 0.402 |
| metal / softmax_768x6144 | 0.064 | 52.360 | 5.997 | 0.958 | 4.207 | 0.870 | 3.800 | 1.006 |
| metal / softmax_64x12289 | 0.110 | 57.809 | 1.846 | 0.703 | 5.245 | 0.546 | 0.948 | 0.525 |
| metal / rmsnorm_37x1537 | 0.063 | 51.975 | 1.513 | 0.689 | 1.947 | 0.257 | 0.315 | 0.366 |
| metal / rmsnorm_256x3072 | 0.085 | 55.634 | 2.102 | 1.027 | 4.056 | 0.364 | 0.840 | 0.424 |
| metal / rmsnorm_768x6144 | 0.063 | 55.740 | 6.653 | 1.351 | 5.784 | 0.589 | 3.589 | 1.117 |
| metal / rmsnorm_64x12289 | 0.073 | 58.415 | 2.159 | 1.394 | 8.799 | 0.484 | 0.901 | 0.458 |
| metal / layernorm_37x1537 | 0.072 | 62.483 | 1.420 | 0.684 | 3.629 | 0.314 | 0.318 | 0.314 |
| metal / layernorm_256x3072 | 0.071 | 62.181 | 2.194 | 1.744 | 6.416 | 0.643 | 1.495 | 0.503 |
| metal / layernorm_768x6144 | 0.082 | 63.611 | 7.063 | 1.631 | 5.829 | 1.037 | 2.711 | 0.822 |
| metal / layernorm_64x12289 | 0.084 | 70.366 | 2.205 | 1.059 | 9.583 | 0.449 | 0.862 | 0.442 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| softmax_37x1537 / native | 4.899 | 7.208 | 5.006 | 215.250 | 1.009× |
| softmax_37x1537 / torch | 19.360 | 70.250 | 30.403 | 326.125 | 3.997× |
| softmax_256x3072 / native | 14.783 | 19.250 | 15.400 | 248.667 | 0.964× |
| softmax_256x3072 / torch | 29.465 | 49.500 | 38.955 | 321.083 | 2.005× |
| softmax_768x6144 / native | 57.235 | 59.958 | 59.404 | 299.250 | 1.004× |
| softmax_768x6144 / torch | 129.708 | 138.625 | 146.019 | 445.333 | 1.231× |
| softmax_64x12289 / native | 18.722 | 27.833 | 23.370 | 303.709 | 1.015× |
| softmax_64x12289 / torch | 33.178 | 70.250 | 39.062 | 402.833 | 2.390× |
| rmsnorm_37x1537 / native | 5.118 | 7.292 | 5.200 | 220.833 | 1.074× |
| rmsnorm_37x1537 / torch | 5.707 | 8.500 | 8.026 | 237.667 | 1.001× |
| rmsnorm_256x3072 / native | 15.473 | 20.500 | 16.132 | 250.916 | 1.008× |
| rmsnorm_256x3072 / torch | 17.225 | 21.083 | 20.381 | 269.583 | 0.975× |
| rmsnorm_768x6144 / native | 60.682 | 61.000 | 63.062 | 284.167 | 0.991× |
| rmsnorm_768x6144 / torch | 78.275 | 82.125 | 88.960 | 339.834 | 1.005× |
| rmsnorm_64x12289 / native | 19.736 | 24.292 | 23.307 | 251.459 | 1.042× |
| rmsnorm_64x12289 / torch | 18.563 | 26.875 | 27.813 | 283.666 | 1.014× |
| layernorm_37x1537 / native | 5.480 | 8.000 | 5.791 | 222.542 | 1.060× |
| layernorm_37x1537 / torch | 8.713 | 11.875 | 12.480 | 243.042 | 1.031× |
| layernorm_256x3072 / native | 17.126 | 20.625 | 17.692 | 239.917 | 0.977× |
| layernorm_256x3072 / torch | 40.521 | 44.583 | 47.265 | 313.208 | 1.001× |
| layernorm_768x6144 / native | 74.799 | 78.792 | 78.826 | 333.083 | 1.001× |
| layernorm_768x6144 / torch | 281.315 | 277.125 | 296.982 | 536.375 | 0.991× |
| layernorm_64x12289 / native | 20.559 | 23.667 | 20.982 | 230.833 | 0.976× |
| layernorm_64x12289 / torch | 57.491 | 62.667 | 66.671 | 289.417 | 1.004× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| softmax_37x1537 | 4.737 | 59.689 | 7.250 | 75.625 | 215.250 | 326.125 |
| softmax_256x3072 | 14.283 | 49.811 | 18.250 | 50.250 | 248.667 | 321.083 |
| softmax_768x6144 | 57.438 | 139.621 | 61.000 | 141.000 | 299.250 | 445.333 |
| softmax_64x12289 | 19.008 | 64.621 | 24.541 | 69.083 | 303.709 | 402.833 |
| rmsnorm_37x1537 | 5.414 | 5.706 | 7.375 | 8.291 | 220.833 | 237.667 |
| rmsnorm_256x3072 | 15.409 | 16.645 | 18.541 | 20.875 | 250.916 | 269.583 |
| rmsnorm_768x6144 | 60.107 | 78.495 | 62.459 | 81.708 | 284.167 | 339.834 |
| rmsnorm_64x12289 | 20.883 | 19.103 | 25.875 | 25.041 | 251.459 | 283.666 |
| layernorm_37x1537 | 5.889 | 8.887 | 9.333 | 10.292 | 222.542 | 243.042 |
| layernorm_256x3072 | 16.693 | 40.609 | 20.417 | 54.959 | 239.917 | 313.208 |
| layernorm_768x6144 | 74.859 | 278.461 | 76.250 | 280.500 | 333.083 | 536.375 |
| layernorm_64x12289 | 20.290 | 57.745 | 23.458 | 65.292 | 230.833 | 289.417 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / softmax_37x1537 | 2 / 2 | 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 1314.048 |
| metal / softmax_256x3072 | 2 / 2 | 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 1312.552 |
| metal / softmax_768x6144 | 2 / 2 | 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 1801.032 |
| metal / softmax_64x12289 | 2 / 2 | 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 1343.278 |
| metal / rmsnorm_37x1537 | 2 / 2 | 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 1266.882 |
| metal / rmsnorm_256x3072 | 2 / 2 | 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 1279.663 |
| metal / rmsnorm_768x6144 | 2 / 2 | 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 1687.656 |
| metal / rmsnorm_64x12289 | 2 / 2 | 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 1385.501 |
| metal / layernorm_37x1537 | 2 / 2 | 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 1212.082 |
| metal / layernorm_256x3072 | 2 / 2 | 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 1297.188 |
| metal / layernorm_768x6144 | 2 / 2 | 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 1801.054 |
| metal / layernorm_64x12289 | 2 / 2 | 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ autot, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 1475.197 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
