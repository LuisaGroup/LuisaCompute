# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T03:53:15.567702+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_1024x1024x1024 | 128×32×1024 / 1 | 1 | 283.557 | 300.711 | 288.033 | 303.555 | 0.94× | 512.625 | 687.500 |
| metal | gemm_4097x4097x4096 | 128×32×1024 / 1 | 1 | 53494.500 | 21599.959 | 53689.275 | 21875.742 | 2.48× | 53262.125 | 21596.500 |
| metal | gemm_2049x4097x1025 | 128×32×4096 / 1 | 1 | 10880.688 | 2682.077 | 11205.362 | 2714.133 | 4.06× | 11012.125 | 2934.916 |
| metal | gemm_1025x1025x1024 | 128×32×4096 / 1 | 1 | 4139.875 | 408.514 | 4155.896 | 424.801 | 10.13× | 4255.459 | 718.542 |
| metal | gemm_129x257x61 | 128×32×4096 / 1 | 1 | 300.512 | 29.613 | 307.167 | 29.777 | 10.15× | 544.417 | 296.583 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_1024x1024x1024 | 0.060 | 32.513 | 2.999 | 1.537 | 3.804 | 1.725 | 1.016 | 0.554 |
| metal / gemm_4097x4097x4096 | 0.051 | 42.991 | 29.357 | 19.555 | 71.988 | 35.124 | 14.650 | 2.967 |
| metal / gemm_2049x4097x1025 | 0.060 | 40.234 | 9.096 | 18.089 | 24.202 | 11.263 | 11.758 | 1.091 |
| metal / gemm_1025x1025x1024 | 0.059 | 41.024 | 2.867 | 1.717 | 11.411 | 1.757 | 1.381 | 0.464 |
| metal / gemm_129x257x61 | 0.057 | 41.309 | 2.572 | 0.877 | 2.776 | 1.628 | 0.333 | 0.301 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_1024x1024x1024 / native | 278.820 | 269.417 | 283.557 | 512.625 | 1.001× |
| gemm_1024x1024x1024 / torch | 287.738 | 278.458 | 300.711 | 687.500 | 1.075× |
| gemm_1024x1024x1024 / system | 279.839 | 269.458 | 288.134 | 505.666 | 1.084× |
| gemm_4097x4097x4096 / native | 51361.500 | 53130.875 | 53494.500 | 53262.125 | 1.009× |
| gemm_4097x4097x4096 / torch | 20354.458 | 20520.083 | 21599.959 | 21596.500 | 0.981× |
| gemm_4097x4097x4096 / system | 22038.333 | 21820.125 | 23031.417 | 22206.125 | 0.996× |
| gemm_2049x4097x1025 / native | 10570.813 | 10540.500 | 10880.688 | 11012.125 | 1.002× |
| gemm_2049x4097x1025 / torch | 2580.952 | 2517.708 | 2682.077 | 2934.916 | 1.027× |
| gemm_2049x4097x1025 / system | 2807.250 | 2621.125 | 2883.833 | 2892.083 | 1.009× |
| gemm_1025x1025x1024 / native | 3801.490 | 3737.750 | 4139.875 | 4255.459 | 0.997× |
| gemm_1025x1025x1024 / torch | 392.484 | 388.708 | 408.514 | 718.542 | 1.059× |
| gemm_1025x1025x1024 / system | 486.929 | 474.500 | 500.050 | 718.292 | 1.056× |
| gemm_129x257x61 / native | 294.508 | 297.250 | 300.512 | 544.417 | 1.000× |
| gemm_129x257x61 / torch | 12.917 | 18.625 | 29.613 | 296.583 | 2.034× |
| gemm_129x257x61 / system | 14.010 | 18.125 | 16.440 | 260.917 | 1.763× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_1024x1024x1024 | 279.288 | 290.100 | 269.833 | 279.834 | 512.625 | 687.500 |
| gemm_4097x4097x4096 | 52486.125 | 20404.542 | 53794.500 | 21396.625 | 53262.125 | 21596.500 |
| gemm_2049x4097x1025 | 10585.250 | 2646.720 | 10849.500 | 2462.375 | 11012.125 | 2934.916 |
| gemm_1025x1025x1024 | 3802.438 | 396.381 | 3715.500 | 376.167 | 4255.459 | 718.542 |
| gemm_129x257x61 | 294.440 | 17.045 | 297.625 | 20.334 | 544.417 | 296.583 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_1024x1024x1024 | mps_matrix_multiplication | 288.134 | 0.984× | 505.666 |
| metal / gemm_4097x4097x4096 | mps_matrix_multiplication | 23031.417 | 2.323× | 22206.125 |
| metal / gemm_2049x4097x1025 | mps_matrix_multiplication | 2883.833 | 3.773× | 2892.083 |
| metal / gemm_1025x1025x1024 | mps_matrix_multiplication | 500.050 | 8.279× | 718.292 |
| metal / gemm_129x257x61 | mps_matrix_multiplication | 16.440 | 18.280× | 260.917 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_1024x1024x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 5371.961 |
| metal / gemm_4097x4097x4096 | 3 / 3 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 1.88% | 17073.605 |
| metal / gemm_2049x4097x1025 | 3 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 105.50% | 7467.188 |
| metal / gemm_1025x1025x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.36% | 5602.526 |
| metal / gemm_129x257x61 | 3 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 1.32% | 4090.445 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
