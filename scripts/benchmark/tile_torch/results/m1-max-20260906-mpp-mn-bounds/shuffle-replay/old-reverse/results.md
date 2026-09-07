# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T04:12:17.174283+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_1024x1024x1024 | 128×32×1024 / 1 | 1 | 286.058 | 300.533 | 290.528 | 308.725 | 0.95× | 499.167 | 531.209 |
| metal | gemm_4097x4097x4096 | 128×32×16 / 1 | 1 | 140979.500 | 21220.000 | 142093.958 | 21459.100 | 6.64× | 141980.542 | 20821.750 |
| metal | gemm_2049x4097x1025 | 128×32×16 / 1 | 1 | 16792.917 | 2839.646 | 17023.025 | 2890.969 | 5.91× | 15900.875 | 3132.959 |
| metal | gemm_1025x1025x1024 | 128×32×16 / 1 | 1 | 2096.866 | 399.947 | 2139.221 | 408.990 | 5.24× | 2285.458 | 639.750 |
| metal | gemm_129x257x61 | 128×32×16 / 1 | 1 | 32.992 | 30.134 | 33.297 | 32.390 | 1.09× | 322.375 | 366.583 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_1024x1024x1024 | 0.069 | 31.538 | 2.914 | 1.587 | 2.793 | 1.251 | 1.010 | 0.402 |
| metal / gemm_4097x4097x4096 | 0.051 | 37.578 | 33.889 | 20.402 | 159.848 | 34.808 | 14.701 | 2.592 |
| metal / gemm_2049x4097x1025 | 0.062 | 38.606 | 8.973 | 1.654 | 30.765 | 9.699 | 7.072 | 1.133 |
| metal / gemm_1025x1025x1024 | 0.056 | 36.801 | 4.311 | 2.133 | 6.155 | 1.206 | 1.103 | 0.420 |
| metal / gemm_129x257x61 | 0.055 | 39.277 | 1.647 | 0.940 | 1.589 | 1.107 | 0.361 | 0.324 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_1024x1024x1024 / native | 278.570 | 268.333 | 286.058 | 499.167 | 1.007× |
| gemm_1024x1024x1024 / torch | 286.400 | 277.875 | 300.533 | 531.209 | 1.080× |
| gemm_1024x1024x1024 / system | 279.889 | 268.417 | 287.866 | 656.375 | 1.092× |
| gemm_4097x4097x4096 / native | 140243.292 | 141116.250 | 140979.500 | 141980.542 | 0.991× |
| gemm_4097x4097x4096 / torch | 20690.500 | 20839.833 | 21220.000 | 20821.750 | 0.985× |
| gemm_4097x4097x4096 / system | 21939.208 | 22218.708 | 23098.125 | 22399.042 | 0.999× |
| gemm_2049x4097x1025 / native | 15928.875 | 15046.208 | 16792.917 | 15900.875 | 1.018× |
| gemm_2049x4097x1025 / torch | 2731.285 | 2472.000 | 2839.646 | 3132.959 | 1.007× |
| gemm_2049x4097x1025 / system | 2826.569 | 2617.125 | 2975.500 | 3023.625 | 1.023× |
| gemm_1025x1025x1024 / native | 2156.722 | 2216.625 | 2096.866 | 2285.458 | 1.011× |
| gemm_1025x1025x1024 / torch | 383.570 | 387.667 | 399.947 | 639.750 | 1.054× |
| gemm_1025x1025x1024 / system | 485.265 | 467.292 | 497.407 | 716.792 | 1.043× |
| gemm_129x257x61 / native | 31.236 | 38.000 | 32.992 | 322.375 | 0.977× |
| gemm_129x257x61 / torch | 15.480 | 20.250 | 30.134 | 366.583 | 2.018× |
| gemm_129x257x61 / system | 14.252 | 17.042 | 15.932 | 269.625 | 1.767× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_1024x1024x1024 | 280.539 | 290.079 | 270.166 | 276.750 | 499.167 | 531.209 |
| gemm_4097x4097x4096 | 140312.792 | 20234.708 | 138646.541 | 19964.417 | 141980.542 | 20821.750 |
| gemm_2049x4097x1025 | 15774.584 | 2718.930 | 14681.375 | 2523.542 | 15900.875 | 3132.959 |
| gemm_1025x1025x1024 | 2181.468 | 385.337 | 2071.167 | 395.208 | 2285.458 | 639.750 |
| gemm_129x257x61 | 30.893 | 19.928 | 38.042 | 20.334 | 322.375 | 366.583 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_1024x1024x1024 | mps_matrix_multiplication | 287.866 | 0.994× | 656.375 |
| metal / gemm_4097x4097x4096 | mps_matrix_multiplication | 23098.125 | 6.104× | 22399.042 |
| metal / gemm_2049x4097x1025 | mps_matrix_multiplication | 2975.500 | 5.644× | 3023.625 |
| metal / gemm_1025x1025x1024 | mps_matrix_multiplication | 497.407 | 4.216× | 716.792 |
| metal / gemm_129x257x61 | mps_matrix_multiplication | 15.932 | 2.071× | 269.625 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_1024x1024x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 5097.405 |
| metal / gemm_4097x4097x4096 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 10783.705 |
| metal / gemm_2049x4097x1025 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 5196.987 |
| metal / gemm_1025x1025x1024 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 2499.321 |
| metal / gemm_129x257x61 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 1979.819 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
