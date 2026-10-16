# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T04:01:00.274095+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_1024x1024x1024 | 128×32×4096 / 1 | 1 | 286.300 | 306.604 | 295.486 | 311.846 | 0.93× | 486.583 | 567.417 |
| metal | gemm_4097x4097x4096 | 128×32×4096 / 1 | 2 | 26040.500 | 21148.083 | 27029.992 | 21649.525 | 1.23× | 25238.833 | 21358.541 |
| metal | gemm_2049x4097x1025 | 128×32×4096 / 1 | 2 | 3845.675 | 2675.554 | 3910.543 | 2728.342 | 1.44× | 4252.875 | 3000.167 |
| metal | gemm_1025x1025x1024 | 128×32×4096 / 1 | 2 | 793.373 | 430.471 | 821.923 | 441.053 | 1.84× | 1048.583 | 992.042 |
| metal | gemm_129x257x61 | 128×32×4096 / 1 | 2 | 150.900 | 31.107 | 153.841 | 33.521 | 4.85× | 369.750 | 262.542 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_1024x1024x1024 | 0.055 | 34.680 | 2.881 | 1.585 | 2.844 | 1.403 | 1.095 | 0.450 |
| metal / gemm_4097x4097x4096 | 0.051 | 42.182 | 29.411 | 20.229 | 41.002 | 32.609 | 20.540 | 2.679 |
| metal / gemm_2049x4097x1025 | 0.053 | 42.604 | 9.073 | 18.162 | 18.160 | 12.000 | 7.738 | 1.244 |
| metal / gemm_1025x1025x1024 | 0.056 | 41.375 | 3.220 | 1.670 | 4.059 | 2.128 | 1.300 | 0.716 |
| metal / gemm_129x257x61 | 0.066 | 42.612 | 1.692 | 1.032 | 2.267 | 1.115 | 0.324 | 0.260 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_1024x1024x1024 / native | 279.564 | 272.958 | 286.300 | 486.583 | 0.995× |
| gemm_1024x1024x1024 / torch | 286.828 | 276.292 | 306.604 | 567.417 | 1.075× |
| gemm_1024x1024x1024 / system | 280.170 | 269.958 | 286.025 | 469.084 | 1.088× |
| gemm_4097x4097x4096 / native | 25642.708 | 25626.375 | 26040.500 | 25238.833 | 0.972× |
| gemm_4097x4097x4096 / torch | 20657.958 | 20442.333 | 21148.083 | 21358.541 | 1.015× |
| gemm_4097x4097x4096 / system | 21990.125 | 22796.042 | 23073.667 | 22546.167 | 1.010× |
| gemm_2049x4097x1025 / native | 3789.442 | 3655.000 | 3845.675 | 4252.875 | 0.992× |
| gemm_2049x4097x1025 / torch | 2626.958 | 2465.750 | 2675.554 | 3000.167 | 0.999× |
| gemm_2049x4097x1025 / system | 2772.542 | 2621.625 | 2849.653 | 2985.166 | 1.027× |
| gemm_1025x1025x1024 / native | 777.622 | 783.417 | 793.373 | 1048.583 | 1.004× |
| gemm_1025x1025x1024 / torch | 385.627 | 379.417 | 430.471 | 992.042 | 1.057× |
| gemm_1025x1025x1024 / system | 487.564 | 479.000 | 500.353 | 708.291 | 1.048× |
| gemm_129x257x61 / native | 148.706 | 151.875 | 150.900 | 369.750 | 0.999× |
| gemm_129x257x61 / torch | 15.316 | 23.917 | 31.107 | 262.542 | 1.902× |
| gemm_129x257x61 / system | 14.947 | 17.292 | 15.782 | 239.833 | 1.764× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_1024x1024x1024 | 278.604 | 289.967 | 268.209 | 276.000 | 486.583 | 567.417 |
| gemm_4097x4097x4096 | 24671.042 | 20834.416 | 25721.584 | 20238.958 | 25238.833 | 21358.541 |
| gemm_2049x4097x1025 | 3758.033 | 2604.339 | 3780.542 | 2525.250 | 4252.875 | 3000.167 |
| gemm_1025x1025x1024 | 777.854 | 387.440 | 756.916 | 380.750 | 1048.583 | 992.042 |
| gemm_129x257x61 | 148.572 | 19.917 | 153.417 | 25.208 | 369.750 | 262.542 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_1024x1024x1024 | mps_matrix_multiplication | 286.025 | 1.001× | 469.084 |
| metal / gemm_4097x4097x4096 | mps_matrix_multiplication | 23073.667 | 1.129× | 22546.167 |
| metal / gemm_2049x4097x1025 | mps_matrix_multiplication | 2849.653 | 1.350× | 2985.166 |
| metal / gemm_1025x1025x1024 | mps_matrix_multiplication | 500.353 | 1.586× | 708.291 |
| metal / gemm_129x257x61 | mps_matrix_multiplication | 15.782 | 9.562× | 239.833 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_1024x1024x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.19% | 5043.071 |
| metal / gemm_4097x4097x4096 | 3 / 3 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 16514.227 |
| metal / gemm_2049x4097x1025 | 3 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 670.65% | 7143.434 |
| metal / gemm_1025x1025x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.62% | 5521.024 |
| metal / gemm_129x257x61 | 3 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 261.41% | 4083.135 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
