# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T03:51:34.072708+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_129x257x61 | 128×32×16 / 1 | 1 | 32.941 | 29.368 | 34.035 | 30.186 | 1.12× | 320.292 | 259.750 |
| metal | gemm_1025x1025x1024 | 128×32×16 / 1 | 1 | 2360.458 | 407.501 | 2420.294 | 415.006 | 5.79× | 2650.750 | 638.958 |
| metal | gemm_2049x4097x1025 | 128×32×16 / 1 | 1 | 15974.188 | 2810.310 | 16078.575 | 2817.574 | 5.68× | 15895.958 | 3682.917 |
| metal | gemm_4097x4097x4096 | 128×32×16 / 1 | 1 | 141215.250 | 20900.959 | 143160.633 | 21836.300 | 6.76× | 141584.833 | 21502.667 |
| metal | gemm_1024x1024x1024 | 128×32×4096 / 1 | 1 | 285.621 | 303.182 | 286.881 | 305.974 | 0.94× | 499.000 | 573.250 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_129x257x61 | 0.064 | 37.558 | 1.656 | 1.113 | 281.903 | 1.430 | 0.323 | 0.360 |
| metal / gemm_1025x1025x1024 | 0.055 | 37.064 | 2.723 | 1.631 | 5.761 | 2.550 | 1.278 | 0.440 |
| metal / gemm_2049x4097x1025 | 0.050 | 38.299 | 8.853 | 2.634 | 25.414 | 8.115 | 8.159 | 3.527 |
| metal / gemm_4097x4097x4096 | 0.050 | 38.755 | 32.888 | 21.506 | 154.689 | 30.436 | 17.602 | 6.308 |
| metal / gemm_1024x1024x1024 | 0.059 | 35.787 | 3.262 | 1.674 | 2.810 | 1.788 | 0.979 | 0.622 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_129x257x61 / native | 31.297 | 37.917 | 32.941 | 320.292 | 0.995× |
| gemm_129x257x61 / torch | 15.363 | 24.333 | 29.368 | 259.750 | 1.883× |
| gemm_129x257x61 / system | 15.215 | 17.083 | 16.069 | 257.750 | 1.704× |
| gemm_1025x1025x1024 / native | 2065.940 | 2240.750 | 2360.458 | 2650.750 | 1.013× |
| gemm_1025x1025x1024 / torch | 388.557 | 374.875 | 407.501 | 638.958 | 1.050× |
| gemm_1025x1025x1024 / system | 484.424 | 469.458 | 501.885 | 707.542 | 1.047× |
| gemm_2049x4097x1025 / native | 15769.604 | 15373.875 | 15974.188 | 15895.958 | 1.016× |
| gemm_2049x4097x1025 / torch | 2665.661 | 2455.500 | 2810.310 | 3682.917 | 1.008× |
| gemm_2049x4097x1025 / system | 2769.054 | 2597.125 | 2854.071 | 3034.583 | 1.016× |
| gemm_4097x4097x4096 / native | 139695.833 | 139235.042 | 141215.250 | 141584.833 | 1.025× |
| gemm_4097x4097x4096 / torch | 20450.458 | 20675.375 | 20900.959 | 21502.667 | 0.982× |
| gemm_4097x4097x4096 / system | 22180.750 | 21256.625 | 22479.917 | 23011.625 | 0.967× |
| gemm_1024x1024x1024 / native | 279.069 | 270.542 | 285.621 | 499.000 | 1.003× |
| gemm_1024x1024x1024 / torch | 286.827 | 275.500 | 303.182 | 573.250 | 1.074× |
| gemm_1024x1024x1024 / system | 279.300 | 269.250 | 288.424 | 501.416 | 1.091× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_129x257x61 | 31.144 | 20.664 | 37.375 | 24.792 | 320.292 | 259.750 |
| gemm_1025x1025x1024 | 2088.994 | 389.665 | 2010.208 | 370.875 | 2650.750 | 638.958 |
| gemm_2049x4097x1025 | 16027.271 | 2670.726 | 15128.416 | 2521.000 | 15895.958 | 3682.917 |
| gemm_4097x4097x4096 | 142506.375 | 20598.459 | 142103.917 | 20517.000 | 141584.833 | 21502.667 |
| gemm_1024x1024x1024 | 279.132 | 290.091 | 272.375 | 277.667 | 499.000 | 573.250 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_129x257x61 | mps_matrix_multiplication | 16.069 | 2.050× | 257.750 |
| metal / gemm_1025x1025x1024 | mps_matrix_multiplication | 501.885 | 4.703× | 707.542 |
| metal / gemm_2049x4097x1025 | mps_matrix_multiplication | 2854.071 | 5.597× | 3034.583 |
| metal / gemm_4097x4097x4096 | mps_matrix_multiplication | 22479.917 | 6.282× | 23011.625 |
| metal / gemm_1024x1024x1024 | mps_matrix_multiplication | 288.424 | 0.990× | 501.416 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_129x257x61 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 2813.414 |
| metal / gemm_1025x1025x1024 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 2725.535 |
| metal / gemm_2049x4097x1025 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 4107.338 |
| metal / gemm_4097x4097x4096 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 12363.595 |
| metal / gemm_1024x1024x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.34% | 5838.986 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
