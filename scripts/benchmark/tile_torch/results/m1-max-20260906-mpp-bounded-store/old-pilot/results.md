# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T05:54:46.554952+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_129x257x61 | 128×32×4096 / 1 | 2 | 31.821 | 31.090 | 32.177 | 31.923 | 1.02× | 318.042 | 269.959 |
| metal | gemm_1025x1025x1024 | 128×32×4096 / 1 | 2 | 572.471 | 476.641 | 584.911 | 488.367 | 1.20× | 708.208 | 665.083 |
| metal | gemm_2049x4097x1025 | 128×32×4096 / 1 | 2 | 3959.910 | 3282.245 | 4096.286 | 3417.651 | 1.21× | 4207.292 | 3673.083 |
| metal | gemm_4097x4097x4096 | 128×32×4096 / 1 | 2 | 25172.709 | 22777.896 | 26174.042 | 23113.933 | 1.11× | 24909.000 | 23168.250 |
| metal | gemm_1024x1024x1024 | 128×32×4096 / 1 | 1 | 341.512 | 384.651 | 354.082 | 403.584 | 0.89× | 510.958 | 688.458 |
| metal | gemm_4096x4096x4096 | 128×32×4096 / 1 | 1 | 20006.979 | 19558.604 | 20110.846 | 19919.383 | 1.02× | 20836.208 | 19564.291 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_129x257x61 | 0.058 | 41.778 | 1.422 | 0.942 | 187.332 | 1.000 | 0.425 | 0.409 |
| metal / gemm_1025x1025x1024 | 0.067 | 43.614 | 4.683 | 1.649 | 2.563 | 1.513 | 1.200 | 0.430 |
| metal / gemm_2049x4097x1025 | 0.057 | 42.753 | 9.005 | 19.563 | 10.874 | 7.209 | 9.360 | 1.094 |
| metal / gemm_4097x4097x4096 | 0.066 | 42.848 | 34.856 | 21.200 | 35.211 | 34.037 | 16.183 | 2.475 |
| metal / gemm_1024x1024x1024 | 0.058 | 36.598 | 3.001 | 3.493 | 3.027 | 1.508 | 1.029 | 0.784 |
| metal / gemm_4096x4096x4096 | 0.064 | 32.883 | 31.790 | 23.044 | 35.782 | 31.540 | 9.873 | 3.636 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_129x257x61 / native | 24.777 | 38.250 | 31.821 | 318.042 | 1.200× |
| gemm_129x257x61 / torch | 18.471 | 46.375 | 31.090 | 269.959 | 1.411× |
| gemm_129x257x61 / system | 16.431 | 17.250 | 20.335 | 236.125 | 1.529× |
| gemm_1025x1025x1024 / native | 542.414 | 483.167 | 572.471 | 708.208 | 1.010× |
| gemm_1025x1025x1024 / torch | 448.157 | 379.458 | 476.641 | 665.083 | 1.042× |
| gemm_1025x1025x1024 / system | 574.396 | 487.083 | 590.453 | 736.333 | 1.031× |
| gemm_2049x4097x1025 / native | 3720.833 | 3620.250 | 3959.910 | 4207.292 | 1.005× |
| gemm_2049x4097x1025 / torch | 3184.740 | 3099.042 | 3282.245 | 3673.083 | 0.998× |
| gemm_2049x4097x1025 / system | 3499.302 | 3347.958 | 3635.703 | 3890.958 | 0.952× |
| gemm_4097x4097x4096 / native | 23868.875 | 24058.583 | 25172.709 | 24909.000 | 1.000× |
| gemm_4097x4097x4096 / torch | 22037.250 | 22138.417 | 22777.896 | 23168.250 | 1.015× |
| gemm_4097x4097x4096 / system | 24111.000 | 24182.417 | 24961.333 | 24624.458 | 1.003× |
| gemm_1024x1024x1024 / native | 326.507 | 289.292 | 341.512 | 510.958 | 1.001× |
| gemm_1024x1024x1024 / torch | 362.108 | 278.125 | 384.651 | 688.458 | 1.045× |
| gemm_1024x1024x1024 / system | 340.393 | 276.458 | 355.575 | 512.250 | 1.034× |
| gemm_4096x4096x4096 / native | 19493.625 | 19988.375 | 20006.979 | 20836.208 | 0.994× |
| gemm_4096x4096x4096 / torch | 18778.250 | 18425.875 | 19558.604 | 19564.291 | 1.012× |
| gemm_4096x4096x4096 / system | 20472.167 | 20430.792 | 21021.854 | 20926.833 | 0.988× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_129x257x61 | 29.417 | 23.762 | 36.709 | 54.708 | 318.042 | 269.959 |
| gemm_1025x1025x1024 | 547.672 | 463.600 | 469.833 | 377.583 | 708.208 | 665.083 |
| gemm_2049x4097x1025 | 3757.028 | 3126.943 | 3717.750 | 3066.709 | 4207.292 | 3673.083 |
| gemm_4097x4097x4096 | 23859.167 | 22251.834 | 23837.208 | 22406.125 | 24909.000 | 23168.250 |
| gemm_1024x1024x1024 | 328.227 | 360.353 | 289.084 | 277.458 | 510.958 | 688.458 |
| gemm_4096x4096x4096 | 19542.730 | 18986.604 | 19853.625 | 18520.375 | 20836.208 | 19564.291 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_129x257x61 | mps_matrix_multiplication | 20.335 | 1.565× | 236.125 |
| metal / gemm_1025x1025x1024 | mps_matrix_multiplication | 590.453 | 0.970× | 736.333 |
| metal / gemm_2049x4097x1025 | mps_matrix_multiplication | 3635.703 | 1.089× | 3890.958 |
| metal / gemm_4097x4097x4096 | mps_matrix_multiplication | 24961.333 | 1.008× | 24624.458 |
| metal / gemm_1024x1024x1024 | mps_matrix_multiplication | 355.575 | 0.960× | 512.250 |
| metal / gemm_4096x4096x4096 | mps_matrix_multiplication | 21021.854 | 0.952× | 20926.833 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_129x257x61 | 1 / 1 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 7530.774 |
| metal / gemm_1025x1025x1024 | 1 / 1 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 3680.303 |
| metal / gemm_2049x4097x1025 | 1 / 1 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 4150.911 |
| metal / gemm_4097x4097x4096 | 1 / 1 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 6794.791 |
| metal / gemm_1024x1024x1024 | 1 / 1 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 3439.307 |
| metal / gemm_4096x4096x4096 | 1 / 1 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 7209.855 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
