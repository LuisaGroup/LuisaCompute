# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T04:10:39.655795+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_129x257x61 | 128×32×4096 / 1 | 2 | 25.942 | 30.257 | 26.762 | 31.600 | 0.86× | 261.916 | 394.209 |
| metal | gemm_1025x1025x1024 | 128×32×4096 / 1 | 2 | 489.759 | 412.155 | 499.174 | 426.395 | 1.19× | 699.208 | 792.750 |
| metal | gemm_2049x4097x1025 | 128×32×4096 / 1 | 2 | 3289.725 | 2737.321 | 3293.235 | 2829.344 | 1.20× | 3326.709 | 2932.708 |
| metal | gemm_4097x4097x4096 | 128×32×4096 / 1 | 2 | 21841.291 | 20266.125 | 22297.883 | 20558.950 | 1.08× | 21771.416 | 21399.834 |
| metal | gemm_1024x1024x1024 | 128×32×1024 / 1 | 1 | 287.822 | 300.605 | 292.147 | 304.426 | 0.96× | 503.375 | 562.250 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_129x257x61 | 0.061 | 41.546 | 1.755 | 0.832 | 2.796 | 0.975 | 0.294 | 0.296 |
| metal / gemm_1025x1025x1024 | 0.056 | 41.151 | 3.199 | 1.830 | 2.904 | 1.460 | 1.074 | 0.422 |
| metal / gemm_2049x4097x1025 | 0.056 | 42.492 | 8.403 | 18.732 | 15.110 | 9.966 | 8.197 | 1.416 |
| metal / gemm_4097x4097x4096 | 0.059 | 41.829 | 31.952 | 21.151 | 35.120 | 31.475 | 17.085 | 1.980 |
| metal / gemm_1024x1024x1024 | 0.062 | 31.774 | 3.137 | 2.613 | 2.640 | 2.250 | 1.029 | 0.422 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_129x257x61 / native | 24.576 | 37.375 | 25.942 | 261.916 | 1.006× |
| gemm_129x257x61 / torch | 17.702 | 24.667 | 30.257 | 394.209 | 1.824× |
| gemm_129x257x61 / system | 14.671 | 17.000 | 16.577 | 224.750 | 1.747× |
| gemm_1025x1025x1024 / native | 470.910 | 475.208 | 489.759 | 699.208 | 1.005× |
| gemm_1025x1025x1024 / torch | 385.704 | 366.000 | 412.155 | 792.750 | 1.057× |
| gemm_1025x1025x1024 / system | 484.429 | 469.625 | 493.075 | 691.958 | 1.044× |
| gemm_2049x4097x1025 / native | 3121.892 | 2992.667 | 3289.725 | 3326.709 | 1.001× |
| gemm_2049x4097x1025 / torch | 2584.554 | 2462.958 | 2737.321 | 2932.708 | 1.017× |
| gemm_2049x4097x1025 / system | 2776.187 | 2605.375 | 2799.576 | 2936.041 | 1.013× |
| gemm_4097x4097x4096 / native | 21216.667 | 21310.208 | 21841.291 | 21771.416 | 0.988× |
| gemm_4097x4097x4096 / torch | 19836.000 | 20529.292 | 20266.125 | 21399.834 | 1.045× |
| gemm_4097x4097x4096 / system | 21904.583 | 21707.125 | 23176.583 | 22542.792 | 1.012× |
| gemm_1024x1024x1024 / native | 278.889 | 269.750 | 287.822 | 503.375 | 1.005× |
| gemm_1024x1024x1024 / torch | 286.981 | 276.833 | 300.605 | 562.250 | 1.076× |
| gemm_1024x1024x1024 / system | 281.697 | 271.625 | 288.487 | 574.291 | 1.079× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_129x257x61 | 24.562 | 22.419 | 36.916 | 23.667 | 261.916 | 394.209 |
| gemm_1025x1025x1024 | 471.954 | 386.983 | 463.083 | 371.250 | 699.208 | 792.750 |
| gemm_2049x4097x1025 | 3130.908 | 2607.946 | 3011.000 | 2451.084 | 3326.709 | 2932.708 |
| gemm_4097x4097x4096 | 20965.333 | 20701.667 | 22185.792 | 20640.625 | 21771.416 | 21399.834 |
| gemm_1024x1024x1024 | 279.672 | 289.812 | 279.375 | 276.166 | 503.375 | 562.250 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_129x257x61 | mps_matrix_multiplication | 16.577 | 1.565× | 224.750 |
| metal / gemm_1025x1025x1024 | mps_matrix_multiplication | 493.075 | 0.993× | 691.958 |
| metal / gemm_2049x4097x1025 | mps_matrix_multiplication | 2799.576 | 1.175× | 2936.041 |
| metal / gemm_4097x4097x4096 | mps_matrix_multiplication | 23176.583 | 0.942× | 22542.792 |
| metal / gemm_1024x1024x1024 | mps_matrix_multiplication | 288.487 | 0.998× | 574.291 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_129x257x61 | 3 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 118.88% | 6522.521 |
| metal / gemm_1025x1025x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.62% | 6028.310 |
| metal / gemm_2049x4097x1025 | 3 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 259.63% | 7910.132 |
| metal / gemm_4097x4097x4096 | 3 / 3 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 14724.254 |
| metal / gemm_1024x1024x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 5072.402 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
