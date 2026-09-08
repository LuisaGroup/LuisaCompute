# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T04:11:31.408805+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_1024x1024x1024 | 128×32×4096 / 1 | 1 | 285.095 | 303.908 | 290.761 | 311.256 | 0.94× | 487.584 | 566.625 |
| metal | gemm_4097x4097x4096 | 128×32×1024 / 1 | 2 | 22422.542 | 20476.625 | 22704.541 | 20898.917 | 1.10× | 21972.500 | 20740.875 |
| metal | gemm_2049x4097x1025 | 128×32×4096 / 1 | 2 | 3166.271 | 2752.555 | 3263.807 | 2921.810 | 1.15× | 3340.083 | 2977.083 |
| metal | gemm_1025x1025x1024 | 128×32×4096 / 1 | 2 | 480.783 | 404.986 | 502.362 | 420.179 | 1.19× | 720.208 | 674.167 |
| metal | gemm_129x257x61 | 128×32×4096 / 1 | 2 | 26.021 | 30.175 | 27.129 | 30.883 | 0.86× | 256.709 | 294.333 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_1024x1024x1024 | 0.121 | 34.727 | 4.411 | 1.622 | 3.008 | 1.188 | 1.098 | 0.442 |
| metal / gemm_4097x4097x4096 | 0.061 | 41.889 | 31.849 | 20.581 | 36.168 | 33.485 | 18.309 | 2.945 |
| metal / gemm_2049x4097x1025 | 0.053 | 40.876 | 8.036 | 18.364 | 13.193 | 10.350 | 8.050 | 1.097 |
| metal / gemm_1025x1025x1024 | 0.060 | 41.511 | 3.730 | 1.732 | 3.589 | 1.453 | 0.849 | 0.451 |
| metal / gemm_129x257x61 | 0.056 | 41.328 | 1.598 | 1.131 | 2.113 | 1.089 | 0.326 | 0.306 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_1024x1024x1024 / native | 281.747 | 269.708 | 285.095 | 487.584 | 0.994× |
| gemm_1024x1024x1024 / torch | 287.645 | 275.458 | 303.908 | 566.625 | 1.082× |
| gemm_1024x1024x1024 / system | 280.533 | 270.875 | 289.220 | 511.250 | 1.083× |
| gemm_4097x4097x4096 / native | 21629.167 | 21773.625 | 22422.542 | 21972.500 | 0.988× |
| gemm_4097x4097x4096 / torch | 20390.125 | 20345.125 | 20476.625 | 20740.875 | 0.987× |
| gemm_4097x4097x4096 / system | 22066.833 | 22371.458 | 22310.000 | 22235.917 | 1.004× |
| gemm_2049x4097x1025 / native | 3102.326 | 2981.708 | 3166.271 | 3340.083 | 1.014× |
| gemm_2049x4097x1025 / torch | 2588.917 | 2513.083 | 2752.555 | 2977.083 | 1.008× |
| gemm_2049x4097x1025 / system | 2754.161 | 2606.500 | 2869.548 | 2958.792 | 1.021× |
| gemm_1025x1025x1024 / native | 467.192 | 475.500 | 480.783 | 720.208 | 1.000× |
| gemm_1025x1025x1024 / torch | 385.105 | 378.833 | 404.986 | 674.167 | 1.072× |
| gemm_1025x1025x1024 / system | 485.994 | 467.500 | 495.834 | 699.416 | 1.052× |
| gemm_129x257x61 / native | 24.854 | 38.708 | 26.021 | 256.709 | 0.988× |
| gemm_129x257x61 / torch | 15.924 | 23.875 | 30.175 | 294.333 | 1.809× |
| gemm_129x257x61 / system | 14.758 | 17.333 | 16.359 | 290.875 | 1.801× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_1024x1024x1024 | 280.128 | 292.022 | 271.875 | 289.250 | 487.584 | 566.625 |
| gemm_4097x4097x4096 | 21732.958 | 19970.375 | 21732.292 | 20062.625 | 21972.500 | 20740.875 |
| gemm_2049x4097x1025 | 3168.431 | 2578.944 | 2997.791 | 2422.083 | 3340.083 | 2977.083 |
| gemm_1025x1025x1024 | 468.468 | 392.406 | 467.916 | 368.166 | 720.208 | 674.167 |
| gemm_129x257x61 | 24.544 | 19.788 | 37.125 | 23.500 | 256.709 | 294.333 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_1024x1024x1024 | mps_matrix_multiplication | 289.220 | 0.986× | 511.250 |
| metal / gemm_4097x4097x4096 | mps_matrix_multiplication | 22310.000 | 1.005× | 22235.917 |
| metal / gemm_2049x4097x1025 | mps_matrix_multiplication | 2869.548 | 1.103× | 2958.792 |
| metal / gemm_1025x1025x1024 | mps_matrix_multiplication | 495.834 | 0.970× | 699.416 |
| metal / gemm_129x257x61 | mps_matrix_multiplication | 16.359 | 1.591× | 290.875 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_1024x1024x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.25% | 5176.722 |
| metal / gemm_4097x4097x4096 | 3 / 3 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 1.66% | 13370.561 |
| metal / gemm_2049x4097x1025 | 3 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 277.42% | 6480.159 |
| metal / gemm_1025x1025x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.37% | 5278.175 |
| metal / gemm_129x257x61 | 3 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 123.26% | 3599.606 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
