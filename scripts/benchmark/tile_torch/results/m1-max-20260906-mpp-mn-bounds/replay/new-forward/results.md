# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T03:52:19.126992+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_129x257x61 | 128×32×4096 / 1 | 1 | 301.561 | 31.280 | 303.438 | 31.855 | 9.64× | 531.000 | 309.458 |
| metal | gemm_1025x1025x1024 | 128×32×1024 / 1 | 1 | 3885.812 | 404.288 | 3900.204 | 410.652 | 9.61× | 4038.084 | 651.209 |
| metal | gemm_2049x4097x1025 | 128×32×4096 / 1 | 1 | 11267.000 | 2744.583 | 11307.533 | 2798.101 | 4.11× | 11445.833 | 3448.291 |
| metal | gemm_4097x4097x4096 | 128×32×1024 / 1 | 1 | 53076.208 | 21483.666 | 54289.600 | 22236.317 | 2.47× | 53793.791 | 21081.000 |
| metal | gemm_1024x1024x1024 | 128×32×1024 / 1 | 1 | 287.751 | 301.242 | 293.431 | 306.930 | 0.96× | 484.625 | 601.250 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_129x257x61 | 0.077 | 41.624 | 1.398 | 0.883 | 2.233 | 1.057 | 0.313 | 0.257 |
| metal / gemm_1025x1025x1024 | 0.057 | 41.212 | 3.079 | 1.770 | 9.600 | 1.788 | 2.924 | 0.434 |
| metal / gemm_2049x4097x1025 | 0.060 | 42.983 | 8.139 | 18.908 | 27.768 | 8.873 | 8.393 | 2.178 |
| metal / gemm_4097x4097x4096 | 0.057 | 42.927 | 29.593 | 21.113 | 67.763 | 29.795 | 15.495 | 5.201 |
| metal / gemm_1024x1024x1024 | 0.058 | 32.773 | 3.058 | 1.839 | 2.624 | 1.486 | 1.137 | 0.570 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_129x257x61 / native | 295.137 | 298.500 | 301.561 | 531.000 | 1.003× |
| gemm_129x257x61 / torch | 12.984 | 20.333 | 31.280 | 309.458 | 1.892× |
| gemm_129x257x61 / system | 13.742 | 17.208 | 16.347 | 272.125 | 1.819× |
| gemm_1025x1025x1024 / native | 3762.375 | 3738.333 | 3885.812 | 4038.084 | 0.998× |
| gemm_1025x1025x1024 / torch | 382.950 | 369.208 | 404.288 | 651.209 | 1.060× |
| gemm_1025x1025x1024 / system | 482.864 | 470.875 | 499.683 | 706.000 | 1.056× |
| gemm_2049x4097x1025 / native | 10573.687 | 10843.958 | 11267.000 | 11445.833 | 1.004× |
| gemm_2049x4097x1025 / torch | 2633.857 | 2452.875 | 2744.583 | 3448.291 | 1.000× |
| gemm_2049x4097x1025 / system | 2774.229 | 2640.167 | 2868.354 | 3014.000 | 1.033× |
| gemm_4097x4097x4096 / native | 52429.375 | 53069.333 | 53076.208 | 53793.791 | 1.009× |
| gemm_4097x4097x4096 / torch | 20634.417 | 20720.375 | 21483.666 | 21081.000 | 0.976× |
| gemm_4097x4097x4096 / system | 21778.750 | 21760.458 | 23073.416 | 22506.458 | 1.032× |
| gemm_1024x1024x1024 / native | 281.412 | 269.167 | 287.751 | 484.625 | 1.000× |
| gemm_1024x1024x1024 / torch | 285.996 | 276.000 | 301.242 | 601.250 | 1.080× |
| gemm_1024x1024x1024 / system | 283.342 | 269.792 | 288.121 | 502.083 | 1.080× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_129x257x61 | 296.464 | 17.510 | 297.458 | 20.166 | 531.000 | 309.458 |
| gemm_1025x1025x1024 | 3794.104 | 387.169 | 3733.708 | 373.458 | 4038.084 | 651.209 |
| gemm_2049x4097x1025 | 10610.938 | 2560.607 | 10886.333 | 2454.958 | 11445.833 | 3448.291 |
| gemm_4097x4097x4096 | 53283.000 | 20198.333 | 52652.042 | 20714.791 | 53793.791 | 21081.000 |
| gemm_1024x1024x1024 | 281.569 | 289.332 | 274.291 | 279.209 | 484.625 | 601.250 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_129x257x61 | mps_matrix_multiplication | 16.347 | 18.448× | 272.125 |
| metal / gemm_1025x1025x1024 | mps_matrix_multiplication | 499.683 | 7.777× | 706.000 |
| metal / gemm_2049x4097x1025 | mps_matrix_multiplication | 2868.354 | 3.928× | 3014.000 |
| metal / gemm_4097x4097x4096 | mps_matrix_multiplication | 23073.416 | 2.300× | 22506.458 |
| metal / gemm_1024x1024x1024 | mps_matrix_multiplication | 288.121 | 0.999× | 502.083 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_129x257x61 | 3 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 1.42% | 5050.828 |
| metal / gemm_1025x1025x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 6093.355 |
| metal / gemm_2049x4097x1025 | 3 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 98.06% | 8477.282 |
| metal / gemm_4097x4097x4096 | 3 / 3 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 3.95% | 17982.109 |
| metal / gemm_1024x1024x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 5499.705 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
