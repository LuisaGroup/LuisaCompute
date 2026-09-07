# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T03:54:08.539436+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_1024x1024x1024 | 128×32×4096 / 1 | 1 | 446.061 | 487.965 | 475.238 | 496.453 | 0.91× | 623.042 | 785.541 |
| metal | gemm_4097x4097x4096 | 128×32×16 / 1 | 1 | 136774.167 | 20885.000 | 140918.866 | 21767.475 | 6.55× | 139589.708 | 20655.083 |
| metal | gemm_2049x4097x1025 | 128×32×16 / 1 | 1 | 15834.458 | 2728.451 | 16269.758 | 2759.069 | 5.80× | 15818.750 | 2963.708 |
| metal | gemm_1025x1025x1024 | 128×32×16 / 1 | 1 | 2207.412 | 400.816 | 2246.402 | 407.620 | 5.51× | 2332.416 | 659.208 |
| metal | gemm_129x257x61 | 128×32×16 / 1 | 1 | 32.804 | 30.123 | 33.930 | 30.638 | 1.09× | 255.959 | 282.375 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_1024x1024x1024 | 0.058 | 36.665 | 3.549 | 1.610 | 3.034 | 1.432 | 0.805 | 0.378 |
| metal / gemm_4097x4097x4096 | 0.057 | 38.566 | 31.601 | 20.519 | 149.960 | 31.739 | 15.157 | 2.330 |
| metal / gemm_2049x4097x1025 | 0.053 | 37.666 | 7.243 | 2.882 | 28.553 | 6.015 | 7.412 | 2.027 |
| metal / gemm_1025x1025x1024 | 0.062 | 37.965 | 3.433 | 1.709 | 7.031 | 1.235 | 0.936 | 0.451 |
| metal / gemm_129x257x61 | 0.050 | 91.521 | 3.825 | 0.856 | 1.417 | 0.971 | 0.287 | 0.309 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_1024x1024x1024 / native | 431.092 | 442.000 | 446.061 | 623.042 | 1.008× |
| gemm_1024x1024x1024 / torch | 460.455 | 365.917 | 487.965 | 785.541 | 1.031× |
| gemm_1024x1024x1024 / system | 429.546 | 359.500 | 435.572 | 586.666 | 1.045× |
| gemm_4097x4097x4096 / native | 141526.583 | 142312.250 | 136774.167 | 139589.708 | 0.994× |
| gemm_4097x4097x4096 / torch | 19698.833 | 20099.417 | 20885.000 | 20655.083 | 1.030× |
| gemm_4097x4097x4096 / system | 21683.417 | 21650.792 | 23502.875 | 23072.084 | 1.010× |
| gemm_2049x4097x1025 / native | 15406.646 | 14874.833 | 15834.458 | 15818.750 | 0.997× |
| gemm_2049x4097x1025 / torch | 2617.000 | 2455.375 | 2728.451 | 2963.708 | 1.014× |
| gemm_2049x4097x1025 / system | 2815.965 | 2625.833 | 2900.139 | 3076.375 | 0.987× |
| gemm_1025x1025x1024 / native | 2080.569 | 2236.542 | 2207.412 | 2332.416 | 1.005× |
| gemm_1025x1025x1024 / torch | 383.736 | 372.042 | 400.816 | 659.208 | 1.061× |
| gemm_1025x1025x1024 / system | 486.873 | 467.125 | 502.370 | 705.792 | 1.047× |
| gemm_129x257x61 / native | 31.015 | 38.708 | 32.804 | 255.959 | 1.006× |
| gemm_129x257x61 / torch | 17.381 | 31.250 | 30.123 | 282.375 | 1.963× |
| gemm_129x257x61 / system | 13.629 | 19.750 | 16.166 | 250.042 | 1.833× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_1024x1024x1024 | 435.308 | 455.379 | 392.834 | 454.417 | 623.042 | 785.541 |
| gemm_4097x4097x4096 | 139994.500 | 19999.625 | 140593.500 | 19637.250 | 139589.708 | 20655.083 |
| gemm_2049x4097x1025 | 15416.416 | 2612.611 | 14667.625 | 2436.750 | 15818.750 | 2963.708 |
| gemm_1025x1025x1024 | 2109.417 | 387.518 | 2227.125 | 372.667 | 2332.416 | 659.208 |
| gemm_129x257x61 | 31.189 | 24.637 | 38.541 | 30.000 | 255.959 | 282.375 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_1024x1024x1024 | mps_matrix_multiplication | 435.572 | 1.024× | 586.666 |
| metal / gemm_4097x4097x4096 | mps_matrix_multiplication | 23502.875 | 5.819× | 23072.084 |
| metal / gemm_2049x4097x1025 | mps_matrix_multiplication | 2900.139 | 5.460× | 3076.375 |
| metal / gemm_1025x1025x1024 | mps_matrix_multiplication | 502.370 | 4.394× | 705.792 |
| metal / gemm_129x257x61 | mps_matrix_multiplication | 16.166 | 2.029× | 250.042 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_1024x1024x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 3.92% | 5239.910 |
| metal / gemm_4097x4097x4096 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 10746.059 |
| metal / gemm_2049x4097x1025 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 5324.407 |
| metal / gemm_1025x1025x1024 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 2456.266 |
| metal / gemm_129x257x61 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 1926.356 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
