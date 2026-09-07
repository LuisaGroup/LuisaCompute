# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T04:00:05.993933+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_129x257x61 | 128×32×1024 / 1 | 2 | 152.268 | 36.876 | 153.477 | 38.127 | 4.13× | 441.667 | 245.583 |
| metal | gemm_1025x1025x1024 | 128×32×4096 / 1 | 2 | 796.387 | 404.551 | 814.728 | 406.258 | 1.97× | 1011.416 | 711.541 |
| metal | gemm_2049x4097x1025 | 128×32×4096 / 1 | 2 | 3884.658 | 2691.345 | 3949.343 | 2723.977 | 1.44× | 4247.250 | 3000.458 |
| metal | gemm_4097x4097x4096 | 128×32×1024 / 1 | 2 | 26021.333 | 21299.291 | 26449.642 | 21717.708 | 1.22× | 26643.208 | 21074.958 |
| metal | gemm_1024x1024x1024 | 128×32×1024 / 1 | 1 | 284.485 | 303.006 | 294.512 | 305.425 | 0.94× | 520.250 | 584.625 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_129x257x61 | 0.060 | 42.388 | 1.595 | 0.918 | 2.064 | 1.070 | 0.364 | 0.335 |
| metal / gemm_1025x1025x1024 | 0.057 | 42.501 | 3.717 | 2.036 | 3.218 | 2.030 | 1.294 | 0.438 |
| metal / gemm_2049x4097x1025 | 0.056 | 41.884 | 8.623 | 19.009 | 18.323 | 10.682 | 7.705 | 1.082 |
| metal / gemm_4097x4097x4096 | 0.071 | 43.706 | 28.838 | 20.954 | 38.372 | 31.566 | 14.717 | 6.703 |
| metal / gemm_1024x1024x1024 | 0.058 | 33.224 | 2.964 | 1.540 | 2.726 | 2.074 | 1.047 | 0.400 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_129x257x61 / native | 148.514 | 153.458 | 152.268 | 441.667 | 0.998× |
| gemm_129x257x61 / torch | 13.980 | 24.750 | 36.876 | 245.583 | 1.907× |
| gemm_129x257x61 / system | 15.186 | 17.375 | 16.028 | 247.916 | 1.671× |
| gemm_1025x1025x1024 / native | 772.292 | 784.375 | 796.387 | 1011.416 | 1.005× |
| gemm_1025x1025x1024 / torch | 388.642 | 374.875 | 404.551 | 711.541 | 1.068× |
| gemm_1025x1025x1024 / system | 484.511 | 468.375 | 497.744 | 728.333 | 1.063× |
| gemm_2049x4097x1025 / native | 3833.775 | 3657.250 | 3884.658 | 4247.250 | 0.994× |
| gemm_2049x4097x1025 / torch | 2631.905 | 2443.333 | 2691.345 | 3000.458 | 1.007× |
| gemm_2049x4097x1025 / system | 2805.229 | 2596.125 | 2815.500 | 3090.416 | 1.017× |
| gemm_4097x4097x4096 / native | 25484.583 | 25578.167 | 26021.333 | 26643.208 | 1.019× |
| gemm_4097x4097x4096 / torch | 20286.000 | 20623.625 | 21299.291 | 21074.958 | 0.993× |
| gemm_4097x4097x4096 / system | 22143.500 | 22166.542 | 22366.208 | 22908.791 | 1.005× |
| gemm_1024x1024x1024 / native | 279.764 | 270.167 | 284.485 | 520.250 | 1.000× |
| gemm_1024x1024x1024 / torch | 287.327 | 276.833 | 303.006 | 584.625 | 1.068× |
| gemm_1024x1024x1024 / system | 280.965 | 269.292 | 289.859 | 512.959 | 1.084× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_129x257x61 | 148.634 | 17.469 | 152.875 | 20.125 | 441.667 | 245.583 |
| gemm_1025x1025x1024 | 778.398 | 391.278 | 757.375 | 373.416 | 1011.416 | 711.541 |
| gemm_2049x4097x1025 | 3789.067 | 2643.012 | 3652.917 | 2486.125 | 4247.250 | 3000.458 |
| gemm_4097x4097x4096 | 25673.458 | 20359.375 | 25751.417 | 20542.041 | 26643.208 | 21074.958 |
| gemm_1024x1024x1024 | 281.049 | 289.563 | 270.000 | 277.792 | 520.250 | 584.625 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_129x257x61 | mps_matrix_multiplication | 16.028 | 9.500× | 247.916 |
| metal / gemm_1025x1025x1024 | mps_matrix_multiplication | 497.744 | 1.600× | 728.333 |
| metal / gemm_2049x4097x1025 | mps_matrix_multiplication | 2815.500 | 1.380× | 3090.416 |
| metal / gemm_4097x4097x4096 | mps_matrix_multiplication | 22366.208 | 1.163× | 22908.791 |
| metal / gemm_1024x1024x1024 | mps_matrix_multiplication | 289.859 | 0.981× | 512.959 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_129x257x61 | 3 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 260.42% | 4985.378 |
| metal / gemm_1025x1025x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.48% | 6227.376 |
| metal / gemm_2049x4097x1025 | 3 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 669.43% | 8485.165 |
| metal / gemm_4097x4097x4096 | 3 / 3 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 1.59% | 17668.646 |
| metal / gemm_1024x1024x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 5063.011 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
