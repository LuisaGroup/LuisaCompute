# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T04:01:50.740839+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_1024x1024x1024 | 128×32×1024 / 1 | 1 | 283.603 | 302.772 | 294.857 | 306.813 | 0.94× | 499.250 | 566.750 |
| metal | gemm_4097x4097x4096 | 128×32×16 / 1 | 1 | 142912.000 | 20695.708 | 144021.659 | 21256.808 | 6.91× | 142261.625 | 21454.709 |
| metal | gemm_2049x4097x1025 | 128×32×16 / 1 | 1 | 15610.625 | 2701.030 | 16148.021 | 2753.570 | 5.78× | 16267.916 | 2989.375 |
| metal | gemm_1025x1025x1024 | 128×32×16 / 1 | 1 | 2146.449 | 404.672 | 2193.356 | 407.570 | 5.30× | 2271.375 | 646.416 |
| metal | gemm_129x257x61 | 128×32×16 / 1 | 1 | 33.072 | 29.182 | 34.327 | 29.961 | 1.13× | 254.167 | 290.833 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_1024x1024x1024 | 0.075 | 33.509 | 2.902 | 1.504 | 2.898 | 1.324 | 0.839 | 0.412 |
| metal / gemm_4097x4097x4096 | 0.062 | 37.643 | 32.316 | 19.355 | 151.080 | 35.779 | 19.893 | 2.507 |
| metal / gemm_2049x4097x1025 | 0.075 | 38.932 | 8.367 | 1.457 | 31.373 | 11.233 | 7.094 | 1.644 |
| metal / gemm_1025x1025x1024 | 0.058 | 37.669 | 3.358 | 1.580 | 8.328 | 1.343 | 1.198 | 0.926 |
| metal / gemm_129x257x61 | 0.067 | 39.383 | 2.689 | 0.926 | 1.496 | 1.385 | 0.293 | 0.349 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_1024x1024x1024 / native | 279.824 | 270.542 | 283.603 | 499.250 | 0.994× |
| gemm_1024x1024x1024 / torch | 287.070 | 288.125 | 302.772 | 566.750 | 1.082× |
| gemm_1024x1024x1024 / system | 278.971 | 272.625 | 288.717 | 482.250 | 1.088× |
| gemm_4097x4097x4096 / native | 142011.250 | 143783.292 | 142912.000 | 142261.625 | 0.994× |
| gemm_4097x4097x4096 / torch | 20752.583 | 20230.250 | 20695.708 | 21454.709 | 0.959× |
| gemm_4097x4097x4096 / system | 22216.083 | 22212.625 | 23479.083 | 23354.791 | 1.019× |
| gemm_2049x4097x1025 / native | 15279.188 | 14884.042 | 15610.625 | 16267.916 | 1.000× |
| gemm_2049x4097x1025 / torch | 2624.000 | 2499.750 | 2701.030 | 2989.375 | 1.003× |
| gemm_2049x4097x1025 / system | 2811.946 | 2604.500 | 2901.048 | 3167.542 | 0.995× |
| gemm_1025x1025x1024 / native | 2143.310 | 2221.792 | 2146.449 | 2271.375 | 1.011× |
| gemm_1025x1025x1024 / torch | 386.858 | 374.500 | 404.672 | 646.416 | 1.045× |
| gemm_1025x1025x1024 / system | 484.513 | 471.500 | 495.208 | 689.125 | 1.040× |
| gemm_129x257x61 / native | 31.062 | 37.917 | 33.072 | 254.167 | 1.003× |
| gemm_129x257x61 / torch | 16.257 | 24.625 | 29.182 | 290.833 | 1.763× |
| gemm_129x257x61 / system | 14.909 | 17.042 | 15.800 | 240.416 | 1.684× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_1024x1024x1024 | 279.645 | 292.044 | 268.625 | 275.083 | 499.250 | 566.750 |
| gemm_4097x4097x4096 | 141837.791 | 20309.083 | 142276.458 | 20258.291 | 142261.625 | 21454.709 |
| gemm_2049x4097x1025 | 15327.604 | 2600.131 | 14696.792 | 2437.291 | 16267.916 | 2989.375 |
| gemm_1025x1025x1024 | 2146.537 | 389.886 | 2013.833 | 386.000 | 2271.375 | 646.416 |
| gemm_129x257x61 | 31.033 | 20.613 | 37.709 | 24.833 | 254.167 | 290.833 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_1024x1024x1024 | mps_matrix_multiplication | 288.717 | 0.982× | 482.250 |
| metal / gemm_4097x4097x4096 | mps_matrix_multiplication | 23479.083 | 6.087× | 23354.791 |
| metal / gemm_2049x4097x1025 | mps_matrix_multiplication | 2901.048 | 5.381× | 3167.542 |
| metal / gemm_1025x1025x1024 | mps_matrix_multiplication | 495.208 | 4.334× | 689.125 |
| metal / gemm_129x257x61 | mps_matrix_multiplication | 15.800 | 2.093× | 240.416 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_1024x1024x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 5211.797 |
| metal / gemm_4097x4097x4096 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 10769.741 |
| metal / gemm_2049x4097x1025 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 5157.495 |
| metal / gemm_1025x1025x1024 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 2453.787 |
| metal / gemm_129x257x61 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 1925.003 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
