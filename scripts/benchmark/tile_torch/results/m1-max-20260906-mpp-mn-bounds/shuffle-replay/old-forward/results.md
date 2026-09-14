# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T04:09:57.319831+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_129x257x61 | 128×32×16 / 1 | 1 | 33.645 | 29.399 | 33.827 | 31.099 | 1.14× | 264.292 | 367.083 |
| metal | gemm_1025x1025x1024 | 128×32×16 / 1 | 1 | 2113.620 | 404.427 | 2118.744 | 419.634 | 5.23× | 2407.333 | 718.667 |
| metal | gemm_2049x4097x1025 | 128×32×16 / 1 | 1 | 15938.083 | 2776.226 | 16158.558 | 2898.125 | 5.74× | 16249.500 | 3061.416 |
| metal | gemm_4097x4097x4096 | 128×32×16 / 1 | 1 | 144142.375 | 21034.250 | 147050.367 | 23017.109 | 6.85× | 141140.208 | 21054.208 |
| metal | gemm_1024x1024x1024 | 128×32×1024 / 1 | 1 | 287.064 | 304.559 | 302.124 | 308.179 | 0.94× | 502.500 | 628.166 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_129x257x61 | 0.055 | 39.905 | 1.465 | 0.696 | 1.249 | 0.449 | 0.313 | 0.302 |
| metal / gemm_1025x1025x1024 | 0.053 | 36.669 | 3.111 | 1.586 | 6.689 | 1.719 | 1.255 | 0.404 |
| metal / gemm_2049x4097x1025 | 0.059 | 39.217 | 8.847 | 1.646 | 25.193 | 10.980 | 7.537 | 1.783 |
| metal / gemm_4097x4097x4096 | 0.063 | 37.642 | 31.693 | 21.443 | 157.496 | 31.600 | 17.906 | 2.943 |
| metal / gemm_1024x1024x1024 | 0.054 | 33.173 | 3.501 | 1.804 | 2.769 | 1.441 | 0.924 | 0.534 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_129x257x61 / native | 31.995 | 37.875 | 33.645 | 264.292 | 0.987× |
| gemm_129x257x61 / torch | 22.968 | 30.125 | 29.399 | 367.083 | 1.764× |
| gemm_129x257x61 / system | 14.986 | 17.500 | 16.609 | 267.750 | 1.754× |
| gemm_1025x1025x1024 / native | 2122.583 | 2236.667 | 2113.620 | 2407.333 | 0.967× |
| gemm_1025x1025x1024 / torch | 385.954 | 375.042 | 404.427 | 718.667 | 1.056× |
| gemm_1025x1025x1024 / system | 486.638 | 466.042 | 498.787 | 694.500 | 1.049× |
| gemm_2049x4097x1025 / native | 15136.375 | 14775.417 | 15938.083 | 16249.500 | 0.996× |
| gemm_2049x4097x1025 / torch | 2609.839 | 2448.042 | 2776.226 | 3061.416 | 0.993× |
| gemm_2049x4097x1025 / system | 2797.625 | 2643.583 | 2857.930 | 3005.625 | 0.995× |
| gemm_4097x4097x4096 / native | 140672.250 | 142828.375 | 144142.375 | 141140.208 | 0.992× |
| gemm_4097x4097x4096 / torch | 20189.875 | 20586.792 | 21034.250 | 21054.208 | 1.015× |
| gemm_4097x4097x4096 / system | 22332.917 | 22008.250 | 22714.250 | 23061.333 | 0.981× |
| gemm_1024x1024x1024 / native | 282.365 | 270.292 | 287.064 | 502.500 | 1.001× |
| gemm_1024x1024x1024 / torch | 288.313 | 275.250 | 304.559 | 628.166 | 1.077× |
| gemm_1024x1024x1024 / system | 284.426 | 273.875 | 293.391 | 568.000 | 1.073× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_129x257x61 | 31.559 | 30.143 | 37.333 | 29.709 | 264.292 | 367.083 |
| gemm_1025x1025x1024 | 2071.641 | 387.657 | 2206.292 | 372.250 | 2407.333 | 718.667 |
| gemm_2049x4097x1025 | 15119.541 | 2594.518 | 14789.709 | 2478.000 | 16249.500 | 3061.416 |
| gemm_4097x4097x4096 | 139773.209 | 20301.958 | 142667.292 | 20430.708 | 141140.208 | 21054.208 |
| gemm_1024x1024x1024 | 281.820 | 292.335 | 269.375 | 276.167 | 502.500 | 628.166 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_129x257x61 | mps_matrix_multiplication | 16.609 | 2.026× | 267.750 |
| metal / gemm_1025x1025x1024 | mps_matrix_multiplication | 498.787 | 4.238× | 694.500 |
| metal / gemm_2049x4097x1025 | mps_matrix_multiplication | 2857.930 | 5.577× | 3005.625 |
| metal / gemm_4097x4097x4096 | mps_matrix_multiplication | 22714.250 | 6.346× | 23061.333 |
| metal / gemm_1024x1024x1024 | mps_matrix_multiplication | 293.391 | 0.978× | 568.000 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_129x257x61 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 2557.672 |
| metal / gemm_1025x1025x1024 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 2382.581 |
| metal / gemm_2049x4097x1025 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 3718.321 |
| metal / gemm_4097x4097x4096 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 11979.933 |
| metal / gemm_1024x1024x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 5159.395 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
