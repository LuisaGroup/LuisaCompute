# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T19:16:51.618961+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_1024x1024x1537 | 128×32×4096 / 1 | 1 | 491.141 | 456.898 | 501.946 | 461.413 | 1.07× | 707.500 | 722.583 |
| metal | gemm_4096x4096x4096 | 128×32×4096 / 1 | 1 | 18631.458 | 17060.542 | 19236.891 | 17486.717 | 1.09× | 18516.708 | 16686.458 |
| metal | gemm_4096x4096x11008 | 128×32×1024 / 1 | 1 | 55957.750 | 48630.333 | 56342.008 | 50023.416 | 1.15× | 56208.083 | 48529.250 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_1024x1024x1537 | 0.053 | 35.247 | 3.780 | 1.190 | 3.734 | 0.889 | 1.204 | 0.590 |
| metal / gemm_4096x4096x4096 | 0.061 | 31.509 | 29.688 | 18.103 | 29.545 | 27.267 | 9.115 | 2.176 |
| metal / gemm_4096x4096x11008 | 0.059 | 34.840 | 81.325 | 23.196 | 71.228 | 62.775 | 20.334 | 10.667 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_1024x1024x1537 / native | 483.632 | 469.250 | 491.141 | 707.500 | 0.995× |
| gemm_1024x1024x1537 / torch | 440.834 | 427.458 | 456.898 | 722.583 | 1.052× |
| gemm_1024x1024x1537 / system | 436.375 | 420.375 | 451.715 | 646.709 | 1.052× |
| gemm_4096x4096x4096 / native | 18039.500 | 17986.167 | 18631.458 | 18516.708 | 1.004× |
| gemm_4096x4096x4096 / torch | 16614.708 | 16685.625 | 17060.542 | 16686.458 | 1.006× |
| gemm_4096x4096x4096 / system | 17871.250 | 17981.500 | 18571.542 | 18054.000 | 1.004× |
| gemm_4096x4096x11008 / native | 55680.583 | 55569.667 | 55957.750 | 56208.083 | 1.002× |
| gemm_4096x4096x11008 / torch | 47416.083 | 47792.542 | 48630.333 | 48529.250 | 1.001× |
| gemm_4096x4096x11008 / system | 50425.167 | 50664.125 | 53156.666 | 51911.666 | 0.989× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_1024x1024x1537 | 482.724 | 443.449 | 472.167 | 425.167 | 707.500 | 722.583 |
| gemm_4096x4096x4096 | 18105.250 | 16696.083 | 18053.334 | 16778.750 | 18516.708 | 16686.458 |
| gemm_4096x4096x11008 | 55775.625 | 47977.750 | 55447.750 | 47859.250 | 56208.083 | 48529.250 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_1024x1024x1537 | mps_matrix_multiplication | 451.715 | 1.087× | 646.709 |
| metal / gemm_4096x4096x4096 | mps_matrix_multiplication | 18571.542 | 1.003× | 18054.000 |
| metal / gemm_4096x4096x11008 | mps_matrix_multiplication | 53156.666 | 1.053× | 51911.666 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_1024x1024x1537 | 4 / 4 | 128×32×128 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 20.32% | 7965.223 |
| metal / gemm_4096x4096x4096 | 4 / 4 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 15078.984 |
| metal / gemm_4096x4096x11008 | 4 / 4 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 34950.186 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
