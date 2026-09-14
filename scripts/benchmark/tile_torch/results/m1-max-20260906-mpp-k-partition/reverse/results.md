# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T19:18:04.135553+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_4096x4096x11008 | 128×32×1024 / 1 | 1 | 55973.208 | 48390.083 | 56077.500 | 50197.666 | 1.16× | 56179.750 | 48476.417 |
| metal | gemm_4096x4096x4096 | 128×32×4096 / 1 | 1 | 17847.625 | 16241.958 | 18579.259 | 16342.600 | 1.10× | 18090.541 | 16103.750 |
| metal | gemm_1024x1024x1537 | 128×32×4096 / 1 | 1 | 498.220 | 454.314 | 499.828 | 460.575 | 1.10× | 709.875 | 678.208 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_4096x4096x11008 | 0.058 | 36.165 | 78.887 | 22.704 | 68.320 | 61.394 | 13.922 | 3.110 |
| metal / gemm_4096x4096x4096 | 0.120 | 53.724 | 32.747 | 16.841 | 32.067 | 25.442 | 10.248 | 1.373 |
| metal / gemm_1024x1024x1537 | 0.053 | 35.599 | 5.183 | 17.854 | 3.750 | 1.408 | 0.828 | 0.431 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_4096x4096x11008 / native | 55920.000 | 55639.583 | 55973.208 | 56179.750 | 0.994× |
| gemm_4096x4096x11008 / torch | 47938.875 | 48017.167 | 48390.083 | 48476.417 | 1.003× |
| gemm_4096x4096x11008 / system | 51118.042 | 50529.917 | 51655.958 | 51223.417 | 0.996× |
| gemm_4096x4096x4096 / native | 17721.042 | 18457.042 | 17847.625 | 18090.541 | 1.013× |
| gemm_4096x4096x4096 / torch | 15795.458 | 15807.000 | 16241.958 | 16103.750 | 0.997× |
| gemm_4096x4096x4096 / system | 16841.708 | 17087.958 | 17017.375 | 17038.708 | 0.996× |
| gemm_1024x1024x1537 / native | 482.343 | 472.250 | 498.220 | 709.875 | 1.001× |
| gemm_1024x1024x1537 / torch | 437.976 | 440.792 | 454.314 | 678.208 | 1.066× |
| gemm_1024x1024x1537 / system | 433.358 | 427.250 | 442.928 | 648.209 | 1.059× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_4096x4096x11008 | 55700.125 | 47737.958 | 55715.750 | 48001.458 | 56179.750 | 48476.417 |
| gemm_4096x4096x4096 | 18011.250 | 15752.917 | 18178.291 | 15769.791 | 18090.541 | 16103.750 |
| gemm_1024x1024x1537 | 484.833 | 446.762 | 471.292 | 423.125 | 709.875 | 678.208 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_4096x4096x11008 | mps_matrix_multiplication | 51655.958 | 1.084× | 51223.417 |
| metal / gemm_4096x4096x4096 | mps_matrix_multiplication | 17017.375 | 1.049× | 17038.708 |
| metal / gemm_1024x1024x1537 | mps_matrix_multiplication | 442.928 | 1.125× | 648.209 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_4096x4096x11008 | 4 / 4 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 34341.249 |
| metal / gemm_4096x4096x4096 | 4 / 4 | 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 13545.073 |
| metal / gemm_1024x1024x1537 | 4 / 4 | 128×32×128 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×4096 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 19.77% | 7244.340 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
