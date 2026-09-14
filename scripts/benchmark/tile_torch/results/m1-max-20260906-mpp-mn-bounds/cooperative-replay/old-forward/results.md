# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T03:59:23.341592+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_129x257x61 | 128×32×16 / 1 | 1 | 33.308 | 30.268 | 33.854 | 31.315 | 1.10× | 280.250 | 268.250 |
| metal | gemm_1025x1025x1024 | 128×32×16 / 1 | 1 | 2211.328 | 418.504 | 2222.957 | 420.093 | 5.28× | 2322.459 | 668.375 |
| metal | gemm_2049x4097x1025 | 128×32×16 / 1 | 1 | 15948.062 | 2768.042 | 16215.713 | 2847.923 | 5.76× | 16643.208 | 2888.417 |
| metal | gemm_4097x4097x4096 | 128×32×16 / 1 | 1 | 142139.875 | 21237.750 | 143182.966 | 22051.908 | 6.69× | 141180.583 | 20479.625 |
| metal | gemm_1024x1024x1024 | 128×32×1024 / 1 | 1 | 282.870 | 302.240 | 292.251 | 308.560 | 0.94× | 533.292 | 536.250 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_129x257x61 | 0.055 | 38.824 | 1.360 | 0.825 | 1.233 | 0.439 | 0.279 | 0.299 |
| metal / gemm_1025x1025x1024 | 0.053 | 36.747 | 2.882 | 1.298 | 4.361 | 1.373 | 0.899 | 0.403 |
| metal / gemm_2049x4097x1025 | 0.049 | 39.240 | 8.620 | 2.296 | 24.856 | 9.142 | 8.162 | 1.114 |
| metal / gemm_4097x4097x4096 | 0.055 | 36.192 | 30.895 | 20.956 | 156.199 | 34.408 | 17.656 | 3.742 |
| metal / gemm_1024x1024x1024 | 0.058 | 31.719 | 3.663 | 1.845 | 2.530 | 1.498 | 0.721 | 0.473 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_129x257x61 / native | 31.085 | 38.458 | 33.308 | 280.250 | 1.006× |
| gemm_129x257x61 / torch | 14.814 | 20.667 | 30.268 | 268.250 | 1.814× |
| gemm_129x257x61 / system | 13.918 | 17.083 | 15.775 | 243.708 | 1.793× |
| gemm_1025x1025x1024 / native | 2118.984 | 2180.500 | 2211.328 | 2322.459 | 1.005× |
| gemm_1025x1025x1024 / torch | 386.206 | 373.500 | 418.504 | 668.375 | 1.055× |
| gemm_1025x1025x1024 / system | 486.587 | 466.417 | 504.939 | 695.959 | 1.034× |
| gemm_2049x4097x1025 / native | 15551.583 | 14502.458 | 15948.062 | 16643.208 | 0.991× |
| gemm_2049x4097x1025 / torch | 2658.411 | 2501.667 | 2768.042 | 2888.417 | 0.997× |
| gemm_2049x4097x1025 / system | 2781.187 | 2627.583 | 2915.826 | 2976.208 | 1.018× |
| gemm_4097x4097x4096 / native | 141349.458 | 142939.917 | 142139.875 | 141180.583 | 1.006× |
| gemm_4097x4097x4096 / torch | 20251.375 | 20470.333 | 21237.750 | 20479.625 | 1.005× |
| gemm_4097x4097x4096 / system | 21973.250 | 21936.375 | 22632.375 | 22640.750 | 0.991× |
| gemm_1024x1024x1024 / native | 278.517 | 268.667 | 282.870 | 533.292 | 1.001× |
| gemm_1024x1024x1024 / torch | 287.171 | 275.750 | 302.240 | 536.250 | 1.073× |
| gemm_1024x1024x1024 / system | 279.779 | 270.125 | 287.019 | 524.750 | 1.088× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_129x257x61 | 31.223 | 19.628 | 37.792 | 20.416 | 280.250 | 268.250 |
| gemm_1025x1025x1024 | 2096.396 | 387.421 | 2230.375 | 372.458 | 2322.459 | 668.375 |
| gemm_2049x4097x1025 | 15468.646 | 2627.387 | 14859.375 | 2479.667 | 16643.208 | 2888.417 |
| gemm_4097x4097x4096 | 142253.666 | 20524.958 | 141839.625 | 21144.000 | 141180.583 | 20479.625 |
| gemm_1024x1024x1024 | 279.128 | 289.321 | 270.041 | 276.709 | 533.292 | 536.250 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_129x257x61 | mps_matrix_multiplication | 15.775 | 2.111× | 243.708 |
| metal / gemm_1025x1025x1024 | mps_matrix_multiplication | 504.939 | 4.379× | 695.959 |
| metal / gemm_2049x4097x1025 | mps_matrix_multiplication | 2915.826 | 5.469× | 2976.208 |
| metal / gemm_4097x4097x4096 | mps_matrix_multiplication | 22632.375 | 6.280× | 22640.750 |
| metal / gemm_1024x1024x1024 | mps_matrix_multiplication | 287.019 | 0.986× | 524.750 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / gemm_129x257x61 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 2551.764 |
| metal / gemm_1025x1025x1024 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 2413.933 |
| metal / gemm_2049x4097x1025 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 4042.477 |
| metal / gemm_4097x4097x4096 | 1 / 3 | 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×16 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 12007.099 |
| metal / gemm_1024x1024x1024 | 3 / 3 | 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False / 128×32×1024 @ 128t, preserve, P=auto, U=1, V=1, cache=False | 0.00% | 4988.788 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
