# TileIR/TVMx vs PyTorch

> Integration diagnostics only, not a position-balanced or low-noise speed
> ranking. See the [observer audit](../m1-max-20260905-device-timing-counter-control/notes.md).

Generated: 2026-09-05T06:35:35.916602+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_32x32x32 | 32×32×32 / 2 | 4 | 4.500 | 30.602 | 5.082 | 31.964 | 0.15× | 1599.459 | 303.084 |
| metal | gemm_128x128x128 | 32×32×32 / 2 | 8 | 11.702 | 35.276 | 11.735 | 36.345 | 0.33× | 230.708 | 303.417 |
| metal | gemm_127x193x61 | 32×32×32 / 2 | 8 | 16.831 | 32.177 | 17.662 | 32.533 | 0.52× | 282.167 | 287.250 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_32x32x32 | 0.063 | 52.591 | 1.289 | 8.301 | 0.751 | 59.618 | 0.281 | 0.325 |
| metal / gemm_128x128x128 | 0.057 | 66.085 | 2.357 | 1.697 | 1.223 | 10.368 | 0.627 | 0.461 |
| metal / gemm_127x193x61 | 0.051 | 77.615 | 2.174 | 0.936 | 1.290 | 8.040 | 0.401 | 0.367 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| gemm_32x32x32 / native | 3.704 | 8.458 | 4.500 | 1599.459 | 0.894× |
| gemm_32x32x32 / torch | 32.145 | 166.625 | 30.602 | 303.084 | 1.737× |
| gemm_32x32x32 / system | 9.002 | 12.083 | 13.097 | 301.084 | 2.414× |
| gemm_128x128x128 / native | 12.499 | 13.708 | 11.702 | 230.708 | 0.949× |
| gemm_128x128x128 / torch | 19.601 | 145.417 | 35.276 | 303.417 | 1.894× |
| gemm_128x128x128 / system | 12.823 | 19.000 | 18.162 | 258.083 | 2.010× |
| gemm_127x193x61 / native | 13.425 | 23.500 | 16.831 | 282.167 | 0.996× |
| gemm_127x193x61 / torch | 30.559 | 18.500 | 32.177 | 287.250 | 1.413× |
| gemm_127x193x61 / system | 14.703 | 18.125 | 21.561 | 259.458 | 1.721× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| gemm_32x32x32 | 3.092 | 36.134 | 7.459 | 20.667 | 1599.459 | 303.084 |
| gemm_128x128x128 | 8.958 | 18.189 | 14.875 | 147.334 | 230.708 | 303.417 |
| gemm_127x193x61 | 12.188 | 16.615 | 23.000 | 38.416 | 282.167 | 287.250 |

## Direct system-library GEMM baselines

Same FP32 inputs, compact row-major strides, alpha=1, beta=0, no transpose or reduced-precision option. CPU uses classic LP64 Accelerate cblas_sgemm; Metal uses MPSMatrixMultiplication (not MPSGraph) with private buffers and one command buffer per timed batch. Timings include API/encoding/submission costs, not setup or uploads. Complete outputs pass the same FP64 oracle. Raw samples and each case's implementation order are recorded in JSON; use compare_system.py for per-case six-order balance.

| Device / case | System implementation | System p50 µs | Native / system | System latency µs |
|---|---|---:|---:|---:|
| metal / gemm_32x32x32 | mps_matrix_multiplication | 13.097 | 0.344× | 301.084 |
| metal / gemm_128x128x128 | mps_matrix_multiplication | 18.162 | 0.644× | 258.083 |
| metal / gemm_127x193x61 | mps_matrix_multiplication | 21.561 | 0.781× | 259.458 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
