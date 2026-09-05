# TileIR/TVMx vs PyTorch

> Measurement-method audit only: background-load variability prevents a stable
> speed ranking. `valid` means correctness and sample coverage. See the
> [assessment and scope](notes.md) before interpreting either timing phase.

Generated: 2026-09-05T06:33:46.928780+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sum_17x257 | 1×257×1 / 2 | 0 | 3.527 | 5.478 | 3.537 | 5.669 | 0.64× | 239.208 | 206.542 |
| metal | sum_64x4096 | 1×4096×1 / 2 | 0 | 5.380 | 20.671 | 6.396 | 20.861 | 0.26× | 208.458 | 230.625 |
| metal | softmax_17x257 | 1×257×1 / 2 | 0 | 4.234 | 33.311 | 4.279 | 36.306 | 0.13× | 202.375 | 347.041 |
| metal | softmax_64x4096 | 1×4096×1 / 2 | 0 | 11.176 | 36.547 | 11.236 | 38.437 | 0.31× | 238.167 | 306.584 |
| metal | rmsnorm_17x257 | 1×257×1 / 2 | 0 | 6.660 | 7.668 | 6.970 | 8.214 | 0.87× | 231.083 | 350.209 |
| metal | rmsnorm_64x4096 | 1×4096×1 / 2 | 0 | 16.256 | 16.064 | 16.647 | 17.937 | 1.01× | 237.250 | 268.042 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sum_17x257 | 0.080 | 50.191 | 1.325 | 5.319 | 49.293 | 258.689 | 0.265 | 0.573 |
| metal / sum_64x4096 | 0.065 | 41.945 | 1.465 | 0.731 | 3.575 | 4.119 | 0.257 | 0.726 |
| metal / softmax_17x257 | 0.068 | 56.881 | 1.196 | 0.687 | 53.717 | 21.016 | 0.281 | 0.380 |
| metal / softmax_64x4096 | 0.064 | 51.659 | 1.556 | 2.858 | 5.216 | 2.428 | 0.477 | 0.383 |
| metal / rmsnorm_17x257 | 0.085 | 59.854 | 1.265 | 0.847 | 0.834 | 3.282 | 0.288 | 0.644 |
| metal / rmsnorm_64x4096 | 0.064 | 54.058 | 2.161 | 0.927 | 3.787 | 6.973 | 0.520 | 0.382 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| sum_17x257 / native | 2.262 | 4.625 | 3.527 | 239.208 | 1.026× |
| sum_17x257 / torch | 2.932 | 42.833 | 5.478 | 206.542 | 1.107× |
| sum_64x4096 / native | 4.432 | 6.792 | 5.380 | 208.458 | 0.838× |
| sum_64x4096 / torch | 18.376 | 18.125 | 20.671 | 230.625 | 0.686× |
| softmax_17x257 / native | 2.968 | 82.083 | 4.234 | 202.375 | 0.906× |
| softmax_17x257 / torch | 9.902 | 49.042 | 33.311 | 347.041 | 5.854× |
| softmax_64x4096 / native | 16.454 | 20.167 | 11.176 | 238.167 | 1.001× |
| softmax_64x4096 / torch | 23.728 | 64.750 | 36.547 | 306.584 | 3.079× |
| rmsnorm_17x257 / native | 4.811 | 8.208 | 6.660 | 231.083 | 0.941× |
| rmsnorm_17x257 / torch | 3.240 | 7.333 | 7.668 | 350.209 | 0.957× |
| rmsnorm_64x4096 / native | 10.117 | 13.417 | 16.256 | 237.250 | 0.976× |
| rmsnorm_64x4096 / torch | 8.877 | 11.875 | 16.064 | 268.042 | 1.076× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| sum_17x257 | 2.355 | 3.190 | 4.666 | 5.083 | 239.208 | 206.542 |
| sum_64x4096 | 3.971 | 12.608 | 8.458 | 17.167 | 208.458 | 230.625 |
| softmax_17x257 | 2.874 | 35.914 | 4.958 | 41.292 | 202.375 | 347.041 |
| softmax_64x4096 | 7.517 | 59.770 | 10.625 | 60.541 | 238.167 | 306.584 |
| rmsnorm_17x257 | 4.379 | 3.360 | 7.375 | 10.167 | 231.083 | 350.209 |
| rmsnorm_64x4096 | 9.867 | 9.191 | 16.750 | 17.250 | 237.250 | 268.042 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
