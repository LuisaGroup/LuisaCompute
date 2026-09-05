# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T07:07:34.451834+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | rmsnorm_1x127 | 1×127×1 / 2 | 0 | 3.410 | 7.896 | 3.436 | 9.063 | 0.43× | 221.750 | 273.667 |
| metal | rmsnorm_17x257 | 1×257×1 / 2 | 0 | 4.790 | 7.534 | 4.911 | 8.313 | 0.64× | 210.917 | 238.000 |
| metal | rmsnorm_64x4096 | 1×4096×1 / 2 | 0 | 10.773 | 12.380 | 10.976 | 12.944 | 0.87× | 224.708 | 340.708 |
| metal | rmsnorm_1024x4096 | 1×4096×1 / 2 | 0 | 72.750 | 75.493 | 74.269 | 75.883 | 0.96× | 321.209 | 321.250 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / rmsnorm_1x127 | 0.113 | 52.983 | 1.517 | 5.102 | 1.173 | 245.340 | 0.256 | 0.330 |
| metal / rmsnorm_17x257 | 0.069 | 59.841 | 1.640 | 1.877 | 60.412 | 0.590 | 0.263 | 0.350 |
| metal / rmsnorm_64x4096 | 0.080 | 57.509 | 2.497 | 1.563 | 4.086 | 10.104 | 0.547 | 0.530 |
| metal / rmsnorm_1024x4096 | 0.076 | 53.896 | 6.622 | 42.559 | 4.656 | 5.311 | 6.758 | 0.756 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| rmsnorm_1x127 / native | 3.044 | 5.167 | 3.410 | 221.750 | 1.011× |
| rmsnorm_1x127 / torch | 7.600 | 18.500 | 7.896 | 273.667 | 1.000× |
| rmsnorm_17x257 / native | 4.387 | 7.042 | 4.790 | 210.917 | 0.999× |
| rmsnorm_17x257 / torch | 5.293 | 15.292 | 7.534 | 238.000 | 1.050× |
| rmsnorm_64x4096 / native | 10.536 | 14.125 | 10.773 | 224.708 | 0.954× |
| rmsnorm_64x4096 / torch | 8.996 | 12.458 | 12.380 | 340.708 | 1.040× |
| rmsnorm_1024x4096 / native | 68.367 | 70.292 | 72.750 | 321.209 | 1.003× |
| rmsnorm_1024x4096 / torch | 68.990 | 71.792 | 75.493 | 321.250 | 0.999× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| rmsnorm_1x127 | 3.298 | 8.778 | 5.125 | 18.375 | 221.750 | 273.667 |
| rmsnorm_17x257 | 4.382 | 5.559 | 7.208 | 14.791 | 210.917 | 238.000 |
| rmsnorm_64x4096 | 10.102 | 9.116 | 13.542 | 13.333 | 224.708 | 340.708 |
| rmsnorm_1024x4096 | 68.520 | 68.838 | 70.042 | 71.125 | 321.209 | 321.250 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
