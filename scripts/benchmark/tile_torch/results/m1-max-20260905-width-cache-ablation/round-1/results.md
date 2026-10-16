# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T11:20:32.036788+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_37x1537 | 1×1537×1 / 1 | 0 | 6.864 | 31.002 | 6.962 | 33.258 | 0.22× | 231.291 | 373.667 |
| metal | rmsnorm_37x1537 | 1×1537×1 / 1 | 0 | 6.769 | 10.338 | 7.328 | 10.668 | 0.65× | 206.250 | 247.167 |
| metal | layernorm_37x1537 | 1×1537×1 / 1 | 0 | 8.134 | 14.922 | 8.259 | 15.232 | 0.55× | 232.458 | 252.000 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 0.069 | 51.216 | 1.222 | 0.715 | 2.048 | 1.898 | 0.285 | 0.269 |
| metal / rmsnorm_37x1537 | 0.064 | 55.037 | 1.417 | 0.675 | 2.046 | 0.259 | 0.692 | 0.325 |
| metal / layernorm_37x1537 | 0.085 | 62.460 | 2.974 | 0.813 | 3.990 | 1.632 | 0.314 | 0.342 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| softmax_37x1537 / native | 4.956 | 7.333 | 6.864 | 231.291 | 1.028× |
| softmax_37x1537 / torch | 16.398 | 58.625 | 31.002 | 373.667 | 4.439× |
| rmsnorm_37x1537 / native | 11.646 | 7.583 | 6.769 | 206.250 | 0.408× |
| rmsnorm_37x1537 / torch | 5.320 | 8.125 | 10.338 | 247.167 | 1.003× |
| layernorm_37x1537 / native | 5.637 | 8.375 | 8.134 | 232.458 | 0.997× |
| layernorm_37x1537 / torch | 9.004 | 10.500 | 14.922 | 252.000 | 0.837× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| softmax_37x1537 | 5.092 | 58.109 | 7.125 | 58.291 | 231.291 | 373.667 |
| rmsnorm_37x1537 | 5.348 | 5.616 | 9.875 | 8.750 | 206.250 | 247.167 |
| layernorm_37x1537 | 5.739 | 8.516 | 8.500 | 10.583 | 232.458 | 252.000 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / softmax_37x1537 | 4 / 4 | 8×8×16 @ 416t, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ 416t, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 5535.399 |
| metal / rmsnorm_37x1537 | 4 / 4 | 8×8×16 @ 416t, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ 416t, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 5564.399 |
| metal / layernorm_37x1537 | 4 / 4 | 8×8×16 @ 416t, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ 416t, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 5297.888 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
