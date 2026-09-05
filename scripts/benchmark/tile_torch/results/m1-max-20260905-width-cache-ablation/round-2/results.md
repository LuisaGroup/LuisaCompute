# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T11:20:53.709397+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_37x1537 | 1×1537×1 / 1 | 0 | 6.830 | 31.917 | 6.879 | 32.789 | 0.21× | 211.875 | 329.125 |
| metal | rmsnorm_37x1537 | 1×1537×1 / 1 | 0 | 6.735 | 9.806 | 8.304 | 10.128 | 0.69× | 217.417 | 234.625 |
| metal | layernorm_37x1537 | 1×1537×1 / 1 | 0 | 7.816 | 15.044 | 7.854 | 16.241 | 0.52× | 215.541 | 247.583 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 0.061 | 52.992 | 1.004 | 0.931 | 2.980 | 2.016 | 0.595 | 0.290 |
| metal / rmsnorm_37x1537 | 0.062 | 54.268 | 3.404 | 0.684 | 2.034 | 0.291 | 0.587 | 0.580 |
| metal / layernorm_37x1537 | 0.074 | 61.761 | 2.803 | 0.801 | 3.808 | 1.706 | 0.308 | 0.328 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| softmax_37x1537 / native | 4.993 | 7.417 | 6.830 | 211.875 | 0.956× |
| softmax_37x1537 / torch | 16.456 | 57.875 | 31.917 | 329.125 | 4.417× |
| rmsnorm_37x1537 / native | 5.426 | 7.500 | 6.735 | 217.417 | 0.964× |
| rmsnorm_37x1537 / torch | 5.606 | 8.125 | 9.806 | 234.625 | 1.006× |
| layernorm_37x1537 / native | 6.048 | 8.250 | 7.816 | 215.541 | 0.951× |
| layernorm_37x1537 / torch | 8.861 | 11.542 | 15.044 | 247.583 | 0.844× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| softmax_37x1537 | 4.777 | 57.432 | 7.167 | 57.875 | 211.875 | 329.125 |
| rmsnorm_37x1537 | 5.232 | 5.472 | 7.250 | 8.791 | 217.417 | 234.625 |
| layernorm_37x1537 | 5.695 | 8.625 | 8.041 | 10.708 | 215.541 | 247.583 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

For host/gpu-control selection, the model column is diagnostic: regret is measured(model pick) / measured(best) - 1 inside the same finite set. Explicit model selection uses only reported whole-kernel costs, not timing labels; no measured regret is inferred by comparing two model scores. Trials still execute for validation and diagnostics, so this is not a compile-only tuning path. GPU-control selection uses no-counter command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / selected pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / softmax_37x1537 | 4 / 4 | 8×8×16 @ 416t, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ 416t, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 5364.801 |
| metal / rmsnorm_37x1537 | 4 / 4 | 8×8×16 @ 416t, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ 416t, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 5344.931 |
| metal / layernorm_37x1537 | 4 / 4 | 8×8×16 @ 416t, preserve, P=1, U=1, V=4, cache=True / 8×8×16 @ 416t, preserve, P=1, U=1, V=4, cache=True | not measured (model selection) | 5223.952 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
