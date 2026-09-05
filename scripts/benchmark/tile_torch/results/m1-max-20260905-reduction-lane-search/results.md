# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T07:09:26.642379+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | rmsnorm_1x127 | 1×127×1 / 2 | 0 | 3.003 | 6.949 | 3.198 | 7.150 | 0.43× | 229.250 | 208.042 |
| metal | rmsnorm_17x257 | 1×257×1 / 2 | 0 | 3.271 | 6.125 | 3.548 | 6.322 | 0.53× | 205.417 | 228.167 |
| metal | rmsnorm_64x4096 | 1×4096×1 / 2 | 0 | 9.069 | 12.177 | 9.165 | 13.011 | 0.74× | 250.542 | 259.750 |
| metal | rmsnorm_1024x4096 | 1×4096×1 / 2 | 0 | 70.676 | 76.346 | 71.522 | 77.651 | 0.93× | 288.000 | 409.667 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / rmsnorm_1x127 | 0.064 | 45.956 | 1.337 | 0.753 | 0.901 | 0.307 | 0.275 | 0.362 |
| metal / rmsnorm_17x257 | 0.066 | 49.733 | 1.410 | 0.547 | 1.325 | 0.268 | 0.262 | 0.275 |
| metal / rmsnorm_64x4096 | 0.065 | 51.339 | 1.998 | 1.284 | 1.893 | 0.329 | 0.440 | 0.420 |
| metal / rmsnorm_1024x4096 | 0.061 | 49.606 | 5.985 | 1.367 | 4.661 | 0.518 | 6.326 | 2.751 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| rmsnorm_1x127 / native | 2.764 | 5.125 | 3.003 | 229.250 | 0.972× |
| rmsnorm_1x127 / torch | 4.290 | 9.042 | 6.949 | 208.042 | 1.009× |
| rmsnorm_17x257 / native | 3.354 | 5.667 | 3.271 | 205.417 | 0.988× |
| rmsnorm_17x257 / torch | 3.526 | 7.000 | 6.125 | 228.167 | 0.964× |
| rmsnorm_64x4096 / native | 9.022 | 12.250 | 9.069 | 250.542 | 0.985× |
| rmsnorm_64x4096 / torch | 8.540 | 12.167 | 12.177 | 259.750 | 1.005× |
| rmsnorm_1024x4096 / native | 67.739 | 138.667 | 70.676 | 288.000 | 0.988× |
| rmsnorm_1024x4096 / torch | 68.930 | 71.208 | 76.346 | 409.667 | 0.995× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| rmsnorm_1x127 | 2.688 | 4.422 | 5.167 | 7.625 | 229.250 | 208.042 |
| rmsnorm_17x257 | 3.391 | 3.523 | 5.666 | 7.125 | 205.417 | 228.167 |
| rmsnorm_64x4096 | 8.743 | 8.550 | 12.167 | 12.959 | 250.542 | 259.750 |
| rmsnorm_1024x4096 | 67.829 | 68.848 | 152.958 | 72.417 | 288.000 | 409.667 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

The model column is diagnostic only: it compares the lowest reported normalized kernel cost with the measured winner inside the same finite candidate set. Model regret uses the selected objective: measured(model pick) / measured(best) - 1; it is not a hardware-optimality claim. Selection defaults to host-wall throughput; an explicit gpu-control objective uses no-counter GPU command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / measured pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / rmsnorm_1x127 | 12 / 12 | 8×8×16 @ autot, preserve, P=auto, U=1, V=1 / 8×8×16 @ 32t, preserve, P=auto, U=1, V=4 | 15.27% | 10923.944 |
| metal / rmsnorm_17x257 | 12 / 12 | 8×8×16 @ autot, preserve, P=auto, U=1, V=2 / 8×8×16 @ 256t, preserve, P=auto, U=1, V=4 | 38.69% | 10830.428 |
| metal / rmsnorm_64x4096 | 12 / 12 | 8×8×16 @ autot, preserve, P=auto, U=1, V=4 / 8×8×16 @ 128t, preserve, P=auto, U=1, V=4 | 15.37% | 11002.620 |
| metal / rmsnorm_1024x4096 | 12 / 12 | 8×8×16 @ 256t, preserve, P=auto, U=1, V=1 / 8×8×16 @ 256t, preserve, P=auto, U=1, V=2 | 0.53% | 14361.153 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
