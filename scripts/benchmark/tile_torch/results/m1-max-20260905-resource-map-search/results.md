# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T09:27:55.717123+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_23x769 | 1×769×1 / 2 | 0 | 3.314 | 32.575 | 3.508 | 40.111 | 0.10× | 379.750 | 300.958 |
| metal | softmax_128x2048 | 1×2048×1 / 2 | 0 | 9.560 | 39.840 | 10.178 | 44.363 | 0.24× | 259.750 | 325.917 |
| metal | softmax_1024x4096 | 1×4096×1 / 2 | 0 | 51.871 | 134.409 | 53.774 | 144.320 | 0.39× | 331.041 | 445.792 |
| metal | softmax_128x8193 | 1×8193×1 / 2 | 0 | 22.101 | 67.256 | 26.211 | 77.494 | 0.33× | 264.875 | 374.250 |
| metal | rmsnorm_23x769 | 1×769×1 / 2 | 0 | 3.363 | 7.432 | 3.693 | 7.845 | 0.45× | 208.500 | 212.958 |
| metal | rmsnorm_128x2048 | 1×2048×1 / 2 | 0 | 8.316 | 19.998 | 8.430 | 22.615 | 0.42× | 260.625 | 284.791 |
| metal | rmsnorm_1024x4096 | 1×4096×1 / 2 | 0 | 55.794 | 77.174 | 57.458 | 77.782 | 0.72× | 353.541 | 323.250 |
| metal | rmsnorm_128x8193 | 1×8193×1 / 2 | 0 | 23.172 | 28.057 | 26.005 | 35.586 | 0.83× | 278.166 | 307.958 |
| metal | layernorm_23x769 | 1×769×1 / 2 | 0 | 4.691 | 9.719 | 4.792 | 10.122 | 0.48× | 236.042 | 220.834 |
| metal | layernorm_128x2048 | 1×2048×1 / 2 | 0 | 8.466 | 21.891 | 8.613 | 22.684 | 0.39× | 241.541 | 256.083 |
| metal | layernorm_1024x4096 | 1×4096×1 / 2 | 0 | 63.947 | 219.842 | 64.210 | 231.200 | 0.29× | 286.708 | 454.125 |
| metal | layernorm_128x8193 | 1×8193×1 / 2 | 0 | 25.074 | 79.573 | 25.185 | 82.280 | 0.32× | 237.833 | 295.916 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_23x769 | 0.063 | 53.412 | 0.957 | 0.467 | 0.973 | 0.641 | 0.349 | 0.301 |
| metal / softmax_128x2048 | 0.064 | 52.574 | 1.526 | 0.885 | 4.708 | 0.567 | 0.518 | 0.378 |
| metal / softmax_1024x4096 | 0.070 | 47.719 | 5.833 | 0.874 | 8.819 | 0.617 | 2.915 | 0.680 |
| metal / softmax_128x8193 | 0.066 | 54.810 | 2.134 | 0.706 | 12.816 | 0.654 | 0.861 | 0.523 |
| metal / rmsnorm_23x769 | 0.070 | 52.386 | 1.363 | 0.770 | 1.440 | 0.266 | 0.253 | 0.453 |
| metal / rmsnorm_128x2048 | 0.086 | 54.879 | 1.498 | 0.696 | 3.133 | 0.303 | 0.481 | 0.453 |
| metal / rmsnorm_1024x4096 | 0.067 | 50.346 | 5.781 | 1.230 | 7.228 | 0.422 | 3.121 | 0.618 |
| metal / rmsnorm_128x8193 | 0.073 | 58.741 | 2.622 | 0.863 | 4.864 | 0.503 | 0.889 | 0.378 |
| metal / layernorm_23x769 | 0.070 | 58.731 | 1.791 | 0.698 | 1.467 | 0.322 | 0.268 | 0.293 |
| metal / layernorm_128x2048 | 0.076 | 64.417 | 2.077 | 1.017 | 6.448 | 0.335 | 0.454 | 0.347 |
| metal / layernorm_1024x4096 | 0.076 | 64.636 | 6.117 | 1.353 | 2.747 | 0.677 | 3.433 | 1.096 |
| metal / layernorm_128x8193 | 0.072 | 67.323 | 2.406 | 1.324 | 7.522 | 0.631 | 1.247 | 0.461 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| softmax_23x769 / native | 3.240 | 6.833 | 3.314 | 379.750 | 0.904× |
| softmax_23x769 / torch | 17.754 | 64.250 | 32.575 | 300.958 | 4.421× |
| softmax_128x2048 / native | 8.225 | 11.167 | 9.560 | 259.750 | 1.032× |
| softmax_128x2048 / torch | 27.439 | 64.875 | 39.840 | 325.917 | 2.712× |
| softmax_1024x4096 / native | 49.522 | 52.542 | 51.871 | 331.041 | 1.018× |
| softmax_1024x4096 / torch | 120.726 | 131.000 | 134.409 | 445.792 | 1.199× |
| softmax_128x8193 / native | 20.861 | 26.167 | 22.101 | 264.875 | 0.975× |
| softmax_128x8193 / torch | 49.657 | 90.250 | 67.256 | 374.250 | 2.147× |
| rmsnorm_23x769 / native | 3.439 | 5.458 | 3.363 | 208.500 | 0.988× |
| rmsnorm_23x769 / torch | 4.081 | 7.750 | 7.432 | 212.958 | 1.054× |
| rmsnorm_128x2048 / native | 7.880 | 11.208 | 8.316 | 260.625 | 1.131× |
| rmsnorm_128x2048 / torch | 11.741 | 20.375 | 19.998 | 284.791 | 1.088× |
| rmsnorm_1024x4096 / native | 53.391 | 57.750 | 55.794 | 353.541 | 1.024× |
| rmsnorm_1024x4096 / torch | 70.702 | 71.417 | 77.174 | 323.250 | 1.000× |
| rmsnorm_128x8193 / native | 21.374 | 28.000 | 23.172 | 278.166 | 0.999× |
| rmsnorm_128x8193 / torch | 25.545 | 33.167 | 28.057 | 307.958 | 1.003× |
| layernorm_23x769 / native | 4.577 | 7.083 | 4.691 | 236.042 | 1.066× |
| layernorm_23x769 / torch | 5.753 | 8.625 | 9.719 | 220.834 | 1.065× |
| layernorm_128x2048 / native | 9.001 | 10.833 | 8.466 | 241.541 | 0.927× |
| layernorm_128x2048 / torch | 18.141 | 17.875 | 21.891 | 256.083 | 0.936× |
| layernorm_1024x4096 / native | 61.441 | 62.958 | 63.947 | 286.708 | 0.988× |
| layernorm_1024x4096 / torch | 205.231 | 200.625 | 219.842 | 454.125 | 1.000× |
| layernorm_128x8193 / native | 23.900 | 28.750 | 25.074 | 237.833 | 1.024× |
| layernorm_128x8193 / torch | 70.394 | 76.125 | 79.573 | 295.916 | 0.998× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| softmax_23x769 | 2.924 | 63.063 | 5.917 | 79.583 | 379.750 | 300.958 |
| softmax_128x2048 | 8.473 | 61.221 | 10.042 | 59.000 | 259.750 | 325.917 |
| softmax_1024x4096 | 49.300 | 125.940 | 50.458 | 130.292 | 331.041 | 445.792 |
| softmax_128x8193 | 20.065 | 91.457 | 24.084 | 88.500 | 264.875 | 374.250 |
| rmsnorm_23x769 | 3.242 | 4.245 | 5.417 | 7.750 | 208.500 | 212.958 |
| rmsnorm_128x2048 | 8.820 | 11.879 | 10.833 | 21.750 | 260.625 | 284.791 |
| rmsnorm_1024x4096 | 54.480 | 70.291 | 91.834 | 71.792 | 353.541 | 323.250 |
| rmsnorm_128x8193 | 21.396 | 23.701 | 25.000 | 32.959 | 278.166 | 307.958 |
| layernorm_23x769 | 4.730 | 5.605 | 7.292 | 8.667 | 236.042 | 220.834 |
| layernorm_128x2048 | 8.117 | 16.985 | 10.792 | 18.583 | 241.541 | 256.083 |
| layernorm_1024x4096 | 60.917 | 205.015 | 70.708 | 201.041 | 286.708 | 454.125 |
| layernorm_128x8193 | 24.733 | 69.745 | 28.541 | 76.208 | 237.833 | 295.916 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

The model column is diagnostic only: it compares the lowest reported normalized kernel cost with the measured winner inside the same finite candidate set. Model regret uses the selected objective: measured(model pick) / measured(best) - 1; it is not a hardware-optimality claim. Selection defaults to host-wall throughput; an explicit gpu-control objective uses no-counter GPU command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / measured pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / softmax_23x769 | 10 / 10 | 8×8×16 @ 128t, preserve, P=1, U=1, V=4, cache=False / 8×8×16 @ 32t, preserve, P=1, U=1, V=4, cache=True | 36.45% | 6999.956 |
| metal / softmax_128x2048 | 9 / 10 | 8×8×16 @ 256t, preserve, P=1, U=1, V=4, cache=False / 8×8×16 @ 256t, preserve, P=1, U=1, V=4, cache=True | 17.50% | 6026.668 |
| metal / softmax_1024x4096 | 8 / 10 | 8×8×16 @ 256t, preserve, P=1, U=1, V=4, cache=False / 8×8×16 @ 1024t, preserve, P=1, U=1, V=4, cache=True | 39.18% | 6961.364 |
| metal / softmax_128x8193 | 5 / 10 | 8×8×16 @ 512t, preserve, P=1, U=1, V=4, cache=False / 8×8×16 @ 1024t, preserve, P=1, U=1, V=4, cache=True | 24.63% | 4418.030 |
| metal / rmsnorm_23x769 | 10 / 10 | 8×8×16 @ 256t, preserve, P=1, U=1, V=4, cache=False / 8×8×16 @ 256t, preserve, P=1, U=1, V=4, cache=True | 11.83% | 6207.688 |
| metal / rmsnorm_128x2048 | 10 / 10 | 8×8×16 @ 256t, preserve, P=1, U=1, V=4, cache=False / 8×8×16 @ 128t, preserve, P=1, U=1, V=4, cache=True | 11.20% | 6329.766 |
| metal / rmsnorm_1024x4096 | 9 / 10 | 8×8×16 @ 512t, preserve, P=1, U=1, V=4, cache=False / 8×8×16 @ 256t, preserve, P=1, U=1, V=4, cache=True | 35.81% | 7371.544 |
| metal / rmsnorm_128x8193 | 8 / 10 | 8×8×16 @ 512t, preserve, P=1, U=1, V=4, cache=False / 8×8×16 @ 256t, preserve, P=1, U=1, V=4, cache=True | 3.91% | 5594.027 |
| metal / layernorm_23x769 | 10 / 10 | 8×8×16 @ 128t, preserve, P=1, U=1, V=4, cache=False / 8×8×16 @ 256t, preserve, P=1, U=1, V=4, cache=False | 23.35% | 6071.188 |
| metal / layernorm_128x2048 | 9 / 10 | 8×8×16 @ 256t, preserve, P=1, U=1, V=4, cache=False / 8×8×16 @ 256t, preserve, P=1, U=1, V=4, cache=True | 15.77% | 6011.563 |
| metal / layernorm_1024x4096 | 8 / 10 | 8×8×16 @ 256t, preserve, P=1, U=1, V=4, cache=False / 8×8×16 @ 128t, preserve, P=1, U=1, V=4, cache=True | 25.35% | 7454.923 |
| metal / layernorm_128x8193 | 5 / 10 | 8×8×16 @ 512t, preserve, P=1, U=1, V=4, cache=False / 8×8×16 @ 512t, preserve, P=1, U=1, V=4, cache=True | 11.56% | 4794.470 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
