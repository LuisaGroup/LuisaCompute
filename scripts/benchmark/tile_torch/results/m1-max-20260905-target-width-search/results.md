# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T07:54:17.732398+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sum_17x257 | 1×257×1 / 2 | 0 | 2.804 | 5.006 | 2.855 | 5.963 | 0.56× | 203.000 | 207.000 |
| metal | sum_64x4096 | 1×4096×1 / 2 | 0 | 4.102 | 26.311 | 4.671 | 31.674 | 0.16× | 250.042 | 308.791 |
| metal | sum_1024x4096 | 1×4096×1 / 2 | 0 | 24.412 | 28.302 | 24.435 | 28.598 | 0.86× | 276.084 | 241.500 |
| metal | sum_7x1537 | 1×1537×1 / 2 | 0 | 2.924 | 8.902 | 3.192 | 9.850 | 0.33× | 223.500 | 258.709 |
| metal | sum_128x8192 | 1×8192×1 / 2 | 0 | 8.574 | 23.955 | 8.958 | 24.948 | 0.36× | 234.208 | 252.167 |
| metal | softmax_17x257 | 1×257×1 / 2 | 0 | 3.746 | 30.065 | 4.576 | 31.412 | 0.12× | 199.459 | 348.584 |
| metal | softmax_64x4096 | 1×4096×1 / 2 | 0 | 8.110 | 33.959 | 8.131 | 36.714 | 0.24× | 240.708 | 333.625 |
| metal | softmax_1024x4096 | 1×4096×1 / 2 | 0 | 63.446 | 134.806 | 63.681 | 135.765 | 0.47× | 299.250 | 398.000 |
| metal | softmax_7x1537 | 1×1537×1 / 2 | 0 | 3.903 | 33.055 | 4.024 | 34.527 | 0.12× | 233.833 | 326.500 |
| metal | softmax_128x8192 | 1×8192×1 / 2 | 0 | 23.084 | 46.584 | 25.482 | 50.334 | 0.50× | 240.625 | 331.167 |
| metal | rmsnorm_17x257 | 1×257×1 / 2 | 0 | 3.191 | 8.335 | 3.600 | 9.527 | 0.38× | 193.833 | 222.209 |
| metal | rmsnorm_64x4096 | 1×4096×1 / 2 | 0 | 9.003 | 15.842 | 9.081 | 22.130 | 0.57× | 274.250 | 300.500 |
| metal | rmsnorm_1024x4096 | 1×4096×1 / 2 | 0 | 68.404 | 75.950 | 70.200 | 76.487 | 0.90× | 332.625 | 322.541 |
| metal | rmsnorm_7x1537 | 1×1537×1 / 2 | 0 | 4.419 | 9.644 | 4.610 | 10.251 | 0.46× | 202.625 | 243.500 |
| metal | rmsnorm_128x8192 | 1×8192×1 / 2 | 0 | 22.081 | 26.149 | 22.709 | 27.296 | 0.84× | 276.667 | 271.916 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sum_17x257 | 0.054 | 38.834 | 0.996 | 0.480 | 42.185 | 0.275 | 0.260 | 0.265 |
| metal / sum_64x4096 | 0.062 | 41.823 | 1.145 | 0.464 | 1.884 | 0.275 | 0.280 | 0.247 |
| metal / sum_1024x4096 | 0.058 | 39.567 | 5.520 | 1.149 | 8.587 | 0.306 | 0.250 | 0.263 |
| metal / sum_7x1537 | 0.066 | 41.599 | 1.047 | 0.429 | 1.324 | 0.234 | 0.343 | 0.262 |
| metal / sum_128x8192 | 0.069 | 41.674 | 2.132 | 0.648 | 15.680 | 0.360 | 0.295 | 0.278 |
| metal / softmax_17x257 | 0.077 | 47.560 | 1.224 | 0.483 | 1.029 | 0.572 | 0.271 | 0.319 |
| metal / softmax_64x4096 | 0.066 | 43.403 | 1.229 | 0.796 | 5.527 | 0.562 | 0.706 | 0.327 |
| metal / softmax_1024x4096 | 0.071 | 46.007 | 5.959 | 1.072 | 7.865 | 0.883 | 4.085 | 0.982 |
| metal / softmax_7x1537 | 0.064 | 53.947 | 1.094 | 0.479 | 0.952 | 0.586 | 0.253 | 0.304 |
| metal / softmax_128x8192 | 0.064 | 50.050 | 2.168 | 0.726 | 5.777 | 0.721 | 1.125 | 0.379 |
| metal / rmsnorm_17x257 | 0.087 | 50.450 | 1.389 | 0.597 | 1.040 | 0.242 | 0.379 | 0.459 |
| metal / rmsnorm_64x4096 | 0.057 | 50.392 | 1.816 | 0.681 | 5.721 | 0.858 | 0.420 | 0.675 |
| metal / rmsnorm_1024x4096 | 0.065 | 49.255 | 5.861 | 1.131 | 16.010 | 0.497 | 2.924 | 0.822 |
| metal / rmsnorm_7x1537 | 0.060 | 48.637 | 1.429 | 0.653 | 1.280 | 0.448 | 0.262 | 0.255 |
| metal / rmsnorm_128x8192 | 0.071 | 50.459 | 2.215 | 1.048 | 2.233 | 0.440 | 1.071 | 0.425 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| sum_17x257 / native | 2.642 | 4.583 | 2.804 | 203.000 | 0.983× |
| sum_17x257 / torch | 3.017 | 5.917 | 5.006 | 207.000 | 1.036× |
| sum_64x4096 / native | 3.565 | 6.833 | 4.102 | 250.042 | 1.000× |
| sum_64x4096 / torch | 19.162 | 23.458 | 26.311 | 308.791 | 0.949× |
| sum_1024x4096 / native | 24.013 | 29.500 | 24.412 | 276.084 | 0.985× |
| sum_1024x4096 / torch | 26.123 | 33.125 | 28.302 | 241.500 | 1.002× |
| sum_7x1537 / native | 2.989 | 4.875 | 2.924 | 223.500 | 1.091× |
| sum_7x1537 / torch | 6.680 | 10.583 | 8.902 | 258.709 | 0.899× |
| sum_128x8192 / native | 8.902 | 11.792 | 8.574 | 234.208 | 0.917× |
| sum_128x8192 / torch | 22.158 | 23.542 | 23.955 | 252.167 | 0.942× |
| softmax_17x257 / native | 3.189 | 7.208 | 3.746 | 199.459 | 1.004× |
| softmax_17x257 / torch | 14.480 | 79.625 | 30.065 | 348.584 | 5.459× |
| softmax_64x4096 / native | 8.198 | 10.833 | 8.110 | 240.708 | 1.058× |
| softmax_64x4096 / torch | 22.225 | 64.625 | 33.959 | 333.625 | 3.240× |
| softmax_1024x4096 / native | 60.926 | 66.333 | 63.446 | 299.250 | 0.993× |
| softmax_1024x4096 / torch | 121.522 | 132.125 | 134.806 | 398.000 | 1.186× |
| softmax_7x1537 / native | 4.413 | 7.625 | 3.903 | 233.833 | 0.989× |
| softmax_7x1537 / torch | 19.783 | 104.250 | 33.055 | 326.500 | 4.900× |
| softmax_128x8192 / native | 22.408 | 27.542 | 23.084 | 240.625 | 0.966× |
| softmax_128x8192 / torch | 38.890 | 71.500 | 46.584 | 331.167 | 2.125× |
| rmsnorm_17x257 / native | 3.427 | 6.583 | 3.191 | 193.833 | 0.943× |
| rmsnorm_17x257 / torch | 3.609 | 8.333 | 8.335 | 222.209 | 1.036× |
| rmsnorm_64x4096 / native | 8.835 | 11.500 | 9.003 | 274.250 | 1.018× |
| rmsnorm_64x4096 / torch | 10.334 | 15.625 | 15.842 | 300.500 | 1.085× |
| rmsnorm_1024x4096 / native | 64.745 | 66.083 | 68.404 | 332.625 | 0.996× |
| rmsnorm_1024x4096 / torch | 68.846 | 72.208 | 75.950 | 322.541 | 1.005× |
| rmsnorm_7x1537 / native | 4.582 | 6.708 | 4.419 | 202.625 | 0.995× |
| rmsnorm_7x1537 / torch | 3.947 | 8.500 | 9.644 | 243.500 | 1.167× |
| rmsnorm_128x8192 / native | 21.331 | 26.875 | 22.081 | 276.667 | 0.979× |
| rmsnorm_128x8192 / torch | 22.855 | 28.583 | 26.149 | 271.916 | 1.005× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| sum_17x257 | 2.651 | 3.139 | 4.666 | 5.292 | 203.000 | 207.000 |
| sum_64x4096 | 3.941 | 18.037 | 6.458 | 24.667 | 250.042 | 308.791 |
| sum_1024x4096 | 23.758 | 26.507 | 30.333 | 33.458 | 276.084 | 241.500 |
| sum_7x1537 | 3.038 | 5.963 | 5.875 | 9.375 | 223.500 | 258.709 |
| sum_128x8192 | 8.164 | 20.842 | 11.708 | 23.750 | 234.208 | 252.167 |
| softmax_17x257 | 3.351 | 62.493 | 7.333 | 81.625 | 199.459 | 348.584 |
| softmax_64x4096 | 8.676 | 59.092 | 10.833 | 60.666 | 240.708 | 333.625 |
| softmax_1024x4096 | 60.152 | 125.631 | 60.792 | 132.166 | 299.250 | 398.000 |
| softmax_7x1537 | 4.225 | 79.125 | 6.625 | 101.167 | 233.833 | 326.500 |
| softmax_128x8192 | 21.835 | 66.839 | 27.875 | 69.625 | 240.625 | 331.167 |
| rmsnorm_17x257 | 3.452 | 3.765 | 6.792 | 7.500 | 193.833 | 222.209 |
| rmsnorm_64x4096 | 8.902 | 11.883 | 11.834 | 19.708 | 274.250 | 300.500 |
| rmsnorm_1024x4096 | 64.486 | 69.316 | 77.042 | 71.958 | 332.625 | 322.541 |
| rmsnorm_7x1537 | 4.558 | 4.819 | 6.625 | 7.542 | 202.625 | 243.500 |
| rmsnorm_128x8192 | 21.212 | 22.997 | 26.375 | 27.792 | 276.667 | 271.916 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

The model column is diagnostic only: it compares the lowest reported normalized kernel cost with the measured winner inside the same finite candidate set. Model regret uses the selected objective: measured(model pick) / measured(best) - 1; it is not a hardware-optimality claim. Selection defaults to host-wall throughput; an explicit gpu-control objective uses no-counter GPU command-buffer throughput, never the instrumented compute-pass probe.

| Device / case | Valid / attempted candidates | Model pick / measured pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / sum_17x257 | 6 / 6 | 8×8×16 @ 96t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 32t, preserve, P=auto, U=1, V=4 | 20.17% | 4176.712 |
| metal / sum_64x4096 | 6 / 6 | 8×8×16 @ 256t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 128t, preserve, P=auto, U=1, V=4 | 7.24% | 3605.302 |
| metal / sum_1024x4096 | 6 / 6 | 8×8×16 @ 256t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 1024t, preserve, P=auto, U=1, V=4 | 6.79% | 3708.057 |
| metal / sum_7x1537 | 6 / 6 | 8×8×16 @ 128t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 96t, preserve, P=auto, U=1, V=4 | 0.51% | 3403.817 |
| metal / sum_128x8192 | 6 / 6 | 8×8×16 @ 512t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 1024t, preserve, P=auto, U=1, V=4 | 35.39% | 3681.175 |
| metal / softmax_17x257 | 6 / 6 | 8×8×16 @ 96t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 96t, preserve, P=auto, U=1, V=4 | 0.00% | 3890.334 |
| metal / softmax_64x4096 | 5 / 6 | 8×8×16 @ 256t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 1024t, preserve, P=auto, U=1, V=4 | 1.29% | 3313.352 |
| metal / softmax_1024x4096 | 5 / 6 | 8×8×16 @ 256t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 1024t, preserve, P=auto, U=1, V=4 | 17.81% | 4342.629 |
| metal / softmax_7x1537 | 6 / 6 | 8×8×16 @ 256t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 128t, preserve, P=auto, U=1, V=4 | 6.94% | 3988.971 |
| metal / softmax_128x8192 | 4 / 6 | 8×8×16 @ 512t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 256t, preserve, P=auto, U=1, V=4 | 16.94% | 3216.794 |
| metal / rmsnorm_17x257 | 6 / 6 | 8×8×16 @ 96t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 96t, preserve, P=auto, U=1, V=4 | 0.00% | 3544.378 |
| metal / rmsnorm_64x4096 | 6 / 6 | 8×8×16 @ 256t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 512t, preserve, P=auto, U=1, V=4 | 4.93% | 3725.834 |
| metal / rmsnorm_1024x4096 | 6 / 6 | 8×8×16 @ 256t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 1024t, preserve, P=auto, U=1, V=4 | 13.25% | 4589.073 |
| metal / rmsnorm_7x1537 | 6 / 6 | 8×8×16 @ 256t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 1024t, preserve, P=auto, U=1, V=4 | 9.95% | 3594.831 |
| metal / rmsnorm_128x8192 | 6 / 6 | 8×8×16 @ 512t, preserve, P=auto, U=1, V=4 / 8×8×16 @ 128t, preserve, P=auto, U=1, V=4 | 0.56% | 4147.413 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
