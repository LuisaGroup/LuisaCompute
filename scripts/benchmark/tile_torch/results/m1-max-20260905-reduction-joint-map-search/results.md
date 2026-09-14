# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T02:07:46.111259+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sum_1x4096 | 1×4096×1 / 2 | 0 | 3.418 | 7.927 | 3.433 | 8.061 | 0.43× | 242.375 | 259.584 |
| metal | sum_64x4096 | 1×4096×1 / 2 | 0 | 4.462 | 16.641 | 4.481 | 16.666 | 0.27× | 204.000 | 279.166 |
| metal | sum_1024x4096 | 1×4096×1 / 2 | 0 | 23.276 | 28.330 | 23.400 | 28.440 | 0.82× | 271.958 | 235.375 |
| metal | sum_17x257 | 1×257×1 / 2 | 0 | 3.097 | 4.442 | 3.099 | 4.683 | 0.70× | 236.791 | 231.708 |
| metal | sum_1024x257 | 1×257×1 / 2 | 0 | 4.428 | 6.179 | 4.441 | 6.184 | 0.72× | 188.125 | 213.708 |
| metal | softmax_1x4096 | 1×4096×1 / 2 | 0 | 3.658 | 27.088 | 4.383 | 28.813 | 0.14× | 214.084 | 305.500 |
| metal | softmax_64x4096 | 1×4096×1 / 2 | 0 | 7.548 | 30.605 | 7.551 | 31.122 | 0.25× | 246.084 | 309.250 |
| metal | softmax_1024x4096 | 1×4096×1 / 2 | 0 | 63.806 | 127.040 | 63.881 | 127.930 | 0.50× | 294.250 | 406.334 |
| metal | softmax_17x257 | 1×257×1 / 2 | 0 | 3.348 | 29.214 | 3.360 | 30.564 | 0.11× | 238.875 | 362.541 |
| metal | softmax_1024x257 | 1×257×1 / 2 | 0 | 6.774 | 33.392 | 6.811 | 33.993 | 0.20× | 236.833 | 316.000 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sum_1x4096 | 0.058 | 38.166 | 1.338 | 0.892 | 0.673 | 0.321 | 0.389 | 0.295 |
| metal / sum_64x4096 | 0.055 | 37.418 | 1.460 | 0.911 | 1.593 | 0.283 | 0.297 | 0.270 |
| metal / sum_1024x4096 | 0.057 | 38.049 | 5.584 | 1.420 | 1.820 | 0.332 | 0.273 | 0.261 |
| metal / sum_17x257 | 0.053 | 40.037 | 1.330 | 0.369 | 0.732 | 0.193 | 0.288 | 0.288 |
| metal / sum_1024x257 | 0.051 | 37.309 | 1.157 | 0.830 | 1.725 | 0.253 | 0.210 | 0.228 |
| metal / softmax_1x4096 | 0.066 | 44.214 | 1.166 | 0.827 | 0.764 | 0.511 | 0.288 | 0.246 |
| metal / softmax_64x4096 | 0.069 | 46.497 | 1.320 | 0.863 | 2.188 | 0.484 | 0.463 | 0.356 |
| metal / softmax_1024x4096 | 0.063 | 46.894 | 4.979 | 1.251 | 2.547 | 0.633 | 2.849 | 0.636 |
| metal / softmax_17x257 | 0.066 | 50.349 | 1.208 | 1.206 | 0.844 | 0.548 | 0.299 | 0.235 |
| metal / softmax_1024x257 | 0.063 | 48.163 | 1.101 | 0.914 | 1.806 | 0.523 | 0.438 | 0.288 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

The model column is diagnostic only: it compares the lowest reported normalized kernel cost with the measured winner inside the same finite candidate set. Model regret is measured(model pick) / measured(best) - 1; it is not a hardware-optimality claim.

| Device / case | Valid / attempted candidates | Model pick / measured pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / sum_1x4096 | 8 / 8 | 8×8×16 @ autot, preserve, P=auto, U=1 / 8×8×16 @ 128t, preserve, P=auto, U=4 | 14.45% | 5331.377 |
| metal / sum_64x4096 | 8 / 8 | 8×8×16 @ autot, preserve, P=auto, U=4 / 8×8×16 @ 128t, preserve, P=auto, U=1 | 24.52% | 5449.958 |
| metal / sum_1024x4096 | 8 / 8 | 8×8×16 @ autot, preserve, P=auto, U=1 / 8×8×16 @ 128t, preserve, P=auto, U=4 | 14.24% | 5499.644 |
| metal / sum_17x257 | 8 / 8 | 8×8×16 @ autot, preserve, P=auto, U=1 / 8×8×16 @ autot, preserve, P=auto, U=1 | 0.00% | 5681.370 |
| metal / sum_1024x257 | 8 / 8 | 8×8×16 @ autot, preserve, P=auto, U=1 / 8×8×16 @ 128t, preserve, P=4, U=1 | 24.67% | 5738.866 |
| metal / softmax_1x4096 | 4 / 8 | 8×8×16 @ autot, preserve, P=auto, U=1 / 8×8×16 @ autot, preserve, P=auto, U=4 | 24.84% | 3764.283 |
| metal / softmax_64x4096 | 4 / 8 | 8×8×16 @ autot, preserve, P=auto, U=1 / 8×8×16 @ 128t, preserve, P=auto, U=1 | 0.01% | 3661.450 |
| metal / softmax_1024x4096 | 4 / 8 | 8×8×16 @ autot, preserve, P=auto, U=1 / 8×8×16 @ 128t, preserve, P=auto, U=4 | 6.80% | 3973.674 |
| metal / softmax_17x257 | 8 / 8 | 8×8×16 @ autot, preserve, P=auto, U=1 / 8×8×16 @ autot, preserve, P=auto, U=4 | 20.92% | 5836.733 |
| metal / softmax_1024x257 | 8 / 8 | 8×8×16 @ autot, preserve, P=auto, U=4 / 8×8×16 @ autot, preserve, P=4, U=1 | 10.58% | 5275.060 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
