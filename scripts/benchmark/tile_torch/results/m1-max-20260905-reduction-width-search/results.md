# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T01:57:32.931038+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sum_1x4096 | 1×4096×1 / 2 | 0 | 2.849 | 7.608 | 2.888 | 8.007 | 0.37× | 229.583 | 209.167 |
| metal | sum_64x4096 | 1×4096×1 / 2 | 0 | 4.503 | 16.816 | 4.508 | 17.418 | 0.27× | 232.000 | 315.500 |
| metal | sum_1024x4096 | 1×4096×1 / 2 | 0 | 23.376 | 28.572 | 23.380 | 28.921 | 0.82× | 258.250 | 264.833 |
| metal | sum_17x257 | 1×257×1 / 2 | 0 | 3.316 | 4.821 | 3.333 | 4.875 | 0.69× | 227.041 | 240.958 |
| metal | sum_1024x257 | 1×257×1 / 2 | 0 | 5.458 | 6.225 | 5.481 | 6.264 | 0.88× | 243.375 | 217.583 |
| metal | softmax_1x4096 | 1×4096×1 / 2 | 0 | 4.856 | 26.831 | 4.903 | 28.446 | 0.18× | 238.167 | 356.333 |
| metal | softmax_64x4096 | 1×4096×1 / 2 | 0 | 8.908 | 31.729 | 8.933 | 31.970 | 0.28× | 224.500 | 337.167 |
| metal | softmax_1024x4096 | 1×4096×1 / 2 | 0 | 63.837 | 128.995 | 64.129 | 129.256 | 0.49× | 291.166 | 406.500 |
| metal | softmax_17x257 | 1×257×1 / 2 | 0 | 3.186 | 29.195 | 3.437 | 29.978 | 0.11× | 224.875 | 335.833 |
| metal | softmax_1024x257 | 1×257×1 / 2 | 0 | 11.809 | 33.417 | 11.890 | 33.739 | 0.35× | 281.375 | 301.708 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sum_1x4096 | 0.057 | 35.493 | 1.207 | 0.636 | 0.662 | 0.331 | 0.406 | 0.266 |
| metal / sum_64x4096 | 0.052 | 37.907 | 1.284 | 0.498 | 1.653 | 0.248 | 0.315 | 0.354 |
| metal / sum_1024x4096 | 0.056 | 37.799 | 5.466 | 1.019 | 1.781 | 0.349 | 0.256 | 0.290 |
| metal / sum_17x257 | 0.054 | 37.811 | 0.990 | 0.781 | 0.825 | 0.221 | 0.260 | 0.303 |
| metal / sum_1024x257 | 0.056 | 38.085 | 1.387 | 1.100 | 1.445 | 0.276 | 0.283 | 0.221 |
| metal / softmax_1x4096 | 0.068 | 47.555 | 1.376 | 0.824 | 0.950 | 0.527 | 0.266 | 0.330 |
| metal / softmax_64x4096 | 0.069 | 44.996 | 1.474 | 0.865 | 3.569 | 0.474 | 0.419 | 0.372 |
| metal / softmax_1024x4096 | 0.068 | 44.968 | 5.240 | 0.886 | 2.599 | 0.509 | 2.815 | 0.660 |
| metal / softmax_17x257 | 0.066 | 48.606 | 1.337 | 0.803 | 1.188 | 0.490 | 0.313 | 0.338 |
| metal / softmax_1024x257 | 0.068 | 45.856 | 1.431 | 0.887 | 1.438 | 0.485 | 0.458 | 0.313 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

The model column is diagnostic only: it compares the lowest reported normalized kernel cost with the measured winner inside the same finite candidate set. Model regret is measured(model pick) / measured(best) - 1; it is not a hardware-optimality claim.

| Device / case | Valid / attempted candidates | Model pick / measured pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / sum_1x4096 | 4 / 4 | 8×8×16 @ 256t, preserve, P=auto, U=1 / 8×8×16 @ 128t, preserve, P=auto, U=1 | 10.53% | 2595.507 |
| metal / sum_64x4096 | 4 / 4 | 8×8×16 @ 256t, preserve, P=auto, U=1 / 8×8×16 @ 128t, preserve, P=auto, U=1 | 24.87% | 2430.687 |
| metal / sum_1024x4096 | 4 / 4 | 8×8×16 @ 256t, preserve, P=auto, U=1 / 8×8×16 @ 128t, preserve, P=auto, U=1 | 7.11% | 2342.200 |
| metal / sum_17x257 | 4 / 4 | 8×8×16 @ 64t, preserve, P=auto, U=1 / 8×8×16 @ 128t, preserve, P=auto, U=1 | 27.64% | 2489.242 |
| metal / sum_1024x257 | 4 / 4 | 8×8×16 @ 64t, preserve, P=auto, U=1 / 8×8×16 @ 64t, preserve, P=auto, U=1 | 0.00% | 2428.046 |
| metal / softmax_1x4096 | 3 / 4 | 8×8×16 @ 256t, preserve, P=auto, U=1 / 8×8×16 @ 128t, preserve, P=auto, U=1 | 22.00% | 2195.210 |
| metal / softmax_64x4096 | 3 / 4 | 8×8×16 @ 256t, preserve, P=auto, U=1 / 8×8×16 @ 256t, preserve, P=auto, U=1 | 0.00% | 1884.361 |
| metal / softmax_1024x4096 | 3 / 4 | 8×8×16 @ 256t, preserve, P=auto, U=1 / 8×8×16 @ 128t, preserve, P=auto, U=1 | 6.35% | 2292.360 |
| metal / softmax_17x257 | 4 / 4 | 8×8×16 @ 64t, preserve, P=auto, U=1 / 8×8×16 @ 256t, preserve, P=auto, U=1 | 30.29% | 2547.658 |
| metal / softmax_1024x257 | 4 / 4 | 8×8×16 @ 64t, preserve, P=auto, U=1 / 8×8×16 @ 64t, preserve, P=auto, U=1 | 0.00% | 2233.199 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
