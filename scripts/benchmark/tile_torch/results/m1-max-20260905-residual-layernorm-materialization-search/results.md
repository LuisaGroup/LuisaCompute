# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T00:39:21.476557+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | residual_layernorm_1x127 | 1×127×1 / 2 | 0 | 3.426 | 10.671 | 3.538 | 10.697 | 0.32× | 240.500 | 247.042 |
| metal | residual_layernorm_17x257 | 1×257×1 / 2 | 0 | 3.655 | 11.705 | 3.683 | 11.745 | 0.31× | 212.834 | 228.209 |
| metal | residual_layernorm_128x1024 | 1×1024×1 / 2 | 0 | 6.321 | 18.592 | 6.361 | 18.714 | 0.34× | 254.792 | 264.292 |
| metal | residual_layernorm_64x4096 | 1×4096×1 / 2 | 0 | 8.324 | 27.046 | 9.406 | 27.106 | 0.31× | 202.375 | 270.500 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / residual_layernorm_1x127 | 0.070 | 48.672 | 1.562 | 0.947 | 0.637 | 0.396 | 0.277 | 0.303 |
| metal / residual_layernorm_17x257 | 0.068 | 50.478 | 1.169 | 1.056 | 0.815 | 0.365 | 0.297 | 0.326 |
| metal / residual_layernorm_128x1024 | 0.072 | 49.154 | 1.713 | 1.454 | 3.923 | 3.631 | 0.433 | 0.353 |
| metal / residual_layernorm_64x4096 | 0.071 | 49.227 | 1.769 | 1.204 | 3.131 | 0.357 | 0.402 | 0.348 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

The model column is diagnostic only: it compares the lowest reported normalized kernel cost with the measured winner inside the same finite candidate set. Model regret is measured(model pick) / measured(best) - 1; it is not a hardware-optimality claim.

| Device / case | Valid / attempted candidates | Model pick / measured pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / residual_layernorm_1x127 | 2 / 2 | 8×8×16 @ 0t, expensive-only / 8×8×16 @ 0t, preserve | 6.82% | 3102.069 |
| metal / residual_layernorm_17x257 | 2 / 2 | 8×8×16 @ 0t, expensive-only / 8×8×16 @ 0t, preserve | 1.80% | 2969.758 |
| metal / residual_layernorm_128x1024 | 2 / 2 | 8×8×16 @ 0t, expensive-only / 8×8×16 @ 0t, preserve | 37.51% | 2853.296 |
| metal / residual_layernorm_64x4096 | 2 / 2 | 8×8×16 @ 0t, expensive-only / 8×8×16 @ 0t, preserve | 43.66% | 2946.795 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
