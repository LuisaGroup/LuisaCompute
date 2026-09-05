# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T00:41:14.181109+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | residual_layernorm_1x127 | 1×127×1 / 2 | 0 | 3.109 | 10.950 | 3.168 | 11.061 | 0.28× | 218.333 | 240.708 |
| metal | residual_layernorm_17x257 | 1×257×1 / 2 | 0 | 3.465 | 11.717 | 3.548 | 11.894 | 0.30× | 238.500 | 246.125 |
| metal | residual_layernorm_128x1024 | 1×1024×1 / 2 | 0 | 5.719 | 18.585 | 5.741 | 18.893 | 0.31× | 222.042 | 269.417 |
| metal | residual_layernorm_64x4096 | 1×4096×1 / 2 | 0 | 9.119 | 26.981 | 9.515 | 27.288 | 0.34× | 234.000 | 259.041 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / residual_layernorm_1x127 | 0.070 | 45.230 | 1.523 | 1.039 | 0.629 | 0.404 | 0.275 | 0.305 |
| metal / residual_layernorm_17x257 | 0.069 | 50.162 | 1.566 | 1.042 | 1.155 | 0.329 | 0.275 | 0.315 |
| metal / residual_layernorm_128x1024 | 0.070 | 48.624 | 1.574 | 0.865 | 2.413 | 0.353 | 0.351 | 0.318 |
| metal / residual_layernorm_64x4096 | 0.069 | 49.050 | 1.831 | 1.118 | 3.927 | 0.321 | 0.448 | 0.331 |

## JIT search

All candidates are recaptured, compiled, and checked against the same FP64 oracle. Invalid candidates are retained in JSON but cannot win. Candidate order rotates across cases. Tables above use a fresh post-selection run, not the search minimum; a revalidation failure remains a failure. This is not a confidence interval or an exhaustive search.

Selection wall time below includes JIT, validation, native/PyTorch measurements, and process overhead; it is excluded from warm timings. Full candidate settings, rejected cases, and raw trial samples are in results.json.

The model column is diagnostic only: it compares the lowest reported normalized kernel cost with the measured winner inside the same finite candidate set. Model regret is measured(model pick) / measured(best) - 1; it is not a hardware-optimality claim.

| Device / case | Valid / attempted candidates | Model pick / measured pick | Model regret | Selection wall ms |
|---|---:|---|---:|---:|
| metal / residual_layernorm_1x127 | 4 / 4 | 8×8×16 @ 64t, preserve / 8×8×16 @ 128t, preserve | 17.81% | 3654.095 |
| metal / residual_layernorm_17x257 | 4 / 4 | 8×8×16 @ 128t, preserve / 8×8×16 @ 256t, preserve | 7.31% | 3763.462 |
| metal / residual_layernorm_128x1024 | 4 / 4 | 8×8×16 @ 256t, preserve / 8×8×16 @ 128t, preserve | 9.69% | 3528.038 |
| metal / residual_layernorm_64x4096 | 2 / 4 | 8×8×16 @ 256t, preserve / 8×8×16 @ 256t, preserve | 0.00% | 3612.195 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
