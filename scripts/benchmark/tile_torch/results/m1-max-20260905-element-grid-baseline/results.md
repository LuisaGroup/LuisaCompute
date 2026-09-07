# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T01:46:00.102714+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | add_1x127 | 1×256×1 / 2 | 0 | 101.592 | 4.208 | 104.428 | 4.333 | 24.14× | 368.833 | 225.375 |
| metal | add_17x257 | 1×256×1 / 2 | 0 | 230.083 | 4.552 | 232.834 | 4.956 | 50.55× | 453.667 | 188.667 |
| metal | add_128x1024 | 1×256×1 / 2 | 0 | 53.788 | 7.726 | 54.032 | 8.198 | 6.96× | 304.875 | 330.083 |
| metal | add_4096x256 | 1×256×1 / 2 | 0 | 100.550 | 27.646 | 102.784 | 28.489 | 3.64× | 324.959 | 267.208 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / add_1x127 | 0.180 | 42.646 | 2.370 | 4.229 | 43.710 | 62.147 | 0.291 | 0.310 |
| metal / add_17x257 | 0.047 | 45.205 | 1.781 | 0.669 | 54.958 | 0.636 | 0.325 | 0.355 |
| metal / add_128x1024 | 0.051 | 43.388 | 2.361 | 0.845 | 50.234 | 8.585 | 0.501 | 0.329 |
| metal / add_4096x256 | 0.045 | 42.640 | 2.632 | 1.984 | 14.328 | 9.603 | 0.810 | 0.425 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
