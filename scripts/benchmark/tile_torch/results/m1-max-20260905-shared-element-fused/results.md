# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T05:36:40.926440+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gelu_add_1x127 | 1×256×1 / 2 | 0 | 3.185 | 10.866 | 3.214 | 11.259 | 0.29× | 505.542 | 216.583 |
| metal | gelu_add_17x257 | 1×256×1 / 2 | 0 | 3.221 | 11.326 | 3.373 | 11.593 | 0.28× | 227.792 | 220.917 |
| metal | gelu_add_128x1024 | 1×256×1 / 2 | 0 | 6.803 | 19.361 | 7.460 | 20.227 | 0.35× | 353.750 | 275.167 |
| metal | gelu_add_4096x256 | 1×256×1 / 2 | 0 | 25.379 | 78.412 | 25.496 | 88.864 | 0.32× | 369.125 | 311.667 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gelu_add_1x127 | 0.058 | 51.462 | 1.343 | 7.214 | 55.974 | 70.094 | 0.322 | 0.307 |
| metal / gelu_add_17x257 | 0.064 | 58.041 | 1.448 | 2.347 | 55.101 | 1.544 | 0.358 | 0.307 |
| metal / gelu_add_128x1024 | 0.061 | 50.785 | 2.058 | 0.935 | 57.865 | 7.743 | 0.648 | 0.321 |
| metal / gelu_add_4096x256 | 0.058 | 50.893 | 3.017 | 3.688 | 2.140 | 8.166 | 0.795 | 0.378 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
