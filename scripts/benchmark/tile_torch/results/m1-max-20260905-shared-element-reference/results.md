# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T05:35:48.160224+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gelu_add_1x127 | 1×256×1 / 2 | 0 | 93.896 | 11.293 | 94.531 | 11.401 | 8.31× | 352.209 | 238.375 |
| metal | gelu_add_17x257 | 1×256×1 / 2 | 0 | 160.628 | 11.344 | 161.897 | 11.460 | 14.16× | 507.084 | 238.833 |
| metal | gelu_add_128x1024 | 1×256×1 / 2 | 0 | 60.632 | 18.735 | 62.314 | 20.167 | 3.24× | 320.167 | 273.584 |
| metal | gelu_add_4096x256 | 1×256×1 / 2 | 0 | 86.277 | 75.264 | 90.236 | 78.202 | 1.15× | 431.750 | 318.500 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gelu_add_1x127 | 0.132 | 45.435 | 1.449 | 4.372 | 70.925 | 115.900 | 0.256 | 0.305 |
| metal / gelu_add_17x257 | 0.074 | 50.502 | 1.353 | 0.653 | 69.157 | 0.615 | 1.519 | 0.339 |
| metal / gelu_add_128x1024 | 0.067 | 47.394 | 1.785 | 1.093 | 61.772 | 8.284 | 0.347 | 0.304 |
| metal / gelu_add_4096x256 | 0.069 | 45.477 | 3.412 | 1.980 | 7.412 | 20.098 | 0.847 | 0.418 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
