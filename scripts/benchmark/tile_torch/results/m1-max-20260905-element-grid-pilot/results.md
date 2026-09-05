# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T01:43:23.154388+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | add_1x127 | 1×256×1 / 2 | 0 | 2.811 | 4.057 | 2.952 | 4.278 | 0.69× | 203.375 | 192.334 |
| metal | add_17x257 | 1×256×1 / 2 | 0 | 2.854 | 4.480 | 3.492 | 4.976 | 0.64× | 239.875 | 215.791 |
| metal | add_128x1024 | 1×256×1 / 2 | 0 | 6.282 | 7.612 | 6.436 | 7.824 | 0.83× | 200.208 | 226.250 |
| metal | add_4096x256 | 1×256×1 / 2 | 0 | 23.530 | 27.775 | 23.561 | 28.666 | 0.85× | 247.458 | 253.167 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / add_1x127 | 0.073 | 42.571 | 1.264 | 4.663 | 49.856 | 62.889 | 0.277 | 0.283 |
| metal / add_17x257 | 0.058 | 46.983 | 1.264 | 1.401 | 49.449 | 1.119 | 1.436 | 0.277 |
| metal / add_128x1024 | 0.053 | 43.775 | 1.662 | 1.211 | 49.655 | 6.630 | 0.343 | 0.292 |
| metal / add_4096x256 | 0.052 | 43.490 | 2.887 | 1.918 | 2.003 | 7.935 | 0.841 | 0.422 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
