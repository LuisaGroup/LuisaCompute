# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T00:34:03.119110+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | residual_layernorm_1x127 | 1×127×1 / 2 | 0 | 3.305 | 10.839 | 3.326 | 10.995 | 0.30× | 229.125 | 236.458 |
| metal | residual_layernorm_17x257 | 1×257×1 / 2 | 0 | 3.625 | 11.851 | 3.643 | 11.910 | 0.31× | 238.125 | 243.208 |
| metal | residual_layernorm_128x1024 | 1×1024×1 / 2 | 0 | 6.142 | 18.458 | 6.178 | 18.670 | 0.33× | 253.791 | 281.500 |
| metal | residual_layernorm_64x4096 | 1×4096×1 / 2 | 0 | 9.495 | 27.044 | 9.531 | 27.256 | 0.35× | 252.875 | 284.542 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / residual_layernorm_1x127 | 0.079 | 48.866 | 1.456 | 3.987 | 0.715 | 59.299 | 0.304 | 0.359 |
| metal / residual_layernorm_17x257 | 0.074 | 50.322 | 1.620 | 0.528 | 0.881 | 1.277 | 0.362 | 0.341 |
| metal / residual_layernorm_128x1024 | 0.070 | 48.967 | 1.287 | 1.330 | 3.822 | 6.923 | 0.433 | 0.386 |
| metal / residual_layernorm_64x4096 | 0.071 | 49.126 | 1.813 | 1.490 | 2.968 | 2.369 | 0.466 | 0.405 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
