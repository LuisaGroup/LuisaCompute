# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T00:33:46.602041+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `expensive-only`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | residual_layernorm_1x127 | 1×127×1 / 2 | 0 | 3.699 | 10.663 | 3.753 | 10.743 | 0.35× | 232.084 | 252.333 |
| metal | residual_layernorm_17x257 | 1×257×1 / 2 | 0 | 3.674 | 11.711 | 3.684 | 11.748 | 0.31× | 244.250 | 258.917 |
| metal | residual_layernorm_128x1024 | 1×1024×1 / 2 | 0 | 8.141 | 18.421 | 8.217 | 18.647 | 0.44× | 205.209 | 266.917 |
| metal | residual_layernorm_64x4096 | 1×4096×1 / 2 | 0 | 13.627 | 26.717 | 13.807 | 26.874 | 0.51× | 240.542 | 271.834 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / residual_layernorm_1x127 | 0.071 | 46.433 | 1.515 | 3.901 | 0.703 | 59.032 | 0.298 | 0.405 |
| metal / residual_layernorm_17x257 | 0.063 | 52.098 | 1.327 | 0.537 | 0.945 | 1.377 | 0.321 | 0.350 |
| metal / residual_layernorm_128x1024 | 0.071 | 48.922 | 1.581 | 1.226 | 2.360 | 6.239 | 0.406 | 0.387 |
| metal / residual_layernorm_64x4096 | 0.071 | 49.093 | 1.764 | 1.094 | 2.822 | 2.301 | 0.464 | 0.369 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
