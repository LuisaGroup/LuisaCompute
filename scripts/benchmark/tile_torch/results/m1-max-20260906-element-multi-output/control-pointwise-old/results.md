# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T05:15:57.355252+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | add_37x1537 | 1×256×1 / 2 | 0 | 5.415 | 7.887 | 5.615 | 7.968 | 0.69× | 247.458 | 462.958 |
| metal | add_1024x4096 | 1×256×1 / 2 | 0 | 146.519 | 140.705 | 147.475 | 163.116 | 1.04× | 409.167 | 674.000 |
| metal | gelu_add_37x1537 | 1×256×1 / 2 | 0 | 4.896 | 14.426 | 4.966 | 14.952 | 0.34× | 330.958 | 275.291 |
| metal | gelu_add_1024x4096 | 1×256×1 / 2 | 0 | 132.895 | 338.788 | 155.652 | 339.020 | 0.39× | 586.417 | 656.375 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / add_37x1537 | 0.167 | 32.817 | 1.777 | 5.132 | 50.878 | 89.180 | 0.432 | 0.563 |
| metal / add_1024x4096 | 0.057 | 30.419 | 9.780 | 41.857 | 52.764 | 15.396 | 4.408 | 0.997 |
| metal / gelu_add_37x1537 | 0.061 | 32.310 | 1.615 | 2.408 | 55.941 | 6.877 | 0.485 | 1.932 |
| metal / gelu_add_1024x4096 | 0.054 | 32.646 | 9.088 | 1.619 | 54.780 | 0.744 | 2.574 | 2.537 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
