# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T15:21:16.154507+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `group`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `True`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `1`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | gemm_2048x2048x2048 | 128×32×1024 / 1 | 1 | 2958.736 | 2789.236 | 2968.947 | 2938.114 | 1.06× | 2466.708 | 2551.500 |
| metal | gemm_4096x4096x4096 | 128×32×1024 / 1 | 1 | 22106.458 | 19513.416 | 22419.125 | 19873.683 | 1.13× | 21953.917 | 19570.417 |
| metal | gemm_8192x8192x8192 | 128×32×1024 / 1 | 1 | 200885.458 | 158585.791 | 205611.425 | 160031.325 | 1.27× | 201889.750 | 155064.375 |
| metal | gemm_256x11008x4096 | 128×32×1024 / 1 | 1 | 3687.069 | 4048.250 | 3694.003 | 5067.784 | 0.91× | 3823.792 | 3699.750 |
| metal | gemm_4096x4096x11008 | FAILED | | | | | | | | |
| metal | gemm_2049x4097x1025 | FAILED | | | | | | | | |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / gemm_2048x2048x2048 | 0.057 | 33.551 | 10.143 | 46.861 | 278.893 | 60.825 | 2.406 | 1.013 |
| metal / gemm_4096x4096x4096 | 0.062 | 33.247 | 41.126 | 5.412 | 297.198 | 29.215 | 10.052 | 4.004 |
| metal / gemm_8192x8192x8192 | 0.058 | 33.134 | 155.373 | 34.279 | 520.157 | 190.120 | 62.712 | 14.010 |
| metal / gemm_256x11008x4096 | 0.297 | 35.536 | 53.524 | 5.595 | 270.784 | 15.613 | 2.467 | 0.634 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
