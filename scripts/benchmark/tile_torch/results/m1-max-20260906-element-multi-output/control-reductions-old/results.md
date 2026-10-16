# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T05:16:04.221317+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_37x1537 | 1×1537×1 / 2 | 0 | 6.179 | 48.173 | 6.277 | 67.135 | 0.13× | 254.708 | 539.875 |
| metal | softmax_1024x4096 | 1×4096×1 / 2 | 0 | 76.979 | 171.482 | 77.915 | 175.839 | 0.45× | 321.833 | 494.542 |
| metal | rmsnorm_37x1537 | 1×1537×1 / 2 | 0 | 6.587 | 8.317 | 6.940 | 9.648 | 0.79× | 308.625 | 324.041 |
| metal | rmsnorm_1024x4096 | 1×4096×1 / 2 | 0 | 72.501 | 97.625 | 85.682 | 101.541 | 0.74× | 714.666 | 1740.375 |
| metal | layernorm_37x1537 | 1×1537×1 / 2 | 0 | 5.953 | 13.823 | 5.960 | 14.121 | 0.43× | 232.666 | 249.083 |
| metal | layernorm_1024x4096 | 1×4096×1 / 2 | 0 | 87.009 | 391.010 | 106.427 | 860.564 | 0.22× | 335.583 | 989.750 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 0.077 | 32.340 | 1.269 | 4.419 | 63.397 | 61.490 | 0.510 | 0.381 |
| metal / softmax_1024x4096 | 0.070 | 32.158 | 5.932 | 41.620 | 69.933 | 5.268 | 4.395 | 1.008 |
| metal / rmsnorm_37x1537 | 0.067 | 33.634 | 1.657 | 0.745 | 61.301 | 65.077 | 0.418 | 0.379 |
| metal / rmsnorm_1024x4096 | 0.067 | 33.810 | 5.632 | 2.303 | 68.708 | 14.842 | 2.708 | 0.922 |
| metal / layernorm_37x1537 | 0.080 | 35.947 | 1.413 | 0.672 | 61.446 | 1.242 | 0.343 | 1.834 |
| metal / layernorm_1024x4096 | 0.082 | 36.645 | 6.459 | 1.478 | 75.680 | 0.514 | 2.958 | 1.750 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
