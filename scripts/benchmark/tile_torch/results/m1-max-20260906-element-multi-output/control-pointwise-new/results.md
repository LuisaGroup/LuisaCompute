# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T05:16:01.465361+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | add_37x1537 | 1×256×1 / 2 | 0 | 4.438 | 9.428 | 4.720 | 14.428 | 0.47× | 375.291 | 244.917 |
| metal | add_1024x4096 | 1×256×1 / 2 | 0 | 144.183 | 153.816 | 164.078 | 156.313 | 0.94× | 552.042 | 960.584 |
| metal | gelu_add_37x1537 | 1×256×1 / 2 | 0 | 4.799 | 15.437 | 4.832 | 15.905 | 0.31× | 226.625 | 418.459 |
| metal | gelu_add_1024x4096 | 1×256×1 / 2 | 0 | 161.240 | 363.651 | 165.327 | 376.641 | 0.44× | 359.792 | 741.458 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / add_37x1537 | 0.052 | 30.062 | 2.038 | 4.323 | 39.090 | 65.816 | 0.331 | 0.337 |
| metal / add_1024x4096 | 0.050 | 29.837 | 9.618 | 40.770 | 42.733 | 11.849 | 2.461 | 1.412 |
| metal / gelu_add_37x1537 | 0.064 | 31.915 | 1.623 | 1.962 | 40.593 | 8.191 | 1.936 | 0.270 |
| metal / gelu_add_1024x4096 | 0.059 | 30.834 | 9.391 | 1.624 | 41.118 | 0.855 | 2.342 | 0.966 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
