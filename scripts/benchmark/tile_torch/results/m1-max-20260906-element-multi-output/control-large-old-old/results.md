# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T05:16:16.224518+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sigmoid_pair_1024x4096 | 1×256×1 / 2 | 0 | 406.002 | 340.929 | 410.131 | 341.780 | 1.19× | 942.084 | 574.958 |
| metal | sigmoid_pair_4096x4096 | 1×256×1 / 2 | 0 | 1548.575 | 1883.195 | 1763.115 | 2087.111 | 0.82× | 2060.542 | 1723.417 |
| metal | gelu_pair_1024x4096 | 1×256×1 / 2 | 0 | 488.798 | 253.920 | 545.661 | 257.127 | 1.93× | 1091.292 | 656.958 |
| metal | gelu_pair_4096x4096 | 1×256×1 / 2 | 0 | 2203.052 | 1342.024 | 2223.819 | 1378.095 | 1.64× | 2963.584 | 1399.250 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sigmoid_pair_1024x4096 | 0.059 | 31.512 | 5.558 | 46.152 | 81.985 | 85.632 | 9.775 | 2.713 |
| metal / sigmoid_pair_4096x4096 | 0.058 | 31.027 | 17.411 | 4.067 | 20.289 | 1.662 | 31.596 | 5.664 |
| metal / gelu_pair_1024x4096 | 0.064 | 32.250 | 5.881 | 38.802 | 53.989 | 0.923 | 9.816 | 0.998 |
| metal / gelu_pair_4096x4096 | 0.061 | 31.092 | 20.522 | 5.399 | 20.891 | 1.988 | 38.744 | 4.563 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
