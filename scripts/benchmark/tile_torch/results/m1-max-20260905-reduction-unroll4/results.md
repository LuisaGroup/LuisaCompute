# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T02:02:35.091289+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sum_1x4096 | 1×4096×1 / 2 | 0 | 3.256 | 7.018 | 3.296 | 7.096 | 0.46× | 247.791 | 230.417 |
| metal | sum_64x4096 | 1×4096×1 / 2 | 0 | 4.593 | 17.564 | 4.612 | 17.695 | 0.26× | 198.291 | 314.042 |
| metal | sum_1024x4096 | 1×4096×1 / 2 | 0 | 25.260 | 27.936 | 26.001 | 28.679 | 0.90× | 270.750 | 273.584 |
| metal | sum_17x257 | 1×257×1 / 2 | 0 | 3.002 | 4.761 | 3.018 | 4.780 | 0.63× | 208.375 | 250.917 |
| metal | sum_1024x257 | 1×257×1 / 2 | 0 | 4.121 | 5.519 | 4.449 | 5.884 | 0.75× | 241.959 | 236.167 |
| metal | softmax_1x4096 | 1×4096×1 / 2 | 0 | 4.617 | 27.142 | 4.666 | 27.259 | 0.17× | 221.125 | 301.833 |
| metal | softmax_64x4096 | 1×4096×1 / 2 | 0 | 7.565 | 31.415 | 7.677 | 31.959 | 0.24× | 204.708 | 288.833 |
| metal | softmax_1024x4096 | 1×4096×1 / 2 | 0 | 67.222 | 126.594 | 67.380 | 127.571 | 0.53× | 311.375 | 426.000 |
| metal | softmax_17x257 | 1×257×1 / 2 | 0 | 2.753 | 29.788 | 2.927 | 43.282 | 0.09× | 223.084 | 392.792 |
| metal | softmax_1024x257 | 1×257×1 / 2 | 0 | 8.294 | 33.524 | 8.367 | 33.777 | 0.25× | 239.417 | 325.125 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sum_1x4096 | 0.088 | 37.717 | 1.147 | 3.738 | 47.324 | 62.739 | 0.383 | 0.509 |
| metal / sum_64x4096 | 0.057 | 38.176 | 1.459 | 0.759 | 50.861 | 3.053 | 0.271 | 0.370 |
| metal / sum_1024x4096 | 0.066 | 38.147 | 5.403 | 38.989 | 4.307 | 0.465 | 0.275 | 0.303 |
| metal / sum_17x257 | 0.063 | 41.005 | 1.361 | 0.332 | 45.973 | 0.238 | 0.274 | 0.309 |
| metal / sum_1024x257 | 0.058 | 38.334 | 1.101 | 0.882 | 49.149 | 0.308 | 0.296 | 0.316 |
| metal / softmax_1x4096 | 0.065 | 44.504 | 1.189 | 0.328 | 55.950 | 35.870 | 0.294 | 0.237 |
| metal / softmax_64x4096 | 0.063 | 49.165 | 1.621 | 0.518 | 57.843 | 3.894 | 0.419 | 0.345 |
| metal / softmax_1024x4096 | 0.075 | 46.901 | 5.339 | 13.816 | 4.027 | 2.806 | 3.040 | 0.626 |
| metal / softmax_17x257 | 0.066 | 50.805 | 1.273 | 0.833 | 52.275 | 2.685 | 0.306 | 0.296 |
| metal / softmax_1024x257 | 0.069 | 47.455 | 1.475 | 11.342 | 54.044 | 2.740 | 0.435 | 0.316 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
