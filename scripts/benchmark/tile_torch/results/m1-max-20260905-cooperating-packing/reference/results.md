# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T12:51:32.226223+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_37x1537 | 1×1537×1 / 2 | 0 | 5.946 | 42.882 | 6.437 | 54.554 | 0.14× | 481.542 | 335.959 |
| metal | softmax_256x3072 | 1×3072×1 / 2 | 0 | 19.270 | 50.926 | 21.150 | 50.997 | 0.38× | 527.584 | 314.291 |
| metal | softmax_768x6144 | 1×6144×1 / 2 | 0 | 91.254 | 194.956 | 94.930 | 201.478 | 0.47× | 286.417 | 668.958 |
| metal | softmax_1024x4096 | 1×4096×1 / 2 | 0 | 72.718 | 168.104 | 79.353 | 180.045 | 0.43× | 329.666 | 731.542 |
| metal | rmsnorm_37x1537 | 1×1537×1 / 2 | 0 | 5.701 | 9.829 | 6.444 | 9.868 | 0.58× | 233.250 | 252.458 |
| metal | rmsnorm_256x3072 | 1×3072×1 / 2 | 0 | 20.639 | 27.221 | 21.574 | 27.260 | 0.76× | 346.625 | 719.500 |
| metal | rmsnorm_768x6144 | 1×6144×1 / 2 | 0 | 95.261 | 129.231 | 96.323 | 132.033 | 0.74× | 301.917 | 367.000 |
| metal | rmsnorm_1024x4096 | 1×4096×1 / 2 | 0 | 71.637 | 113.614 | 81.906 | 115.843 | 0.63× | 308.500 | 257.125 |
| metal | layernorm_37x1537 | 1×1537×1 / 2 | 0 | 6.892 | 13.585 | 6.941 | 14.133 | 0.51× | 227.541 | 321.250 |
| metal | layernorm_256x3072 | 1×3072×1 / 2 | 0 | 22.341 | 65.181 | 27.396 | 72.245 | 0.34× | 262.375 | 458.334 |
| metal | layernorm_768x6144 | 1×6144×1 / 2 | 0 | 99.329 | 407.143 | 117.016 | 407.359 | 0.24× | 666.916 | 624.500 |
| metal | layernorm_1024x4096 | 1×4096×1 / 2 | 0 | 100.912 | 314.586 | 102.026 | 315.638 | 0.32× | 346.250 | 498.166 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 0.125 | 53.905 | 1.703 | 5.038 | 64.158 | 55.103 | 0.336 | 0.325 |
| metal / softmax_256x3072 | 0.064 | 54.286 | 2.101 | 1.508 | 67.910 | 5.207 | 0.872 | 0.432 |
| metal / softmax_768x6144 | 0.077 | 53.741 | 6.982 | 1.234 | 70.137 | 43.840 | 5.678 | 0.863 |
| metal / softmax_1024x4096 | 0.078 | 54.057 | 6.056 | 1.009 | 6.655 | 7.144 | 4.865 | 0.705 |
| metal / rmsnorm_37x1537 | 0.067 | 57.819 | 1.419 | 1.251 | 59.477 | 240.254 | 0.737 | 0.420 |
| metal / rmsnorm_256x3072 | 0.079 | 57.035 | 2.299 | 1.133 | 4.595 | 6.806 | 0.902 | 0.497 |
| metal / rmsnorm_768x6144 | 0.064 | 56.718 | 6.971 | 19.565 | 5.896 | 9.153 | 5.563 | 0.924 |
| metal / rmsnorm_1024x4096 | 0.069 | 55.548 | 6.298 | 1.112 | 7.418 | 1.714 | 4.985 | 0.682 |
| metal / layernorm_37x1537 | 0.077 | 66.342 | 3.052 | 0.732 | 61.196 | 1.154 | 0.405 | 1.988 |
| metal / layernorm_256x3072 | 0.071 | 65.254 | 2.439 | 1.814 | 7.457 | 0.509 | 1.111 | 0.397 |
| metal / layernorm_768x6144 | 0.077 | 62.519 | 7.290 | 17.763 | 5.477 | 0.859 | 3.796 | 3.105 |
| metal / layernorm_1024x4096 | 0.095 | 63.163 | 7.453 | 3.348 | 6.568 | 1.425 | 3.570 | 0.653 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
