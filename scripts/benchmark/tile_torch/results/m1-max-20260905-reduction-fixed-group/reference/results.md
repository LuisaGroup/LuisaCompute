# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T14:04:24.477584+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_37x769 | 1×769×1 / 2 | 0 | 5.999 | 33.433 | 6.431 | 39.902 | 0.18× | 247.166 | 433.000 |
| metal | softmax_1024x1024 | 1×1024×1 / 2 | 0 | 17.780 | 76.891 | 18.133 | 113.701 | 0.23× | 241.959 | 429.958 |
| metal | softmax_16384x257 | 1×257×1 / 2 | 0 | 49.556 | 198.009 | 50.082 | 207.135 | 0.25× | 390.042 | 527.292 |
| metal | softmax_4096x1024 | 1×1024×1 / 2 | 0 | 53.159 | 120.924 | 54.549 | 123.383 | 0.44× | 328.959 | 501.417 |
| metal | rmsnorm_37x769 | 1×769×1 / 2 | 0 | 6.099 | 7.314 | 6.227 | 7.411 | 0.83× | 226.167 | 282.750 |
| metal | rmsnorm_1024x1024 | 1×1024×1 / 2 | 0 | 20.798 | 24.114 | 22.160 | 25.209 | 0.86× | 301.917 | 370.208 |
| metal | rmsnorm_16384x257 | 1×257×1 / 2 | 0 | 52.697 | 68.576 | 53.474 | 69.571 | 0.77× | 344.916 | 545.959 |
| metal | rmsnorm_4096x1024 | 1×1024×1 / 2 | 0 | 61.229 | 77.819 | 61.836 | 81.036 | 0.79× | 394.542 | 383.667 |
| metal | layernorm_37x769 | 1×769×1 / 2 | 0 | 7.345 | 15.332 | 8.461 | 17.239 | 0.48× | 232.917 | 371.916 |
| metal | layernorm_1024x1024 | 1×1024×1 / 2 | 0 | 20.672 | 47.462 | 21.237 | 50.086 | 0.44× | 384.166 | 318.834 |
| metal | layernorm_16384x257 | 1×257×1 / 2 | 0 | 56.049 | 128.877 | 59.307 | 134.872 | 0.43× | 337.333 | 400.375 |
| metal | layernorm_4096x1024 | 1×1024×1 / 2 | 0 | 58.077 | 153.755 | 61.234 | 155.904 | 0.38× | 354.416 | 440.709 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_37x769 | 0.372 | 249.891 | 1.719 | 5.398 | 88.579 | 61.293 | 0.332 | 0.327 |
| metal / softmax_1024x1024 | 0.122 | 53.773 | 3.053 | 2.445 | 79.999 | 6.421 | 1.188 | 0.425 |
| metal / softmax_16384x257 | 0.069 | 54.374 | 7.760 | 1.451 | 76.004 | 149.520 | 3.335 | 0.982 |
| metal / softmax_4096x1024 | 0.132 | 54.585 | 8.122 | 0.845 | 11.972 | 2.907 | 3.082 | 1.269 |
| metal / rmsnorm_37x769 | 0.071 | 59.678 | 1.477 | 1.099 | 69.686 | 246.891 | 0.396 | 0.340 |
| metal / rmsnorm_1024x1024 | 0.068 | 55.202 | 2.685 | 1.498 | 79.971 | 1.052 | 1.222 | 1.251 |
| metal / rmsnorm_16384x257 | 0.064 | 57.564 | 6.715 | 1.440 | 67.041 | 28.051 | 4.021 | 1.212 |
| metal / rmsnorm_4096x1024 | 0.066 | 52.892 | 7.285 | 1.596 | 5.423 | 0.398 | 3.715 | 1.215 |
| metal / layernorm_37x769 | 0.128 | 70.714 | 1.324 | 1.083 | 82.978 | 1.399 | 0.491 | 0.335 |
| metal / layernorm_1024x1024 | 0.081 | 61.953 | 2.162 | 0.901 | 79.949 | 0.357 | 1.311 | 0.502 |
| metal / layernorm_16384x257 | 0.082 | 65.714 | 6.995 | 1.869 | 70.965 | 25.384 | 4.202 | 1.233 |
| metal / layernorm_4096x1024 | 0.327 | 68.013 | 7.712 | 1.200 | 4.207 | 0.467 | 3.845 | 1.034 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
