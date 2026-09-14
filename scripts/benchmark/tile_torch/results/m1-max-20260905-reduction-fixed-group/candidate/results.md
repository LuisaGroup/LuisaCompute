# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T14:07:15.675314+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_37x769 | 1×769×1 / 2 | 0 | 6.138 | 31.583 | 7.393 | 32.937 | 0.19× | 250.750 | 349.917 |
| metal | softmax_1024x1024 | 1×1024×1 / 2 | 0 | 21.889 | 86.126 | 22.325 | 106.399 | 0.25× | 263.583 | 431.250 |
| metal | softmax_16384x257 | 1×257×1 / 2 | 0 | 53.366 | 200.338 | 57.692 | 209.435 | 0.27× | 376.834 | 629.375 |
| metal | softmax_4096x1024 | 1×1024×1 / 2 | 0 | 65.477 | 116.669 | 66.139 | 128.358 | 0.56× | 346.458 | 499.834 |
| metal | rmsnorm_37x769 | 1×769×1 / 2 | 0 | 5.502 | 8.593 | 5.503 | 9.566 | 0.64× | 313.167 | 270.958 |
| metal | rmsnorm_1024x1024 | 1×1024×1 / 2 | 0 | 24.499 | 25.055 | 29.681 | 25.520 | 0.98× | 283.125 | 307.375 |
| metal | rmsnorm_16384x257 | 1×257×1 / 2 | 0 | 55.528 | 72.855 | 55.854 | 73.346 | 0.76× | 404.209 | 392.208 |
| metal | rmsnorm_4096x1024 | 1×1024×1 / 2 | 0 | 64.305 | 79.469 | 68.611 | 84.358 | 0.81× | 1175.709 | 466.542 |
| metal | layernorm_37x769 | 1×769×1 / 2 | 0 | 5.834 | 14.784 | 6.296 | 16.424 | 0.39× | 283.375 | 350.542 |
| metal | layernorm_1024x1024 | 1×1024×1 / 2 | 0 | 22.283 | 50.681 | 22.367 | 54.156 | 0.44× | 263.416 | 286.208 |
| metal | layernorm_16384x257 | 1×257×1 / 2 | 0 | 57.383 | 129.605 | 57.392 | 133.506 | 0.44× | 543.458 | 409.209 |
| metal | layernorm_4096x1024 | 1×1024×1 / 2 | 0 | 59.184 | 154.410 | 61.207 | 159.972 | 0.38× | 339.542 | 478.875 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_37x769 | 0.078 | 61.791 | 1.105 | 4.602 | 76.307 | 63.350 | 0.320 | 0.290 |
| metal / softmax_1024x1024 | 0.080 | 54.430 | 1.915 | 1.683 | 72.523 | 6.733 | 1.031 | 0.409 |
| metal / softmax_16384x257 | 0.069 | 57.953 | 6.025 | 1.717 | 64.314 | 52.765 | 2.904 | 0.824 |
| metal / softmax_4096x1024 | 0.065 | 57.611 | 6.319 | 0.837 | 6.808 | 5.665 | 3.733 | 1.564 |
| metal / rmsnorm_37x769 | 0.072 | 60.623 | 2.771 | 0.987 | 69.056 | 71.396 | 0.367 | 0.532 |
| metal / rmsnorm_1024x1024 | 0.069 | 57.761 | 2.603 | 1.076 | 100.675 | 0.993 | 2.769 | 0.533 |
| metal / rmsnorm_16384x257 | 0.065 | 84.421 | 6.952 | 26.585 | 79.991 | 0.452 | 3.819 | 0.843 |
| metal / rmsnorm_4096x1024 | 0.081 | 74.118 | 6.477 | 1.568 | 7.557 | 0.432 | 5.422 | 0.962 |
| metal / layernorm_37x769 | 0.077 | 82.147 | 1.789 | 1.099 | 80.769 | 0.638 | 0.338 | 0.333 |
| metal / layernorm_1024x1024 | 0.160 | 75.351 | 2.823 | 0.969 | 86.358 | 0.462 | 1.747 | 0.556 |
| metal / layernorm_16384x257 | 0.079 | 81.256 | 7.023 | 30.942 | 90.998 | 0.543 | 4.102 | 0.966 |
| metal / layernorm_4096x1024 | 0.077 | 71.103 | 6.953 | 3.165 | 5.646 | 0.505 | 5.002 | 0.958 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
