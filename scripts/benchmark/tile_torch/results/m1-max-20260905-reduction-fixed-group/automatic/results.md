# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T14:47:22.057524+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_37x769 | 1×769×1 / 2 | 0 | 5.875 | 57.720 | 6.576 | 59.331 | 0.10× | 300.416 | 396.708 |
| metal | softmax_1024x1024 | 1×1024×1 / 2 | 0 | 27.492 | 66.830 | 32.772 | 71.927 | 0.41× | 306.458 | 551.917 |
| metal | softmax_16384x257 | 1×257×1 / 2 | 0 | 93.784 | 234.538 | 94.263 | 272.301 | 0.40× | 658.750 | 585.541 |
| metal | softmax_4096x1024 | 1×1024×1 / 2 | 0 | 74.113 | 136.854 | 74.701 | 165.357 | 0.54× | 332.500 | 1359.792 |
| metal | rmsnorm_37x769 | 1×769×1 / 2 | 0 | 5.659 | 11.004 | 5.683 | 13.435 | 0.51× | 224.958 | 223.167 |
| metal | rmsnorm_1024x1024 | 1×1024×1 / 2 | 0 | 25.939 | 32.789 | 28.641 | 39.597 | 0.79× | 500.625 | 284.792 |
| metal | rmsnorm_16384x257 | 1×257×1 / 2 | 0 | 64.278 | 84.594 | 77.716 | 97.646 | 0.76× | 333.291 | 372.834 |
| metal | rmsnorm_4096x1024 | 1×1024×1 / 2 | 0 | 67.723 | 106.458 | 74.101 | 107.141 | 0.64× | 371.292 | 435.709 |
| metal | layernorm_37x769 | 1×769×1 / 2 | 0 | 5.412 | 10.453 | 5.855 | 11.619 | 0.52× | 228.833 | 331.875 |
| metal | layernorm_1024x1024 | 1×1024×1 / 2 | 0 | 26.233 | 61.236 | 26.367 | 61.749 | 0.43× | 296.583 | 397.834 |
| metal | layernorm_16384x257 | 1×257×1 / 2 | 0 | 81.025 | 143.505 | 95.189 | 154.091 | 0.56× | 325.208 | 433.709 |
| metal | layernorm_4096x1024 | 1×1024×1 / 2 | 0 | 72.999 | 213.232 | 75.065 | 214.159 | 0.34× | 425.917 | 460.792 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_37x769 | 0.193 | 75.008 | 2.390 | 6.630 | 68.171 | 65.552 | 0.408 | 0.349 |
| metal / softmax_1024x1024 | 0.074 | 54.827 | 2.720 | 1.700 | 63.089 | 4.055 | 1.229 | 0.489 |
| metal / softmax_16384x257 | 0.067 | 56.647 | 6.359 | 1.224 | 64.228 | 52.111 | 3.227 | 0.760 |
| metal / softmax_4096x1024 | 0.067 | 48.999 | 5.705 | 1.284 | 7.596 | 3.321 | 3.827 | 1.033 |
| metal / rmsnorm_37x769 | 0.076 | 57.272 | 1.767 | 0.983 | 59.505 | 271.166 | 0.331 | 0.411 |
| metal / rmsnorm_1024x1024 | 0.078 | 54.447 | 3.509 | 1.403 | 65.994 | 1.191 | 1.000 | 0.553 |
| metal / rmsnorm_16384x257 | 0.065 | 57.918 | 7.920 | 2.500 | 4.749 | 18.814 | 3.947 | 2.365 |
| metal / rmsnorm_4096x1024 | 0.068 | 58.348 | 8.293 | 1.476 | 6.012 | 0.382 | 3.696 | 0.672 |
| metal / layernorm_37x769 | 0.086 | 64.756 | 1.391 | 0.757 | 61.654 | 1.244 | 0.317 | 0.742 |
| metal / layernorm_1024x1024 | 0.073 | 61.095 | 3.326 | 1.852 | 62.195 | 0.335 | 2.600 | 0.398 |
| metal / layernorm_16384x257 | 0.080 | 63.438 | 7.737 | 2.019 | 63.421 | 18.885 | 3.981 | 0.692 |
| metal / layernorm_4096x1024 | 0.087 | 62.164 | 18.534 | 2.462 | 8.603 | 3.352 | 2.770 | 0.705 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
