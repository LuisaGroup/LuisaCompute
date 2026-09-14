# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T12:59:11.114246+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_37x1537 | 1×1537×1 / 2 | 0 | 7.460 | 113.854 | 8.478 | 129.588 | 0.07× | 293.958 | 3302.625 |
| metal | softmax_256x3072 | 1×3072×1 / 2 | 0 | 20.845 | 45.663 | 21.636 | 51.651 | 0.46× | 235.209 | 336.833 |
| metal | softmax_768x6144 | 1×6144×1 / 2 | 0 | 97.103 | 185.370 | 104.244 | 204.795 | 0.52× | 412.208 | 825.583 |
| metal | softmax_1024x4096 | 1×4096×1 / 2 | 0 | 86.868 | 175.136 | 86.885 | 192.913 | 0.50× | 270.167 | 454.667 |
| metal | rmsnorm_37x1537 | 1×1537×1 / 2 | 0 | 6.741 | 14.993 | 6.880 | 15.090 | 0.45× | 247.250 | 246.666 |
| metal | rmsnorm_256x3072 | 1×3072×1 / 2 | 0 | 20.595 | 39.975 | 25.442 | 41.698 | 0.52× | 287.375 | 254.333 |
| metal | rmsnorm_768x6144 | 1×6144×1 / 2 | 0 | 105.798 | 127.717 | 106.607 | 128.474 | 0.83× | 289.208 | 334.458 |
| metal | rmsnorm_1024x4096 | 1×4096×1 / 2 | 0 | 91.767 | 95.095 | 99.638 | 111.187 | 0.97× | 298.459 | 478.292 |
| metal | layernorm_37x1537 | 1×1537×1 / 2 | 0 | 6.768 | 14.327 | 6.798 | 15.536 | 0.47× | 273.333 | 368.916 |
| metal | layernorm_256x3072 | 1×3072×1 / 2 | 0 | 22.962 | 65.803 | 23.086 | 66.688 | 0.35× | 436.125 | 492.125 |
| metal | layernorm_768x6144 | 1×6144×1 / 2 | 0 | 108.284 | 403.727 | 117.096 | 408.109 | 0.27× | 424.833 | 853.542 |
| metal | layernorm_1024x4096 | 1×4096×1 / 2 | 0 | 119.145 | 305.519 | 120.974 | 310.655 | 0.39× | 467.000 | 538.750 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 0.096 | 64.502 | 1.470 | 3.942 | 62.522 | 52.137 | 0.742 | 0.458 |
| metal / softmax_256x3072 | 0.070 | 53.175 | 3.111 | 3.077 | 69.559 | 8.232 | 2.807 | 0.403 |
| metal / softmax_768x6144 | 0.083 | 54.705 | 6.836 | 1.241 | 74.474 | 43.210 | 4.459 | 0.738 |
| metal / softmax_1024x4096 | 0.082 | 54.324 | 5.710 | 0.924 | 72.657 | 5.817 | 3.863 | 0.747 |
| metal / rmsnorm_37x1537 | 0.089 | 64.180 | 1.884 | 2.892 | 61.922 | 68.845 | 0.306 | 1.140 |
| metal / rmsnorm_256x3072 | 0.073 | 56.619 | 2.211 | 0.809 | 65.340 | 4.710 | 0.916 | 0.396 |
| metal / rmsnorm_768x6144 | 0.062 | 54.863 | 7.512 | 19.647 | 73.592 | 6.361 | 4.229 | 0.686 |
| metal / rmsnorm_1024x4096 | 0.068 | 56.583 | 5.663 | 1.345 | 81.635 | 3.389 | 3.204 | 0.837 |
| metal / layernorm_37x1537 | 0.075 | 73.094 | 1.411 | 0.972 | 66.569 | 0.635 | 0.606 | 2.082 |
| metal / layernorm_256x3072 | 0.085 | 66.368 | 2.201 | 2.409 | 75.028 | 0.324 | 0.958 | 0.415 |
| metal / layernorm_768x6144 | 0.083 | 65.048 | 7.321 | 17.380 | 81.437 | 0.799 | 3.831 | 0.954 |
| metal / layernorm_1024x4096 | 0.069 | 63.425 | 5.967 | 3.463 | 78.815 | 0.693 | 3.201 | 0.923 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
