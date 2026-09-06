# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T05:16:07.912560+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_37x1537 | 1×1537×1 / 2 | 0 | 7.678 | 52.946 | 9.857 | 56.860 | 0.15× | 352.750 | 511.834 |
| metal | softmax_1024x4096 | 1×4096×1 / 2 | 0 | 64.279 | 191.727 | 75.968 | 202.588 | 0.34× | 478.000 | 559.667 |
| metal | rmsnorm_37x1537 | 1×1537×1 / 2 | 0 | 7.288 | 9.415 | 9.085 | 10.080 | 0.77× | 222.625 | 240.167 |
| metal | rmsnorm_1024x4096 | 1×4096×1 / 2 | 0 | 79.088 | 96.380 | 83.721 | 117.142 | 0.82× | 345.875 | 328.875 |
| metal | layernorm_37x1537 | 1×1537×1 / 2 | 0 | 6.954 | 14.862 | 7.184 | 14.910 | 0.47× | 253.292 | 299.375 |
| metal | layernorm_1024x4096 | 1×4096×1 / 2 | 0 | 83.613 | 283.175 | 93.509 | 286.144 | 0.30× | 353.666 | 460.958 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_37x1537 | 0.078 | 30.961 | 1.769 | 4.067 | 43.616 | 61.789 | 0.394 | 0.385 |
| metal / softmax_1024x4096 | 0.075 | 33.202 | 5.376 | 43.280 | 53.081 | 3.150 | 2.880 | 1.134 |
| metal / rmsnorm_37x1537 | 0.066 | 33.402 | 2.763 | 1.082 | 45.791 | 64.020 | 0.359 | 2.009 |
| metal / rmsnorm_1024x4096 | 0.062 | 33.572 | 5.745 | 1.452 | 51.428 | 12.400 | 4.618 | 3.091 |
| metal / layernorm_37x1537 | 0.082 | 37.214 | 1.408 | 0.679 | 44.833 | 2.021 | 0.363 | 0.370 |
| metal / layernorm_1024x4096 | 0.072 | 36.961 | 6.415 | 1.333 | 52.122 | 0.498 | 3.741 | 0.791 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
