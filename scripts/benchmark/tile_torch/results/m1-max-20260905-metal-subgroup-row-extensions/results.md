# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T23:48:22.159442+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared FP32 exp materialization; the DSL and execution hierarchy remain target-independent. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | layernorm_1x127 | 1×127×1 / 2 | 0 | 4.500 | 8.400 | 4.582 | 8.446 | 0.54× | 212.417 | 248.375 |
| metal | layernorm_17x257 | 1×257×1 / 2 | 0 | 5.714 | 8.821 | 5.785 | 8.978 | 0.65× | 236.125 | 242.292 |
| metal | layernorm_128x1024 | 1×1024×1 / 2 | 0 | 7.542 | 13.726 | 7.562 | 13.929 | 0.55× | 251.417 | 273.792 |
| metal | layernorm_64x4096 | 1×4096×1 / 2 | 0 | 12.413 | 24.313 | 12.482 | 25.310 | 0.51× | 272.167 | 283.708 |
| metal | cross_entropy_1x127 | 1×127×1 / 2 | 0 | 4.513 | 107.246 | 4.583 | 108.702 | 0.04× | 227.917 | 438.125 |
| metal | cross_entropy_17x257 | 1×257×1 / 2 | 0 | 3.449 | 107.695 | 3.481 | 109.682 | 0.03× | 229.666 | 445.167 |
| metal | cross_entropy_128x1024 | 1×1024×1 / 2 | 0 | 4.290 | 110.171 | 4.469 | 112.051 | 0.04× | 235.875 | 452.125 |
| metal | cross_entropy_64x4096 | 1×4096×1 / 2 | 0 | 5.838 | 112.263 | 5.859 | 115.286 | 0.05× | 232.084 | 443.875 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / layernorm_1x127 | 0.070 | 46.296 | 1.463 | 3.896 | 0.656 | 58.966 | 0.299 | 0.367 |
| metal / layernorm_17x257 | 0.074 | 51.295 | 1.555 | 0.612 | 0.744 | 0.704 | 0.359 | 0.340 |
| metal / layernorm_128x1024 | 0.072 | 48.431 | 1.393 | 1.158 | 1.829 | 2.545 | 0.417 | 0.398 |
| metal / layernorm_64x4096 | 0.077 | 48.605 | 1.666 | 1.436 | 4.186 | 6.461 | 0.617 | 0.391 |
| metal / cross_entropy_1x127 | 0.075 | 45.095 | 1.734 | 1.033 | 0.656 | 46.151 | 0.309 | 0.552 |
| metal / cross_entropy_17x257 | 0.073 | 51.230 | 1.498 | 0.604 | 0.822 | 8.440 | 0.279 | 0.553 |
| metal / cross_entropy_128x1024 | 0.084 | 45.960 | 1.619 | 1.038 | 1.798 | 9.541 | 0.340 | 0.555 |
| metal / cross_entropy_64x4096 | 0.075 | 46.251 | 1.376 | 0.983 | 4.791 | 5.998 | 0.301 | 0.538 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
