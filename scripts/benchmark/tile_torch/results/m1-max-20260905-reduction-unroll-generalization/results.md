# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T02:15:45.413259+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | rmsnorm_1x127 | 1×127×1 / 2 | 0 | 3.755 | 7.149 | 3.909 | 7.338 | 0.53× | 231.709 | 249.000 |
| metal | rmsnorm_17x257 | 1×257×1 / 2 | 0 | 5.379 | 6.104 | 5.421 | 7.789 | 0.88× | 261.000 | 218.292 |
| metal | rmsnorm_128x1024 | 1×1024×1 / 2 | 0 | 6.613 | 8.786 | 6.645 | 8.829 | 0.75× | 233.375 | 237.166 |
| metal | rmsnorm_64x4096 | 1×4096×1 / 2 | 0 | 10.739 | 12.577 | 11.311 | 12.599 | 0.85× | 259.375 | 231.458 |
| metal | layernorm_1x127 | 1×127×1 / 2 | 0 | 4.553 | 7.958 | 4.875 | 8.510 | 0.57× | 234.083 | 223.292 |
| metal | layernorm_17x257 | 1×257×1 / 2 | 0 | 5.373 | 9.329 | 5.621 | 9.385 | 0.58× | 274.750 | 232.917 |
| metal | layernorm_128x1024 | 1×1024×1 / 2 | 0 | 7.513 | 13.970 | 7.546 | 14.044 | 0.54× | 228.167 | 227.208 |
| metal | layernorm_64x4096 | 1×4096×1 / 2 | 0 | 12.852 | 23.315 | 13.028 | 24.554 | 0.55× | 268.291 | 271.667 |
| metal | residual_layernorm_1x127 | 1×127×1 / 2 | 0 | 3.513 | 11.256 | 3.524 | 11.310 | 0.31× | 230.500 | 248.250 |
| metal | residual_layernorm_17x257 | 1×257×1 / 2 | 0 | 3.251 | 11.918 | 3.322 | 12.235 | 0.27× | 249.458 | 243.667 |
| metal | residual_layernorm_128x1024 | 1×1024×1 / 2 | 0 | 5.595 | 19.022 | 5.930 | 19.246 | 0.29× | 242.042 | 286.917 |
| metal | residual_layernorm_64x4096 | 1×4096×1 / 2 | 0 | 9.895 | 27.105 | 9.947 | 27.341 | 0.37× | 278.750 | 261.083 |
| metal | cross_entropy_1x127 | 1×127×1 / 2 | 0 | 3.300 | 107.530 | 3.308 | 108.468 | 0.03× | 235.416 | 426.625 |
| metal | cross_entropy_17x257 | 1×257×1 / 2 | 0 | 3.620 | 110.813 | 3.642 | 112.137 | 0.03× | 230.792 | 431.584 |
| metal | cross_entropy_128x1024 | 1×1024×1 / 2 | 0 | 4.479 | 114.252 | 4.488 | 116.363 | 0.04× | 242.541 | 435.750 |
| metal | cross_entropy_64x4096 | 1×4096×1 / 2 | 0 | 5.931 | 113.245 | 6.014 | 113.763 | 0.05× | 228.083 | 472.500 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / rmsnorm_1x127 | 0.064 | 45.064 | 1.501 | 4.153 | 45.241 | 62.680 | 0.305 | 0.481 |
| metal / rmsnorm_17x257 | 0.064 | 55.322 | 1.705 | 0.809 | 52.362 | 0.516 | 0.338 | 0.315 |
| metal / rmsnorm_128x1024 | 0.067 | 50.453 | 1.613 | 1.052 | 50.128 | 3.672 | 0.404 | 0.319 |
| metal / rmsnorm_64x4096 | 0.062 | 52.414 | 1.752 | 0.868 | 54.972 | 5.844 | 0.459 | 0.317 |
| metal / layernorm_1x127 | 0.070 | 52.659 | 1.580 | 1.103 | 46.915 | 0.591 | 0.241 | 0.327 |
| metal / layernorm_17x257 | 0.080 | 64.273 | 1.376 | 0.521 | 54.469 | 0.252 | 0.405 | 0.334 |
| metal / layernorm_128x1024 | 0.072 | 58.630 | 1.628 | 1.066 | 54.090 | 0.306 | 0.384 | 0.297 |
| metal / layernorm_64x4096 | 0.072 | 58.883 | 1.711 | 0.664 | 61.938 | 0.290 | 0.522 | 0.386 |
| metal / residual_layernorm_1x127 | 0.068 | 49.140 | 1.229 | 0.781 | 45.603 | 0.539 | 0.258 | 0.282 |
| metal / residual_layernorm_17x257 | 0.075 | 52.276 | 1.539 | 0.468 | 47.997 | 0.249 | 0.341 | 0.289 |
| metal / residual_layernorm_128x1024 | 0.070 | 51.204 | 1.578 | 1.117 | 50.285 | 0.363 | 0.380 | 0.370 |
| metal / residual_layernorm_64x4096 | 0.071 | 54.707 | 1.841 | 1.287 | 58.788 | 0.550 | 0.573 | 0.365 |
| metal / cross_entropy_1x127 | 0.086 | 46.357 | 1.484 | 1.030 | 46.640 | 40.779 | 0.268 | 0.506 |
| metal / cross_entropy_17x257 | 0.077 | 60.997 | 1.553 | 0.549 | 51.948 | 7.993 | 0.268 | 0.558 |
| metal / cross_entropy_128x1024 | 0.070 | 50.921 | 1.535 | 1.496 | 50.960 | 6.726 | 0.259 | 0.556 |
| metal / cross_entropy_64x4096 | 0.074 | 50.885 | 1.597 | 1.079 | 57.113 | 6.102 | 0.280 | 0.525 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
