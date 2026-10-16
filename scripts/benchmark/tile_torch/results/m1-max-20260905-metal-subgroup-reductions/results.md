# TileIR/TVMx vs PyTorch

Generated: 2026-09-04T23:14:31.938143+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` uses the reference worker mapping.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; `functional.rms_norm` has no `out=` overload and its returned-output allocation remains inside warm timing. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared FP32 exp materialization; the DSL and execution hierarchy remain target-independent. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sum_1x127 | 1×127×1 / 2 | 0 | 3.268 | 7.211 | 3.282 | 7.236 | 0.45× | 197.667 | 246.208 |
| metal | sum_17x257 | 1×257×1 / 2 | 0 | 3.106 | 4.340 | 3.120 | 4.347 | 0.72× | 220.458 | 226.666 |
| metal | sum_128x1024 | 1×1024×1 / 2 | 0 | 3.387 | 5.604 | 3.416 | 5.623 | 0.60× | 226.250 | 237.875 |
| metal | sum_64x4096 | 1×4096×1 / 2 | 0 | 4.721 | 16.119 | 4.759 | 16.169 | 0.29× | 227.042 | 282.667 |
| metal | softmax_1x127 | 1×127×1 / 2 | 0 | 3.578 | 26.111 | 3.585 | 26.824 | 0.14× | 226.875 | 328.709 |
| metal | softmax_17x257 | 1×257×1 / 2 | 0 | 3.305 | 26.594 | 3.319 | 27.452 | 0.12× | 228.042 | 340.584 |
| metal | softmax_128x1024 | 1×1024×1 / 2 | 0 | 5.385 | 30.376 | 5.415 | 30.819 | 0.18× | 244.042 | 302.459 |
| metal | softmax_64x4096 | 1×4096×1 / 2 | 0 | 8.881 | 31.029 | 8.919 | 31.485 | 0.29× | 240.750 | 301.875 |
| metal | rmsnorm_1x127 | 1×127×1 / 2 | 0 | 3.904 | 7.155 | 4.040 | 7.291 | 0.55× | 232.208 | 235.125 |
| metal | rmsnorm_17x257 | 1×257×1 / 2 | 0 | 5.335 | 6.154 | 5.395 | 6.164 | 0.87× | 231.875 | 249.125 |
| metal | rmsnorm_128x1024 | 1×1024×1 / 2 | 0 | 6.673 | 8.707 | 6.724 | 8.779 | 0.77× | 255.292 | 249.083 |
| metal | rmsnorm_64x4096 | 1×4096×1 / 2 | 0 | 11.177 | 12.392 | 11.193 | 12.577 | 0.90× | 256.375 | 249.209 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sum_1x127 | 0.083 | 32.297 | 1.156 | 3.839 | 0.595 | 59.311 | 0.255 | 0.348 |
| metal / sum_17x257 | 0.057 | 36.334 | 1.241 | 0.335 | 0.686 | 0.602 | 0.307 | 0.307 |
| metal / sum_128x1024 | 0.053 | 33.322 | 1.297 | 0.791 | 1.696 | 0.915 | 0.341 | 0.342 |
| metal / sum_64x4096 | 0.055 | 35.370 | 1.504 | 0.565 | 2.957 | 3.126 | 0.337 | 0.330 |
| metal / softmax_1x127 | 0.069 | 38.660 | 1.297 | 1.118 | 0.638 | 40.486 | 0.306 | 0.352 |
| metal / softmax_17x257 | 0.068 | 41.223 | 1.215 | 0.367 | 0.753 | 3.691 | 0.385 | 0.358 |
| metal / softmax_128x1024 | 0.064 | 38.491 | 0.966 | 0.929 | 1.698 | 3.750 | 0.438 | 0.361 |
| metal / softmax_64x4096 | 0.071 | 38.571 | 1.057 | 0.700 | 3.448 | 2.133 | 0.497 | 0.354 |
| metal / rmsnorm_1x127 | 0.068 | 40.153 | 1.544 | 1.354 | 0.640 | 1.829 | 0.319 | 0.336 |
| metal / rmsnorm_17x257 | 0.067 | 43.923 | 1.605 | 0.558 | 0.693 | 0.227 | 0.366 | 0.322 |
| metal / rmsnorm_128x1024 | 0.066 | 41.296 | 1.597 | 1.309 | 1.764 | 2.421 | 0.409 | 0.356 |
| metal / rmsnorm_64x4096 | 0.084 | 41.065 | 1.679 | 1.055 | 2.999 | 5.323 | 0.521 | 0.397 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
