# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T01:58:49.028280+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sum_1x4096 | 1×4096×1 / 2 | 0 | 3.266 | 8.028 | 3.299 | 8.071 | 0.41× | 217.292 | 229.125 |
| metal | sum_64x4096 | 1×4096×1 / 2 | 0 | 4.064 | 16.354 | 4.436 | 16.497 | 0.25× | 225.250 | 270.666 |
| metal | sum_1024x4096 | 1×4096×1 / 2 | 0 | 26.178 | 28.885 | 26.279 | 29.013 | 0.91× | 260.209 | 272.125 |
| metal | sum_17x257 | 1×257×1 / 2 | 0 | 3.028 | 4.471 | 3.103 | 4.479 | 0.68× | 232.625 | 220.250 |
| metal | sum_1024x257 | 1×257×1 / 2 | 0 | 4.486 | 6.179 | 4.558 | 6.226 | 0.73× | 229.541 | 229.041 |
| metal | softmax_1x4096 | 1×4096×1 / 2 | 0 | 3.616 | 28.264 | 3.653 | 30.156 | 0.13× | 224.666 | 341.541 |
| metal | softmax_64x4096 | 1×4096×1 / 2 | 0 | 9.076 | 32.912 | 9.139 | 35.643 | 0.28× | 234.375 | 310.750 |
| metal | softmax_1024x4096 | 1×4096×1 / 2 | 0 | 67.024 | 128.430 | 67.927 | 128.795 | 0.52× | 301.791 | 422.083 |
| metal | softmax_17x257 | 1×257×1 / 2 | 0 | 3.576 | 30.061 | 4.217 | 30.497 | 0.12× | 244.167 | 354.792 |
| metal | softmax_1024x257 | 1×257×1 / 2 | 0 | 7.536 | 33.636 | 8.521 | 33.814 | 0.22× | 269.000 | 325.125 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sum_1x4096 | 0.058 | 36.736 | 1.185 | 4.015 | 36.536 | 62.193 | 0.353 | 0.398 |
| metal / sum_64x4096 | 0.051 | 37.639 | 1.450 | 0.858 | 2.917 | 3.327 | 0.285 | 0.306 |
| metal / sum_1024x4096 | 0.056 | 37.875 | 5.491 | 36.641 | 3.589 | 0.395 | 0.296 | 0.253 |
| metal / sum_17x257 | 0.059 | 41.429 | 0.926 | 0.289 | 0.704 | 0.224 | 0.275 | 0.228 |
| metal / sum_1024x257 | 0.055 | 40.572 | 1.400 | 1.279 | 48.664 | 0.289 | 0.270 | 0.256 |
| metal / softmax_1x4096 | 0.063 | 42.852 | 1.232 | 0.296 | 0.737 | 36.346 | 0.310 | 0.327 |
| metal / softmax_64x4096 | 0.068 | 44.961 | 1.387 | 0.577 | 3.515 | 3.849 | 0.492 | 0.359 |
| metal / softmax_1024x4096 | 0.067 | 45.010 | 5.077 | 13.988 | 4.209 | 2.917 | 3.132 | 0.564 |
| metal / softmax_17x257 | 0.069 | 48.092 | 1.348 | 1.303 | 0.687 | 3.401 | 0.318 | 0.289 |
| metal / softmax_1024x257 | 0.064 | 45.160 | 1.406 | 0.439 | 52.478 | 2.231 | 0.493 | 0.339 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
