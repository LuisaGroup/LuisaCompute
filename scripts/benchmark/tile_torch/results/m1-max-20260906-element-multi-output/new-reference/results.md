# TileIR/TVMx vs PyTorch

Generated: 2026-09-06T05:09:27.630008+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | sigmoid_pair_1x127 | 1×256×1 / 2 | 0 | 117.616 | 22.418 | 118.928 | 24.030 | 5.25× | 750.833 | 401.333 |
| metal | sigmoid_pair_37x1537 | 1×256×1 / 2 | 0 | 266.014 | 25.597 | 266.673 | 26.292 | 10.39× | 510.459 | 369.833 |
| metal | sigmoid_pair_1024x4096 | 1×256×1 / 2 | 0 | 396.370 | 298.194 | 430.109 | 350.136 | 1.33× | 696.000 | 568.166 |
| metal | sigmoid_pair_4096x4096 | 1×256×1 / 2 | 0 | 1751.493 | 1991.567 | 1941.737 | 2117.960 | 0.88× | 1959.792 | 2169.209 |
| metal | gelu_pair_1x127 | 1×256×1 / 2 | 0 | 128.173 | 12.880 | 128.652 | 13.670 | 9.95× | 387.208 | 221.875 |
| metal | gelu_pair_37x1537 | 1×256×1 / 2 | 0 | 293.964 | 26.240 | 294.072 | 33.986 | 11.20× | 951.208 | 346.666 |
| metal | gelu_pair_1024x4096 | 1×256×1 / 2 | 0 | 533.160 | 233.226 | 541.263 | 236.489 | 2.29× | 657.708 | 523.708 |
| metal | gelu_pair_4096x4096 | 1×256×1 / 2 | 0 | 2217.042 | 1310.184 | 2252.000 | 1345.065 | 1.69× | 2552.541 | 1302.833 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / sigmoid_pair_1x127 | 0.052 | 30.573 | 1.909 | 5.473 | 13.229 | 70.430 | 0.625 | 0.530 |
| metal / sigmoid_pair_37x1537 | 0.061 | 33.574 | 1.775 | 0.790 | 11.239 | 5.023 | 0.821 | 0.359 |
| metal / sigmoid_pair_1024x4096 | 0.094 | 32.150 | 6.714 | 41.601 | 82.293 | 12.420 | 10.307 | 1.035 |
| metal / sigmoid_pair_4096x4096 | 0.054 | 31.578 | 20.439 | 3.994 | 20.988 | 4.566 | 32.998 | 6.845 |
| metal / gelu_pair_1x127 | 0.068 | 29.147 | 1.128 | 9.459 | 9.343 | 0.555 | 0.523 | 0.284 |
| metal / gelu_pair_37x1537 | 0.065 | 32.154 | 1.283 | 0.594 | 11.473 | 1.540 | 0.658 | 0.289 |
| metal / gelu_pair_1024x4096 | 0.062 | 30.157 | 5.869 | 17.540 | 82.791 | 0.853 | 10.187 | 1.287 |
| metal / gelu_pair_4096x4096 | 0.060 | 30.128 | 21.088 | 7.534 | 22.547 | 1.501 | 41.642 | 3.643 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| sigmoid_pair_1x127 / native | 99.792 | 101.875 | 117.616 | 750.833 | 1.005× |
| sigmoid_pair_1x127 / torch | 12.193 | 46.417 | 22.418 | 401.333 | 0.788× |
| sigmoid_pair_37x1537 / native | 256.262 | 329.875 | 266.014 | 510.459 | 1.017× |
| sigmoid_pair_37x1537 / torch | 11.076 | 14.542 | 25.597 | 369.833 | 0.919× |
| sigmoid_pair_1024x4096 / native | 371.226 | 336.375 | 396.370 | 696.000 | 0.909× |
| sigmoid_pair_1024x4096 / torch | 289.510 | 250.625 | 298.194 | 568.166 | 0.994× |
| sigmoid_pair_4096x4096 / native | 1518.090 | 1763.917 | 1751.493 | 1959.792 | 1.044× |
| sigmoid_pair_4096x4096 / torch | 1356.017 | 1380.333 | 1991.567 | 2169.209 | 1.286× |
| gelu_pair_1x127 / native | 125.318 | 116.417 | 128.173 | 387.208 | 1.008× |
| gelu_pair_1x127 / torch | 4.419 | 113.625 | 12.880 | 221.875 | 0.999× |
| gelu_pair_37x1537 / native | 286.976 | 258.500 | 293.964 | 951.208 | 1.008× |
| gelu_pair_37x1537 / torch | 17.969 | 15.500 | 26.240 | 346.666 | 0.507× |
| gelu_pair_1024x4096 / native | 456.141 | 466.583 | 533.160 | 657.708 | 0.827× |
| gelu_pair_1024x4096 / torch | 170.218 | 630.500 | 233.226 | 523.708 | 1.068× |
| gelu_pair_4096x4096 / native | 1975.583 | 1860.083 | 2217.042 | 2552.541 | 1.060× |
| gelu_pair_4096x4096 / torch | 992.958 | 1116.000 | 1310.184 | 1302.833 | 1.032× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| sigmoid_pair_1x127 | 114.249 | 9.611 | 101.959 | 113.542 | 750.833 | 401.333 |
| sigmoid_pair_37x1537 | 259.004 | 10.329 | 247.417 | 13.750 | 510.459 | 369.833 |
| sigmoid_pair_1024x4096 | 343.101 | 287.869 | 428.292 | 635.208 | 696.000 | 568.166 |
| sigmoid_pair_4096x4096 | 1589.278 | 1768.175 | 1405.000 | 2012.959 | 1959.792 | 2169.209 |
| gelu_pair_1x127 | 126.046 | 5.068 | 114.833 | 7.625 | 387.208 | 221.875 |
| gelu_pair_37x1537 | 282.257 | 9.510 | 262.291 | 63.417 | 951.208 | 346.666 |
| gelu_pair_1024x4096 | 390.586 | 218.350 | 435.667 | 199.458 | 657.708 | 523.708 |
| gelu_pair_4096x4096 | 1950.573 | 1024.423 | 1822.042 | 1209.500 | 2552.541 | 1302.833 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
