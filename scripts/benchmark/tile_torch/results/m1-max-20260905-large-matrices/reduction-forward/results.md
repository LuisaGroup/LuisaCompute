# TileIR/TVMx vs PyTorch

Generated: 2026-09-05T16:15:42.579502+00:00

Hardware: Apple M1 Max; macOS-26.6.2-arm64-arm-64bit-Mach-O. PyTorch 2.14.0; FP32; 8 CPU threads.

Native root execution request: `auto`. Explicit scopes fail on unsupported targets; `auto` admits proved target mapping families and otherwise retains the reference worker mapping. Inspect each row's execution plans for actual realization.

Native TIRx vectorization: `True`; experimental automatic CPU packing: `False`. Automatic packing is opt-in and preserves inner serial/reduction order. Disabling TIRx vectorization does not disable LLVM's own optimizations.

Both sides use device-resident inputs. Native outputs are preallocated. PyTorch uses preallocated `out=` storage where its operator exposes it; the functional RMSNorm, LayerNorm, residual LayerNorm, and cross-entropy calls used here return new outputs, so their allocation remains inside warm timing. Every row records its output policy. Warm timings include host dispatch/binding overhead, exclude transfers and compilation, and are NOT GPU hardware-event times. PyTorch is eager (no torch.compile).

Native GEMM retains an MMA in TileIR. CPU matrix realization: `reference`. CBLAS is selected only from a proved whole-kernel contract and is visible as one provider call in generated LLVM; reference keeps contraction loops. CPU array math: `reference`. Accelerate consumes only proved FP32 add/max/min recurrences and a versioned compiler-owned shared pure-Tile materialization whose expression is revalidated as exp; the DSL and execution hierarchy remain target-independent. Shared-Tile lowering policy: `preserve`. Cooperative-matrix capability requested: `False`. Eligible Metal group MMA can use native FP32 SIMD-group matrices. Base pipeline window: `2`; tuned choices appear per row. Window 1 retains ordered execution, 2 permits safe software prefetching. Neither mode claims hardware-asynchronous transfers. Sort is not included in this performance comparison.

Ratio = native / PyTorch; greater than 1 means native is slower. P50 is per-call batched throughput; latency columns synchronize each individual call. All values are microseconds.

| Device | Operator / M×N[×K] | Block / window | Matrix calls | Native p50 | Torch p50 | Native p90 | Torch p90 | Ratio | Native latency | Torch latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal | softmax_37x8191 | 1×8191×1 / 2 | 0 | 12.311 | 34.058 | 12.346 | 43.672 | 0.36× | 269.584 | 377.458 |
| metal | softmax_1024x8192 | 1×8192×1 / 2 | 0 | 160.418 | 272.847 | 162.463 | 278.870 | 0.59× | 413.292 | 558.833 |
| metal | softmax_1024x16384 | 1×16384×1 / 2 | 0 | 404.618 | 598.853 | 409.680 | 617.478 | 0.68× | 616.500 | 917.458 |
| metal | softmax_4096x8192 | 1×8192×1 / 2 | 0 | 820.560 | 1062.054 | 823.801 | 1082.379 | 0.77× | 1037.625 | 1287.459 |
| metal | softmax_8192x4096 | 1×4096×1 / 2 | 0 | 803.530 | 936.328 | 818.071 | 952.783 | 0.86× | 1031.291 | 1140.000 |
| metal | softmax_16384x4096 | 1×4096×1 / 2 | 0 | 1611.852 | 1831.086 | 1632.417 | 1838.352 | 0.88× | 1900.875 | 2062.333 |
| metal | rmsnorm_37x8191 | 1×8191×1 / 2 | 0 | 12.010 | 17.080 | 12.281 | 17.887 | 0.70× | 248.334 | 240.125 |
| metal | rmsnorm_1024x8192 | 1×8192×1 / 2 | 0 | 158.913 | 168.636 | 160.985 | 171.023 | 0.94× | 432.875 | 404.250 |
| metal | rmsnorm_1024x16384 | 1×16384×1 / 2 | 0 | 412.848 | 424.295 | 415.889 | 432.790 | 0.97× | 1045.833 | 719.083 |
| metal | rmsnorm_4096x8192 | 1×8192×1 / 2 | 0 | 806.751 | 824.130 | 816.440 | 843.497 | 0.98× | 1057.542 | 1093.584 |
| metal | rmsnorm_8192x4096 | 1×4096×1 / 2 | 0 | 806.989 | 817.479 | 826.358 | 840.033 | 0.99× | 1054.250 | 1022.500 |
| metal | rmsnorm_16384x4096 | 1×4096×1 / 2 | 0 | 1615.365 | 1666.765 | 1647.676 | 1682.197 | 0.97× | 1796.042 | 2069.042 |
| metal | layernorm_37x8191 | 1×8191×1 / 2 | 0 | 13.670 | 40.731 | 13.847 | 41.008 | 0.34× | 245.625 | 278.458 |
| metal | layernorm_1024x8192 | 1×8192×1 / 2 | 0 | 164.273 | 522.811 | 167.244 | 529.106 | 0.31× | 446.041 | 813.375 |
| metal | layernorm_1024x16384 | 1×16384×1 / 2 | 0 | 427.665 | 1112.396 | 431.040 | 1130.177 | 0.38× | 838.167 | 1372.708 |
| metal | layernorm_4096x8192 | 1×8192×1 / 2 | 0 | 861.168 | 2128.927 | 871.544 | 2146.808 | 0.40× | 1127.958 | 2578.250 |
| metal | layernorm_8192x4096 | 1×4096×1 / 2 | 0 | 819.095 | 1637.142 | 840.931 | 1646.274 | 0.50× | 1140.917 | 1850.250 |
| metal | layernorm_16384x4096 | 1×4096×1 / 2 | 0 | 1643.012 | 3215.986 | 1651.296 | 3300.313 | 0.51× | 1811.792 | 3918.084 |

## Setup and cold-call phases

Times below are milliseconds. Native compile includes the bridge/compiler call; lazy device compilation can also occur on first invocation. These are process-cold calls, not a guarantee that OS/driver disk caches are cold.

| Device / case | Capture | Native compile | Native alloc/upload | Torch alloc/upload | Native first call | Torch first call | Native download | Torch download |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metal / softmax_37x8191 | 0.070 | 83.543 | 1.389 | 5.698 | 86.997 | 53.886 | 0.454 | 0.466 |
| metal / softmax_1024x8192 | 0.062 | 51.391 | 9.712 | 2.075 | 11.541 | 42.096 | 5.884 | 1.101 |
| metal / softmax_1024x16384 | 0.068 | 57.225 | 19.459 | 16.326 | 78.210 | 8.419 | 16.559 | 1.579 |
| metal / softmax_4096x8192 | 0.093 | 54.012 | 36.244 | 5.864 | 14.902 | 4.882 | 29.643 | 3.728 |
| metal / softmax_8192x4096 | 0.091 | 56.397 | 36.776 | 17.835 | 75.228 | 9.004 | 27.181 | 3.766 |
| metal / softmax_16384x4096 | 0.076 | 57.069 | 112.873 | 17.068 | 25.368 | 8.781 | 44.634 | 5.702 |
| metal / rmsnorm_37x8191 | 0.313 | 59.224 | 1.740 | 2.116 | 76.853 | 70.408 | 0.486 | 0.328 |
| metal / rmsnorm_1024x8192 | 0.064 | 54.860 | 10.504 | 23.509 | 6.705 | 7.864 | 7.106 | 1.121 |
| metal / rmsnorm_1024x16384 | 0.070 | 57.093 | 21.992 | 17.434 | 88.108 | 5.927 | 14.099 | 1.777 |
| metal / rmsnorm_4096x8192 | 0.069 | 55.540 | 37.688 | 4.777 | 12.934 | 1.242 | 29.494 | 3.300 |
| metal / rmsnorm_8192x4096 | 0.073 | 54.633 | 40.071 | 18.608 | 14.523 | 10.453 | 26.263 | 3.355 |
| metal / rmsnorm_16384x4096 | 0.072 | 54.779 | 79.831 | 11.409 | 23.060 | 5.139 | 59.944 | 11.039 |
| metal / layernorm_37x8191 | 0.084 | 70.687 | 2.105 | 1.946 | 77.043 | 7.775 | 0.478 | 0.378 |
| metal / layernorm_1024x8192 | 0.085 | 65.844 | 10.378 | 15.799 | 9.274 | 0.921 | 6.850 | 1.153 |
| metal / layernorm_1024x16384 | 0.071 | 67.480 | 21.129 | 17.410 | 84.488 | 10.202 | 13.198 | 2.019 |
| metal / layernorm_4096x8192 | 0.075 | 64.653 | 42.733 | 4.452 | 14.227 | 8.293 | 26.329 | 7.367 |
| metal / layernorm_8192x4096 | 0.080 | 66.551 | 40.426 | 18.304 | 75.938 | 14.267 | 27.404 | 3.270 |
| metal / layernorm_16384x4096 | 0.086 | 65.583 | 96.752 | 10.820 | 22.064 | 12.839 | 58.814 | 6.861 |

## GPU command-buffer control (no encoder probes)

These samples collect completed command-buffer GPUStartTime/GPUEndTime without encoder hooks or counter attachments. They include GPU work and gaps inside each command buffer (including any blits), not CPU encoding or completion notification. They are not individual-kernel timestamps. Probe/control ratios compare identical batch sizes in alternating-order samples; they diagnose timing perturbation, not a correction factor. Prefer this no-counter control for cross-framework GPU comparisons when counters perturb execution.

| Case / path | GPU batch µs/op | GPU single µs | E2E batch µs/op | E2E single µs | Counter / control GPU batch |
|---|---:|---:|---:|---:|---:|
| softmax_37x8191 / native | 11.798 | 13.500 | 12.311 | 269.584 | 0.982× |
| softmax_37x8191 / torch | 26.729 | 62.000 | 34.058 | 377.458 | 2.820× |
| softmax_1024x8192 / native | 157.669 | 164.625 | 160.418 | 413.292 | 0.994× |
| softmax_1024x8192 / torch | 254.645 | 235.333 | 272.847 | 558.833 | 1.103× |
| softmax_1024x16384 / native | 392.143 | 373.167 | 404.618 | 616.500 | 1.013× |
| softmax_1024x16384 / torch | 572.759 | 535.708 | 598.853 | 917.458 | 1.026× |
| softmax_4096x8192 / native | 793.654 | 754.000 | 820.560 | 1037.625 | 0.996× |
| softmax_4096x8192 / torch | 1007.636 | 947.208 | 1062.054 | 1287.459 | 0.992× |
| softmax_8192x4096 / native | 787.807 | 747.417 | 803.530 | 1031.291 | 1.001× |
| softmax_8192x4096 / torch | 886.206 | 836.208 | 936.328 | 1140.000 | 1.008× |
| softmax_16384x4096 / native | 1579.815 | 1471.542 | 1611.852 | 1900.875 | 1.005× |
| softmax_16384x4096 / torch | 1746.693 | 1620.708 | 1831.086 | 2062.333 | 1.001× |
| rmsnorm_37x8191 / native | 11.382 | 14.833 | 12.010 | 248.334 | 1.024× |
| rmsnorm_37x8191 / torch | 12.962 | 16.708 | 17.080 | 240.125 | 1.001× |
| rmsnorm_1024x8192 / native | 156.078 | 175.542 | 158.913 | 432.875 | 0.998× |
| rmsnorm_1024x8192 / torch | 161.862 | 166.958 | 168.636 | 404.250 | 1.004× |
| rmsnorm_1024x16384 / native | 404.490 | 382.792 | 412.848 | 1045.833 | 0.997× |
| rmsnorm_1024x16384 / torch | 413.426 | 395.625 | 424.295 | 719.083 | 0.987× |
| rmsnorm_4096x8192 / native | 797.022 | 749.750 | 806.751 | 1057.542 | 0.995× |
| rmsnorm_4096x8192 / torch | 816.870 | 751.125 | 824.130 | 1093.584 | 1.003× |
| rmsnorm_8192x4096 / native | 787.200 | 755.250 | 806.989 | 1054.250 | 1.001× |
| rmsnorm_8192x4096 / torch | 797.590 | 752.833 | 817.479 | 1022.500 | 1.018× |
| rmsnorm_16384x4096 / native | 1589.065 | 1482.125 | 1615.365 | 1796.042 | 0.995× |
| rmsnorm_16384x4096 / torch | 1614.743 | 1489.542 | 1666.765 | 2069.042 | 1.000× |
| layernorm_37x8191 / native | 12.822 | 15.833 | 13.670 | 245.625 | 1.001× |
| layernorm_37x8191 / torch | 34.682 | 38.125 | 40.731 | 278.458 | 1.020× |
| layernorm_1024x8192 / native | 159.753 | 164.750 | 164.273 | 446.041 | 0.999× |
| layernorm_1024x8192 / torch | 505.452 | 480.792 | 522.811 | 813.375 | 1.000× |
| layernorm_1024x16384 / native | 421.987 | 395.458 | 427.665 | 838.167 | 1.011× |
| layernorm_1024x16384 / torch | 1100.611 | 1027.125 | 1112.396 | 1372.708 | 0.987× |
| layernorm_4096x8192 / native | 804.751 | 745.583 | 861.168 | 1127.958 | 1.000× |
| layernorm_4096x8192 / torch | 2045.646 | 1920.500 | 2128.927 | 2578.250 | 0.991× |
| layernorm_8192x4096 / native | 797.767 | 752.542 | 819.095 | 1140.917 | 0.990× |
| layernorm_8192x4096 / torch | 1605.828 | 1485.500 | 1637.142 | 1850.250 | 1.002× |
| layernorm_16384x4096 / native | 1577.627 | 1477.000 | 1643.012 | 1811.792 | 1.006× |
| layernorm_16384x4096 / torch | 3160.722 | 2958.375 | 3215.986 | 3918.084 | 1.002× |

## Instrumented compute-pass diagnostics versus end-to-end dispatch

Device numbers use real Metal compute-pass start/end counters, calibrated to nanoseconds. They exclude CPU encoding, queue wait before GPU execution, and completion notification. Host-wall numbers above are separate, uninstrumented samples. A pass may contain multiple dispatches: batched GPU time is divided by its own recorded repetition count (at most 64), and a multi-kernel eager operator is not mislabeled as one kernel. Compute-pass time includes GPU dispatch/barrier work inside the pass, not only arithmetic instructions. Do not subtract independently sampled medians to infer CPU cost.

Counter attachments can perturb execution substantially. Compare against the command-buffer control above; without that control, instrumentation overhead is unvalidated. These probe samples are diagnostics, not an uninstrumented kernel-speed ranking.

| Case | Native probe batch µs/op | Torch probe batch µs/op | Native probe single µs | Torch probe single µs | Native E2E single µs | Torch E2E single µs |
|---|---:|---:|---:|---:|---:|---:|
| softmax_37x8191 | 11.464 | 61.179 | 14.750 | 61.125 | 269.584 | 377.458 |
| softmax_1024x8192 | 157.428 | 257.848 | 170.292 | 250.500 | 413.292 | 558.833 |
| softmax_1024x16384 | 397.430 | 575.701 | 379.334 | 557.333 | 616.500 | 917.458 |
| softmax_4096x8192 | 790.503 | 1007.775 | 754.042 | 949.417 | 1037.625 | 1287.459 |
| softmax_8192x4096 | 787.992 | 896.880 | 742.584 | 843.042 | 1031.291 | 1140.000 |
| softmax_16384x4096 | 1591.034 | 1747.676 | 1462.166 | 1621.417 | 1900.875 | 2062.333 |
| rmsnorm_37x8191 | 11.363 | 13.251 | 14.458 | 16.750 | 248.334 | 240.125 |
| rmsnorm_1024x8192 | 155.600 | 161.547 | 154.041 | 165.583 | 432.875 | 404.250 |
| rmsnorm_1024x16384 | 403.870 | 412.773 | 384.709 | 393.750 | 1045.833 | 719.083 |
| rmsnorm_4096x8192 | 794.711 | 822.175 | 744.125 | 760.875 | 1057.542 | 1093.584 |
| rmsnorm_8192x4096 | 788.045 | 801.236 | 746.125 | 754.958 | 1054.250 | 1022.500 |
| rmsnorm_16384x4096 | 1586.935 | 1616.069 | 1496.750 | 1497.750 | 1796.042 | 2069.042 |
| layernorm_37x8191 | 12.865 | 35.174 | 15.958 | 38.292 | 245.625 | 278.458 |
| layernorm_1024x8192 | 160.019 | 505.524 | 165.208 | 475.875 | 446.041 | 813.375 |
| layernorm_1024x16384 | 422.839 | 1085.099 | 397.375 | 1027.459 | 838.167 | 1372.708 |
| layernorm_4096x8192 | 806.964 | 2043.118 | 756.875 | 1920.250 | 1127.958 | 2578.250 |
| layernorm_8192x4096 | 792.194 | 1609.092 | 748.292 | 1485.458 | 1140.917 | 1850.250 |
| layernorm_16384x4096 | 1584.481 | 3182.602 | 1472.334 | 2958.583 | 1811.792 | 3918.084 |

Raw samples, numerical errors, device identities, compiler version, binary hash, source revision, and thread settings are in [results.json](results.json).
