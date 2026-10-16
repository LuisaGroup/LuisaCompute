# TileIR/TVMx, PyTorch, and direct BLAS/MPS GEMM

Compact row-major FP32 C=A×B, alpha=1, beta=0, no transpose or prepacking. All three full outputs use the same deterministic inputs and FP64 oracle (atol=rtol=1e-4).

Native schedules are frozen from the supplied reports; their historical timing scores are not used. Fresh JIT, setup and upload are excluded. Warm times include host API/encoding/submission overhead and synchronization; these are not GPU hardware-event times. MPSMatrixMultiplication uses private buffers, MPSKernelOptionsNone, and one command buffer per batch; PyTorch uses its default eager MPS path, not a claim of the same internal MPS kernel.

CPU uses Accelerate's classic LP64 cblas_sgemm. Thread environment is recorded, not a measurement of actual library worker counts. The runner clears optional PyTorch MPS fast-math, prefer-Metal and fallback overrides before importing PyTorch.

Every shape gets all six implementation orders, and case order rotates. Values are medians of per-round p50s in microseconds. Ratios are paired round medians [min, max], not confidence intervals. Native/system >1 means Tile is slower. No failing or slow round is removed.

| Backend / shape M×N×K | Valid rounds | Tile µs | PyTorch µs | BLAS/MPS µs | Tile / BLAS or MPS [range] |
|---|---:|---:|---:|---:|---:|
| cpu / gemm_1024x1024x1024 | 6/6 | 10208.031 | 1028.191 | 1112.837 | 9.494× [8.104, 10.146] |
| metal / gemm_1024x1024x1024 | 6/6 | 318.295 | 286.703 | 281.503 | 1.134× [1.005, 1.172] |
| cpu / gemm_1024x128x256 | 6/6 | 560.994 | 64.554 | 62.511 | 8.984× [7.604, 10.937] |
| metal / gemm_1024x128x256 | 6/6 | 22.116 | 31.407 | 24.964 | 0.895× [0.859, 0.929] |
| cpu / gemm_127x193x61 | 6/6 | 72.661 | 6.559 | 5.825 | 12.494× [10.960, 15.948] |
| metal / gemm_127x193x61 | 6/6 | 10.922 | 28.971 | 16.139 | 0.678× [0.638, 0.725] |
| cpu / gemm_128x128x128 | 6/6 | 56.549 | 4.907 | 4.182 | 13.538× [8.833, 16.818] |
| metal / gemm_128x128x128 | 6/6 | 12.190 | 28.774 | 14.314 | 0.839× [0.803, 0.888] |
| cpu / gemm_256x1024x128 | 6/6 | 643.461 | 68.028 | 66.604 | 9.660× [7.720, 10.395] |
| metal / gemm_256x1024x128 | 6/6 | 17.953 | 31.224 | 18.842 | 0.946× [0.897, 1.018] |
| cpu / gemm_32x32x32 | 6/6 | 5.429 | 0.891 | 0.312 | 17.577× [11.769, 21.993] |
| metal / gemm_32x32x32 | 6/6 | 5.352 | 28.603 | 9.756 | 0.555× [0.502, 0.578] |
| cpu / gemm_512x512x512 | 6/6 | 1474.382 | 142.577 | 137.491 | 10.694× [10.070, 11.853] |
| metal / gemm_512x512x512 | 6/6 | 53.771 | 47.645 | 50.643 | 1.061× [1.042, 1.088] |
| cpu / gemm_513x257x129 | 6/6 | 708.094 | 45.001 | 43.015 | 16.475× [14.030, 18.019] |
| metal / gemm_513x257x129 | 6/6 | 26.362 | 34.384 | 35.901 | 0.732× [0.709, 0.752] |

Failed measurements: 0. Binary stability check: True.

Raw timing/latency samples, cold phases, all correctness errors, schedules, device/API identities, compiler versions and binary hashes: [results.json](results.json).
