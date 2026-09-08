# TileIR/TVMx, PyTorch, and direct BLAS/MPS GEMM

Compact row-major FP32 C=A×B, alpha=1, beta=0, no transpose or prepacking. All three full outputs use the same deterministic inputs and FP64 oracle (atol=rtol=1e-4).

Native schedules are frozen from the supplied reports; their historical timing scores are not used. Fresh JIT, setup and upload are excluded. Warm times include host API/encoding/submission overhead and synchronization; these are not GPU hardware-event times. MPSMatrixMultiplication uses private buffers, MPSKernelOptionsNone, and one command buffer per batch; PyTorch uses its default eager MPS path, not a claim of the same internal MPS kernel.

CPU uses Accelerate's classic LP64 cblas_sgemm. Thread environment is recorded, not a measurement of actual library worker counts. The runner clears optional PyTorch MPS fast-math, prefer-Metal and fallback overrides before importing PyTorch.

Every shape gets all six implementation orders, and case order rotates. Values are medians of per-round p50s in microseconds. Ratios are paired round medians [min, max], not confidence intervals. Native/system >1 means Tile is slower. No failing or slow round is removed.

| Backend / shape M×N×K | Valid rounds | Tile µs | PyTorch µs | BLAS/MPS µs | Tile / BLAS or MPS [range] |
|---|---:|---:|---:|---:|---:|
| cpu / gemm_1024x1024x1024 | 6/6 | 5919.062 | 1020.527 | 1027.681 | 5.747× [5.363, 6.120] |
| cpu / gemm_1024x128x256 | 6/6 | 172.539 | 65.562 | 62.591 | 2.750× [2.405, 3.362] |
| cpu / gemm_127x193x61 | 6/6 | 38.110 | 6.645 | 5.908 | 6.467× [5.630, 7.311] |
| cpu / gemm_128x128x128 | 6/6 | 17.899 | 4.826 | 4.182 | 4.280× [3.877, 5.221] |
| cpu / gemm_256x1024x128 | 6/6 | 234.142 | 71.485 | 67.019 | 3.489× [3.121, 3.742] |
| cpu / gemm_32x32x32 | 6/6 | 4.721 | 0.876 | 0.348 | 14.492× [9.567, 17.429] |
| cpu / gemm_512x512x512 | 6/6 | 733.980 | 146.263 | 138.193 | 5.280× [4.625, 5.747] |
| cpu / gemm_513x257x129 | 6/6 | 282.882 | 45.868 | 43.100 | 6.565× [5.622, 7.132] |

Failed measurements: 0. Binary stability check: True.

Raw timing/latency samples, cold phases, all correctness errors, schedules, device/API identities, compiler versions and binary hashes: [results.json](results.json).
