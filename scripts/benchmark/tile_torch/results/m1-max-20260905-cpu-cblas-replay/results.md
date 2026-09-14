# TileIR/TVMx, PyTorch, and direct BLAS/MPS GEMM

Compact row-major FP32 C=A×B, alpha=1, beta=0, no transpose or prepacking. All three full outputs use the same deterministic inputs and FP64 oracle (atol=rtol=1e-4).

Native schedules are frozen from the supplied reports; their historical timing scores are not used. Fresh JIT, setup and upload are excluded. Warm times include host API/encoding/submission overhead and synchronization; these are not GPU hardware-event times. MPSMatrixMultiplication uses private buffers, MPSKernelOptionsNone, and one command buffer per batch; PyTorch uses its default eager MPS path, not a claim of the same internal MPS kernel.

CPU uses Accelerate's classic LP64 cblas_sgemm. Thread environment is recorded, not a measurement of actual library worker counts. The runner clears optional PyTorch MPS fast-math, prefer-Metal and fallback overrides before importing PyTorch.

Every shape gets all six implementation orders, and case order rotates. Values are medians of per-round p50s in microseconds. Ratios are paired round medians [min, max], not confidence intervals. Native/system >1 means Tile is slower. No failing or slow round is removed.

| Backend / shape M×N×K | Valid rounds | Tile µs | PyTorch µs | BLAS/MPS µs | Tile / BLAS or MPS [range] |
|---|---:|---:|---:|---:|---:|
| cpu / gemm_1024x1024x1024 | 6/6 | 1000.745 | 904.583 | 969.581 | 1.025× [0.974, 1.115] |
| cpu / gemm_1024x128x256 | 6/6 | 62.717 | 63.192 | 61.232 | 1.024× [1.020, 1.051] |
| cpu / gemm_127x193x61 | 6/6 | 6.253 | 6.820 | 5.772 | 1.081× [1.006, 1.093] |
| cpu / gemm_128x128x128 | 6/6 | 4.544 | 5.000 | 4.107 | 1.102× [1.052, 1.123] |
| cpu / gemm_256x1024x128 | 6/6 | 65.720 | 66.392 | 64.262 | 1.022× [1.013, 1.043] |
| cpu / gemm_32x32x32 | 6/6 | 0.523 | 0.960 | 0.419 | 1.206× [0.808, 1.424] |
| cpu / gemm_512x512x512 | 6/6 | 132.359 | 139.356 | 130.299 | 1.012× [0.953, 1.057] |
| cpu / gemm_513x257x129 | 6/6 | 43.832 | 45.208 | 43.155 | 1.019× [0.976, 1.061] |

Failed measurements: 0. Binary stability check: True.

Raw timing/latency samples, cold phases, all correctness errors, schedules, device/API identities, compiler versions and binary hashes: [results.json](results.json).
