# TileIR/TVMx, PyTorch, and direct BLAS/MPS GEMM

Compact row-major FP32 C=A×B, alpha=1, beta=0, no transpose or prepacking. All three full outputs use the same deterministic inputs and FP64 oracle (atol=rtol=1e-4).

Native schedules are frozen from the supplied reports; their historical timing scores are not used. Fresh JIT, setup and upload are excluded. Warm times include host API/encoding/submission overhead and synchronization; these are not GPU hardware-event times. MPSMatrixMultiplication uses private buffers, MPSKernelOptionsNone, and one command buffer per batch; PyTorch uses its default eager MPS path, not a claim of the same internal MPS kernel.

CPU uses Accelerate's classic LP64 cblas_sgemm. Thread environment is recorded, not a measurement of actual library worker counts. The runner clears optional PyTorch MPS fast-math, prefer-Metal and fallback overrides before importing PyTorch.

Every shape gets all six implementation orders, and case order rotates. Values are medians of per-round p50s in microseconds. Ratios are paired round medians [min, max], not confidence intervals. Native/system >1 means Tile is slower. No failing or slow round is removed.

| Backend / shape M×N×K | Valid rounds | Tile µs | PyTorch µs | BLAS/MPS µs | Tile / BLAS or MPS [range] |
|---|---:|---:|---:|---:|---:|
| cpu / gemm_1024x1024x1024 | 6/6 | 984.515 | 930.311 | 965.743 | 1.031× [0.893, 1.234] |
| cpu / gemm_1024x128x256 | 6/6 | 62.717 | 63.152 | 61.323 | 1.023× [1.019, 1.028] |
| cpu / gemm_127x193x61 | 6/6 | 6.287 | 6.791 | 6.030 | 1.047× [1.012, 1.075] |
| cpu / gemm_128x128x128 | 6/6 | 4.518 | 4.961 | 4.073 | 1.105× [1.085, 1.148] |
| cpu / gemm_256x1024x128 | 6/6 | 65.597 | 65.877 | 64.332 | 1.020× [1.007, 1.026] |
| cpu / gemm_32x32x32 | 6/6 | 0.503 | 0.918 | 0.390 | 1.254× [1.071, 1.484] |
| cpu / gemm_512x512x512 | 6/6 | 130.099 | 139.469 | 131.055 | 0.995× [0.988, 1.002] |
| cpu / gemm_513x257x129 | 6/6 | 43.612 | 43.701 | 43.356 | 1.005× [0.990, 1.035] |

Failed measurements: 0. Binary stability check: True.

Raw timing/latency samples, cold phases, all correctness errors, schedules, device/API identities, compiler versions and binary hashes: [results.json](results.json).
