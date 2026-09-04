# TileIR/TVMx, PyTorch, and direct BLAS/MPS GEMM

Compact row-major FP32 C=A×B, alpha=1, beta=0, no transpose or prepacking. All three full outputs use the same deterministic inputs and FP64 oracle (atol=rtol=1e-4).

Native schedules are frozen from the supplied reports; their historical timing scores are not used. Fresh JIT, setup and upload are excluded. Warm times include host API/encoding/submission overhead and synchronization; these are not GPU hardware-event times. MPSMatrixMultiplication uses private buffers, MPSKernelOptionsNone, and one command buffer per batch; PyTorch uses its default eager MPS path, not a claim of the same internal MPS kernel.

CPU uses Accelerate's classic LP64 cblas_sgemm. Thread environment is recorded, not a measurement of actual library worker counts. The runner clears optional PyTorch MPS fast-math, prefer-Metal and fallback overrides before importing PyTorch.

Every shape gets all six implementation orders, and case order rotates. Values are medians of per-round p50s in microseconds. Ratios are paired round medians [min, max], not confidence intervals. Native/system >1 means Tile is slower. No failing or slow round is removed.

| Backend / shape M×N×K | Valid rounds | Tile µs | PyTorch µs | BLAS/MPS µs | Tile / BLAS or MPS [range] |
|---|---:|---:|---:|---:|---:|
| cpu / gemm_1024x1024x1024 | 6/6 | 5541.276 | 982.044 | 988.617 | 5.585× [5.287, 5.981] |
| cpu / gemm_1024x128x256 | 6/6 | 181.695 | 64.136 | 62.506 | 2.910× [2.338, 3.955] |
| cpu / gemm_127x193x61 | 6/6 | 33.366 | 6.512 | 5.901 | 5.699× [5.084, 7.272] |
| cpu / gemm_128x128x128 | 6/6 | 15.703 | 4.863 | 4.193 | 3.748× [3.393, 4.512] |
| cpu / gemm_256x1024x128 | 6/6 | 224.925 | 67.263 | 66.765 | 3.364× [2.761, 3.678] |
| cpu / gemm_32x32x32 | 6/6 | 4.543 | 0.861 | 0.317 | 14.807× [12.317, 17.749] |
| cpu / gemm_512x512x512 | 6/6 | 629.262 | 144.988 | 137.159 | 4.585× [4.148, 5.189] |
| cpu / gemm_513x257x129 | 6/6 | 404.701 | 44.747 | 42.770 | 9.440× [7.913, 9.675] |

Failed measurements: 0. Binary stability check: True.

Raw timing/latency samples, cold phases, all correctness errors, schedules, device/API identities, compiler versions and binary hashes: [results.json](results.json).
