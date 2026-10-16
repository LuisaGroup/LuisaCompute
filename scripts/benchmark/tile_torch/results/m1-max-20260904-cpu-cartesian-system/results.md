# TileIR/TVMx, PyTorch, and direct BLAS/MPS GEMM

Compact row-major FP32 C=A×B, alpha=1, beta=0, no transpose or prepacking. All three full outputs use the same deterministic inputs and FP64 oracle (atol=rtol=1e-4).

Native schedules are frozen from the supplied reports; their historical timing scores are not used. Fresh JIT, setup and upload are excluded. Warm times include host API/encoding/submission overhead and synchronization; these are not GPU hardware-event times. MPSMatrixMultiplication uses private buffers, MPSKernelOptionsNone, and one command buffer per batch; PyTorch uses its default eager MPS path, not a claim of the same internal MPS kernel.

CPU uses Accelerate's classic LP64 cblas_sgemm. Thread environment is recorded, not a measurement of actual library worker counts. The runner clears optional PyTorch MPS fast-math, prefer-Metal and fallback overrides before importing PyTorch.

Every shape gets all six implementation orders, and case order rotates. Values are medians of per-round p50s in microseconds. Ratios are paired round medians [min, max], not confidence intervals. Native/system >1 means Tile is slower. No failing or slow round is removed.

| Backend / shape M×N×K | Valid rounds | Tile µs | PyTorch µs | BLAS/MPS µs | Tile / BLAS or MPS [range] |
|---|---:|---:|---:|---:|---:|
| cpu / gemm_1024x1024x1024 | 6/6 | 7820.391 | 974.972 | 983.890 | 7.845× [7.574, 8.567] |
| cpu / gemm_1024x128x256 | 6/6 | 265.142 | 64.493 | 62.515 | 4.235× [4.172, 8.571] |
| cpu / gemm_127x193x61 | 6/6 | 36.397 | 6.565 | 5.856 | 6.223× [5.628, 7.801] |
| cpu / gemm_128x128x128 | 6/6 | 22.589 | 4.893 | 4.174 | 5.399× [4.924, 6.368] |
| cpu / gemm_256x1024x128 | 6/6 | 299.889 | 68.757 | 67.066 | 4.475× [4.169, 5.632] |
| cpu / gemm_32x32x32 | 6/6 | 4.451 | 0.909 | 0.299 | 15.212× [11.302, 17.150] |
| cpu / gemm_512x512x512 | 6/6 | 922.129 | 145.061 | 136.418 | 6.802× [6.288, 7.953] |
| cpu / gemm_513x257x129 | 6/6 | 386.947 | 43.539 | 42.993 | 9.004× [7.857, 11.390] |

Failed measurements: 0. Binary stability check: True.

Raw timing/latency samples, cold phases, all correctness errors, schedules, device/API identities, compiler versions and binary hashes: [results.json](results.json).
