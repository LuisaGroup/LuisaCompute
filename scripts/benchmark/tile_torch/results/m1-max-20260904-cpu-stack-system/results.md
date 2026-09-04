# TileIR/TVMx, PyTorch, and direct BLAS/MPS GEMM

Compact row-major FP32 C=A×B, alpha=1, beta=0, no transpose or prepacking. All three full outputs use the same deterministic inputs and FP64 oracle (atol=rtol=1e-4).

Native schedules are frozen from the supplied reports; their historical timing scores are not used. Fresh JIT, setup and upload are excluded. Warm times include host API/encoding/submission overhead and synchronization; these are not GPU hardware-event times. MPSMatrixMultiplication uses private buffers, MPSKernelOptionsNone, and one command buffer per batch; PyTorch uses its default eager MPS path, not a claim of the same internal MPS kernel.

CPU uses Accelerate's classic LP64 cblas_sgemm. Thread environment is recorded, not a measurement of actual library worker counts. The runner clears optional PyTorch MPS fast-math, prefer-Metal and fallback overrides before importing PyTorch.

Every shape gets all six implementation orders, and case order rotates. Values are medians of per-round p50s in microseconds. Ratios are paired round medians [min, max], not confidence intervals. Native/system >1 means Tile is slower. No failing or slow round is removed.

| Backend / shape M×N×K | Valid rounds | Tile µs | PyTorch µs | BLAS/MPS µs | Tile / BLAS or MPS [range] |
|---|---:|---:|---:|---:|---:|
| cpu / gemm_1024x1024x1024 | 6/6 | 11492.371 | 1061.770 | 1019.098 | 11.267× [9.181, 12.233] |
| cpu / gemm_1024x128x256 | 6/6 | 474.023 | 64.959 | 62.742 | 7.553× [6.964, 9.676] |
| cpu / gemm_127x193x61 | 6/6 | 51.995 | 6.582 | 5.948 | 8.746× [7.794, 10.357] |
| cpu / gemm_128x128x128 | 6/6 | 39.090 | 4.912 | 4.198 | 9.314× [8.599, 9.700] |
| cpu / gemm_256x1024x128 | 6/6 | 514.310 | 68.825 | 67.164 | 7.669× [6.354, 9.407] |
| cpu / gemm_32x32x32 | 6/6 | 5.419 | 0.874 | 0.306 | 17.342× [11.392, 18.552] |
| cpu / gemm_512x512x512 | 6/6 | 1667.844 | 139.957 | 138.044 | 11.809× [11.070, 15.045] |
| cpu / gemm_513x257x129 | 6/6 | 510.996 | 45.312 | 43.106 | 11.828× [10.895, 14.115] |

Failed measurements: 0. Binary stability check: True.

Raw timing/latency samples, cold phases, all correctness errors, schedules, device/API identities, compiler versions and binary hashes: [results.json](results.json).
