# Tile lowering comparison

FP32 C=A×B; compact row-major, no transpose or prepacking. All five complete outputs use identical inputs and an FP64 oracle (atol=rtol=1e-4). Native and handwritten MPP use the same fixed configuration; TIRx has a separate recorded schedule. No search or minimum-of-rounds selection occurs here.

All table values are synchronized host-wall batched throughput (µs/call), including each runtime's dispatch, encoding/submission and synchronization overhead. Compilation, allocation and transfers are excluded. They are NOT GPU kernel times. Raw MPS/MPP GPU intervals are retained only in JSON; no cross-metric ratios are computed. Torch is its default eager MPS path, not a claim that it uses the direct MPS benchmark's kernel.

Ten rounds balance positions and pairwise precedence. Fewer rounds are explicitly an unbalanced smoke/exploration run. No failed or slow row is removed. Ratios below are paired round medians, not confidence intervals.

Order balanced: True. Rounds: 10.

| M×N×K | Valid rounds | Tile→MPP | Tile→TIRx | Hand MPP | MPS | Torch | Native/MPS | Native/hand MPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32×32×32 | 10/10 | 4.387 | 4.840 | 3.055 | 11.181 | 26.950 | 0.398 | 1.477 |
| 128×128×128 | 10/10 | 9.478 | 6.827 | 5.496 | 16.768 | 27.247 | 0.571 | 1.724 |
| 512×512×512 | 10/10 | 49.355 | 53.313 | 47.683 | 53.305 | 48.668 | 0.929 | 1.044 |
| 1024×1024×1024 | 10/10 | 295.407 | 320.019 | 272.073 | 278.567 | 291.049 | 1.061 | 1.086 |
| 256×1024×128 | 10/10 | 19.745 | 19.350 | 17.613 | 20.721 | 29.183 | 0.948 | 1.111 |
| 1024×128×256 | 10/10 | 19.333 | 20.096 | 18.924 | 27.859 | 29.421 | 0.683 | 1.021 |
| 127×193×61 | 10/10 | 8.821 | 8.755 | 7.756 | 18.588 | 27.196 | 0.474 | 1.145 |
| 513×257×129 | 10/10 | 31.778 | 22.783 | 25.867 | 37.182 | 34.654 | 0.851 | 1.221 |

Artifacts unchanged: True.

Raw samples, configs, ordering, failed paths and provenance: [results.json](results.json).
