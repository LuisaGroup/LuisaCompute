# Tile lowering comparison

FP32 C=A×B; compact row-major, no transpose or prepacking. All five complete outputs use identical inputs and an FP64 oracle (atol=rtol=1e-4). Native and handwritten MPP use the same fixed configuration; TIRx has a separate recorded schedule. No search or minimum-of-rounds selection occurs here.

All table values are synchronized host-wall batched throughput (µs/call), including each runtime's dispatch, encoding/submission and synchronization overhead. Compilation, allocation and transfers are excluded. They are NOT GPU kernel times. Raw MPS/MPP GPU intervals are retained only in JSON; no cross-metric ratios are computed. Torch is its default eager MPS path, not a claim that it uses the direct MPS benchmark's kernel.

Ten rounds balance positions and pairwise precedence. Fewer rounds are explicitly an unbalanced smoke/exploration run. No failed or slow row is removed. Ratios below are paired round medians, not confidence intervals.

Order balanced: True. Rounds: 10.

| M×N×K | Valid rounds | Tile→MPP | Tile→TIRx | Hand MPP | MPS | Torch | Native/MPS | Native/hand MPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32×32×32 | 10/10 | 4.315 | 4.948 | 2.999 | 10.915 | 26.925 | 0.406 | 1.510 |
| 128×128×128 | 10/10 | 9.525 | 10.792 | 5.473 | 16.726 | 27.428 | 0.573 | 1.736 |
| 512×512×512 | 10/10 | 49.885 | 53.196 | 47.623 | 52.843 | 48.701 | 0.942 | 1.048 |
| 1024×1024×1024 | 10/10 | 294.952 | 319.361 | 271.927 | 278.677 | 290.723 | 1.058 | 1.085 |
| 256×1024×128 | 10/10 | 19.522 | 19.220 | 17.589 | 20.610 | 29.321 | 0.941 | 1.109 |
| 1024×128×256 | 10/10 | 19.226 | 21.392 | 18.752 | 28.118 | 28.980 | 0.687 | 1.028 |
| 127×193×61 | 10/10 | 8.509 | 9.190 | 7.671 | 18.552 | 27.375 | 0.469 | 1.105 |
| 513×257×129 | 10/10 | 31.808 | 22.957 | 25.785 | 37.747 | 34.674 | 0.842 | 1.234 |

Artifacts unchanged: True.

Raw samples, configs, ordering, failed paths and provenance: [results.json](results.json).
