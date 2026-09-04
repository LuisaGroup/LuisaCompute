# Tile lowering comparison

FP32 C=A×B; compact row-major, no transpose or prepacking. All complete outputs use identical inputs and an FP64 oracle (atol=rtol=1e-4). Native and handwritten MPP use the same fixed configuration; TIRx has a separate recorded schedule. No search or minimum-of-rounds selection occurs here.

All table values are synchronized host-wall batched throughput (µs/call), including each runtime's dispatch, encoding/submission and synchronization overhead. Compilation, allocation and transfers are excluded. They are NOT GPU kernel times. Raw MPS/MPP GPU intervals are retained only in JSON; no cross-metric ratios are computed. Torch is its default eager MPS path, not a claim that it uses the direct MPS benchmark's kernel.

14 rounds balance positions and pairwise precedence. Fewer rounds are explicitly an unbalanced smoke/exploration run. No failed or slow row is removed. Ratios below are paired round medians, not confidence intervals.

Order balanced: True. Rounds: 14.

Native/hand MPP and TIRx/Luisa use fast math off; TIRx/Luisa fast explicitly uses it on. TVMx's current Metal runtime hardcodes fast math on. Optional TIRx Runtime controls must have identical generated Metal source hashes and matching threadgroup widths. Compiler language/resource options and submission APIs can still differ; this is not an isolated GPU execution-overhead measurement.

| M×N×K | Valid rounds | Tile→MPP | TIRx/TVM | Hand MPP | MPS | Torch | TIRx/Luisa | TIRx/Luisa fast | Native/MPS | Native/hand MPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32×32×32 | 14/14 | 4.573 | 4.968 | 3.083 | 11.274 | 27.034 | 5.431 | 5.435 | 0.403 | 1.512 |
| 128×128×128 | 14/14 | 9.420 | 6.815 | 5.536 | 16.952 | 27.194 | 7.427 | 7.368 | 0.555 | 1.704 |
| 512×512×512 | 14/14 | 50.042 | 53.484 | 47.827 | 53.312 | 48.866 | 53.996 | 53.964 | 0.942 | 1.045 |
| 1024×1024×1024 | 14/14 | 294.958 | 320.225 | 272.627 | 279.207 | 291.332 | 320.784 | 320.667 | 1.056 | 1.082 |
| 256×1024×128 | 14/14 | 19.352 | 19.274 | 17.638 | 20.870 | 29.133 | 19.689 | 19.684 | 0.927 | 1.098 |
| 1024×128×256 | 14/14 | 19.404 | 20.144 | 19.022 | 28.189 | 29.226 | 20.365 | 20.425 | 0.686 | 1.022 |
| 127×193×61 | 14/14 | 8.797 | 8.829 | 7.877 | 18.698 | 27.083 | 9.310 | 9.367 | 0.471 | 1.118 |
| 513×257×129 | 14/14 | 31.747 | 22.526 | 26.063 | 37.570 | 34.884 | 23.078 | 22.906 | 0.851 | 1.224 |

Artifacts unchanged: True.

Raw samples, configs, ordering, failed paths and provenance: [results.json](results.json).
