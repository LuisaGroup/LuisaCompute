# Tile lowering comparison

FP32 C=A×B; compact row-major, no transpose or prepacking. All complete outputs use identical inputs and an FP64 oracle (atol=rtol=1e-4). Native and handwritten MPP use the same fixed configuration; TIRx has a separate recorded schedule. No search or minimum-of-rounds selection occurs here.

All table values are synchronized host-wall batched throughput (µs/call), including each runtime's dispatch, encoding/submission and synchronization overhead. Compilation, allocation and transfers are excluded. They are NOT GPU kernel times. Raw MPS/MPP GPU intervals are retained only in JSON; no cross-metric ratios are computed. Torch is its default eager MPS path, not a claim that it uses the direct MPS benchmark's kernel.

12 rounds balance positions and pairwise precedence. Fewer rounds are explicitly an unbalanced smoke/exploration run. No failed or slow row is removed. Ratios below are paired round medians, not confidence intervals.

Order balanced: True. Rounds: 12.

Native/hand MPP and TIRx/Luisa use fast math off; TIRx/Luisa fast explicitly uses it on. TVMx's current Metal runtime hardcodes fast math on. Optional TIRx Runtime controls must have identical generated Metal source hashes and matching threadgroup widths. Compiler language/resource options and submission APIs can still differ; this is not an isolated GPU execution-overhead measurement.

Optional TIRx→MPP is the patched TVM Metal code generator using non-owning memory inputs, not the native MPP emitter. It reuses the frozen TIRx geometry; recorded planner costs still describe the SIMD-group reference family, not MPP's internal instruction count or register use.

| M×N×K | Valid rounds | Tile→MPP | TIRx/TVM | Hand MPP | MPS | Torch | TIRx→MPP/TVM | Native/MPS | Native/hand MPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32×32×32 | 12/12 | 4.638 | 4.894 | 3.052 | 11.077 | 26.997 | 4.747 | 0.422 | 1.534 |
| 128×128×128 | 12/12 | 9.268 | 6.851 | 5.474 | 16.871 | 27.501 | 6.032 | 0.550 | 1.700 |
| 512×512×512 | 12/12 | 49.726 | 53.003 | 47.557 | 52.903 | 48.574 | 49.147 | 0.942 | 1.047 |
| 1024×1024×1024 | 12/12 | 293.304 | 318.676 | 271.393 | 277.756 | 290.453 | 347.856 | 1.056 | 1.080 |
| 256×1024×128 | 12/12 | 19.549 | 19.033 | 17.502 | 20.787 | 29.088 | 18.089 | 0.949 | 1.120 |
| 1024×128×256 | 12/12 | 19.311 | 19.958 | 18.906 | 27.828 | 29.140 | 18.971 | 0.693 | 1.026 |
| 127×193×61 | 12/12 | 8.799 | 8.689 | 7.751 | 18.507 | 27.183 | 8.492 | 0.477 | 1.135 |
| 513×257×129 | 12/12 | 31.595 | 22.497 | 25.838 | 37.685 | 34.623 | 21.087 | 0.839 | 1.219 |

Artifacts unchanged: True.

Raw samples, configs, ordering, failed paths and provenance: [results.json](results.json).
