# Tile lowering comparison

FP32 C=A×B; compact row-major, no transpose or prepacking. All complete outputs use identical inputs and an FP64 oracle (atol=rtol=1e-4). Native and handwritten MPP use the same fixed configuration; TIRx has a separate recorded schedule. No search or minimum-of-rounds selection occurs here.

All table values are synchronized host-wall batched throughput (µs/call), including each runtime's dispatch, encoding/submission and synchronization overhead. Compilation, allocation and transfers are excluded. They are NOT GPU kernel times. Raw MPS/MPP GPU intervals are retained only in JSON; no cross-metric ratios are computed. Torch is its default eager MPS path, not a claim that it uses the direct MPS benchmark's kernel.

14 rounds balance positions and pairwise precedence. Fewer rounds are explicitly an unbalanced smoke/exploration run. No failed or slow row is removed. Ratios below are paired round medians, not confidence intervals.

Order balanced: True. Rounds: 14.

Native/hand MPP and TIRx/Luisa use fast math off; TIRx/Luisa fast explicitly uses it on. TVMx's current Metal runtime hardcodes fast math on. Optional TIRx Runtime controls must have identical generated Metal source hashes and matching threadgroup widths. Compiler language/resource options and submission APIs can still differ; this is not an isolated GPU execution-overhead measurement.

Optional TIRx→MPP is the patched TVM Metal code generator using non-owning memory inputs, not the native MPP emitter. It reuses the frozen TIRx geometry; recorded planner costs still describe the SIMD-group reference family, not MPP's internal instruction count or register use.

Optional TIRx→MPP views enables proven read-only snapshot forwarding, with a separately frozen schedule. It is not a same-geometry ablation unless the recorded schedules match; original TIRx and non-forwarding MPP remain controls.

| M×N×K | Valid rounds | Tile→MPP | TIRx/TVM | Hand MPP | MPS | Torch | TIRx→MPP/TVM | TIRx→MPP views/TVM | Native/MPS | Native/hand MPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32×32×32 | 14/14 | 4.618 | 4.953 | 3.066 | 11.179 | 27.640 | 4.806 | 3.225 | 0.405 | 1.520 |
| 128×128×128 | 14/14 | 9.275 | 6.754 | 5.371 | 16.597 | 28.051 | 6.138 | 5.448 | 0.559 | 1.708 |
| 512×512×512 | 14/14 | 49.875 | 53.331 | 47.897 | 53.326 | 48.794 | 49.373 | 43.523 | 0.937 | 1.042 |
| 1024×1024×1024 | 14/14 | 294.725 | 320.162 | 272.558 | 278.687 | 291.133 | 348.847 | 291.736 | 1.058 | 1.081 |
| 256×1024×128 | 14/14 | 19.746 | 19.231 | 17.749 | 20.815 | 29.594 | 18.336 | 16.618 | 0.950 | 1.112 |
| 1024×128×256 | 14/14 | 19.355 | 20.080 | 18.916 | 28.213 | 29.551 | 19.027 | 17.655 | 0.686 | 1.025 |
| 127×193×61 | 14/14 | 8.714 | 8.823 | 7.845 | 18.520 | 27.824 | 8.552 | 6.409 | 0.475 | 1.119 |
| 513×257×129 | 14/14 | 31.842 | 22.732 | 25.905 | 37.960 | 34.762 | 21.090 | 21.178 | 0.839 | 1.229 |

Artifacts unchanged: True.

Raw samples, configs, ordering, failed paths and provenance: [results.json](results.json).
