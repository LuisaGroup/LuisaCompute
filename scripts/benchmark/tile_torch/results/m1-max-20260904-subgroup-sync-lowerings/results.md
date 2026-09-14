# Tile lowering comparison

FP32 C=A×B; compact row-major, no transpose or prepacking. All complete outputs use identical inputs and an FP64 oracle (atol=rtol=1e-4). Native and handwritten MPP use the same fixed configuration; TIRx has a separate recorded schedule. No search or minimum-of-rounds selection occurs here.

All table values are synchronized host-wall batched throughput (µs/call), including each runtime's dispatch, encoding/submission and synchronization overhead. Compilation, allocation and transfers are excluded. They are NOT GPU kernel times. Raw MPS/MPP GPU intervals are retained only in JSON; no cross-metric ratios are computed. Torch is its default eager MPS path, not a claim that it uses the direct MPS benchmark's kernel.

14 rounds balance positions and pairwise precedence. Fewer rounds are explicitly an unbalanced smoke/exploration run. No failed or slow row is removed. Ratios below are paired round medians, not confidence intervals.

Order balanced: True. Rounds: 14.

Native/hand MPP and TIRx/Luisa use fast math off; TIRx/Luisa fast explicitly uses it on. TVMx's current Metal runtime hardcodes fast math on. Optional TIRx Runtime controls must have identical generated Metal source hashes and matching threadgroup widths. Compiler language/resource options and submission APIs can still differ; this is not an isolated GPU execution-overhead measurement.

Optional TIRx→MPP is the patched TVM Metal code generator using non-owning memory inputs, not the native MPP emitter. It reuses the frozen TIRx geometry; recorded planner costs still describe the SIMD-group reference family, not MPP's internal instruction count or register use.

Optional TIRx→MPP views enables proven read-only snapshot forwarding, with a separately frozen schedule. It is not a same-geometry ablation unless the recorded schedules match; original TIRx and non-forwarding MPP remain controls.

MPP-view subgroup-fence policy override: elide. The default is retention; requesting elision still requires a reported whole-group independence proof. This policy is not assumed profitable.

| M×N×K | Valid rounds | Tile→MPP | TIRx/TVM | Hand MPP | MPS | Torch | TIRx→MPP/TVM | TIRx→MPP views/TVM | Native/MPS | Native/hand MPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32×32×32 | 14/14 | 5.298 | 5.834 | 3.517 | 12.839 | 29.546 | 5.522 | 3.729 | 0.412 | 1.510 |
| 128×128×128 | 14/14 | 10.673 | 8.350 | 6.553 | 18.755 | 29.609 | 7.639 | 6.705 | 0.568 | 1.622 |
| 512×512×512 | 14/14 | 65.329 | 71.514 | 61.175 | 66.748 | 63.776 | 65.109 | 63.531 | 0.974 | 1.071 |
| 1024×1024×1024 | 14/14 | 400.534 | 429.297 | 355.017 | 369.750 | 384.889 | 451.513 | 378.932 | 1.084 | 1.120 |
| 256×1024×128 | 14/14 | 25.111 | 25.147 | 22.713 | 25.493 | 33.128 | 23.882 | 21.711 | 0.964 | 1.096 |
| 1024×128×256 | 14/14 | 24.438 | 25.985 | 24.536 | 33.094 | 33.215 | 24.150 | 22.507 | 0.741 | 0.992 |
| 127×193×61 | 14/14 | 10.297 | 10.252 | 9.371 | 21.450 | 30.913 | 9.927 | 7.827 | 0.482 | 1.085 |
| 513×257×129 | 14/14 | 41.063 | 29.243 | 33.226 | 46.924 | 45.948 | 26.868 | 27.099 | 0.865 | 1.228 |

Artifacts unchanged: True.

Raw samples, configs, ordering, failed paths and provenance: [results.json](results.json).
