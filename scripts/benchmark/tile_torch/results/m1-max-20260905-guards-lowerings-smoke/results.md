# Tile lowering comparison

FP32 C=A×B; compact row-major, no transpose or prepacking. All complete outputs use identical inputs and an FP64 oracle (atol=rtol=1e-4). Native and handwritten MPP use the same fixed configuration; TIRx has a separate recorded schedule. No search or minimum-of-rounds selection occurs here.

All table values are synchronized host-wall batched throughput (µs/call), including each runtime's dispatch, encoding/submission and synchronization overhead. Compilation, allocation and transfers are excluded. They are NOT GPU kernel times. Raw MPS/MPP GPU intervals are retained only in JSON; no cross-metric ratios are computed. Torch is its default eager MPS path, not a claim that it uses the direct MPS benchmark's kernel.

14 rounds balance positions and pairwise precedence. Fewer rounds are explicitly an unbalanced smoke/exploration run. No failed or slow row is removed. Ratios below are paired round medians, not confidence intervals.

Order balanced: False. Rounds: 2.

Native/hand MPP and TIRx/Luisa use fast math off; TIRx/Luisa fast explicitly uses it on. TVMx's current Metal runtime hardcodes fast math on. Optional TIRx Runtime controls must have identical generated Metal source hashes and matching threadgroup widths. Compiler language/resource options and submission APIs can still differ; this is not an isolated GPU execution-overhead measurement.

Optional TIRx→MPP is the patched TVM Metal code generator using non-owning memory inputs, not the native MPP emitter. It reuses the frozen TIRx geometry; recorded planner costs still describe the SIMD-group reference family, not MPP's internal instruction count or register use.

Optional TIRx→MPP views enables proven read-only snapshot forwarding, with a separately frozen schedule. It is not a same-geometry ablation unless the recorded schedules match; original TIRx and non-forwarding MPP remain controls.

MPP-view subgroup-fence policy override: elide. The default is retention; requesting elision still requires a reported whole-group independence proof. This policy is not assumed profitable.

| M×N×K | Valid rounds | Tile→MPP | TIRx/TVM | Hand MPP | MPS | Torch | TIRx→MPP/TVM | TIRx→MPP views/TVM | Native/MPS | Native/hand MPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32×32×32 | 2/2 | 4.156 | 4.310 | 2.578 | 10.058 | 29.665 | 4.226 | 2.795 | 0.413 | 1.613 |
| 128×128×128 | 2/2 | 8.051 | 6.347 | 4.842 | 14.668 | 29.587 | 5.738 | 4.801 | 0.549 | 1.663 |
| 512×512×512 | 2/2 | 49.910 | 53.751 | 47.951 | 52.925 | 50.103 | 49.491 | 48.350 | 0.943 | 1.041 |
| 1024×1024×1024 | 2/2 | 301.724 | 328.730 | 277.439 | 286.280 | 298.859 | 353.954 | 297.052 | 1.054 | 1.088 |
| 256×1024×128 | 2/2 | 18.781 | 18.642 | 16.942 | 19.454 | 32.319 | 17.588 | 16.024 | 0.965 | 1.109 |
| 1024×128×256 | 2/2 | 18.191 | 19.520 | 18.377 | 25.691 | 32.261 | 18.449 | 16.885 | 0.708 | 0.990 |
| 127×193×61 | 2/2 | 7.763 | 7.716 | 7.014 | 17.403 | 30.340 | 7.380 | 6.063 | 0.446 | 1.107 |
| 513×257×129 | 2/2 | 30.740 | 21.847 | 25.139 | 36.847 | 35.703 | 20.286 | 20.603 | 0.834 | 1.223 |

Artifacts unchanged: True.

Raw samples, configs, ordering, failed paths and provenance: [results.json](results.json).
