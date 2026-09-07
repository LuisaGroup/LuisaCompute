# Tile lowering comparison

FP32 C=A×B; compact row-major, no transpose or prepacking. All complete outputs use identical inputs and an FP64 oracle (atol=rtol=1e-4). Native and handwritten MPP use the same fixed configuration; TIRx has a separate recorded schedule. No search or minimum-of-rounds selection occurs here.

All table values are synchronized host-wall batched throughput (µs/call), including each runtime's dispatch, encoding/submission and synchronization overhead. Compilation, allocation and transfers are excluded. They are NOT GPU kernel times. Raw MPS/MPP GPU intervals are retained only in JSON; no cross-metric ratios are computed. Torch is its default eager MPS path, not a claim that it uses the direct MPS benchmark's kernel.

14 rounds balance positions and pairwise precedence. Fewer rounds are explicitly an unbalanced smoke/exploration run. No failed or slow row is removed. Ratios below are paired round medians, not confidence intervals.

Order balanced: True. Rounds: 14.

Native/hand MPP and TIRx/Luisa use fast math off; TIRx/Luisa fast explicitly uses it on. TVMx's current Metal runtime hardcodes fast math on. Optional TIRx Runtime controls must have identical generated Metal source hashes and matching threadgroup widths. Compiler language/resource options and submission APIs can still differ; this is not an isolated GPU execution-overhead measurement.

Optional TIRx→MPP is the patched TVM Metal code generator using non-owning memory inputs, not the native MPP emitter. It reuses the frozen TIRx geometry; current MPP reports use the separately versioned metal_mpp_memory_v2 relative-work model, not an instruction count, measured register use, or calibrated time prediction.

Optional TIRx→MPP views enables proven read-only snapshot forwarding, with a separately frozen schedule. It is not a same-geometry ablation unless the recorded schedules match; original TIRx and non-forwarding MPP remain controls.

MPP-view subgroup-fence policy override: reported. The default is retention; requesting elision still requires a reported whole-group independence proof. This policy is not assumed profitable.

| M×N×K | Valid rounds | Tile→MPP | TIRx/TVM | Hand MPP | MPS | Torch | TIRx→MPP/TVM | TIRx→MPP views/TVM | Native/MPS | Native/hand MPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32×32×32 | 14/14 | 4.055 | 4.456 | 2.809 | 10.081 | 26.899 | 4.143 | 2.982 | 0.403 | 1.531 |
| 128×128×128 | 14/14 | 9.221 | 6.706 | 5.441 | 16.904 | 27.218 | 5.647 | 5.335 | 0.553 | 1.725 |
| 512×512×512 | 14/14 | 48.933 | 52.248 | 46.802 | 52.428 | 47.745 | 50.352 | 42.413 | 0.937 | 1.041 |
| 1024×1024×1024 | 14/14 | 287.137 | 312.623 | 266.105 | 272.572 | 284.654 | 320.933 | 270.675 | 1.054 | 1.079 |
| 256×1024×128 | 14/14 | 19.361 | 18.408 | 17.286 | 20.350 | 28.668 | 17.893 | 16.025 | 0.953 | 1.117 |
| 1024×128×256 | 14/14 | 17.853 | 19.547 | 18.508 | 26.270 | 28.655 | 18.169 | 16.500 | 0.674 | 1.007 |
| 127×193×61 | 14/14 | 8.491 | 7.961 | 7.127 | 16.915 | 26.997 | 7.780 | 8.861 | 0.482 | 1.135 |
| 513×257×129 | 14/14 | 29.488 | 21.211 | 24.424 | 35.043 | 34.002 | 20.068 | 20.607 | 0.842 | 1.222 |

Artifacts unchanged: True.

Raw samples, configs, ordering, failed paths and provenance: [results.json](results.json).
