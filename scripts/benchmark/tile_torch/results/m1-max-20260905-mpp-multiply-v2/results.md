# Tile lowering comparison

FP32 C=A×B; compact row-major, no transpose or prepacking. All complete outputs use identical inputs and an FP64 oracle (atol=rtol=1e-4). Native and handwritten MPP use the same fixed configuration; TIRx has a separate recorded schedule. No search or minimum-of-rounds selection occurs here.

All table values are synchronized host-wall batched throughput (µs/call), including each runtime's dispatch, encoding/submission and synchronization overhead. Compilation, allocation and transfers are excluded. They are NOT GPU kernel times. Raw MPS/MPP GPU intervals are retained only in JSON; no cross-metric ratios are computed. Torch is its default eager MPS path, not a claim that it uses the direct MPS benchmark's kernel.

14 rounds balance positions and pairwise precedence. Fewer rounds are explicitly an unbalanced smoke/exploration run. No failed or slow row is removed. Ratios below are paired round medians, not confidence intervals.

Order balanced: True. Rounds: 14.

Native/hand MPP and TIRx/Luisa use fast math off; TIRx/Luisa fast explicitly uses it on. TVMx's current Metal runtime hardcodes fast math on. Optional TIRx Runtime controls must have identical generated Metal source hashes and matching threadgroup widths. Compiler language/resource options and submission APIs can still differ; this is not an isolated GPU execution-overhead measurement.

Optional TIRx→MPP is the patched TVM Metal code generator using non-owning memory inputs, not the native MPP emitter. It reuses the frozen TIRx geometry; recorded planner costs still describe the SIMD-group reference family, not MPP's internal instruction count or register use.

Optional TIRx→MPP views enables proven read-only snapshot forwarding, with a separately frozen schedule. It is not a same-geometry ablation unless the recorded schedules match; original TIRx and non-forwarding MPP remain controls.

MPP-view subgroup-fence policy override: reported. The default is retention; requesting elision still requires a reported whole-group independence proof. This policy is not assumed profitable.

| M×N×K | Valid rounds | Tile→MPP | TIRx/TVM | Hand MPP | MPS | Torch | TIRx→MPP/TVM | TIRx→MPP views/TVM | Native/MPS | Native/hand MPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32×32×32 | 14/14 | 4.577 | 4.875 | 3.052 | 11.148 | 27.161 | 4.696 | 3.163 | 0.411 | 1.513 |
| 128×128×128 | 14/14 | 9.273 | 6.709 | 5.443 | 17.005 | 27.298 | 5.984 | 5.400 | 0.545 | 1.707 |
| 512×512×512 | 14/14 | 48.969 | 52.218 | 46.866 | 52.376 | 47.744 | 48.285 | 42.615 | 0.936 | 1.046 |
| 1024×1024×1024 | 14/14 | 287.194 | 312.544 | 266.476 | 272.694 | 284.805 | 341.674 | 273.996 | 1.053 | 1.078 |
| 256×1024×128 | 14/14 | 19.377 | 18.839 | 17.304 | 20.424 | 28.767 | 17.916 | 16.258 | 0.949 | 1.121 |
| 1024×128×256 | 14/14 | 18.996 | 19.646 | 18.515 | 28.097 | 28.758 | 18.789 | 17.230 | 0.674 | 1.026 |
| 127×193×61 | 14/14 | 8.579 | 8.656 | 7.693 | 18.191 | 27.417 | 8.537 | 6.317 | 0.472 | 1.121 |
| 513×257×129 | 14/14 | 31.115 | 22.322 | 25.599 | 37.170 | 33.993 | 20.737 | 20.739 | 0.839 | 1.212 |

Artifacts unchanged: True.

Raw samples, configs, ordering, failed paths and provenance: [results.json](results.json).
