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
| 2048×2048×2048 | 14/14 | 3253.892 | 4207.540 | 2882.614 | 3012.126 | 3196.295 | 3596.367 | 3143.822 | 1.103 | 1.126 |
| 4096×4096×4096 | 14/14 | 30026.438 | 35439.313 | 27701.459 | 30651.021 | 27377.125 | 32445.667 | 29807.812 | 1.004 | 1.098 |
| 8192×8192×8192 | 14/14 | 461685.979 | 369681.646 | 412922.500 | 295216.667 | 278915.208 | 399020.292 | 300684.375 | 1.585 | 1.115 |
| 256×11008×4096 | 14/14 | 6529.361 | 6214.403 | 7155.625 | 4101.250 | 3896.336 | 6004.903 | 4275.901 | 1.562 | 0.946 |
| 4096×4096×11008 | 0/14 | INCOMPLETE | — | — | — | — | — | — | — | — |
| 2049×4097×1025 | 0/14 | INCOMPLETE | — | — | — | — | — | — | — | — |

## Separate GPU command-buffer controls

These no-counter GPU intervals include work and gaps inside completed command buffers, not host encoding or waits. They are not isolated-kernel timestamps. Handwritten MPP uses its existing direct command-buffer timer; other paths use the shared timing helper's uninstrumented control phase. Compute-pass counters remain diagnostic JSON only. Do not subtract independently measured GPU/host medians to infer dispatch cost.

| M×N×K | Valid rounds | Tile→MPP GPU µs | TIRx/TVM GPU µs | Hand MPP GPU µs | MPS GPU µs | Torch GPU µs | TIRx→MPP/TVM GPU µs | TIRx→MPP views/TVM GPU µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048×2048×2048 | 14/14 | 3200.658 | 4088.627 | 2832.007 | 3017.902 | 2989.184 | 3712.788 | 3011.923 |
| 4096×4096×4096 | 14/14 | 29977.667 | 39555.750 | 27168.667 | 30148.000 | 27074.396 | 33065.458 | 29273.125 |
| 8192×8192×8192 | 14/14 | 476116.729 | 438198.104 | 412663.042 | 248049.833 | 237612.208 | 421804.229 | 271091.708 |
| 256×11008×4096 | 14/14 | 6581.181 | 6127.958 | 6902.250 | 3944.170 | 3809.634 | 5718.271 | 4264.245 |
| 4096×4096×11008 | 0/14 | INCOMPLETE | — | — | — | — | — | — |
| 2049×4097×1025 | 0/14 | INCOMPLETE | — | — | — | — | — | — |

Artifacts unchanged: True.

Raw samples, configs, ordering, failed paths and provenance: [results.json](results.json).
