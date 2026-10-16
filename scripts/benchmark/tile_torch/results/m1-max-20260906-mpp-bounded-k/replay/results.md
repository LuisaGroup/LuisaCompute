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
| 128×128×61 | 14/14 | 9.660 | 13.561 | 8.181 | 10.098 | 29.177 | 13.262 | 8.845 | 0.969 | 1.194 |
| 1024×1024×1537 | 14/14 | 512.048 | 1277.968 | 491.592 | 445.931 | 456.918 | 1188.600 | 521.213 | 1.150 | 1.042 |
| 4096×4096×11008 | 14/14 | 74207.958 | 74431.187 | 73260.834 | 51991.499 | 50808.562 | 75908.459 | 57790.291 | 1.425 | 1.009 |
| 8192×8192×8192 | 14/14 | 417633.667 | 257562.688 | 379300.270 | 163072.813 | 170370.604 | 283810.583 | 193842.396 | 2.522 | 1.101 |

## Separate GPU command-buffer controls

These no-counter GPU intervals include work and gaps inside completed command buffers, not host encoding or waits. They are not isolated-kernel timestamps. Handwritten MPP uses its existing direct command-buffer timer; other paths use the shared timing helper's uninstrumented control phase. Compute-pass counters remain diagnostic JSON only. Do not subtract independently measured GPU/host medians to infer dispatch cost.

| M×N×K | Valid rounds | Tile→MPP GPU µs | TIRx/TVM GPU µs | Hand MPP GPU µs | MPS GPU µs | Torch GPU µs | TIRx→MPP/TVM GPU µs | TIRx→MPP views/TVM GPU µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128×128×61 | 14/14 | 9.148 | 12.933 | 7.968 | 8.938 | 13.889 | 12.603 | 8.592 |
| 1024×1024×1537 | 14/14 | 500.481 | 1256.384 | 482.893 | 433.547 | 437.150 | 1164.593 | 511.423 |
| 4096×4096×11008 | 14/14 | 73837.125 | 90557.271 | 72699.500 | 53208.125 | 54887.771 | 87732.792 | 60221.083 |
| 8192×8192×8192 | 14/14 | 412910.000 | 271885.438 | 378928.313 | 220077.896 | 210055.750 | 291378.813 | 241151.958 |

Artifacts unchanged: True.

Raw samples, configs, ordering, failed paths and provenance: [results.json](results.json).
