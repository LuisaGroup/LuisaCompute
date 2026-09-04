# CPU TIRx storage realization: useful, not the missing GEMM microkernel

The bounded stack candidate improves the paired median on all eight frozen
GEMM shapes, but 512³ regresses in two of four pairs. The policy remains
**opt-in, default zero**. These are improvements over our previous lowering,
not over PyTorch. The CPU performance goal is not closed.

## Controlled comparison

Both variants use the same C++ Tile kernel, compact FP32 row-major tensors,
4×16×32 tiles, worker execution, a two-stage scheduling window, CPU automatic
element SIMD, and eight requested threads. There is no parameter search,
packing outside the measured kernel, BLAS replacement, or precision change.
The candidate adds only `max_cpu_stack_bytes=8192`; the reference uses the
pre-change executable with its own adjacent libraries frozen at
`/tmp/luisa-cpu-lowering.fk1jC1/baseline-bin`. A separate loader diagnostic
confirmed that bundle supplies its Luisa libraries; both use the same external
TVM compiler/runtime. The reference has no new stack-budget CLI argument.

Four rounds alternate reference/candidate and native/Torch order and rotate
shapes. Each measurement uses seven 30 ms samples and a 200 ms warmup.
Times are synchronized host wall time, including dispatch; JIT, allocation of
input/output tensors, and upload/download are excluded. Kernel-internal
workspace allocation is included. Ratios below are medians of paired round
ratios; ranges are observed ranges, not confidence intervals.

| M×N×K | Previous TIRx µs | Stack candidate µs | Previous / candidate [range] |
|---|---:|---:|---:|
| 32³ | 6.507 | 6.132 | 1.058 [1.012, 1.765] |
| 128³ | 69.788 | 43.722 | 1.598 [1.444, 2.124] |
| 512³ | 2141.448 | 1908.219 | 1.144 [0.935, 1.334] |
| 1024³ | 14076.709 | 12351.330 | 1.140 [1.020, 1.248] |
| 256×1024×128 | 740.390 | 492.523 | 1.495 [1.451, 2.017] |
| 1024×128×256 | 638.819 | 515.912 | 1.229 [1.145, 1.770] |
| 127×193×61 | 90.795 | 58.891 | 1.542 [1.072, 1.708] |
| 513×257×129 | 739.177 | 529.624 | 1.390 [1.276, 1.851] |

## Evidence and limits

- All 64 native and 64 Torch outputs pass the full FP64 oracle: 30,043,136
  output elements, maximum absolute error zero on the deterministic dyadic
  inputs. This is not a claim of zero error on arbitrary FP32 data.
- Every one of the 32 reference LLVM modules has four static workspace
  allocation call sites per logical work item; all 32 candidates have zero.
  LLVM can eliminate the accumulator arrays into SSA values, but this does
  not prove the absence of machine-register spills. Operand snapshots and
  the separate per-row K loops still exist.
- All 34 executable/adjacent-library/external-TVM artifact paths passed both
  the runner's before/after check and an independent rehash. All 64 archived
  LLVM files were independently hash-checked. LLVM TBAA names contain
  capture-specific buffer identities, so fresh equivalent JITs can have
  different raw hashes; the archived files are not normalized or rewritten.
- The new non-dyadic C++ numerical regressions cover ragged/transposed GEMM,
  repeated changed inputs, aliased external tensors, exact/cumulative aligned
  budgets, disabled planning, explicit Memory with/without placement, and
  address escape. A full-suite failure exposed an empty-Memory-marker
  cleanup regression; it was fixed before these measurements.
- Both full build trees finish successfully. Final 25-test cohorts each
  pass 23 tests: the pre-existing Metal fence-string assertions remain in
  `test_tile_tirx_cooperative_metal` and `test_tile_tirx_memory_metal`. They
  expect `mem_flags(3)` while the unrelated worktree edit emits `2`; neither
  the assertions nor that edit were weakened/overwritten here. All CPU
  tests, including the empty-Memory regression, pass. The 49 Python harness
  tests also pass.
- Twelve additional CPU add/sum/softmax cases pass full output checks with
  the candidate, including ragged extents and temporaries exceeding its
  budget. Their [single-pass timings](../m1-max-20260904-cpu-stack-ops/results.md)
  are exploratory, not a controlled speedup claim.

The legal storage choice is not a calibrated profitability model. Even after
it, 1024³ takes roughly 12 ms versus roughly 1 ms for Torch in this A/B session.
Next structural questions are multi-row reuse of B, a bounded register
microtile, recurrence residency across K tiles, and task/cache partitioning.
These need separate toggles and balanced validation; a shape lookup table
would not establish a cost model.

See [raw samples, policies, commands and hashes](results.json), the
[six-order TIRx/Torch/BLAS comparison](../m1-max-20260904-cpu-stack-system/results.md),
and the earlier independent
[seven-path Metal comparison](../m1-max-20260904-subgroup-sync-lowerings/notes.md).
The latter is a prior measurement, not a rerun of GPU performance after this
CPU-only change. Native MPP, original TIRx, TIRx→MPP, MPS and Torch remain
separate selectable comparison paths.
