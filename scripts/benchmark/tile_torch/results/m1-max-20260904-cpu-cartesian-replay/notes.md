# CPU Cartesian packing: fixed-geometry A/B, 2026-09-04 UTC

The new opt-in realization improves seven of eight paired median ratios over
single-row packing, but **not uniformly**. The 32³ paired median is 0.940×
(slower in three of four pairs), and 513×257×129 regresses in one pair. Keep the
default lane budget at 16; this is not evidence of Torch/BLAS parity.

| M×N×K | Single-row µs | Cartesian µs | Paired speedup median [range] |
|---|---:|---:|---:|
| 32³ | 4.588 | 4.584 | 0.940 [0.909, 1.194] |
| 128³ | 32.046 | 25.690 | 1.397 [1.028, 1.507] |
| 512³ | 1414.340 | 1053.471 | 1.365 [1.195, 1.611] |
| 1024³ | 11187.410 | 8083.594 | 1.331 [1.269, 1.459] |
| 256×1024×128 | 410.364 | 297.457 | 1.408 [1.255, 1.453] |
| 1024×128×256 | 418.689 | 257.752 | 1.625 [1.227, 1.796] |
| 127×193×61 | 44.327 | 38.126 | 1.116 [1.059, 2.105] |
| 513×257×129 | 482.570 | 403.025 | 1.156 [0.829, 1.263] |

Times are medians of per-round p50 synchronized host-wall timings. Paired
ratios are calculated within each round, so their median need not equal the
ratio of the displayed medians. Ranges are observations, not confidence
intervals. All rounds are retained, including the slow first 32³ pair.

## What changed

One binary/library set, identical 4×16×32 tile, worker binding, pipeline window
2, automatic vectorization, and 8192-byte compiler-temporary stack budget.
Reference: logical vector budget 16. Candidate: 64. No tile/shape search or
per-size decision table is used. The common stack override prevents the
preceding storage optimization from being counted again as a packing gain.

Two innermost independent element axes become Cartesian register packs.
Sequences and rectangular serial loops are distributed across separate row
vectors. For the 512³ probe, the generated inner LLVM loop now has four independent
16-float accumulator PHIs and four FMA calls sharing one B vector load,
instead of four separate K loops, each loading B. Serial K order and arithmetic
expressions are unchanged. This is a loop/layout realization, not an external
GEMM call or a generated-source rewrite. Compiler-temporary copy loops also
use the same generic policy; timing measures the complete policy, not an
isolated FMA instruction.

The first prototype flattened `(row,column)` into one 64-lane vector. Native
TIRx emitted extensive scalar address extraction and vector assembly. Its
one-pass 512³ probe was approximately 4.95 ms versus 1.83 ms single-row; these
unbalanced diagnostic timings are **not** used in the table. That emission was
removed. [Its raw LLVM](pack64-512.ll) is retained as negative evidence. All
262144 outputs of that rejected probe, the single-row probe, and the final
jammed-row probe matched the FP64 oracle exactly.

## Verification and provenance

- Four counterbalanced rounds, eight shapes, two policies: 64 native and 64
  separately timed Torch outputs, **128/128 valid**, 30,043,136 checked elements.
- Seven samples × 30 ms, 200 ms warmup, requested eight CPU threads. Timing
  includes dispatch and internal workspace handling; JIT, input/output setup,
  and transfers are recorded separately, outside warm timing.
- Same deterministic dyadic FP32 inputs and full FP64 oracle for both paths;
  maximum absolute error zero. This input family is not an exhaustive
  floating-point accuracy test.
- Additional C++ regressions use sinusoidal inputs, nonzero C, repeated calls,
  transposed A/B, both row/column tails, larger/multiple row packs, windows 1/2,
  workspace/stack storage, K-order cancellation, nonzero index minima,
  element-dependent K bounds, conditional recurrences, non-MMA subtraction,
  and in-place C/D without a noalias promise.
- Full builds completed for both original and MPP-extended TVM installations.
  Both selected CTest cohorts are **23/25**, not green: the same preexisting
  Metal fence assertions expect `mem_flags(3)`, while the separate worktree
  edit in `cooperative.cpp` emits 2. No assertion was weakened. See
  [patched log](ctest-patched-final.log) and [original log](ctest-original-final.log).
- Python harness: 51/51 tests. Source checker passed the changed C++ units.
- All 19 recorded executable/shared-library artifact paths remained unchanged
  and were independently rehashed after all timing;
  64 raw generated LLVM modules are archived by SHA256. Raw LLVM may differ
  between equivalent JITs because TVM's TBAA names include allocation identities.
  Archived source is not normalized or rewritten.
- Apple M1 Max, macOS 26.6.2, generic LLVM CPU target. Exact Torch build, compiler
  identities, frozen source plan, environment, commands, and samples are in
  [results.json](results.json). No build, test, or profiler ran during timing;
  ordinary OS/user activity is not controlled.

Original TIRx CPU/Metal, independent TIRx→MPP, and native Metal MPP remain
separate paths. This CPU change does not replace or retune the seven-way
[Metal/MPS/MPP/Torch comparison](../m1-max-20260904-subgroup-sync-lowerings/notes.md).

The [fresh six-order Torch/BLAS follow-up](../m1-max-20260904-cpu-cartesian-system/notes.md)
quantifies the remaining library gap. It does not use these A/B measurements
as its comparison baseline.
