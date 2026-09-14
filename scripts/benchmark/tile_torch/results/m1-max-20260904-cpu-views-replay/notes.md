# CPU immutable-input views: fixed-geometry A/B, 2026-09-04 UTC

Removing proved immutable input snapshots improves all four pairs for the five
regular shapes from 128³ upward: paired median speedups are 1.45–1.76× over the
current snapshot lowering. This is **not Torch/BLAS parity**. Keep forwarding
default-off: tiny-shape timing is noisy, and the two ragged GEMMs still take
the snapshot fallback.

| M×N×K | Snapshot µs | Input-view option µs | Paired speedup median [range] |
|---|---:|---:|---:|
| 32³ | 4.666 | 4.931 | 1.010 [0.754, 1.217] |
| 128³ | 24.099 | 14.992 | 1.636 [1.267, 2.094] |
| 512³ | 988.463 | 617.067 | 1.507 [1.461, 1.735] |
| 1024³ | 8215.518 | 5605.283 | 1.453 [1.386, 1.530] |
| 256×1024×128 | 331.239 | 191.638 | 1.759 [1.589, 1.838] |
| 1024×128×256 | 265.420 | 190.182 | 1.485 [1.225, 1.778] |
| 127×193×61 | 35.034 | 33.504 | 0.981 [0.918, 1.401] |
| 513×257×129 | 377.179 | 392.023 | 0.937 [0.860, 1.111] |

Times are medians of per-round p50s; ratios are paired within rounds, so their
median need not equal the ratio of displayed times. Ranges are observations,
not confidence intervals. The candidate is slower in 2/4 tiny-shape pairs,
2/4 first-ragged pairs, and 3/4 second-ragged pairs. No round is removed.

## Actual lowering, not just the flag

The 512³ [reference LLVM](sources/3a07a17e4771a15cbc813f65b07f457042da3ca9208f81b1adac8f5af031569c.ll)
has 1024-float and 256-float staging arrays (including pipeline versions).
The [candidate LLVM](sources/8951e3f7562045cb12ea6e932750738df7ee89fa51ed7d3b45adc054c69c09ae.ll)
has neither array, and its ordered K loop still has four 16-float accumulators
sharing one B vector load. No LLVM text is rewritten and no GEMM library is
called by the Tile implementation.

**Important limitation:** both ragged GEMMs retain those staging arrays. In
every one of their four paired rounds, raw LLVM differs only in allocation-
identity TBAA strings; instructions are unchanged. Their slower paired timing
must not be attributed to extra guards introduced by forwarding, since this
realization did not forward their A/B snapshots. These are retained no-op
controls demonstrating timing variability and incomplete optimization coverage.

LLVM forwarding can preserve a proved lazy guarded read, including nonzero
padding; the new positive native-IR test verifies actual allocation removal.
That does not establish coverage of general ragged GEMM domains. Mutable
sources/indices/predicates/fills, aliases, explicit Memory, escapes, and
unproved consumer domains retain snapshots. A bounded indirect consumer test
also exercises the current conservative fallback.

## Controls and verification

- One binary/library set; both variants use 4×16×32 tiles, worker binding,
  pipeline window 2, automatic vectorization, 8192-byte stack budget, and
  64 logical SIMD-pack lanes. Only input forwarding changes. There is no
  parameter search or per-shape dispatch table.
- Geometry comes from the old CPU pilot plan, but none of its timing scores
  is reused. Both variants and Torch are freshly captured/timed in balanced
  orders, with rotating case order.
- Four rounds × eight shapes × two policies: 64 Tile and 64 Torch full outputs,
  **128/128 valid**, 30,043,136 checked elements, maximum absolute error zero.
  Deterministic dyadic FP32 inputs use a full FP64 oracle. This is not exhaustive
  floating-point validation; C++ tests additionally use non-dyadic inputs,
  nonzero C, transposes, repeated calls, tails, and cancellation-sensitive K.
- Seven samples × 30 ms, 200 ms warmup, requested eight CPU threads. Warm host
  wall time includes dispatch/internal workspace handling. JIT, allocations,
  transfers, and cold-call phases are separate. No build, test, or profiler
  ran during timing; ordinary OS/user activity is not controlled.
- Both complete builds succeeded. Both selected CPU/Metal CTest cohorts are
  **23/25**, not green: the existing Metal fence assertions expect
  `mem_flags(3)` while the separate worktree edit emits 2. Those assertions
  were not weakened. See [patched](ctest-patched-final.log) and
  [original](ctest-original-final.log) logs. All added CPU view tests pass.
- Python harness: 54/54 tests; changed C++ translation units pass the source
  checker. All 19 recorded executable/library paths were stable and
  independently rehashed after timing. All 64 archived raw LLVM hashes were
  independently verified; source is neither normalized nor rewritten.
- Apple M1 Max, macOS 26.6.2, Torch 2.14.0. Full commands, compiler/runtime
  identities, dirty revision, schedules, and samples are in [results.json](results.json).

The independent [six-order Torch/BLAS comparison](../m1-max-20260904-cpu-views-system/notes.md)
measures the remaining library gap. The [other-operator smoke run](../m1-max-20260904-cpu-views-ops/notes.md)
checks add, sum, and softmax without claiming a repeated performance result.
Original TIRx CPU/Metal, TIRx→MPP, native MPP, hand MPP, MPS, and Torch remain
separate comparisons. This CPU run does not retime the existing
[seven-way Metal report](../m1-max-20260904-subgroup-sync-lowerings/notes.md).
