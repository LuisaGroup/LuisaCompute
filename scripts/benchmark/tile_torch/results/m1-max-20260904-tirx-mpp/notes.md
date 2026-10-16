# Independent TIRx MPP realization: validated, not a universal replacement

The optional TVM MPP codegen is numerically valid on the tested subset. It
reduces paired median host-wall time on seven of the eight frozen workloads,
but makes 1024³ **9.2% slower than the original TIRx SIMD-group path** in all
twelve rounds. Keep the original lowering and do not change automatic
selection or import its cost coefficients into the MPP family.

For 1024³ the per-path medians are native MPP **293.304 µs**, original TIRx
**318.676 µs**, handwritten MPP **271.393 µs**, MPS **277.756 µs**, Torch
**290.453 µs**, and TIRx→MPP **347.856 µs**. Native/MPS paired median time
ratio is **1.055783**: the approximately 5.6% native gap is not closed.
These are synchronized device-resident **host-wall batched times**, not GPU
kernel durations. See the [complete table](results.md) and [raw samples](results.json).

## What was held fixed

- Apple M1 Max, macOS 26.6.2; Torch 2.14.0, NumPy 2.5.2. Exact versions,
  commands, removed environment overrides and compiler/library hashes are
  in the report metadata.
- Six independent implementations, eight shapes and twelve counterbalanced
  rounds. Seven samples per path, approximately 30 ms per throughput sample,
  200 ms warmup. No search, discarded slow rows, or reused historical times.
- **576/576 full outputs valid**. The deterministic benchmark inputs are
  dyadic; separate kernel tests use non-dyadic inputs and changed buffers.
  All 96 TIRx→MPP outputs were exactly equal to this benchmark's FP64 oracle.
- Both TIRx paths recapture the same DSL and use the frozen TIRx schedule.
  Every pair's complete recorded plan is identical after excluding the
  `metal_mpp` identity bit: geometry, shared storage, barrier sites, copy
  batches and reference costs all match. Private pipeline prefetch is zero
  in this cohort. This is not a claim of identical generated ISA or MSL
  compilation policy: the new path uses MSL 4 and an opaque MPP operation.
- Native and handwritten MPP use their own previously selected matching
  atom/cohort configuration. They do not share the TIRx execution schedule.
- No build, test or profiling process ran during timing. All 22 recorded
  executables/shared libraries were rehashed after completion and unchanged.
  The 16 content-addressed TIRx sources were rehashed independently and match.

The two frozen plan inputs are
[`mpp-search`](../m1-max-20260904-mpp-search/results.json) and
[`joint-search`](../m1-max-20260904-joint-search/results.json); only their
configurations, not timings, are reused here. The TVM source base is
`c7b458e946bc4266915da582457476bdcd9705ae`, with tvm-ffi
`12dbf053b3d9ba4ebd9da3123b1aeca79cf74229`. The isolated build used the
[native C++ extension](../../../../../src/tile/bridge/tirx/patches/README.md),
whose patch SHA-256 is
`7ee767cb77334897b12ba2dcb485f87304f77ac1e1e361bed044b82636885c44`.
The ordinary installed TVM compiler was not replaced.

## Paired comparison, not ratio of independent minima

The ratio is the median of twelve within-round TIRx→MPP/original-TIRx time
ratios. Below one means less time. These are descriptive paired medians, not
confidence intervals or guaranteed speedups on another GPU/compiler.

| M×N×K | TIRx→MPP / original TIRx | MPP slower rounds |
|---|---:|---:|
| 32×32×32 | 0.964049 | 0/12 |
| 128×128×128 | 0.891631 | 0/12 |
| 512×512×512 | 0.929895 | 0/12 |
| 1024×1024×1024 | 1.091519 | 12/12 |
| 256×1024×128 | 0.952717 | 0/12 |
| 1024×128×256 | 0.953042 | 0/12 |
| 127×193×61 | 0.980535 | 2/12 |
| 513×257×129 | 0.929059 | 0/12 |

## Correctness and retained limitations

Both Luisa configurations completed a full build before tests. The patched
TVM also completed a full build. The selected CPU/Metal Tile, native Runtime,
and system regression cohort passed **23/23 in each configuration**. The
benchmark driver's unit suite passed **43/43**.

MPP tests cover nonzero and literal C, repeated calls with changed data,
all sixteen independent A/B/C/D row/column-major combinations, distinct
accumulator fragments with a nonzero fragment index, global/shared inputs,
K=8/24 and K tails, pipeline versions, ragged output, Luisa buffer offsets,
guards, resource moves and disjoint aliases. Unsupported 8×8 rectangles,
capacity overflow, fragment bounds, stride and operand-role mismatches are
rejected. The unpatched build verifies the explicit extension capability
error. No Python compiler boundary or native-emitter fallback is used.

Two existing Metal fence-string tests were deliberately not counted in those
23-test cohorts: `test_tile_tirx_cooperative_metal` and
`test_tile_tirx_memory_metal`. A separate post-benchmark run confirmed their
existing failures at `test_tirx_cooperative.cpp:100` and
`test_tirx_memory.cpp:168`. They expect `mem_flags(3)`, while the pre-existing
worktree change emits `mem_flags(2)`. Neither that change nor those assertions
was overwritten here. This report does **not** claim a completely green
repository suite or establish that the weaker fence is generally sufficient.

Two representation traps were fixed before accepting any performance result:
MPP input transposition needed descriptor flags, and column-major C/D
transfers needed explicit cooperative-coordinate-to-memory mapping on the
tested SDK. Simply exchanging inline-tensor strides produced wrong results.
The final tests keep those cases as regressions.

## Consequence for the planner

The 1024³ source has 64×64 outer tiles, 256 threads split into a 2×4 subgroup
grid, 32×16 MPP output per subgroup, and **32 outer K steps of width 32**.
It retains shared A/B snapshots and two group-barrier sites per K step; the
cooperative accumulator remains live across the loop. This is visible in
the [captured source](sources/dd9c9558b7034ab150dd6ce5e6d808dd28ed93ece06bccbe71bc4f036af5c29e.metal).
By contrast, native/handwritten MPP hand the whole K dimension to their MPP
operations. The experiment does not isolate which of these choices, MSL
compilation, or internal MPP scheduling causes the difference.

The next bounded search must therefore key cost/legality by **implementation
family, memory forwarding versus staging, local shape, participation/cohort,
K granularity, recurrence and edge mode**, not just substitute an intrinsic
while keeping SIMD-group coefficients. Test larger K operations or proved
readonly-view forwarding against shared staging; keep real load/effect,
resource and bounds proofs. Unknown internal MPP registers/instruction counts
remain unknown. Validate candidates first, then freeze and replay on multiple
sizes including held-out tails. Do not make this benchmark's size table a
production dispatch rule.
