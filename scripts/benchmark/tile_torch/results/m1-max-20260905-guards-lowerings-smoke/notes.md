# Seven-path Metal maintenance check after the CPU guard repair

**112/112 complete outputs pass** across eight shapes and two rounds for:
native Tile→MPP, original Tile→TIRx, handwritten MPP, direct MPS, eager Torch,
staged TIRx→MPP, and read-only-view TIRx→MPP. The full FP64 oracle checks
26,287,744 elements, with maximum absolute error zero on the dyadic inputs.

This is explicitly an **unbalanced smoke run**, not a new claim of performance
parity or improvement. Seven paths require fourteen rounds to balance all
positions and pairwise precedence; only two ran here. All timings, including
slow rows, remain in [results.md](results.md) and [results.json](results.json).
The earlier [fourteen-round comparison](../m1-max-20260904-subgroup-sync-lowerings/notes.md)
remains the stronger performance evidence; it must not be pooled with this
separate session or selected against these two rounds.

## Independence and unchanged device code

- All **48** TIRx Metal source records were independently rehashed. For every
  shape and both rounds, original TIRx, staged TIRx→MPP and forwarding
  TIRx→MPP source hashes exactly match the previous fourteen-round report.
  There are 23 unique archived Metal files. The CPU guard repair therefore
  did not alter these tested TIRx device programs. Native/handwritten source
  identity is not inferred from this check; their binaries are fingerprinted
  separately and their full numerical outputs are validated.
- All **22** executable/compiler/runtime library paths remained unchanged and
  were independently rehashed after timing, as were all three frozen plans.
- Original TIRx and staged TIRx→MPP retain their separate frozen schedules;
  native/handwritten MPP share their own MPP plan. The view path uses the same
  explicit, default-off subgroup-fence elision candidate as the previous
  report. This is neither an interchangeable lowering alias nor a new policy
  that removes fences without the independence proof.
- Native/handwritten MPP have fast math off; TVM Metal has it on. Those compiler
  policies are disclosed, not claimed equivalent. Planner costs still identify
  `simdgroup_reference_geometry`, not measured MPP internal register counts.
- Five samples target 20 ms after 100 ms warmup. Timings are synchronized
  device-resident host wall, including each Runtime's submission overhead,
  not GPU kernel times. No build, test or profiler ran alongside measurement.

Both selected CPU/Metal CTest cohorts remain **23/25** due to the two existing
Metal fence-flag assertions, not due to the new CPU guard cases. Logs and
the controlled CPU improvement are in the
[guard-repair A/B](../m1-max-20260905-cpu-guards-replay/notes.md). TIRx lowering
remains independently maintained and compared with MPP, MPS and Torch.
