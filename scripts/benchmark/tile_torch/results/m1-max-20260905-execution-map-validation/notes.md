# Execution-map and cost-policy implementation validation

Date: 2026-09-05, Asia/Shanghai. Branch: `codex/tile-programming-design`.
Apple M1 Max, macOS 26.6.2 arm64.

## Implemented and checked

- Automatic GPU program/element coordinate fusion, with speculative guarded
  input forwarding committed only when the complete realization is proved.
- Separate reduction collaboration and packing, backend-overridable cost
  coefficients/row objective, and bounded ordered stripe unrolling.
- Exact packing/thread/unroll requests fail if unrealizable, including a
  worker-scope conflict or a stripe budget exceeded by an exact width.
- Replayable CLI metadata, bounded Cartesian JIT search, independent row-count
  and width cases, full-output candidate/winner validation and frozen replay.

```bash
cmake --build cmake-build-tirx -j 8
ctest --test-dir cmake-build-tirx -R '^test_tile_' --output-on-failure -j 1
python3 -m unittest discover -s scripts/benchmark/tile_torch -p 'test_*.py'
```

The final submitted-source full build succeeds and all 32/32 Tile tests pass
in 95.99 seconds (30 unit tests, two integration tests). Coverage includes
physical CPU/Metal execution, Tile/XIR, native Runtime, memory, pipeline and
matrix paths. The Python benchmark contract suite passes 72/72.

As in the previous checkpoint, the user's unowned `metal::mem_flags(2)` edit
conflicts with two branch-source assertions that expect `3`. It was temporarily
set to the branch value for the full submitted-source run, then immediately
restored to `2`. It is not included in this patch, no assertion was weakened,
and the normal workspace binary is rebuilt with the user's value. Neither
new element-grid nor reduction realization calls that cooperative-MMA helper.

New executable tests cover ragged 2-D tiles, negative read origins and zero
fill, non-dyadic nonlinear arithmetic, explicit worker preservation, a moved
in-place snapshot, odd packing and inactive tail rows, partial unroll tails,
custom policy selection, invalid policy scores, target/resource constraints
and preservation of the caller's coefficient prior.

The repository clangd syntax checker was run on all seven changed C++ source
files with `cmake-build-tirx/compile_commands.json`; the changed-line formatter
and `git diff --check` are also part of final QA. No RTTI or TVM type is added
to the cost-policy interface. After restoring the user's fence value and
rebuilding, the focused CPU/Metal execution and planner cohort passes again
(3/3).

The Sphinx HTML build succeeds with 24 pre-existing non-Tile warnings and no
Tile cross-reference or syntax warnings. The rendered HTML contains the new
mapping/policy sections and updated status values. Regenerating readable
benchmark Markdown leaves every raw JSON byte unchanged; SHA256 verification
passes for all 123 archived Metal sources in this checkpoint's run folders.
The non-artifact staged diff is whitespace-clean. Raw compiler-generated
Metal keeps its original trailing blank lines so its content-addressed source
hashes remain valid; those archived bytes are excluded from the whitespace
check, not reformatted.

## Performance evidence and QA

The [element-grid replay](../m1-max-20260905-element-grid-replay/notes.md)
contains 32 complete native outputs. The
[reduction mapping replay](../m1-max-20260905-reduction-joint-map-replay/notes.md)
contains 80. A separate
[norm/loss smoke cohort](../m1-max-20260905-reduction-unroll-generalization/results.json)
checks all 16 RMSNorm/LayerNorm/residual-LayerNorm/cross-entropy outputs at
unroll factor four. No benchmark ran concurrently with compilation or tests.

Independent QA recomputes all headline medians and paired ratios from raw
rows, verifies no duplicate `(case, round, variant)` records, confirms every
row's complete validity and checks the replay's before/after fingerprint
result. Generated Metal is archived by SHA256. The noisy unchanged-policy
control and small regressions remain in the report; search minima are not
substituted for independent repeated measurements.

## Remaining boundaries

Element fusion currently supports one compact store domain, one linear
ordering and no explicit execution binding. A general coalescing/layout
permutation solver, arbitrary custom output strides and multi-effect fusion
remain future work. Reduction programs requiring several SIMD groups still
occupy a whole threadgroup; packing such programs needs a converged barrier
protocol for inactive tails. The default cost prior has not gained calibrated
traffic, spill or whole-machine occupancy terms. Backend policy hooks provide
the separation needed to add them; they do not claim that calibration is done.
XIR's policy surface and performance are not changed by this checkpoint.
