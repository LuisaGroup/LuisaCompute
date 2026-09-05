# Reduction access demand and joint resource search

## Technical summary

The TIRx reduction planner now exposes conservative global/private payload
read/write demand to backend cost policies. Staged/JIT tuning can jointly
enumerate input reload/cache and execution mappings, preserving invalid
candidates and freshly recompiling the selected winner. This checkpoint is
an implementation and regression result, **not a calibrated cost model or a
new performance claim**. The prior fixed-width input-cache replay remains
separate evidence.

## Feature contract and scope

`ReductionCandidate` and `GroupPlan` carry an availability flag, per-program
demand and per-worker demand. The latter sums the longest worker stripe for
each distributed domain, rounded independently by its ownership map.
Identical buffer loads count once per statement/expression; later statements
and phases are separate. Both lazy branches and zero-filled tails count
potential demand. Scalar setup, carry variables and collective scaffolding
are outside this payload metric. Unsupported constructs mark the complete
feature unavailable with zero values. No physical cache, DRAM, register or
spill count is inferred.

Optional global/private byte coefficients are finite, nonnegative and zero
by default. The legacy scalar/collective/setup score is therefore unchanged.
Backend policies can use the complete facts without changing the bridge's
ownership, immutability or cumulative resource-budget checks. The coefficients
used in unit tests are synthetic arithmetic checks, not M1 Max calibration.

`--tune-reduction-input-caches reload,cache` adds a finite Cartesian dimension
to the existing JIT search. Every configuration is separately captured and
compiled; invalid configurations retain their reason, and a fresh capture
validates the winner without publishing its search minimum as acceptance
evidence. The old cost score does not prune this measured search. Cache and
mapping defaults are unchanged.

## Verification

The complete `cmake-build-tirx` tree built successfully before the tests.
All five changed C++ translation units passed `scripts/check_cpp_syntax.py`
against that compile database; `git-clang-format --diff HEAD` reports no
formatting changes for the six changed C++ files.

- Python benchmark suite: **87 tests pass**, including feature-schema
  validation, cache/width product and budget, rejected candidates, fresh
  winner capture, and a winner with a worse old analytic score.
- Planner unit executable: **5,941 assertions in nine tests pass**.
- Focused Metal input-reuse test: **89,942 assertions pass**, exercising the
  existing 22 numerical configurations with additional access accounting
  checks. Same-expression `x*x` contributes one global read, cross-phase
  reuse changes the correct demand, and the padded tail remains conservative.
- Full `test_tile_*` CTest cohort: **31/33 pass**. The two existing Metal
  generated-source assertion failures require `mem_flags(3)` while the user's
  untouched local `cooperative.cpp` uses `mem_flags(2)`. Numerical checks pass;
  neither tests nor that local edit are changed or submitted. This is not a
  claim of a clean whole-repository test run.

The first focused test invocation used a nonmatching filter and ran zero
assertions; it was not treated as validation. The corrected filter below is
the passing invocation. The full [CTest log](ctest.log) preserves all failures.

```sh
cmake --build cmake-build-tirx -j 8
uv run --no-project --python 3.13 --with numpy python -m unittest discover -s scripts/benchmark/tile_torch -p 'test_*.py'
cmake-build-tirx/bin/test_tile_tirx_planner
cmake-build-tirx/bin/test_tile_tirx_execution metal tile_execution_reduction_input_cache
ctest --test-dir cmake-build-tirx -R '^test_tile_' --output-on-failure -j1
```

## Planned independent evidence

The next experiment fixes V=4/U=1/P=1 and a 64-scalar private budget, and
searches W={32,128,256,512,1024} × {reload,cache} for softmax, RMSNorm and
LayerNorm at 23×769, 128×2048, 1024×4096 and 128×8193. Three shapes differ
from the earlier fixed-width cache cohort; 1024×4096 is an anchor. The ragged
8193-column case crosses the private-budget boundary at narrow widths.

The reference will be the best valid reload member of that same measured
family; the candidate will be its joint-search winner. Freeze both before
four balanced replay rounds, retain identical-source controls and any failed
acceptance, and validate complete outputs. Use no-counter GPU command-buffer
execution as the stated selection metric, alongside separate batched and
single-call E2E dispatch phases. Do not label command-buffer time as isolated
kernel time or subtract independent phase medians to estimate overhead.

## Report and decision boundary

This technical report extends the existing repository Markdown/Sphinx surface;
it does not replace historical sections or add a parallel reporting app. No
quantitative chart is warranted for this implementation-only checkpoint.
Subsequent measured evidence must determine whether joint resource/mapping
selection generalizes. Full-device service, occupancy and physical private
state remain missing from the calibrated model; this interface alone does
not close the overall library-performance goal.
