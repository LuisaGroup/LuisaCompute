# Shared element SSA and dual timing: validation checkpoint

September 5, 2026. This checkpoint implements same-domain producer
scalarization and separate Metal GPU/host dispatch measurement. It does not
claim completion of the general Tile/library performance goal.

- Full selected-tree build, followed by all `^test_tile_` CTests: **33/33
  passed**, including the new Metal timestamp integration test.
- The submitted source keeps `metal::mem_flags(3)`. The user's unowned local
  `mem_flags(2)` edit was temporarily restored to the submitted value only for
  that full regression, then restored to `2` immediately. Neither the edit
  nor weakened assertions enter the commit.
- After restoring `2`, another full build and focused execution CPU/Metal,
  planner and GPU timing regression passed **4/4**.
- Python benchmark contracts: **77/77 passed**. New tests cover device timing
  units/denominators, scope, host instrumentation exclusion, calibration,
  sample coverage, and invalid values, plus GELU and generic input views.
- Project clangd checks passed for all eight changed C++/Objective-C++
  translation units. Formatting and `git diff --check` passed for maintained
  code/documentation. Archived generated Metal sources preserve their exact
  hashed bytes, including the compiler's trailing blank lines.
- Sphinx HTML build succeeded with the same **24 pre-existing non-Tile
  warnings** (missing Doxygen XML, unrelated toctree/tutorial references and
  a tutorial code fence). The rendered status report's new SSA/timing section
  was inspected in the in-app browser; its text and caveats are readable.

Commands:

```sh
cmake --build cmake-build-tirx -j 8
ctest --test-dir cmake-build-tirx -R '^test_tile_' --output-on-failure -j 1
ctest --test-dir cmake-build-tirx \
  -R '^test_tile_(metal_timing|tirx_(execution(_metal)?|planner))$' \
  --output-on-failure -j 2
python3 -m unittest discover -s scripts/benchmark/tile_torch -p 'test_*.py'
uv run --no-project --python 3.13 --with orjson python \
  scripts/check_cpp_syntax.py --compile-commands-dir cmake-build-tirx <changed-source>
```

The [full regression log](tile-tests.log), [restored-worktree focused log](restored-tests.log),
[build log](build.log) and [Sphinx log](sphinx.log) are preserved here.
The independently recomputed four-round shared-SSA ratios are in the
[A/B note](../m1-max-20260905-shared-element-replay/notes.md).
The initial [GPU/dispatch smoke](../m1-max-20260905-device-timing-smoke-v2/notes.md)
is explicitly not a balanced performance claim. Wider GPU-timed operator
cohorts, native/MPS path coverage and repeated comparisons are subsequent work.

The benchmark helper has no global hooks outside an explicit capture interval,
no TVMx/PyTorch private ABI dependency, and no pinned dependency changes.
Raw GPU samples, compute-pass and command-buffer counts, calibration and
helper-library hashes are saved in each requested run. Unsupported measurement
coverage is an error, never silently replaced by a host clock.
