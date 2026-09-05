# Shared-Tile materialization implementation validation

Date: 2026-09-05 Asia/Shanghai. Machine: Apple M1 Max, macOS 26.6.2 arm64.
Branch: `codex/tile-programming-design`.

## Validated surface

This checkpoint covers the structural shared-Tile policy, target-specific
worker-stripe resource bound, residual-LayerNorm benchmark/JIT controls,
report replay metadata and the integrated documentation. It does not remeasure
the performance reports; those preserve their own binary/compiler/source
hashes and raw samples.

## Build and native tests

```bash
cmake --build cmake-build-tirx -j 8
ctest --test-dir cmake-build-tirx -R '^test_tile_' \
  --output-on-failure -j 1
```

The complete build succeeds. The final submitted-source Tile cohort passes
32/32 in 96.77 seconds: 30 unit-labeled tests and two separately registered integration
tests. It includes physical Metal execution, memory, pipeline, cooperative and
matrix paths as well as CPU/XIR/native Runtime coverage.

An unowned working-tree change sets `metal::mem_flags(2)` where the branch
source has `3`; that edit intentionally disagrees with two generated-source
assertions. For the submitted-source cohort only, the value was temporarily
restored to `3`, rebuilt and tested, then the user's `2` was immediately put
back. The edit is excluded from this change and no assertion was weakened.

Focused executable results:

```text
test_tile_tirx_planner:             5,891 assertions / 7 tests
test_tile_tirx_execution (CPU):    33,071 assertions / 22 tests
test_tile_tirx_execution (Metal):  38,363 assertions / 22 tests
```

The execution tests cover generic shared `tanh`, both cheap-arithmetic
materialization policies, compact LayerNorm stripes, physical Metal numerical
results and the pre-existing reduction/view/ownership cases. Planner coverage
includes the zero stripe-budget rejection.

## Benchmark contracts and artifacts

```bash
python3 -m unittest discover \
  -s scripts/benchmark/tile_torch -p 'test_*.py'
```

All 69/69 tests pass. Coverage includes operation lists/tolerances, native CLI
argument padding, mandatory replay policy metadata, finite Cartesian JIT
products, candidate budgets and fresh winner validation. All six selected
`results.json` files parse successfully, and their recorded hashes remain
unchanged after mechanically regenerating readable `results.md` files.

`git clang-format --diff HEAD -- <changed C++ files>` reports that it would
modify no changed lines. A whole-file Clang 22 dry run on `cpu.cpp` still
reports three formatting differences that reproduce on `HEAD` at the same
pre-existing call-construction lines; they are outside this patch.
`git diff --check` and `xmllint --noout` on the new SVG pass.

## Documentation build

```bash
cd docs
uv run --no-project --python 3.13 \
  --with sphinx --with sphinx-rtd-theme --with breathe --with myst-parser \
  sphinx-build -b html . _build/tile-docs-final --keep-going
```

The clean 17-page HTML build succeeds. Its 24 warnings are outside the Tile
changes: ten missing Doxygen XML notices because Doxygen was not run, two
pre-existing pages omitted from the toctree, eleven tutorial heading links and
one tutorial lexer warning. There are no Tile-page cross-reference,
literal-include, image or syntax warnings.
