# CPU provider and documentation validation

This records the final September 5, 2026 validation after adding shared
exp materialization, structural reduction contracts, CPU CBLAS/Accelerate
realizations, root launch-cost control and predicated-store fast/slow
versioning. Branch: `codex/tile-programming-design`; this note belongs to the
same source/report snapshot as the implementation. Generated artifacts retain
their independent hashes because a Git revision alone does not reproduce a
performance environment.

## Build and focused structural tests

The complete incremental CMake build succeeded:

```sh
cmake --build /tmp/luisa-tvm-mpp.VaKmzx/luisa-build -j 8
```

The following focused tests passed after that complete rebuild:

```sh
/tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/test_tile_tirx_execution \
  cpu tile_execution_cpu_parallel_launch_cost
/tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/test_tile_tirx_execution \
  cpu tile_execution_shared_exp_materialization
/tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/test_tile_tirx_execution \
  cpu tile_execution_cpu_accelerate_math
/tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/test_tile_tirx_execution \
  cpu tile_execution_auto_vector_guards
/tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/test_tile_tirx_execution \
  metal tile_execution_cpu_accelerate_math
```

They respectively completed with 10, 2, 115, 31,542 and 2 passing assertions.
The Metal case verifies that the CPU-only provider policy is rejected rather
than silently lowered or run on CPU.

The first broad CTest attempt was intentionally discarded: only the two
focused binaries had been rebuilt after extending public `PlannerOptions`, so
old test executables passed the old struct ABI into the new dylib and reported
impossible option values. A full incremental build rebuilt every dependent
test before the results below. This was build staleness, not a source/runtime
failure and is not counted as a regression result.

## Complete Tile cohort

```sh
ctest --test-dir /tmp/luisa-tvm-mpp.VaKmzx/luisa-build \
  -L unit_tile --output-on-failure
ctest --test-dir /tmp/luisa-tvm-mpp.VaKmzx/luisa-build \
  -R '^test_tile_(xir_runtime|xir_llm|native_runtime)$' --output-on-failure
```

An ownership-audit run against the local worktree first produced 29/31 label
passes. The only failures were:

- `test_tile_tirx_cooperative_metal`
- `test_tile_tirx_memory_metal`

Both assert generated `metal::mem_flags(3)` while an unowned worktree hunk in
`src/tile/bridge/tirx/cooperative.cpp` emitted `metal::mem_flags(2)`. The tests
still executed their numerical paths; no assertion was weakened.

The submitted source excludes that hunk and preserves `mem_flags(3)`. After a
complete incremental rebuild, the exact 31-test `unit_tile` command above
passed **31/31**. The separately registered XIR Runtime, XIR LLM and native
Runtime tests passed **3/3**, so the submitted-source `test_tile_*` cohort is
**34/34 passing**. The unowned local `mem_flags(2)` edit was restored after
staging and remains outside this report's source snapshot.

## Benchmark-contract and documentation checks

```sh
uv run --no-project --python 3.13 --with numpy \
  python -m unittest discover -s scripts/benchmark/tile_torch -p 'test_*.py'
xmllint --noout docs/_static/tile/tirx-realization-pipeline.svg
git diff --check
```

Python passes **64/64**; the SVG is well-formed; `git diff --check` reports no
errors. Sphinx 9.1 read and rendered all four changed Tile documents and copied
the new SVG/download artifacts. Strict `-W` exits nonzero only for two existing
repository-wide `toc.not_included` warnings:
`docs/custom_agility_sdk.md` and `docs/source/coro_suspend_extensions.md`.
No warning originates in a changed Tile document.

Performance evidence is intentionally separate from test activity:

- [CPU CBLAS replay](../m1-max-20260905-cpu-cblas-v2-replay/notes.md)
- [CPU array-math policy replay](../m1-max-20260905-cpu-accelerate-ops-replay/notes.md)
- [Metal MPP v2 replay](../m1-max-20260905-mpp-cost-v2-replay/notes.md)
