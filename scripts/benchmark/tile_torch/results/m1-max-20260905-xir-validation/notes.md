# TileIR→XIR/SIMD implementation validation

This artifact retains the September 5, 2026 evidence for the initial direct
XIR bridge, CPU planner, SIMD PHI repair, bounds-proof cleanup and LLM operator
suite. The worktree was on `codex/tile-programming-design`, HEAD
`b8c3c54f81f2a4ad947e295f1f75e57605bf8833`, with uncommitted implementation
changes. Paths and hashes below identify the actual inputs; the commit alone
does not reproduce them.

## Final results

- A complete selected-tree CMake build succeeded with SIMD, Metal and TIRx
  enabled. SIMD and TVM were configured with LLVM 21.1.8. Build warnings about
  deployment targets and the external half header are retained in the log.
- The final 25-test CPU/Metal cohort passed **23/25**. The only failures were
  `test_tile_tirx_cooperative_metal` and `test_tile_tirx_memory_metal`, which
  expect `mem_flags(3)` while a pre-existing/unowned worktree change emits
  `mem_flags(2)`. No assertion was weakened and that source was not reverted.
- `test_tile_xir_llm` passed **22,307 assertions in 3 test groups**. It covers
  21 kernel/shape captures. Each is compiled through XIR/SIMD and native-target
  TIRx CPU and checked against an independent FP64 reference. The complete test
  took 11.80 seconds in the focused final run; test duration includes JIT and
  is not a kernel performance metric.
- The focused XIR/runtime/PHI cohort passed 4/4 in 17.20 seconds after the
  bounds proof. Existing wide SIMD codegen and Runtime regression tests also
  pass in the final 25-test cohort.
- The C++ syntax checker reported no errors for all new/changed XIR, planner,
  runtime, benchmark, PHI and relevant Metal files. Two pre-existing unused
  include warnings in `lower.cpp` were removed, then that file and the changed
  runtime test were checked again with no issues.
- Python benchmark-contract tests passed **60/60** with NumPy installed.

## Failure-to-fix evidence

The first combined process crashed while simultaneously loading TVM's LLVM 21
and SIMD's LLVM 22. `xir-first.log` preserves that run; `xir-llvm21.log` shows
that selecting matching LLVM exposed numeric failures instead of the loader/
analysis crash. The XIR-only LLVM-22 smoke output was numerically correct, so
the report treats this as a combined configuration conflict, not an intrinsic
LLVM-22 or bridge failure.

`simd-phi-valid-before.log` contains 62 wrong-result failures from a pure XIR
PHI-cycle regression. The fix snapshots every edge-assignment source before
updating any destination. `xir-phi-after.log` and the final logs show the same
test and the Tile runtime tests passing. The implementation did not relax the
state-slot interference model.

The first LLM suite was manually terminated after 332.28 seconds while still
compiling; no numeric result was reported. After shared SSA cleanup and removal
of unconditional diagnostic assembly generation, a second run was terminated
after 244.06 seconds. Its live `sample` was entirely in LLVM machine scheduling
and register-pressure analysis at the sample instant. This motivated a separate
checked integer interval proof before XIR/Schedule expansion. The final suite
then completed. These interrupted wall times are diagnostic observations, not
a controlled performance A/B.

## Exact commands

Configure the actual build (TVM dylibs were the locally patched C++ build):

```bash
cmake -S /Users/mike/CLionProjects/luisa \
  -B /tmp/luisa-tvm-mpp.VaKmzx/luisa-build \
  -D LUISA_COMPUTE_ENABLE_SIMD=ON \
  -D embree_DIR=/opt/homebrew/opt/embree/lib/cmake/embree-4.4.1 \
  -D LLVM_DIR=/opt/homebrew/opt/llvm@21/lib/cmake/llvm
cmake --build /tmp/luisa-tvm-mpp.VaKmzx/luisa-build --parallel 8
```

Final selected regression cohort:

```bash
ctest --test-dir /tmp/luisa-tvm-mpp.VaKmzx/luisa-build \
  -R '^(test_tile_(xir.*|native.*|layout|ir|dsl|memory|tirx_(execution|matrix|cooperative|memory|targets)(_metal)?)|test_simd_(phi_parallel_copy|llvm_schedule_codegen|runtime_widths|local_memory|arithmetic|warp_uniformity|xir_to_schedule))$' \
  --output-on-failure
```

Focused final cohort and benchmark contracts:

```bash
ctest --test-dir /tmp/luisa-tvm-mpp.VaKmzx/luisa-build \
  -R '^(test_tile_xir|test_tile_xir_runtime|test_tile_xir_llm|test_simd_phi_parallel_copy)$' -V
uv run --no-project --python 3.13 --with numpy \
  python -m unittest discover -s scripts/benchmark/tile_torch -p 'test_*.py'
```

Each C++ syntax check used:

```bash
uv run --no-project --python 3.13 --with orjson \
  python scripts/check_cpp_syntax.py SOURCE.cpp \
  --compile-commands-dir /tmp/luisa-tvm-mpp.VaKmzx/luisa-build \
  --clangd /opt/homebrew/opt/llvm@21/bin/clangd
```

## Files in this artifact

- `xir-final-build.log`, `xir-final-tests.log`, `xir-bounds-tests.log`:
  complete final build, broad selected cohort and focused verbose cohort.
- `python-tests-numpy.log`: 60 benchmark-contract tests.
- `syntax-*.log`: final syntax checks for the touched C++ surface.
- `simd-phi-valid-before.log`, `xir-phi-after.log`: red/green PHI evidence.
- `xir-first.log`, `xir-llvm21.log`: LLVM coexistence investigation.
- `xir-llm.log`, `xir-cleanup-tests.log`, `xir-llm-sample.txt`:
  the two stopped compile-expansion investigations and process sample.
- `manifest.sha256`: content hashes of the retained evidence files.

The balanced execution-performance data is separate in
[`../m1-max-20260905-xir-simd/`](../m1-max-20260905-xir-simd/notes.md), so test
or profiler activity is not mixed with timing.
