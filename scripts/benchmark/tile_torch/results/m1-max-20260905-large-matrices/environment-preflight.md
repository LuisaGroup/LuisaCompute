# Compiler capability preflight

The first `view-pilot/` used the ordinary `cmake-build-tirx` loader search
paths. All six cases were rejected before MPP compilation:

```text
This TVM build lacks Metal MPP memory contract v2; use SIMD-group lowering or the documented TVM patch
```

These are retained failed capability probes, **not six correctness failures,
valid timing rows, or fallbacks to a different lowering**. That build normally
loads the unpatched wheel under `/tmp/luisa-tvmx-venv/` used for the recent
reduction experiments.

The previously validated patched compiler still exists under
`/tmp/luisa-tvm-mpp.VaKmzx/build/lib`. A 32×32×32 MPP-view loader probe with
the same current executable, explicit `DYLD_LIBRARY_PATH` pointing there,
and the existing Metal timing helper exited zero. The dyld diagnostics showed
the compiler, runtime, FFI, Metal runtime and runtime-extra libraries all
loaded from that exact patched directory; Luisa libraries still loaded from
`cmake-build-tirx/bin`. The JSON reported the MPP v2 cost basis, real MPP calls,
zero SIMD-group MMA calls, and both probe/control timing phases. This tiny
probe establishes loading/capability only, not a performance or correctness
result.

The proper six-case pilot is saved separately as `view-pilot-patched/` and
uses that explicit loader environment. The seven-path GEMM replay keeps the
same environment and fingerprints all five patched libraries, their linked
LLVM library, native executables, adjacent Luisa libraries and the timing
helper. Loader overrides are now recorded in the benchmark JSON as well as
the commands. No C++ source, schedule, coefficient or binary was changed to
repair the capability mismatch. The reduction coverage uses the ordinary
wheel environment and records its own artifact boundary separately.

This environment correction is recorded before the corrected pilot or replay;
it does not change the predeclared dimensions, mapping choices, round count,
math policy, tolerances or failure-retention rule.
