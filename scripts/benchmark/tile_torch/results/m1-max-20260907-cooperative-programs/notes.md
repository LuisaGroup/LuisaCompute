# Automatic cooperative programs: bounded improvement, remaining attention gap

September 7, 2026; Apple M1 Max; branch `codex/tile-programming-design`.
The baseline was frozen before this change at the d52daad15 worktree. Exact
compiler, executable, adjacent library and generated-source fingerprints are
in each `results.json`; both runs report unchanged artifacts.

## What changed

An automatic root parallel program with reassociable MMA can now try the
existing cooperative group mapper. Admission trusts the parallel/independent
element contracts; it checks supported effects, allocations and hierarchy
constraints instead of proving source independence again. Leading/interleaved
singleton axes can be projected out of a matrix iteration space without
changing the original buffer layout. Pipeline versioning reserves capacity
for a possible group before adding extra buffer versions.

This is a generic candidate-admission/resource-budget repair in the
**TIRx-to-Metal** route, not a new calibrated cost policy, a general joint
solver, native-MPP work or an XIR/SIMD speedup. Kernel names and the benchmark
dimensions are not in the admission code. The remaining matcher, geometry
score and allocation accounting are bounded/conservative.

## Protocol and results

Each cohort uses all six order permutations of native/baseline/Torch, five
samples, 20 ms target sample duration and 100 ms warmup. Both native variants
use FP32, `fast_math=false`, `relaxed_precision=false` and the existing MMA
reassociation permission. The source is the same online causal GQA program,
with the same block size and exported inputs in each comparison.

Torch is functional SDPA with GQA and an explicit bottom-right causal mask;
its output allocation is included. Native output is preallocated. This is
not a matched fused-implementation/math-policy guarantee, MPSGraph benchmark,
KV-paging workload or model-level measurement. JIT, input transfer and setup
are outside warm timings; their separate fields remain in the JSON.

| Cohort: B,Hq,Hkv,Q,K,D,Dv | Block | Old GPU µs | New GPU µs | Torch GPU µs | Paired new/Torch | Paired new/old |
|---|---|---:|---:|---:|---:|---:|
| Prefill: 1,4,2,64,128,64,64 | 8×16 | 7263.604 | 80.612 | 27.876 | 2.894× | 0.011113× |
| Decode: 1,8,2,1,2048,64,64 | 1×32 | 47977.771 | 327.875 | 32.348 | 10.132× | 0.006830× |

“GPU” above means the **uninstrumented command-buffer GPU control**, including
GPU work and intra-buffer gaps, not isolated kernel time. Entries are medians
of round medians; ratios are medians of paired round ratios. Both new paths
beat the old path in every GPU/E2E throughput round, but lose to Torch in
every GPU/E2E throughput round. New/Torch GPU ratio ranges are 2.832–2.992
and 4.974–10.564; the decode Torch control has a slow outlier.

| Cohort | New E2E batch µs | Torch E2E batch µs | Paired ratio | New single-call E2E µs | Torch single-call E2E µs | Paired ratio |
|---|---:|---:|---:|---:|---:|---:|
| Prefill | 100.087 | 52.877 | 1.885× | 416.667 | 475.667 | 0.899× |
| Decode | 335.382 | 42.579 | 7.867× | 574.167 | 297.375 | 1.922× |

Prefill single-call E2E loses in one of six rounds; its median win does not
reverse the GPU-throughput conclusion. All 36 complete outputs pass the
independent FP64 comparison (`atol=rtol=5e-5`), checking 304,128 elements.
Native also performs two output checks with guard regions per invocation.

Instrumented compute-pass medians remain separate diagnostics:

| Cohort | New compute µs | Torch compute µs | New counter/control ratio | Torch counter/control ratio |
|---|---:|---:|---:|---:|
| Prefill | 97.985 | 66.981 | 1.388× | 3.932× |
| Decode | 331.515 | 41.118 | 1.101× | 1.759× |

The observer materially perturbs Torch, especially prefill. Do not rank the
implementations by the instrumented values or relabel either timer as an
unperturbed isolated kernel measurement.

## Emitted structure and remaining work

Prefill selects one group plan with **64 threads/group**, versus the old
32 whole-program workers. Its first contraction uses SIMD-group matrix
instructions; the later `P*V` update remains scalar because its accumulator
initialization is an expression. Pipeline prologue/body duplication accounts
for repeated call sites; that is not two tensorized semantic contractions.

Decode selects one group plan with **1024 threads/group**, versus eight
whole-program workers. Its query block has M=1, so neither contraction fits
the current 8×8 matrix atom. It still benefits from cooperative element
distribution. Both sources retain many shared temporaries and barriers;
reductions inside this composed family are not yet redistributed through
SIMD-group reduction intrinsics. No hardware profile has apportioned the
remaining time among these costs. This geometry is not established optimal.

Before the capacity repair, a two-order pilot still left decode at eight
workers: an extra K/V pipeline version precluded the group. That unsuccessful
pilot is diagnostic, not part of these six-round tables. Ordered stages with
one storage version remain legal. A true joint resource/time planner should
compare the alternatives rather than always pre-reserving a conservative
whole-body sum.

Next: reusable reduction/accumulator materialization rules, joint element and
matrix participation choices, liveness-based storage and measured geometry
selection. Broad held-out attention shapes, native MPP, SIMD and fresh direct
MPS comparisons are still needed; the performance goal is not complete.

## Validation and reproducibility

The full selected build completed before the test/benchmark runs. The new
matrix semantics test passes 17,189 assertions (automatic/disabled/explicit
worker/arithmetic controls, modular program coordinates, unit-axis positions,
same-instance C/D aliasing, and multi-shape attention). Full CTest reports
35/37 passes under `-R tile`; two passing sparse-resource tests match that
regex, so the Tile-only count is **33/35**. The two failures are the existing
cooperative/memory source-string expectations for the user-owned barrier edit.
See `ctest.xml`. No all-green worktree claim is made.

Both frozen and new binaries include the same unrelated worktree change
`metal::mem_flags(3)` to `(2)`. It is excluded from this checkpoint's code
commit. Timings therefore describe the fingerprinted worktree artifacts, not
an assertion that a clean checkout generates identical code. The archived
Metal sources preserve that distinction.

Run the raw-sample/provenance auditor from the repository root:

```sh
python3 scripts/benchmark/tile_torch/results/m1-max-20260907-cooperative-programs/audit.py
python3 scripts/test_tile_execution_calculus.py
```

The first reuses the preceding checkpoint's raw-value reader, reconciles
four metrics and every paired range/count, checks source identities, and
rejects missing rows, forged ratios, wrong GPU units, source corruption and
wrong round order. The second is a finite mathematical reference model:
nine tests, not a verification of the compiler or a completeness proof.

Final non-native validation also passes all 110 benchmark Python tests and
the auditor's five negative cases. The refreshed documentation passes checks
for 50 HTML pages, 4,048 local links/assets and 199 compatibility anchors.
Headless Chromium checks 39 desktop/mobile section and table views; representative
Cypress, solver and rigidity sections were also visually inspected. See
`docs-qa.json` and `qa_docs.cjs`. Screenshots are temporary local QA artifacts,
not benchmark evidence. Sphinx reports ten pre-existing missing Doxygen XML
warnings, so this is not a successful full API-documentation build.

The related-work expansion and mathematical reference examples are design
work, not newly implemented compiler guarantees or additional performance
measurements. Native code has not changed since the recorded build and tests.

Replay command (substitute an ABI-coherent old baseline directory and build
directory; preserve adjacent old libraries):

```sh
env DYLD_LIBRARY_PATH=/tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin:/Users/mike/.cache/luisa-tile/tvm-fragment-build.0ToXYr/lib \
uv run --offline --no-project --python 3.13 --with numpy --with torch \
python scripts/benchmark/tile_torch/compare_llm.py \
  --native /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/benchmark_tile_native \
  --baseline /tmp/luisa-cooperative-baseline.ZJ6pUs/benchmark_tile_native \
  --build-dir /tmp/luisa-tvm-mpp.VaKmzx/luisa-build --backend metal \
  --case attention:1,4,2,64,128,64,64 --attention-block 8 16 \
  --output /tmp/luisa-cooperative-replay-prefill \
  --rounds 6 --samples 5 --sample-ms 20 --warmup-ms 100 \
  --metal-device-timing /tmp/luisa-tvm-mpp.VaKmzx/luisa-build/bin/libluisa-benchmark-metal-timing.dylib \
  --compiler-artifact /Users/mike/.cache/luisa-tile/tvm-fragment-build.0ToXYr/lib/libtvm_compiler.dylib
```

For decode, use case `attention:1,8,2,1,2048,64,64`, block `1 32` and a
different output directory. Do not overwrite archived evidence or run native
build/tests/profiling concurrently with timing. Compiler legality tests are
separate from performance experiments.
