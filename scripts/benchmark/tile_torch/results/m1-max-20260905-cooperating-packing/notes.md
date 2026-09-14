# Cooperating-program packing: implementation and fixed-plan evidence

September 5, 2026; Apple M1 Max, Metal FP32, TIRx/TVM runtime.

## Conclusion

Multiple cooperating row programs can now share a physical threadgroup without
sharing their private stripes or reduction partials. The new mapping remains
an **explicit/JIT candidate**: the fixed 256-worker-per-row comparison below
usually becomes slower when packing two programs. Automatic packing, private
budgets and cost coefficients are unchanged. This is a capability and model
diagnosis checkpoint, not general parity with Torch/MPS or a new default.

The canonical narrative and complete per-case table live in
[the repository performance documentation](../../../../../docs/source/performance/tile/reductions.md#cooperating-program-packing).
This note owns the experiment provenance and validation record; it is not a
separate report application.

## Implementation boundary

For P programs per group and S SIMD groups per program:

```text
T = 32 * S * P                         whole physical group width
W = 32 * S                             cooperating workers per program
program_in_group = thread / W
worker_in_program = thread % W
partial_base = program_in_group * S
physical_groups = ceil_div(logical_programs, P)
```

Each reduction allocates P*S shared partials, while private stripes use W.
The first and second `simd_sum/max/min` collectives remain uniform. A partial
final group replays the last valid logical coordinate, including the original
loop minimum, and predicates only external stores. This keeps data-dependent
input addresses valid and group barriers unconditional. Inactive slots still
read and compute; the optional service policy counts those extra reads, not
phantom external writes.

Admission remains conservative. The central noalias, pure-expression,
independent-store and private-ownership proofs still apply. For cooperating
packing, every reduction-containing enclosing loop must have unit extent,
constant minimum and unit serial step. Even uniform repeated loops are
rejected: scratch reuse needs an additional read-before-next-write fence
proof. A packed tail also rejects any external buffer both read and written.
Unknown proofs reject this realization rather than moving a fence under a
program-tail predicate.

`reduction_programs_per_group=0` retains the established automatic family
(only single-subgroup programs are packed). An explicit positive P can use
several subgroups per program. Exact `threads_per_group` specifies T, not W;
with T=0, the bounded solver searches fitting widths for that P. No hardware
scope or memory object is added to the DSL.

## Validation

The final sources completed the full selected CMake build:

```sh
cmake --build cmake-build-tirx --parallel 6
ctest --test-dir cmake-build-tirx -R '^test_tile_' --output-on-failure \
  --output-log scripts/benchmark/tile_torch/results/m1-max-20260905-cooperating-packing/ctest.log
uv run --no-project --python 3.13 --with numpy --with torch \
  python -m unittest discover -s scripts/benchmark/tile_torch -p 'test_*.py'
```

- **31/33 Tile CTests pass**, including CPU/Metal execution and planner tests.
  The [complete log](ctest.log) retains two generated-source assertion failures:
  `test_tile_tirx_cooperative_metal` at `test_tirx_cooperative.cpp:100` and
  `test_tile_tirx_memory_metal` at `test_tirx_memory.cpp:168`. The unrelated local
  `cooperative.cpp` change from submitted `mem_flags(3)` to `2` remains untouched
  and is not part of this commit. Their numerical checks pass; this is not an
  all-green worktree.
- **36 new numeric configurations** cover four operators (sum, softmax,
  no-affine LayerNorm, paired min/max) across nine layouts. Coverage includes
  widths 64/96/128/416/512, automatic cooperating width, P=2/3/8, V=1/4/8,
  U=1/3, cached and uncached input, nonzero row origin, rows fewer than P,
  ragged columns, and a 1024-thread group. All outputs and sentinel guard
  rows/unused columns are checked after three dispatches with the unchanged
  `2e-6 + 2e-5*abs(reference)` tolerance.
- **Six typed raw-IR proof cases** accept direct/unit-wrapper reductions and
  reject repeated/row-varying wrappers, read/write tail replay and an
  over-limit 1056-thread group. A typed `DeviceArtifact` audit finds exactly
  one group fence, outside any conditional, for each accepted case.
- **89/89 Python benchmark tests pass**, including consistent S=3/P=3/T=288
  metadata and independent service-cost arithmetic for inactive tail reads.
  The C++ planner also checks the read/write service distinction.
- Selected-database clangd finds no issues in the reduction emitter and two
  changed test files. The planner has no errors and ten warnings at untouched
  lines. Changed-line clang-format and `git diff --check` pass for authored
  files. Captured MSL is retained byte-for-byte, including TVMx's trailing
  blank line; whitespace normalization would invalidate its recorded hash.

An initial scalar-output test used the same named axis twice, which capture
correctly rejected. The test now uses distinct unit row/column axes. No
numerical tolerance or failure assertion was relaxed to pass that test.

The repository documentation was rebuilt in the fresh `reorganized` output
and the existing local preview output. `scripts/check_docs.py` checks all
37 HTML pages, 2,874 local links/assets and 37 compatibility anchors; all
targets resolve. The twelve new documented paired medians/ranges match the
independent receipt exactly at displayed precision. Browser review verifies
the normal documentation navigation, mapping diagram, new section/table and
an old reduction URL redirecting to its new performance section. No standalone
report app or new site theme is introduced. Strict Sphinx still exits nonzero
for ten pre-existing missing-Doxygen-XML API warnings; this is handwritten-page
QA, not a complete API-documentation build or responsive-device test suite.

## Predeclared comparison and measurement

The [protocol](protocol.md) predates candidate timing. Three operators run at
37×1537, 256×3072, 768×6144 and 1024×4096. Both variants use W=256/S=8,
V=4/U=1, immutable input caching, preserved shared SSA, a 64-scalar private
budget and the unchanged analytic policy. Reference P=1/T=256; candidate
P=2/T=512. This holds per-row ownership and recurrence fixed, but changes
physical grouping; neither is an optimal-width claim.

The [reference](reference/results.json) and [candidate](candidate/results.json)
pilots each use three host samples, 10 ms windows and 100 ms warm-up. They
validate 48 outputs in total and supply frozen parameters/sources only.
Pilots have no GPU timing phase and their times are excluded from acceptance.
The reference pilot predates a formatting-only library rebuild. The paired
replay instead uses identical final artifacts for both variants and verifies
that each variant's plan and generated source still match its pilot.

The [four-round replay](replay/results.json) uses fresh capture/JIT, nine
samples, 30 ms target windows and 200 ms warm-up. Variant-first order and
native/Torch-first order are each balanced 2:2 per case. No builds, tests or
profilers ran concurrently. All 192 outputs pass the runner's complete
independent FP64 oracle; the runner's pre-existing tolerances are atol=2e-6
for softmax/RMSNorm, atol=1e-5 for affine LayerNorm, and rtol=2e-5 for all.
Every case, regression and raw round is retained. Exact commands are in the
JSON, including native options and all frozen-plan records.

The native path uses preallocated output. Eager Torch MPS softmax uses `out=`;
functional RMSNorm/LayerNorm allocate returned outputs in timing. There is no
direct MPS/MPP, compiled-Torch, CUDA or CPU comparison in this experiment.
Recorded Torch is 2.14.0; the native plan/binary and Torch configuration,
fast-math environment, runtime and compiler fingerprints are retained.

GPU throughput and single-call GPU time come from completed command-buffer
timestamps **without encoder hooks or counter attachments**. They include
GPU work and gaps inside each buffer, not CPU encoding/completion notification,
and are not isolated-kernel measurements. Host batch throughput and synchronized
single-call latency use separate uninstrumented phases. The instrumented
compute-pass phase remains diagnostic: Torch probe/control throughput ratios
range from 0.762 to 4.558. No independently sampled medians are subtracted and
no prior speedups are multiplied into this experiment.

## Independent audit and findings

```sh
python3 scripts/benchmark/tile_torch/results/m1-max-20260905-cooperating-packing/audit.py
# On the original, unchanged build only, also check current artifact contents:
python3 scripts/benchmark/tile_torch/results/m1-max-20260905-cooperating-packing/audit.py \
  --check-current-artifacts
```

The standalone [audit](audit.py) imports no production statistics, cost or
selection helper. It independently enumerates worker ownership; reconstructs
private/shared storage, access demand, analytic score and physical launch;
checks generated intrinsics/fence counts; recomputes all four timing metrics
from raw samples; and verifies exact source/plan identity and balanced order.
The saved [receipt](audit.json) additionally checked all 21 current artifacts
against the replay's unchanged fingerprints. It verifies recorded full-output
validation, but does not pretend to repeat the FP64 oracle from archived arrays:
those arrays are not retained in this evidence directory.

| Acceptance metric | Positive paired medians | All four pairs improve | All four pairs regress |
|---|---:|---:|---:|
| GPU batch throughput | 2/12 | 2/12 | 8/12 |
| E2E batch throughput | 2/12 | 1/12 | 9/12 |
| GPU single-call | 3/12 | 1/12 | 4/12 |
| E2E single-call | 6/12 | 1/12 | 1/12 |

Ratios are paired P1/P2 time ratios: above one favors P2. Per-case medians,
observed min–max ranges (not confidence intervals), absolute microseconds
and Torch comparisons are in [the complete readable replay](replay/results.md)
and the independent receipt. Only LayerNorm 256×3072 and 768×6144 improve
GPU throughput in every pair (median 1.058× and 1.044×); the latter's E2E
range crosses one. RMSNorm and LayerNorm 1024×4096 instead regress in every
GPU/E2E batch pair (GPU median gains 0.837× and 0.818×).

The analytic score prefers P2 for every case because its setup term shrinks
from 16 to 8. That is a model defect exposed by newly expressible grouping.
Lower active-group concurrency and cross-program fence coupling are plausible
explanations, not measured occupancy/ISA diagnoses. Absolute times shift
between rounds, so retain counterbalanced pairs rather than correcting by an
unmeasured clock/load factor. Ten P2 candidates beat eager Torch GPU throughput
in every pair and all twelve beat its E2E throughput; that external fact does
not overturn the losses against the stronger P1 variant.

Next: compare narrower cooperating/packed factorizations with the established
single-subgroup family at fixed total group size, retain negative controls,
and obtain stable independent measurements before fitting grouping costs.
No additional cohort was run or used to prune these twelve cases.
