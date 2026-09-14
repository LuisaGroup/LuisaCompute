# Generic program traversal: representation and diagnostic checkpoint

## Technical summary

An explicit TIRx Metal-group plan can now permute physical program ordinals
through bounded rectangles of the last two logical axes. Earlier axes remain
independent batches. Structural export retains the axis factorization, and
the permutation is applied after resource/recurrence/matrix planning. It
changes neither per-program effects nor worker distribution, memory layout,
allocation, synchronization or pipeline stage order. No kernel-name/size
dispatch table or DSL entity is introduced.

This is a generic **realization candidate**, not a calibrated automatic policy.
Both options default to one. Unsupported targets/bindings, disabled planning,
invalid extents and missing/inconsistent axis metadata reject exact requests.
CPU, automatic pointwise/reduction packing, and native MPP do not acquire this
emitter or an implied performance gain. The existing Sphinx planner reference
owns the design; its validation and route-results pages own the current
evidence. This supporting record follows that structure rather than creating
another report application.

## Correctness and default controls

- 480 grid/rectangle combinations are checked by an independent constructive
  enumeration, plus invalid/overflowing requests and a signed-64-bit boundary.
- Eight Metal executions combine batched program coordinates, partial tiles,
  a three-tap neighborhood pipeline and optional local reduction.
- Twenty-four matrix programs run two complete non-dyadic input sets each:
  reference SIMD-group versus MPP, staged versus view inputs, transposes,
  M/N/K tails, nonzero recurrence state and two traversal rectangles.
- The [final integration receipt](correctness-v3/results.json) records a
  successful full build and 17/19 passing invocations. Metal memory and
  cooperative tests retain three existing assertions requiring `mem_flags(3)`;
  the user-owned local barrier edit emits `mem_flags(2)`. No assertion or user
  change is removed, and the aggregate receipt deliberately remains failed.
- The [syntax receipt](syntax/results.json) passes eleven changed C++
  translation units and 101 Python tests at that point in development.
  A [final 102-test Python receipt](python-final.json) additionally covers
  exact traversal preservation/rejection in frozen replay.
- The [default-control audit](defaults/audit.json) validates all 36 native/
  Torch outputs. Six Metal sources are byte-identical to the frozen pre-change
  binaries; three CPU LLVM sources differ only by a bijective renaming of
  allocation-identity TBAA labels. Every instruction, metadata reference,
  width and offset suffix is preserved. Raw source files remain untouched.

The malformed-metadata test initially mutated a temporary copy-on-write TIRx
reference, so it failed to inject its invalid shape. Keeping the typed node
reference fixes the test fixture; the numerical programs had already passed.
[That failed attempt](correctness-v2/results.json) is retained. Similarly,
the [initial default comparison](defaults/results.json) required raw CPU bytes
to match and reports failure. The independent audit checks a bijective label
correspondence rather than ignoring metadata. Its first draft incorrectly
expected GEMM tolerances on add/activations; the actual benchmark uses exact
add and stricter activation/softmax tolerances. The corrected auditor checks
those policies without changing any measurement or tolerance.

## Experimental scope and metrics

The [predeclared protocol](protocol.md) fixes six FP32 GEMM shapes through
8192³, eight K/traversal combinations, 128×64 outputs, 256 threads, eight
independent 32×32 subgroup operations, input views, direct output, pipeline
window one and retained subgroup fences. Two rounds rotate then reverse both
shape and candidate order. Five samples, 20 ms requested windows and 100 ms
warmup are recorded for each GPU/E2E batch/single view. Native, Torch and
direct MPS controls are fresh in each call; their internal order rotates but
two rounds are **not** a complete six-order provider balance.

All providers use preallocated output and the complete deterministic FP64
dyadic oracle at atol=rtol=1e-4. Non-dyadic coverage belongs to the separate
native tests. Existing TVM/Torch arithmetic settings are unchanged, not
claimed strictly identical. GPU control means no-counter command-buffer
intervals including work and gaps, not an isolated kernel timestamp. Probe
counter samples remain diagnostics. Host E2E includes warm dispatch and
synchronization; cold JIT, setup, upload and download are separate.

No task-owned benchmark, build or profiler runs concurrently with the screen.
Desktop activity is not controlled; no idle-clock, cache-miss or occupancy
measurement is claimed. Both compiler stacks include the same uncommitted
barrier edit, excluded from the checkpoint. A screen minimum must not become
an accepted speedup, coefficient fit or default selection. A promising fixed
choice needs an independently frozen, balanced replay and held-out operators.

## Screen results and next decision

The completed [screen](screen/results.json) and [independent audit](screen/audit.json)
cover **288 complete outputs and 5,256,855,648 checked elements**, all with
zero maximum error. There are 39 unique MSL sources; two-round source identity,
exact grids, fixed local plans, all four timing views and six malformed-data
rejections pass the audit. The supported mapping family is enlarged, but no
default, calibrated cost or accepted performance objective is changed.

For the fixed 4×8 rectangle, 4096³, 8192³ and 4096×4096×11008 improve against
their same-BK row-major controls at both K sizes in both orders. Small/ragged
cases regress or reverse. Every one of the 96 native/Torch GPU comparisons
remains slower than Torch. The existing Sphinx route-results page reports
the exact six-shape lookup, not per-shape selected minima.

Identity cases matter: at 512³ the 4×8 rectangle spans the entire program
grid width and produces byte-identical row-major source. One measured ratio
is nevertheless 1.141. The audit flags every source-identical control, so
this is evidence of timing variation rather than a compiler regression or
cache-cost estimate. Neither two-order sensitivity nor in-sample minima
establish a generally beneficial traversal. A separately frozen balanced
replay, access-derived reuse/address features, and held-out composed operators
are still needed. No claim of isolated-kernel, native-MPP, XIR or SIMD gain
is made by this checkpoint.

## Reproduction and environment recovery

The source baseline is `5be8d24b7`. Before editing the public planner structs,
its benchmark, Tile library and bridge library were copied into
`/tmp/luisa-program-walk-baseline.xgtt5L`. Loader ordering keeps this ABI-consistent
triple separate from the new binaries; the fallback Runtime libraries and
external TVM libraries are shared. Artifact hashes are recorded before/after
each phase. The traversal screen also fingerprints the external TVM dylibs.

The [first correctness build](correctness/results.json) stopped before tests
because `/tmp/apache-tvm-tirx` lost headers and Git metadata. Read-only checks
found no active build/checkout process; the cause was not established. A new
checkout in `/Users/mike/.cache/luisa-tile/tvm-pinned.9TMqn1` uses TVM commit
`c7b458e946bc4266915da582457476bdcd9705ae`, FFI commit
`12dbf053b3d9ba4ebd9da3123b1aeca79cf74229`, and the repository's four sequential
MPP patches. Every surviving old header/source matches; FFI headers are also
identical. Only Luisa's cached include paths change. The old source is left
untouched, and no external compiler/runtime library is rebuilt or replaced.

From the repository root, with the recorded build and loader paths available:

```sh
uv run --offline --no-project --python 3.13 --with numpy --with torch --with orjson python \
  scripts/benchmark/tile_torch/results/m1-max-20260906-program-order/experiment.py \
  --phase check --output /path/to/new-correctness-directory
uv run --offline --no-project --python 3.13 --with numpy --with torch python \
  scripts/benchmark/tile_torch/results/m1-max-20260906-program-order/experiment.py \
  --phase screen --output /path/to/new-screen-directory
uv run --offline --no-project --python 3.13 python \
  scripts/benchmark/tile_torch/results/m1-max-20260906-program-order/audit.py screen
```

The audit command reads the saved `screen/` cohort beside this file; review
new output directories separately. The auditor checks the complete visit
sequence, physical grid coverage, unchanged local plans, generated sources,
output receipts, and independently reconstructs TIRx/Torch/MPS GPU medians
from raw nanosecond intervals and dispatch counts. Malformed reports exercise
missing/duplicate rows, changed artifacts, output size, ignored traversal and
invalid GPU divisors. Its four-metric table retains both rounds; no confidence
interval or causal claim is made. Exact candidate lookup and order sensitivity
belong in a table, not a selected-minimum ranking chart.

## Documentation review

The reader-facing changes extend the existing Sphinx Tile planner, coverage,
results and validation pages; this supporting directory is the reproducibility
record, not a second documentation hierarchy. The local link checker passes
48 HTML pages, 3,713 links/assets and 199 compatibility anchors. The strict
Sphinx build exits 1 with ten missing-Doxygen-XML warnings in the existing API
reference: Doxygen and its XML output are unavailable here. This is not a
claim that the complete API documentation build is green.

Headless Chrome 152.0.7977.76 checks the two result tables and the planner
mapping section at 1280- and 390-pixel widths. All six section screenshots
were visually inspected: text is readable, desktop tables fit, and narrow
tables/code retain the existing theme's horizontal scrolling without
page-level overflow. There are no page-script errors. Screenshots remain
local review artifacts; the compact browser receipt is saved beside these
notes. No new dashboard, hosting surface or Sphinx navigation branch is added.
