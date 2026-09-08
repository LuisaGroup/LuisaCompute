# Bounded M/N inputs: legal large-K plans, remaining library gap

The final TIRx MPP emitter improves selected GPU batch time substantially on
three large ragged FP32 GEMMs, but does not meet the general MPS/Torch goal.
The answer-first table and qualifications live in the existing
[route report](../../../../../docs/source/performance/tile/results.md#bounded-m-n-inputs-remove-an-admission-barrier),
with the implementation contract in the existing matrix reference. This file
owns reproducibility and validation, not a second documentation hierarchy.

## Evidence and measurement contract

[protocol.md](protocol.md) fixes the five shapes, three shared candidate
requests, old/new/new/old order, reversed shape/candidate order, fresh selected
replays, full FP64 checks, five samples and separate GPU/E2E metrics. This is
an exploratory M1 Max desktop session, not an isolated-device or held-out
calibration experiment. FP32 inputs are deterministic dyadic values for the
timed benchmark; separate semantic tests use non-dyadic and nonfinite values.
All GEMM outputs are preallocated. Reassociation/fast-math differences across
TVM, MPS and Torch remain recorded qualifications, not matched arithmetic.

Each session contains four `results.json`/`results.md` reports and generated
Metal source files named by SHA256. Its `execution.json` records exact commands,
loader paths and 35 artifact hashes before/after. The frozen old compiler uses
the original MPP-v2 plus bounded-K capability. Unchanged support libraries are
loaded from the selected Luisa build; loader paths preserve old-bridge priority.

- [replay/](replay/execution.json): the initial per-output scan, reproduced by
  [initial-per-element.patch](initial-per-element.patch) after bounded-K.
- [cooperative-replay/](cooperative-replay/execution.json): subgroup-striped K
  scans plus static interior views, reproduced by
  [cooperative-scan.patch](cooperative-scan.patch).
- [shuffle-replay/](shuffle-replay/execution.json): lane-owned classifications
  and shuffle distribution, using the final repository MNK patch.

Each session validates 192 complete outputs and 995,833,488 elements, retaining
16 explicit old resource/geometry rejections. Across all three there are 576
outputs and 2,987,500,464 checked elements. Rejected requests have no timing
denominator. The initial root-level loader failures and zero-assertion pilot
are preserved and **excluded**; see the protocol's diagnostic notes.

## Independent validation

[audit.py](audit.py) does not import the benchmark's statistics or validators.
It checks complete shape/candidate coverage and order; rejection causes;
full-output receipts; fixed thread/subgroup/fragment plans; actual shared
allocations; source hashes and identity across order reversals; unchanged
aligned controls; selected versus freshly replayed schedules; GPU nanosecond
conversion and five-sample medians; and all four comparison metrics.
Five deliberately corrupted reports must fail the audit. The latest compiler
artifacts are rehashed separately; older sessions retain their historical hashes.

```sh
uv run --offline --no-project --python 3.13 python \
  scripts/benchmark/tile_torch/results/m1-max-20260906-mpp-mn-bounds/audit.py \
  --current-artifacts --self-test
```

The final Metal matrix suite passes **2,334 assertions in 30 tests**:
[receipt](matrix-shuffle-wide/receipt.json), [log](matrix-shuffle-wide/test.log).
Its new low-level probe covers 168 contracts × six payloads = 1,008 complete
outputs, including 16×64/64×16 fragments spanning multiple classification
chunks. Padded output elements are observed. Signed-zero checks apply to the
explicit empty-operand realization; nonempty MPP results use the documented
numeric/NaN/Inf oracle. Eighteen malformed typed calls are rejected before
launch. High-level tests retain nonzero padding, extra masks, unequal K,
alias/mutation/escape boundaries, transposes and pipeline recurrence checks.

Every recorded binary test is preceded by a complete selected-tree CMake build.
The final optional patch passes `git apply --check` against copies of the
bounded-K baseline; applying it reproduces both external codegen files exactly.
The user-owned barrier flag change in `cooperative.cpp` remains untouched;
this is not a whole-worktree all-green claim.

The final current-TVM CPU suite passes 3,058 assertions
([receipt](cpu-final/receipt.json)); the current bridge linked to frozen
pre-MNK TVM passes 1,875 assertions ([receipt](old-tvm-final/receipt.json)).
The latter exercises the old M/N fallback; capability-specific tests return
without executing new contracts. Syntax/tidy checks report no errors;
existing warnings outside the new code remain, so warning-free status is not
claimed.

The nine selected planner/pipeline/PoC CTests pass
([log](selected-ctests.log)); all 95 Python benchmark unit tests also pass
([log](python-tests.log)). These targeted results do not include the two
older barrier-source assertions affected by the unrelated local change.

Documentation QA retains the existing Sphinx report/reference surfaces.
A fresh `sphinx-build -E -a -b html -W --keep-going` completes rendering but
exits with ten existing missing-Doxygen-XML warnings in `api_reference.rst`,
and no new Tile warning. `scripts/check_docs.py` passes 48 HTML pages, 3,655
local links/assets and 199 compatibility anchors. The rendered five-row
table matches the independent audit exactly at desktop and mobile widths
([automated receipt](docs-qa.json)). Manual screenshot inspection confirms
readable text, no page-level horizontal overflow, and local horizontal
scrolling that exposes every mobile table column. The temporary screenshot
paths in the receipt are local QA evidence, not portable published assets.

## Interpretation and next questions

Forwarding removes A/B staging, but ragged output still uses 16 KiB of shared
storage and four barrier sites. The next structural step is a proved masked
direct-output realization, not an assumption that padded outputs are dead.
Physical-K and empty/partial-subgroup work also need explicit cost features:
the common BK=16 candidate still regresses on the small shape, and nominal K
is not the actual memory/contraction interval. No coefficients are fitted to
this small noisy cohort and no schedule becomes a production default.

The new capability is optional, preserves older TVM ABIs and leaves forwarding
default-off. It does not add a DSL primitive, assume an opaque MPP lane layout,
reduce input precision, or select a CPU fallback. Native MPP, SIMD and other
operators are outside this performance checkpoint.

Visual contract: exact five-shape/two-order lookup in the existing Sphinx
benchmark table, using its neutral theme and locally scrollable mobile table.
No trend, fitted curve or statistical interval is supported by two observations.
The table exposes new/old times and contemporaneous MPS/Torch ratios; all four
metrics remain available in the audit. The technical summary, definitions,
methods, limitations, next action and open questions are split between the
owning report/reference pages and these linked notes.
