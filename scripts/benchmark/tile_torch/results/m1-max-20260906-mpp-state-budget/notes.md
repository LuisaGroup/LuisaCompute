# MPP state budgets: candidate coverage improves; performance unaccepted

## Technical summary

The planner now budgets the explicit state of the selected realization:
`2*rm*rn` logical scalars/lane for MPP's cooperative output versus
`2*(rm*rn+rm+rn)` for the reference emitter's A/B/C fragments. The default
budget stays 64. Explicit minimums are four for MPP's legal 8x16 descriptor
and six for the reference 8x8 atom. Descriptor, coverage, shared-memory and
numerical checks remain independent. This is not a claim about MPP's internal
register allocation or spills, and it does not change native Metal/SIMD.

The independently [audited search](search/audit.json) admits four additional
fixed candidates per tested shape. Correctness is accepted; speed and model
calibration are not. The existing Sphinx matrix reference owns the design and
the route-results page owns the reader-facing status. This supporting record
follows the user's requested documentation structure, not a new report app.

## Scope and definitions

Four shapes: 1024³, 4096³, 1025×1025×1024, 4096×4096×11008. Per compiler and
shape: output blocks 128x32, 128x64, 64x128 and 64x64; BK=4096; threads
64/128/256; pipeline window/copy batch one; explicit MPP direct input views;
FP32; preallocated outputs; no new numerical or fence-elision permission.

Baseline libraries were frozen after a full build of commit 52e17083a plus
the user-owned shared-only barrier edit. Both variants include that same edit;
it is not committed by this change. Native programs use the recorded old/new
loader precedence. See [protocol.md](protocol.md), [experiment.py](experiment.py)
and [search/results.json](search/results.json) for exact commands and hashes.
The 64-scalar search predates the subsequent explicit four-scalar minimum
correction; this later front-door correction does not change its 64-scalar
candidate family or scores. Final correctness tests cover both changes.

Timing has four separate views: synchronized warm batched and single-call
host E2E, and no-counter Metal command-buffer GPU batch and single-call
intervals. The latter include GPU work/gaps, not isolated kernel timestamps.
Instrumented pass measurements are retained as diagnostics only. Search uses
five samples, 20 ms requested windows and 100 ms warmup. No new coefficients
are fitted; Torch and direct MPS are separate library controls.

## Admission and complete-output evidence

Each old compiler search accepts six of twelve requests; each new search
accepts ten. Across four shapes and two variants this is 96 attempts, 64
accepted trials, and eight additional fresh winner compilations. Each accepted
trial and fresh winner checks native, Torch and direct MPS completely against
the deterministic FP64 oracle at atol=rtol=1e-4: 216 outputs and 1,925,296,182
elements, with zero maximum absolute error for these dyadic inputs. Forty-two
new native matrix fixtures separately cover non-dyadic values, partial blocks,
both input orientations, nonzero initializers, one/multiple K steps and
staged/direct input paths. Existing sentinel and observation-boundary tests
remain in the full matrix suite.

The [independent auditor](audit.py) does not import benchmark validation,
percentile or selection functions. It recomputes GPU denominators, GPU/E2E
medians, model/measurement selections, output/state coverage and generated-source
hashes. It confirms byte-identical sources for all 24 common candidates;
these controls are not compiler speedups. A compact exact admission table in
the architecture reference is sufficient; a timing chart would promote data
that is not accepted for speed comparisons. A runnable data-quality companion
is [review.ipynb](review.ipynb).

## Validation and retained failures

The [final full-build/test receipt](final-correctness/results.json) contains
14,319 planner assertions and 4,019 Metal matrix assertions, plus CPU matrix,
CPU/Metal execution, basic/neural/algorithm PoCs and native Metal Runtime.
The independent admission loop checks 4,992 combinations. New exact-budget
tests fail with the old bridge at the expected planner-admission boundary;
these are negative controls, not successful old-library executions.

Retained harness/fixture diagnoses:

- [planner/receipt.json](planner/receipt.json) records an initial mistaken
  `cpu` argument to a no-device planner test. It ran zero assertions and was
  rejected by the receipt checker; it is not a pass.
- The first direct planner run expected the legal 88-scalar reference mapping
  to win despite the default pressure prior. Neutralizing that prior **in the
  boundary test only** separates admission from profitability. Production
  coefficients are unchanged. The old-library comparison, exhaustive coverage
  and final tests verify the admission boundary instead of relying on that
  mistaken assertion.
- The first syntax-check invocation lacked `orjson`; rerunning with the
  repository checker's dependency succeeds. Full native builds and numerical
  tests are independent gates, not replaced by syntax checks.

## Why performance remains unaccepted

Fresh winner GPU time divided by that winner's search-trial GPU time is
1.23–1.88 across all eight case/variant pairs. This is not a paired old/new
speed ratio or a confidence interval. Concurrent desktop/video activity was
observed with read-only process inspection during the run; AC power was
present and `pmset` reported no recorded thermal/performance warning. Those
observations do not identify a causal GPU load or prove thermal stability.
Changing load and minimum-selection bias remain competing explanations.

Assessment: **ready to share for admission/correctness; needs new evidence for
speed or calibration.** No process or application was closed. No benchmark
minimum is promoted to a default or to a Torch/MPS parity claim. The scripted
held-out shapes and six-round frozen replay are prepared but have not been run
in a quiet acceptance window. This is an explicit missing validation step,
not a successful or pending live performance process.

## Next model work and open questions

The model still counts nominal K and some source scalar domains that the
selected realization removes. For example, the 1024³/128x32x4096 direct-output
source has a physical K of 1024 and only tensor run/store, while its plan
records nominal-K work and 12,288 independent source elements. These are
ranking priors, not realized instruction counts. Carry elimination, operand
paths, physical K and edge fractions should feed generic proof-derived cost
features. Correct candidate admission must precede calibration, but does not
by itself establish that the current model selects the fastest candidate.

Use a quiet-window search and independent frozen replay before promotion;
keep the full registered cohort and any reversals. Extend matrix scoring with
reusable realization facts, then test held-out ranking regret rather than
adding a kernel-name/shape dispatch rule. The broad PyTorch-equivalent
performance goal remains open.
