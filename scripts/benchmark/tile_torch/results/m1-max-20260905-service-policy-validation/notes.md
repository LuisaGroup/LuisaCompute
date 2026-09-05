# Whole-launch reduction cost policy

## Technical summary

The reduction solver now accepts a complete machine objective from a backend
policy, instead of selecting by local program work and multiplying every
winner by the same program-count prior afterward. The old policy remains the
default. An opt-in service policy combines distributed scalar/collective work,
physical subgroup launch demand, and global/private payload access features.
No hardware limit or numerical permission is delegated to the cost model.

A six-coefficient nonnegative fit is frozen in [calibration.json](calibration.json)
using only the preceding 101 valid search trials. Leave-one-shape-out model
selection has 4.81% mean and 22.60% maximum observed ranking regret. These are
model-selection diagnostics, **not held-out acceptance results**. The held-out
experiment defined below has not yet run at this checkpoint. Neither this fit
nor the structural implementation alone justifies changing production defaults.

## Machine demand is separate from the local critical path

The existing `reduction_score` hook is preserved. Its compatibility wrapper
returns local work, the old `ceil(programs / 64)` prior, and their product.
The solver now minimizes the returned **kernel score**, leaving that complete
objective intact in `GroupPlan`. A custom backend can override `reduction_cost`
without a hidden second multiplication. Synthetic tests deliberately make
the cheapest whole-launch choice have the most expensive local score.

```text
Proved IR payload and ownership
             |
             v
  legal (width, packing) candidates <--- hard limits / private budget
             |
             v
  backend reduction_cost(candidate)
       /             |             \
 local work    subgroup demand    global service
       \             |             /
             whole-kernel score
                     |
                     v
      minimum score -> realization

Staged/JIT: independently capture reload and cache
           -> compare whole-kernel scores -> fresh winner JIT
```

The optional service objective uses these definitions:

```text
waves = max(1, threadgroups * subgroups/program * programs/group / capacity)
local = scalar_rounds * R + reductions * subgroups/program * K
        + worker_private_access_bytes * P
score = D + local * waves + programs * program_global_access_bytes * G
        + worker_global_access_bytes * W
```

Packed-tail inactive subgroups count toward physical launch demand; only
logical programs multiply payload bytes. The continuous saturation term is
a fitted prior, not an integer residency calculation. `capacity=512` is not
a queried M1 Max occupancy limit. The six coefficients share fitted GPU
microsecond units here; the generic C++ policy does not mandate time units.
Access bytes remain conservative source-level demand, not DRAM transactions,
cache hits, register counts, spills or emitted instructions.

Missing payload facts reject the explicit service profile. Mixing a fallback
score in abstract issue rounds with a fitted time score would make the ranking
invalid. Nonfinite/negative coefficients and invalid whole-launch estimates
also fail closed. Defaults and existing score overrides retain their previous
behavior.

## Calibration data, objective and uncertainty

The sole training source is
[the previous joint search](../m1-max-20260905-resource-map-search/results.json):
softmax, RMSNorm and LayerNorm at 23×769, 128×2048, 1024×4096 and 128×8193.
All use FP32, V=4/U=1/P=1 and the same 64-scalar private-state budget. The
101 valid trials are fit; 19 resource-infeasible trials remain in the source
but supply no timing label. Fresh search winners and independent replay
measurements are **not** additional training rows.

[fit.py](fit.py) recomputes each timing label from raw no-counter
command-buffer samples. It minimizes the sum of squared relative timing
errors with six nonnegative coefficients, using column-scaled exhaustive
active-set enumeration. Capacity is selected from {64,128,256,512,1024,2048}
by mean per-case leave-one-shape-out ranking regret; tied regrets prefer
smaller capacity. The selected capacity is then refit on all 101 trials.
Kernel names and per-shape winner tables are not model features. Each trial
has equal relative-error weight; cases have differing counts of legal trials.

The table records sensitivity to all six tested capacities. Regret is
`measured(model pick) / measured(best valid trial in that case) - 1`; the
12 case regrets are equally weighted. It is not regret against a hardware
optimum, and minima of noisy training measurements remain optimistic.

| Capacity prior | CV mean regret | CV maximum regret | Training relative RMSE |
|---:|---:|---:|---:|
| 64 | 6.51% | 29.68% | 16.28% |
| 128 | 7.28% | 29.68% | 16.61% |
| 256 | 7.28% | 29.68% | 17.08% |
| 512, selected | 4.81% | 22.60% | 17.58% |
| 1024 | 4.81% | 22.60% | 17.82% |
| 2048 | 6.51% | 29.68% | 18.14% |

The private-access coefficient fits to zero, which does **not** show that
private storage is free. These correlated features and this small cohort
cannot identify register pressure or spill behavior. Small LayerNorm remains
a counterexample: the fitted choice has 22.60% regret in the noisy training
search. The model also omits operation issue mix and target code-generation
effects, so its predicted microseconds must not replace executed measurements.

## Frozen held-out protocol

Before observing new labels, freeze the coefficients, sources and this design:

- Operations: softmax, RMSNorm, LayerNorm.
- Shapes: **37×1537, 256×3072, 768×6144, 64×12289**; none are fitting rows.
- Reference: old analytic automatic width, reload, V=4/U=1/P=1.
- Candidate: service-policy automatic width, plus separately captured
  reload/cache alternatives selected by **model score only**. Both searches
  consider all legal whole-subgroup widths, including non-powers of two.
- Plan collection: five samples, 10 ms batches, 100 ms warm-up. These runs
  execute and validate alternatives but do not use timing labels to select.
- Acceptance replay: freeze complete selected plans; four independently
  JIT-compiled rounds, nine samples, 30 ms batches and 200 ms warm-up. Balance
  A/B and native/Torch order independently, rotate cases, retain every case.
- Verify full outputs against the FP64 oracle, all plan/source hashes and
  compiler/runtime artifacts. Retain regressions and mixed individual pairs.

The comparison includes resource and width decisions; it does not isolate a
single coefficient's effect or claim an exhaustive measured optimum. Widths
not in the five-width training search are deliberate model extrapolations.
Passing this one-device holdout is not enough to enable a universal default.

GPU primary means **no-counter command-buffer execution**, including any GPU
gaps/blits in that buffer, not isolated-kernel time. Report separately the
uninstrumented E2E batched throughput and synchronized single-call latency.
Do not subtract independently sampled GPU and host medians. Instrumented
compute-pass timing is diagnostic only. Inputs are device-resident and JIT,
uploads and cold initialization are outside warm timings. Torch is eager;
softmax has preallocated output on both sides, whereas its functional norms
allocate the returned output during timing. This is TIRx/TVM runtime, not
Luisa native Metal, MPP/MPS or the XIR route.

## Verification and next steps

The complete `cmake-build-tirx` build passes. The Python benchmark contract
suite passes **89 tests**, including model-only selection with contradictory
timing labels, fresh winner JIT, exact profile replay, and independent
recomputation of the emitted objective. The planner executable passes
**5,988 assertions in ten tests**. Full Tile CTest is **31/33**, with the same
two known Metal source assertions requiring `mem_flags(3)` while the user's
untouched local `cooperative.cpp` uses `mem_flags(2)`. The CPU and Metal
execution suites, including the new machine-objective selection test, pass.
The full [CTest log](ctest.log) retains the failures; this is not a clean
whole-repository test claim. All six C++ files pass the changed-line formatter.
All five changed translation units pass clangd syntax checks; existing
style/performance warnings are retained rather than rewritten as part of this
change.

Independent SciPy NNLS solutions agree with all six active-set calibration
fits (`rtol=1e-8`, `atol=1e-12`). No held-out timing label has been observed
or used at this implementation checkpoint.

The first full build found Apple's floating `from_chars` API needs
macOS 26; the benchmark now uses a locale-independent classic stream parser
to preserve the existing macOS 13 deployment target. Failed builds are not
counted as verification.

Open questions remain: how well does the simple service surface extrapolate
to ragged widths, larger live state, other operation mixtures and other Apple
GPUs? Does a guarded incumbent policy improve small-kernel decisions? Can
plan-only JIT avoid executing alternatives once the calibrated candidate
family is trusted? None of these are resolved by the current fit.

This is an additive update to the repository-native technical report. Its
structure covers summary, model, evidence/metrics, methods, uncertainty,
verification and further questions. The exact-lookup capacity table supports
model-selection audit; a trend chart is omitted because these are alternative
priors, not temporal observations. The execution flow describes the API
boundary without encoding unsupported physical occupancy claims.
