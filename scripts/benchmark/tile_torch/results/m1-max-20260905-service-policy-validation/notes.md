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
model-selection diagnostics, **not held-out acceptance results**. The subsequent
four-round shape-held-out replay improves GPU throughput in every pair for
nine cases, but two small cases regress in every pair and one is mixed.
At 768×6144, softmax/RMSNorm/LayerNorm gain **1.360×/1.287×/1.231× GPU** and
**1.372×/1.280×/1.233× E2E throughput** over the old automatic planner.
All 288 outputs from plan collection and replay pass executed validation.
The defaults remain unchanged: the regressions are acceptance failures,
not omitted exceptions or reasons to retune on the held-out labels.

## Held-out results: larger cases improve, two small cases regress

The implementation, fit and protocol were committed and pushed as `47314e616`
before these labels were collected. This is a **shape holdout**, not an
operator/device holdout: the same three FP32 row operators run on M1 Max
with Torch 2.14.0. Both variants fix V=4/U=1/P=1. The reference is the old
automatic width with reload, not a measured-search winner. The candidate
jointly chooses width and reuse by the frozen model, not by new timing labels.

GPU means no-counter **command-buffer execution**, not isolated-kernel time.
Times are medians of four per-round p50 values; gains are medians of paired
reference/candidate ratios. Their displayed medians need not have exactly
that ratio. Ranges are observed min–max, not confidence intervals. Sources:
[replay results](../m1-max-20260905-service-policy-replay/results.json) and
the complete independent [audit receipt](audit.txt).

| 768×6144 op | Old GPU µs | Model GPU µs | GPU gain [range] | E2E gain |
|---|---:|---:|---:|---:|
| softmax | 78.017 | 57.414 | 1.360× [1.353, 1.364] | 1.372× |
| RMSNorm | 78.097 | 60.753 | 1.287× [1.279, 1.289] | 1.280× |
| LayerNorm | 92.931 | 75.251 | 1.231× [1.218, 1.243] | 1.233× |

Every anchor GPU and E2E-throughput pair improves. The mappings change from
W=384 to W=192/256/256, with input caching enabled. Candidate-run Torch GPU
medians are 127.085/78.642/281.360 µs and paired native/Torch ratios are
0.452/0.772/0.267. These gains combine width and reuse, not one isolated pass.

The other nine cases retain both regressions and the mixed small RMSNorm
result. They must be included when deciding whether to change defaults.

| Operation / shape | GPU gain [range] | E2E throughput gain [range] |
|---|---:|---:|
| softmax 37×1537 | 0.904× [0.846, 0.932] | 0.920× [0.908, 0.945] |
| softmax 256×3072 | 1.075× [1.037, 1.112] | 1.072× [1.053, 1.081] |
| softmax 64×12289 | 1.152× [1.149, 1.172] | 1.160× [1.122, 1.206] |
| RMSNorm 37×1537 | 0.987× [0.894, 1.043] | 1.017× [1.002, 1.023] |
| RMSNorm 256×3072 | 1.089× [1.059, 1.146] | 1.092× [1.050, 1.097] |
| RMSNorm 64×12289 | 1.089× [1.074, 1.122] | 1.093× [1.083, 1.098] |
| LayerNorm 37×1537 | 0.876× [0.818, 0.928] | 0.847× [0.821, 0.879] |
| LayerNorm 256×3072 | 1.147× [1.118, 1.163] | 1.147× [1.134, 1.164] |
| LayerNorm 64×12289 | 1.224× [1.187, 1.243] | 1.223× [1.200, 1.232] |

The model selects W=416/cache for all three 37×1537 cases instead of
W=192/reload. Softmax and LayerNorm regress in **every GPU and E2E-throughput
pair**; RMSNorm GPU is mixed. The experiment does not isolate whether width,
caching or their interaction causes the regressions. A fixed 2×2 width/reuse
ablation is the next diagnostic, not a post-hoc winner substitution.

Eleven candidate cases beat eager Torch in every GPU-throughput pair.
37×1537 RMSNorm is the exception: 5.157 versus 5.021 µs, paired time ratio
1.024 [0.949, 1.070]. All twelve beat Torch E2E batched throughput in every
pair. Torch softmax has preallocated output; its functional norms allocate
returned outputs inside warm timing. These are route/operator measurements,
not general framework superiority or a production LLM benchmark.

## GPU and end-to-end dispatch remain separate measurements

The following anchor values are median µs, shown as native / Torch. Batched
throughput and synchronized single-call latency are distinct phases. Large
throughput gains do not establish comparable single-call host-latency gains:
only one of twelve cases improves E2E single-call latency in every A/B pair.

| 768×6144 op | GPU batch µs/op | E2E batch µs/op | GPU single µs | E2E single µs |
|---|---:|---:|---:|---:|
| softmax | 57.414 / 127.085 | 58.321 / 136.194 | 80.625 / 138.125 | 285.563 / 439.895 |
| RMSNorm | 60.753 / 78.642 | 62.432 / 83.321 | 80.479 / 96.271 | 321.375 / 338.583 |
| LayerNorm | 75.251 / 281.360 | 76.873 / 290.837 | 96.271 / 289.229 | 340.417 / 532.854 |

Do not subtract GPU and host medians to infer dispatch overhead. Small cases
can even have a lower independently sampled host median than GPU median.
Instrumented compute-pass timing remains diagnostic: Torch's throughput
probe/control ratio spans 0.895–4.357 in this replay. It is neither the
selection metric nor a correction factor for the primary no-counter phase.

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
fits (`rtol=1e-8`, `atol=1e-12`). No held-out timing label had been observed
or used at the implementation checkpoint; the fit remains byte-identical
through the completed replay.

The [independent audit](audit.py) checks all 32 widths per resource choice
against a separately implemented fixture access/ownership oracle, then checks
that JIT resource selection used model costs, not measured labels. Plan
collection has 48 valid measurements / 96 outputs; four-round replay has
96 valid measurements / 192 outputs. All generated source hashes and complete
plans match their frozen originals. The replay checks 21 unchanged artifacts,
including the executable, bridge, three TVM libraries, timing helper and
calibration file. The offline receipt audits recorded executed correctness;
the output arrays themselves are not archived or revalidated offline.

Executable SHA-256 is
`26f6c817d6aebba2c011b2bea3faacf786dbf677f37700261067d065d81b15f9`;
TIRx bridge SHA-256 is
`886bda1ebbff189396d1a0b8cfc8f79a38396703e6a29796dc8ebf327fd902b1`.
The implementation did not change between plan collection and replay.

Reproduce the offline audit from the repository root:

```sh
uv run --no-project --python 3.13 python \
  scripts/benchmark/tile_torch/results/m1-max-20260905-service-policy-validation/audit.py replay
```

The Sphinx HTML build succeeds with only two existing unrelated toctree
warnings. Rendered Section 9.9 was checked at the default 1280×720 viewport:
all five result-table columns fit, and the negative results and separate
GPU/E2E timing qualifications remain readable directly below the table.

The first full build found Apple's floating `from_chars` API needs
macOS 26; the benchmark now uses a locale-independent classic stream parser
to preserve the existing macOS 13 deployment target. Failed builds are not
counted as verification.

Next, isolate W=192/416 × reload/cache at the three regressing/mixed small
cases, keeping the current profile frozen. A future model revision needs a
new acceptance cohort rather than reusing these labels as an untouched holdout.
Do not enable this profile or caching globally on the current evidence.

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
