# Reduction access demand and joint resource search

## Technical summary

The TIRx reduction planner now exposes conservative global/private payload
read/write demand to backend cost policies. Staged/JIT tuning can jointly
enumerate input reload/cache and execution mappings, preserving invalid
candidates and freshly recompiling the selected winner. Implementation and
regression evidence were committed as `d579211f9`.

The subsequent four-round replay compares the joint-search winner against
the **best measured reload width in the same search family**. At 1024×4096,
softmax, RMSNorm and LayerNorm gain **1.200× / 1.214× / 1.234× GPU throughput**
and **1.199× / 1.221× / 1.248× E2E throughput**, with every paired round
positive. All 11 changed-source cases have positive median GPU gains; seven
improve in every GPU pair and four have mixed individual pairs. One unchanged
LayerNorm case is retained as a control. All 192 replay and 226 search/fresh
winner outputs pass executed validation; the independent audit passes.
This is still **not a calibrated replacement cost model**, default-policy
comparison or exhaustive hardware optimum.

## GPU execution and end-to-end dispatch

GPU below means **no-counter command-buffer execution**, including GPU work,
gaps and any blits inside the buffer; it excludes CPU encoding/completion
notification and is not isolated-kernel time. E2E batched throughput and
synchronized single-call latency are separate uninstrumented host phases.
Inputs are device-resident, JIT/transfers outside warm timing. Torch is eager
2.14.0: softmax has preallocated output on both sides, while functional norms
allocate their return values inside Torch warm timing. Native is TIRx/TVM
runtime, not the Luisa native Metal, MPP, MPS or direct XIR route.

The anchor results below use medians of four per-round p50 times; gains are
medians of paired reference/candidate ratios. Their displayed time medians
need not have exactly that ratio. Brackets are observed min–max, not confidence
intervals. The reference is tuned reload, not the prior fixed-W=512 baseline.

| 1024×4096 op | Reload GPU µs | Joint GPU µs | GPU gain [range] | E2E gain |
|---|---:|---:|---:|---:|
| softmax | 59.162 | 49.179 | 1.200× [1.195, 1.210] | 1.199× |
| RMSNorm | 64.211 | 52.826 | 1.214× [1.198, 1.221] | 1.221× |
| LayerNorm | 75.660 | 61.316 | 1.234× [1.210, 1.240] | 1.248× |

Softmax uses W=1024 on both sides, RMSNorm changes W=1024→256, and LayerNorm
uses W=128 on both sides. Thus RMSNorm's gain combines width and input reuse;
it is not a fixed-width caching effect. The smaller/new-shape evidence is
retained below, including all mixed GPU pairs and the unchanged-source control.

| Operation / shape | GPU gain [range] | E2E gain [range] |
|---|---:|---:|
| softmax 23×769 | 1.048× [0.992, 1.105] | 1.050× [1.023, 1.063] |
| softmax 128×2048 | 1.141× [0.980, 1.169] | 1.068× [1.052, 1.126] |
| softmax 128×8193 | 1.152× [1.056, 1.244] | 1.145× [1.123, 1.219] |
| RMSNorm 23×769 | 1.211× [0.967, 1.246] | 1.105× [1.098, 1.109] |
| RMSNorm 128×2048 | 1.035× [1.000, 1.084] | 1.060× [1.019, 1.110] |
| RMSNorm 128×8193 | 1.046× [1.037, 1.057] | 1.042× [1.018, 1.063] |
| LayerNorm 23×769, control | 0.924× [0.909, 1.023] | 1.010× [0.985, 1.047] |
| LayerNorm 128×2048 | 1.136× [1.058, 1.178] | 1.117× [1.100, 1.169] |
| LayerNorm 128×8193 | 1.095× [1.069, 1.098] | 1.097× [1.088, 1.114] |

The minimum RMSNorm 128×2048 GPU gain is 0.999619× before rounding, so it
belongs to the mixed group. All 11 changed-source cases improve in every E2E
throughput pair. The control's apparent GPU regression is not an implementation
change; it exposes small-kernel measurement variability and is not a noise
correction to subtract. These finite observations do not justify a universal
cache default or a fitted ranking model.

All candidate cases are also faster than eager Torch in each paired GPU
throughput measurement on this cohort. At the anchor, paired native/Torch
GPU-time ratios are 0.401 / 0.761 / 0.297. The next table keeps GPU and E2E,
batch and single-call measurements distinct; values are median µs, written
as native / Torch. It describes this measured deployment path and allocation
policy, not general framework superiority or a production-network benchmark.

| 1024×4096 op | GPU batch µs/op | E2E batch µs/op | GPU single µs | E2E single µs |
|---|---:|---:|---:|---:|
| softmax | 49.179 / 122.653 | 50.156 / 131.108 | 55.729 / 130.271 | 293.730 / 416.479 |
| RMSNorm | 52.826 / 69.742 | 53.938 / 74.218 | 71.708 / 79.979 | 303.354 / 323.521 |
| LayerNorm | 61.316 / 205.799 | 62.284 / 213.604 | 73.396 / 203.646 | 297.042 / 468.771 |

Do not subtract GPU and host phase medians to infer dispatch overhead. Their
sampling phases differ; the full report even contains small cases where a
host median falls below the independently sampled GPU median. Instrumented
compute-pass timing remains diagnostic: Torch's command-buffer probe/control
ratio spans 0.878–4.959 across this replay, so probe timing is not used for
selection or cross-framework performance claims.

## Feature contract and scope

`ReductionCandidate` and `GroupPlan` carry an availability flag, per-program
demand and per-worker demand. The latter sums the longest worker stripe for
each distributed domain, rounded independently by its ownership map.
Identical buffer loads count once per statement/expression; later statements
and phases are separate. Both lazy branches and zero-filled tails count
potential demand. Scalar setup, carry variables and collective scaffolding
are outside this payload metric. Unsupported constructs mark the complete
feature unavailable with zero values. No physical cache, DRAM, register or
spill count is inferred.

Optional global/private byte coefficients are finite, nonnegative and zero
by default. The legacy scalar/collective/setup score is therefore unchanged.
Backend policies can use the complete facts without changing the bridge's
ownership, immutability or cumulative resource-budget checks. The coefficients
used in unit tests are synthetic arithmetic checks, not M1 Max calibration.

`--tune-reduction-input-caches reload,cache` adds a finite Cartesian dimension
to the existing JIT search. Every configuration is separately captured and
compiled; invalid configurations retain their reason, and a fresh capture
validates the winner without publishing its search minimum as acceptance
evidence. The old cost score does not prune this measured search. Cache and
mapping defaults are unchanged.

## Verification

The complete `cmake-build-tirx` tree built successfully before the tests.
All five changed C++ translation units passed `scripts/check_cpp_syntax.py`
against that compile database; `git-clang-format --diff HEAD` reports no
formatting changes for the six changed C++ files.

- Python benchmark suite: **87 tests pass**, including feature-schema
  validation, cache/width product and budget, rejected candidates, fresh
  winner capture, and a winner with a worse old analytic score.
- Planner unit executable: **5,941 assertions in nine tests pass**.
- Focused Metal input-reuse test: **89,942 assertions pass**, exercising the
  existing 22 numerical configurations with additional access accounting
  checks. Same-expression `x*x` contributes one global read, cross-phase
  reuse changes the correct demand, and the padded tail remains conservative.
- Full `test_tile_*` CTest cohort: **31/33 pass**. The two existing Metal
  generated-source assertion failures require `mem_flags(3)` while the user's
  untouched local `cooperative.cpp` uses `mem_flags(2)`. Numerical checks pass;
  neither tests nor that local edit are changed or submitted. This is not a
  claim of a clean whole-repository test run.

The first focused test invocation used a nonmatching filter and ran zero
assertions; it was not treated as validation. The corrected filter below is
the passing invocation. The full [CTest log](ctest.log) preserves all failures.

```sh
cmake --build cmake-build-tirx -j 8
uv run --no-project --python 3.13 --with numpy python -m unittest discover -s scripts/benchmark/tile_torch -p 'test_*.py'
cmake-build-tirx/bin/test_tile_tirx_planner
cmake-build-tirx/bin/test_tile_tirx_execution metal tile_execution_reduction_input_cache
ctest --test-dir cmake-build-tirx -R '^test_tile_' --output-on-failure -j1
```

## Joint search and frozen acceptance protocol

The experiment fixes V=4/U=1/P=1 and a 64-scalar private budget, and
searches W={32,128,256,512,1024} × {reload,cache} for softmax, RMSNorm and
LayerNorm at 23×769, 128×2048, 1024×4096 and 128×8193. Three shapes differ
from the earlier fixed-width cache cohort; 1024×4096 is an anchor. The ragged
8193-column case crosses the private-budget boundary at narrow widths.

The completed search records **101 valid and 19 rejected trials**, followed
by 12 fresh winner JITs: all **226 executed native/Torch outputs** pass
validation. The independent audit recomputes raw timing medians, checks all
resource facts, and confirms the six W=512 anchor sources are byte-identical
to the prior input-cache implementation. Extracting these facts has not
changed those kernels. Search timings are not independent acceptance evidence.

The reference is the best valid reload member of that same measured family;
the candidate is its joint-search winner. Both catalogs were frozen before
four balanced replay rounds, without a gain filter. Eleven candidate kernels
cache the input; the smallest LayerNorm chooses reload and becomes an
identical-source control. Widths differ in three reference/candidate pairs.

The complete frozen mapping is exact-lookup evidence, not a performance
ranking. Cache consumes the existing cumulative private budget: at N=8193,
W=256 softmax/LayerNorm would need 66 scalars and is rejected; W=512 needs
34 and is legal. RMSNorm has only the input stripe and admits W=256 with
33 scalars. Thus cache/reload cannot be selected independently of ownership.

| Operation / shape | Reload reference W | Joint candidate W | Candidate input |
|---|---:|---:|---|
| softmax 23×769 | 32 | 32 | cache |
| softmax 128×2048 | 256 | 256 | cache |
| softmax 1024×4096 | 1024 | 1024 | cache |
| softmax 128×8193 | 1024 | 1024 | cache |
| RMSNorm 23×769 | 1024 | 256 | cache |
| RMSNorm 128×2048 | 128 | 128 | cache |
| RMSNorm 1024×4096 | 1024 | 256 | cache |
| RMSNorm 128×8193 | 256 | 256 | cache |
| LayerNorm 23×769 | 256 | 256 | reload/control |
| LayerNorm 128×2048 | 128 | 256 | cache |
| LayerNorm 1024×4096 | 128 | 128 | cache |
| LayerNorm 128×8193 | 512 | 512 | cache |

No-counter GPU command-buffer execution is the stated selection metric,
alongside separate batched and single-call E2E dispatch phases. Do not label
command-buffer time as isolated kernel time or subtract independent phase
medians to estimate overhead. The four-round replay retains all 12 cases,
balances reference/candidate and native/Torch order independently, rotates
case order, and freshly captures/JIT-compiles all 96 measurements. It uses
nine samples, 30 ms batches and 200 ms warmup; search uses five samples,
10 ms batches and 100 ms warmup. All 192 replay outputs are checked against
the harness's FP64 oracle.

The executable, adjacent Tile libraries and explicitly fingerprinted TVMx
compiler/runtime and timing libraries are unchanged between both replay
variants. Every source hash and complete execution plan equals its frozen
catalog entry, including all new access facts. The independent
[audit receipt](audit.txt) recomputes raw GPU/E2E medians and verifies exact
coverage, artifact identity and both order balances.

## What the new facts explain, and what they do not

At N=4096/W=512/V=4, caching adds eight scalar rounds per worker in all
three operators, while reducing global read demand by 32 bytes and increasing
private read/write demand by 64/32 bytes. The old scalar-round prior therefore
ranks the cached plan eight units worse. With its unit scalar coefficient
and the optional global/private byte coefficients `g,p`, the score difference
would instead be `8 - 32*g + 96*p`. This is an algebraic sensitivity check,
not fitted coefficients or a prediction in microseconds.

The search's old-model regret ranges from 3.91% to 39.18% relative to its
in-search GPU minimum. That noisy in-cohort quantity must not be described as
independent performance gain. Adding a global term can rank same-width input
reuse sensibly, but cannot alone establish full-device service, live-state
occupancy, tail-code or cross-width ranking. No replacement policy, lookup
table or default is promoted from this search.

Raw [search measurements](../m1-max-20260905-resource-map-search/results.json),
[reference catalog](reference-plan.json), [candidate catalog](candidate-plan.json)
and [independent audit](analyze.py) retain exact configurations, source hashes,
correctness records and all failures. The audit uses the standard library,
not the benchmark's selection or statistics helpers. Executed numeric arrays
are not archived: offline checking validates recorded correctness, not those
arrays again. The [frozen replay](../m1-max-20260905-resource-map-replay/notes.md)
contains the complete raw measurements and reproduction command. Three shapes
are new relative to the preceding fixed-width cache cohort, but they are not
held-out model validation: their parameters were tuned on those shapes before
independent acceptance.

## Report and decision boundary

This technical report extends the existing repository Markdown/Sphinx surface;
it does not replace historical sections or add a parallel reporting app.
The mapping table is for exact multi-field lookup; it implies no ordered
trend or speedup. Subsequent timing tables retain paired ranges, multiple
metrics and controls rather than drawing a trend from four chosen shapes.
The replay supports joint resource/mapping selection on this finite family,
with stronger seven-case and qualified four-case GPU evidence separated.
Keep the selected configurations available through explicit JIT parameters;
do not install a winner table or universal cache default. Next, calibrate
full-device global/private service and collective/issue demand with explicit
train/holdout separation, retain an incumbent during independent acceptance,
and add live-state/tail features before relying on the model to prune. Physical
private state, another device/dtype, production LLM blocks and the other
lowering routes remain open. This does not close the overall performance goal.

Sphinx HTML builds successfully with only the two pre-existing unrelated
not-in-toctree warnings (`custom_agility_sdk.md` and
`coro_suspend_extensions.md`). The rendered Section 9.8, its definitions and
all five headline table columns were visually checked at the default
1280×720 browser viewport. No mobile or alternate-breakpoint certification
is implied. The existing report sections and historical evidence were kept;
the reporting/validation workflow adds bounded comparisons and uncertainty
rather than replacing earlier outcomes with the latest search minima.
