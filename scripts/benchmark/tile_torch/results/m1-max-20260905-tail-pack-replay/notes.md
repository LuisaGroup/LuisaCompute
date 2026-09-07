# Guard complete reduction packs once

## Technical summary

The TIRx reduction mapper now guards complete worker packs as a unit and
handles the unique partial worker separately. It preserves logical ownership,
private storage, each worker's FP32 recurrence, and all collective/barrier
positions. No layout, cache policy or cost coefficient changes in this A/B.

At 37×1537, four-round paired end-to-end batch gains over the previous emitter
are **1.134× / 1.207× / 1.210×** for softmax/RMSNorm/LayerNorm; every pair
improves. GPU control improves in every softmax pair, but not every norm pair.
Large background variability also affects identical-source controls. Keep
this as bounded structural and E2E evidence, not universal GPU/MPS dominance
or a new cost-model calibration dataset. All **192 replay outputs** validate.

## Ownership proves a cheaper tail partition

For domain size N, W workers and V consecutive elements per worker, write
`r = N mod (W*V)`, `a = floor(r/V)`, `b = r mod V`. Complete W×V chunks keep
their existing emission. The final chunk is partitioned as follows:

```text
worker w < a        -> complete pack: e = 0 .. V-1, one worker guard
worker w = a, b > 0 -> partial pack:  e = 0 .. b-1, one worker guard
all other workers  -> no tail elements
                       |
                       v
                 uniform collective / barrier
```

These worker sets are disjoint and cover exactly `[0,r)`. No inactive element
is loaded, stored or used to index private storage. Both forms visit a
worker's elements in the same order, and collectives remain outside the
worker guards. This is an implementation change to the proved blocked-cyclic
map, not a new DSL primitive or permission to reorder arithmetic.

The original [2×2 diagnostic](../m1-max-20260905-width-cache-ablation/notes.md)
found a width-dependent regression. For N=1537/W=416/V=4, the old all-tail
path emitted four guarded scalar operations per phase, even though 384
workers own complete packs. New generated softmax/LayerNorm source has
**14 instead of 24** `if` statements; RMSNorm has **8 instead of 14**.
These are source counts, not measured branch instructions or register counts.

## Fixed-plan performance: small E2E gains survive all four pairs

The cohort is the same twelve M1 Max FP32 cases as the earlier service-policy
holdout: three operators at 37×1537, 256×3072, 768×6144 and 64×12289.
This is an implementation A/B, **not a new shape holdout**. Both sides use
the identical frozen width, V=4/U=1/P=1, cache choice and six-coefficient
profile. Every plan is independently verified unchanged by [audit.py](audit.py).

Gains below are medians of within-round old/new ratios; ranges are observed
min–max, not confidence intervals. E2E batch timing is uninstrumented warm
host-wall µs/op with device-resident inputs, dispatch and final synchronization
included. GPU means no-counter command-buffer execution including gaps/blits,
**not an isolated kernel**. The two metrics use separate sampling phases.

| 37×1537 operator | E2E gain [range] | GPU gain [range] |
|---|---:|---:|
| softmax | 1.134× [1.053, 1.250] | 1.157× [1.118, 1.224] |
| RMSNorm | 1.207× [1.188, 1.229] | 1.222× [0.578, 1.260] |
| LayerNorm | 1.210× [1.142, 1.252] | 1.175× [0.508, 1.216] |

All twelve small-case E2E pairs improve. GPU norm medians favor the new emitter,
but the last pair regresses sharply, so an all-round GPU win is not established.
Synchronized single-call E2E latency remains mixed for softmax/RMSNorm
(median gains 0.984×/0.986×); LayerNorm improves in all four pairs,
1.158× [1.017, 1.349]. Batch improvements do not automatically transfer to
single-dispatch host latency.

The other nine cases remain in the evidence, including six byte-identical
generated-source controls. Their exact results follow; source identity is a
control, not permission to remove inconvenient timings.

| Case | Source changed? | GPU gain [range] | E2E gain [range] |
|---|---|---:|---:|
| softmax 256×3072 | no | 1.006× [0.791, 1.780] | 1.016× [0.995, 1.105] |
| softmax 768×6144 | no | 1.000× [0.819, 1.004] | 1.001× [0.999, 1.041] |
| softmax 64×12289 | yes | 1.006× [0.874, 1.034] | 1.008× [0.939, 1.022] |
| RMSNorm 256×3072 | no | 0.921× [0.487, 1.006] | 1.041× [0.941, 1.269] |
| RMSNorm 768×6144 | no | 1.001× [0.866, 1.072] | 1.000× [0.994, 1.173] |
| RMSNorm 64×12289 | yes | 1.060× [0.892, 1.454] | 1.066× [0.971, 1.084] |
| LayerNorm 256×3072 | no | 0.991× [0.847, 1.020] | 0.999× [0.870, 1.007] |
| LayerNorm 768×6144 | no | 1.060× [0.992, 1.407] | 0.995× [0.973, 0.998] |
| LayerNorm 64×12289 | yes | 1.065× [0.916, 1.251] | 1.067× [1.050, 1.087] |

The identical-source controls show why the full cohort cannot establish a
stable new GPU ranking. Later host samples also rise substantially for both
native and Torch. A post-run process snapshot found substantial background
media/compositor CPU activity; it is not a continuous GPU trace and does not
prove the cause of each slow sample. No application was terminated, no round
was omitted, and no counter-based correction or minimum-only ranking is used.

## Torch comparisons and their limits

For the small cases, candidate/Torch paired E2E batch time ratios are
0.159/0.543/0.405; all four pairs favor the candidate. GPU ratios have
medians 0.233/0.807/0.598, but RMSNorm and LayerNorm each exceed Torch in the
last pair. These are eager Torch 2.14.0 MPS operator comparisons, not direct
MPS, native MPP, XIR or compiled-graph results. Torch softmax preallocates its
output; functional norms allocate returned outputs inside timing. Raw GPU,
E2E batch and single-call times for every case/provider are in [audit.json](audit.json).

## Validation, protocol and reproducibility

[results.json](results.json) retains all 96 measurements and 192 executed
output validations against the FP64 oracle. Four rounds independently JIT
both binaries, use nine samples, 30 ms host batches and 200 ms warm-up,
balance variant and native/Torch order, and rotate cases. GPU batches use
the existing at-most-64 repetition cap. [audit.py](audit.py) reuses the prior
independent ownership/access oracle, recomputes medians from raw samples,
checks twelve identical full plans and stable per-variant sources, and
verifies the source-control subset. No production statistics helper is used.
The output arrays themselves are not archived for offline reexecution.

The replay records **43 unchanged compiler/runtime artifacts**, including
separate old/new bridge snapshots, the three TVM libraries and the timing
helper. The old executable/bridge hashes are `26f6c817…` / `886bda1e…`;
the complete new hashes, commands and frozen source report are in metadata.
The prior binary was copied before rebuilding; no build ran during replay.

The full selected CMake build passes. Full Tile CTest remains **31/33**:
the two known generated-source assertions conflict with the user's untouched
local `mem_flags(2)` change. Numerical tests pass, including **28 new tail-pack
configurations** at widths 192/416, with and without caching, for N=1536,
1537,1538,1539,1663,1664,1665. Existing V=1/2/4/8, packed-program, guarded-input
and CPU execution cases also pass. The full
[CTest log](../m1-max-20260905-width-cache-ablation/ctest.log) retains the
failures. Python benchmark contracts pass **89/89**. Both changed C++ files
pass changed-line formatting and clangd syntax checks with the selected
`cmake-build-tirx` compilation database; initial auto-detection used the wrong
database and its missing-header diagnostics were not treated as source bugs.

## Next work and unresolved questions

Keep the generic tail emitter and its numerical coverage, but leave the
calibrated service policy opt-in. Repeat the mapping comparison on the new
emitter in a quieter GPU window before revising its issue/control-flow model.
In particular, do not fit noisy GPU-control labels, assume the old 416/cache
choice is now optimal, or multiply old/new ratios across separate experiments
to claim the earlier regression is fully closed.

The next cost-policy revision needs realized tail/control-flow demand and
new independent validation; it must remain separate from legality and
ownership proofs. Stable GPU norm gains, general single-dispatch improvements,
cross-device behavior and the broader PyTorch performance goal remain unproven.

<!-- Presentation contract: exact factorial and paired-control lookup are
the table tasks; a trend or single aggregate bar would hide interaction and
unchanged-source controls. Keep all ranges, units and adjacent limitations.
This extends the existing repository Markdown/Sphinx report, not a parallel
dashboard or replacement artifact. -->
