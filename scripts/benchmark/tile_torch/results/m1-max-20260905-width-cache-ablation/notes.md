# Reduction width and input reuse

## Technical summary

The fixed 2×2 diagnostic attributes the small-row end-to-end regression to
the **wider execution mapping**, not to input caching. Holding reuse fixed,
416 workers are slower than 192 in every round for all three operators.
Caching improves five of the six fixed-width comparisons in every round;
192-worker softmax is mixed. The compiler also emits one guard per scalar in
an all-tail pack, which motivated the subsequent
[tail-pack lowering A/B](../m1-max-20260905-tail-pack-replay/notes.md).
No model coefficient, winner table or default was changed from these labels.

## Fixed-width evidence separates the two decisions

This is M1 Max FP32 at **37×1537**, with V=4/U=1/P=1 and immutable input
reuse either disabled (`reload`) or enabled (`cache`). Times below are warm,
device-resident **end-to-end batch µs/op**, medians of four per-round p50s.
Allocation/upload/JIT are outside the native timing; dispatch and final
synchronization are included. The table is for exact lookup across the two
factors, not a time series. Source: the independently recomputed
[audit](audit.json) and all four raw round reports linked below.

| Operator | 192 reload | 192 cache | 416 reload | 416 cache |
|---|---:|---:|---:|---:|
| softmax | 6.058 | 5.851 | 7.802 | 6.506 |
| RMSNorm | 6.929 | 5.896 | 7.644 | 6.611 |
| LayerNorm | 6.793 | 6.416 | 9.300 | 7.559 |

Within-round `192 / 416` time ratios are 0.775/0.910/0.734 with reload,
and 0.904/0.892/0.838 with cache (softmax/RMSNorm/LayerNorm order).
Every ratio is below one in every round. At 416 workers, paired cache gains
are 1.196× [1.165, 1.212], 1.134× [1.112, 1.194] and
1.230× [1.134, 1.266]. The intervals are observed min–max, not confidence
intervals. At 192 workers, softmax's cache gain is only
1.041× [0.993, 1.076]; do not claim a universal caching benefit.

The previous model's combined `192/reload → 416/cache` choice still loses
for softmax and LayerNorm in every E2E pair (0.932× and 0.884× median gains).
RMSNorm is mixed, 1.043× [0.982, 1.087]. Thus the two-factor diagnostic is
consistent with the earlier holdout failure without replacing it by a
post-hoc timing winner.

## GPU control is retained, but is not a stable ranking here

The no-counter control uses completed command-buffer GPU timestamps,
including GPU gaps/blits; it is **not isolated-kernel time**. Some samples
are strongly bimodal: first-round 192/reload softmax spans 4.128–15.650 µs/op
with the same two command buffers per sample. Every raw sample remains in
the audit. No minimum-only statistic, outlier deletion, timing correction or
cost-model fit is made from this diagnostic. E2E and GPU were measured in
separate phases; their medians must not be subtracted.

Torch 2.14.0 eager MPS was also executed and validated for each measurement.
Its softmax output is preallocated, while functional norms allocate returned
outputs inside timing. This experiment isolates our width/reuse choices;
it is not a new general PyTorch/MPS performance acceptance claim.

## Four balanced permutations and complete validation

[run_ablation.py](run_ablation.py) fixed the four width/reuse permutations
before execution. Each cell appears once in each trial position and runs
native-first/Torch-first twice. Each trial is independently captured and JIT
compiled. The existing harness additionally measures its model-selected
choice afresh; those twelve measurements are retained but **excluded** from
factorial effects. The analysis uses all 48 fixed trial measurements.

All **120 output validations** pass (48 trials plus 12 fresh measurements,
two implementations each). The [manifest](manifest.json) records 23 unchanged
artifacts across all four rounds. [audit.py](audit.py) independently
recomputes raw GPU/E2E medians, checks the full cell matrix and order balance,
and verifies twelve stable generated-source hashes. It audits recorded
executed validation; it does not reexecute unarchived output arrays.

Raw data: [round 1](round-1/results.json), [round 2](round-2/results.json),
[round 3](round-3/results.json), [round 4](round-4/results.json).
The reference executable and bridge hashes match the prior frozen service
checkpoint, `26f6c817…` and `886bda1e…`. Runtime snapshots for the subsequent
code A/B were copied before rebuilding, not reconstructed from timing labels.

## Next step: fix code generation before refitting the model

For N=1537, W=416 and V=4 there are no complete W×V chunks. Nevertheless,
384 workers own full four-element packs, one owns a single element, and
31 own no elements. The old emitter guards each scalar separately in every
load/reduction/element/store domain. This exposes a generic lowering issue,
not a reason to special-case a kernel name or prohibit non-power-of-two widths.

The next A/B keeps all twelve previously selected service-policy plans fixed
and changes only tail-pack emission. A future model revision must account
for realized issue/control-flow costs, then use new stable, independent
measurements. Whether the model still over-selects wide groups **after** that
repair remains an open question; these pre-repair labels cannot answer it.
