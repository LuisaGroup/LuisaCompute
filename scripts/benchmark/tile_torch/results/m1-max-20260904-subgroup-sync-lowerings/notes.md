# Seven independent paths, with an explicit subgroup-sync candidate

The fourteen-round, eight-shape comparison validates **784/784 complete
outputs** across native Tile→MPP, original Tile→TIRx, handwritten MPP, direct
MPS, eager Torch, staged TIRx→MPP and forwarding TIRx→MPP. TIRx remains an
independent maintained lowering, not an alias for Luisa's native emitter.

This run explicitly selects `--tirx-view-subgroup-fences elide` for the last
path. It is **not the default**: fixed-geometry A/B measurements found that
removing all group fences slowed 512³ in every round. The compiler therefore
separates the independence proof from its profitability policy. See the
[controlled A/B report](../m1-max-20260904-subgroup-sync-replay/notes.md).

For 1024³, the forwarding/elision candidate measured **378.932 µs**, versus
original TIRx **429.297 µs**, staged TIRx→MPP **451.513 µs**, native Tile→MPP
**400.534 µs**, handwritten MPP **355.017 µs**, Torch **384.889 µs**, and MPS
**369.750 µs**. Its paired median time is 2.0% below Torch, but it is slower
in 4/14 rounds. It remains **2.3% behind MPS**, slower in 13/14 rounds.
Native Tile→MPP remains 8.4% behind MPS by paired median. This is progress in
candidate representation and validation, not closure of the overall goal.

For 512³, the candidate is **63.531 µs** versus Torch **63.776 µs** and MPS
**66.748 µs**; it is slower than Torch in 6/14 rounds. Its near-Torch result
must not obscure the causal A/B regression against the same view geometry
with retained fences. No universal elision policy is justified.

All times here are synchronized device-resident **host-wall batched times**,
including each runtime's dispatch, encoding/submission and synchronization,
not GPU kernel durations. See the [complete seven-path table](results.md)
and [raw measurements](results.json). JIT, allocation and transfers are
excluded; raw GPU intervals from MPS/handwritten MPP are not mixed into ratios.

## Paired results

Each value is the median of fourteen within-round ratios, with the
forwarding/elision candidate in the numerator. Below one means less time.
These are descriptive paired medians, not confidence intervals.

| M×N×K | / Original TIRx | / Staged TIRx→MPP | / Torch | / MPS | Slower than Torch |
|---|---:|---:|---:|---:|---:|
| 32×32×32 | 0.633283 | 0.668604 | 0.125265 | 0.284321 | 0/14 |
| 128×128×128 | 0.790294 | 0.877529 | 0.220735 | 0.361760 | 0/14 |
| 512×512×512 | 0.885651 | 0.970341 | 0.990750 | 0.943013 | 6/14 |
| 1024×1024×1024 | 0.884214 | 0.835920 | 0.979900 | 1.023224 | 4/14 |
| 256×1024×128 | 0.866738 | 0.894664 | 0.644418 | 0.835545 | 0/14 |
| 1024×128×256 | 0.866066 | 0.924530 | 0.680021 | 0.676883 | 0/14 |
| 127×193×61 | 0.774449 | 0.804891 | 0.255160 | 0.362467 | 0/14 |
| 513×257×129 | 0.921777 | 1.005400 | 0.595688 | 0.575065 | 0/14 |

The six interior cases have proved independent subgroups, zero shared storage
and zero remaining compiler group barriers. Both ragged cases fail the
independence proof and retain staged storage and five static fence sites.
For 513×257×129, the staged and forwarding paths have identical generated
source in every round; the 0.5% difference is not an optimization benefit.

## Controlled identities and retained baselines

All schedules were frozen before measurement, with no search or minimum-of-
rounds selection. Native and handwritten MPP use the same separate
[MPP plan](../m1-max-20260904-mpp-search/results.json). Original/staged TIRx
use the [joint plan](../m1-max-20260904-joint-search/results.json); forwarding
uses the [view plan](../m1-max-20260904-tirx-views-plan.json), with the explicit
elision override recorded in metadata and every command. The latter may
change tile/K geometry relative to the other paths; the full seven-path
improvement cannot be credited to barrier removal alone.

Every original-TIRx and staged-TIRx→MPP source hash matches the preceding
view report, for all shapes and all rounds. Every forwarding/elision source
matches the corresponding A/B prototype source. Thus splitting the default-
off policy from the proof did not change the candidate's emitted device code.
All 22 binaries/libraries matched before/after hashes and were independently
rechecked after measurement; all 23 content-addressed Metal files were also
rehashed. No build, test or profiler ran alongside timing.

Absolute times, including unmodified controls and Torch, are higher than in
the preceding report. Historical medians are not used as an implementation
A/B and no specific thermal/clock explanation is asserted. This report uses
within-session paired results only.

Native/handwritten MPP use fast math and relaxed precision off. TVM's Metal
runtime hardcodes fast math on; the original path uses MSL 3, MPP uses MSL 4.
Those differences are disclosed, not treated as equivalent compiler policies.
The optional same-source TVM/Luisa Runtime controls remain available, but are
not included in this seven-path run. Adding them changes the balancing period.
Planner cost fields remain `cost_basis=simdgroup_reference_geometry`, not
measured internal MPP instruction or register counts.

## Validation and follow-up

Apple M1 Max, macOS 26.6.2, Torch 2.14.0 and NumPy 2.5.2. Seven samples per
measurement, 30-ms target sample duration, 200-ms warmup. Fourteen rounds
balance path positions and pairwise precedence; shape order rotates. There
were no failed or discarded rows. All **184,014,208 output elements** passed
the FP64 oracle (`atol=rtol=1e-4`), maximum observed absolute error zero on the
deterministic dyadic inputs. Non-dyadic/changed-input C++ regressions also pass.

Both patched and unpatched TVM Luisa configurations completed full builds.
The CPU/Metal native/TIRx/system cohort is **23/25** in each, with only the
two existing fence-flag assertion failures; see the A/B report for exact
tests and the unowned `mem_flags(2)` worktree change. The new isolation,
fixed-point forwarding, policy-off and rejection cases pass. The benchmark
contract tests pass **47/47**. Original CPU/Metal lowering remains tested;
an unpatched TVM cannot silently accept an MPP request.

The next cost-model work needs a legal synchronization choice, K granularity,
cohort geometry and working-set/reuse features, calibrated on held-out shapes.
Counting fewer static barriers is not sufficient. The broader CPU realization
gap is unchanged by this Metal experiment, and native MPP still trails its
handwritten/library controls. Those remain work, not completed goals.

Reproduce after a full build and correctness checks, using the command in the
[previous seven-path report](../m1-max-20260904-tirx-views/notes.md), adding
`--tirx-view-subgroup-fences elide` and a **new** output directory. Exact plans,
artifact hashes, per-launch commands and requested/realized policies are in
this report's JSON. No Python source generation or hidden library fallback
is part of either compiler path.
