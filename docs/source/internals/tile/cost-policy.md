(backend-owned-execution-cost-policy)=
# Backend execution cost policies

This reference owns the overridable policy API, candidate features, scoring units and tuning boundaries. The [planner](planner.md) owns proof/search separation; [Metal reductions](reductions.md) owns the realization. Cost estimates are not hardware measurements.

```{contents} On this page
:local:
:depth: 2
```

The TIRx bridge now exports a C++ `ExecutionCostPolicy` interface, with an
`AnalyticExecutionCostPolicy` default implementation. Backend code can
override calibration or the row-program objective without changing candidate
generation, proofs, IR or the solver:

```text
Backend: device limits + calibration/policy
                  |                  |
Bridge: prove -> enumerate legal -> score -> select -> realize
          |           |               ^
          +-- hard constraints        +-- overridable policy
```

`coefficients(limits, basis, prior)` returns the model used by matrix and
reduction planning. `reduction_score(candidate, model)` preserves the local
program-score hook; `reduction_cost(candidate, model)` may replace the complete
machine objective. Its immutable feature record includes logical programs,
threads, subgroups/program, programs/group, shared bytes, private stripe
scalars, reduction count, scalar rounds, ordered unroll factor and consecutive
elements per worker. It also includes the exact physical threadgroup count,
useful scalar element count and useful lane-work fraction
`elements / (scalar_rounds * cooperating_workers)`. These are ownership and
launch facts, not profiler measurements of occupancy or SIMD issue. A backend
can inherit the analytic policy and override either score level. No TVM type or
RTTI crosses this interface.

The solver minimizes `ReductionCost::kernel_score`, not local work. The
compatibility implementation returns the historical program score and
`ceil(programs / preferred_concurrent_programs)` product, preserving existing
policies. A backend-provided whole-kernel score is never multiplied by that
prior again. `ServiceExecutionCostPolicy` is an optional reusable objective
over subgroup launch demand and global/private payload access; a typed
`ReductionServiceModel` supplies its coefficients without device-name tests
inside the solver. Missing access facts reject an explicit service profile
instead of mixing incompatible fallback units. The
{download}`frozen calibration and held-out protocol <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-service-policy-validation/notes.md>`
records the current experiment; this profile is not a new default.

The policy is borrowed through `PlannerOptions::cost_policy` only for the
synchronous compile call; it is not retained by a shader or serialized.
Nonfinite/negative row scores and invalid coefficients fail compilation. A
cheap score cannot waive ownership, numerical permission, resource limits or
an exact mapping constraint. Matrix Pareto DP still requires its additive
coefficient model; this change does not advertise an arbitrary nonlinear
matrix objective, a calibrated universal policy, or an XIR policy interface.

Reduction search now treats packing independently from collaboration.
`reduction_programs_per_group=0` keeps automatic packing of single-subgroup
programs. A positive value fixes packing P and now also admits `S>1` with
per-program partial slots and a proved uniform fence sequence. An exact
thread count is the whole physical group, `T=32*S*P`; without one, fitting
cooperating widths are searched. Packed tails replay a valid program and
suppress external stores, so the service policy counts their extra reads.
Repeated reduction-containing enclosing loops and unsafe read/write replay
are rejected. The [reduction mapping reference](reductions.md#explicit-packing-of-cooperating-programs)
owns the exact admission and ownership contract. This remains an explicit
candidate family: the [fixed replay](../../performance/tile/reductions.md#cooperating-program-packing)
does not justify expanding automatic defaults. Invalid combinations fail
instead of silently falling back. `reduction_unroll_factor` in `[1,16]` controls
bounded partial unrolling of ordered worker stripes, with a separate tail;
it creates no extra accumulators and does not reassociate their recurrence.
The default remains one because measurements show mixed effects.

The benchmark's staged/JIT Cartesian product can now include thread count,
packing, unrolling and materialization. Every candidate and fresh winner is
checked in full; a separate frozen-plan replay is required before claiming a
speedup. The default analytic prior is intentionally unchanged by these
in-cohort trials: it does not yet price unrolling or full-machine row waves.

[Target-width acceptance](../../performance/tile/reductions.md#target-aware-widths-gpu-and-dispatch-acceptance)
contains both improving and regressing search winners. Keep an incumbent and
an independent acceptance phase; useful lane-work fraction alone is not a
complete physical group, memory-service or issue-cost model. State the GPU/E2E
selection objective explicitly.

## Input reuse and access-demand features

The optional `cache_reduction_inputs` extends the materialization choice to
proved immutable inputs reused across distinct consumer domains. It is a
bridge/planner policy, not a Tile DSL memory declaration. It retains only
compiler snapshots admitted by the immutable-view audit; the existing
same-worker audit and cumulative private-stripe budget still decide legality.
An exact cache request rejects cross-worker gather or excess live allocation
instead of silently falling back. Same-domain `x*x` alone is not reuse.

The [fixed-width cache/reload comparison](../../performance/tile/reductions.md#budgeted-immutable-input-reuse)
is the evidence for separating input reuse from width selection. The historical
zero-byte-cost prior penalizes the extra private-copy rounds without rewarding
removed global reloads. It must not prune the measured cache candidate family.

**Resource-sensitive payload access demand** is recorded in `ReductionCandidate` and the realized `GroupPlan`. Each has an
availability flag and separate global/private read/write bytes, both per
logical program and per maximum worker stripe. Worker demand rounds each
distributed domain independently with the candidate's ownership map. Identical
loads count once within one statement/expression, but not across statements
or phases; both lazy branches and zero-filled tails contribute conservative
potential demand. Scalar carry/setup and collective scaffolding are excluded.
These are logical IR facts, **not measured DRAM traffic, register accesses or
spills**. Unsupported payload constructs make the whole feature unavailable
and zero-valued, rather than exposing misleading partial counts.

The analytic policy can optionally add global/private byte-service terms to
its historical scalar/collective/setup prior. Both coefficients default to
zero pending calibration; a backend can instead use the complete program and
worker demands in its overridden objective. Ownership, immutability and
resource-budget proofs remain separate from scoring.

The staged/JIT driver now includes `reload,cache` as a Cartesian dimension
alongside width, packing, unrolling, worker ownership and shared-Tile
materialization. Every combination is separately captured and compiled;
invalid candidates stay in the search record and the winner is freshly JITed
and validated. The old analytic score does not prune this measured search.
The [joint resource/execution comparison](../../performance/tile/reductions.md#joint-resource-and-execution-mapping)
retains the search, rejections, fresh winner validation and independent replay;
its timing tables are not duplicated in this API reference.

At N=8193 the resource/ownership interaction is explicit: W=256 cached
softmax/LayerNorm requires 66 private scalars, exceeding the 64-scalar budget;
W=512 requires 34 and is legal. RMSNorm needs only one stripe, so W=256 with
33 scalars is legal. No kernel-name rule makes those decisions.

At N=4096/W=512, caching adds eight scalar rounds, removes 32 global-read
bytes and adds 64/32 private read/write bytes per worker. With the existing
unit scalar coefficient, the optional access terms change its score delta
from `8` to `8 - 32*g + 96*p`. This sensitivity is not a calibration or a
microsecond prediction. New shapes in a tuned replay are not held-out model
validation; full-device service, private live state and independent incumbent
acceptance remain necessary before the model can safely prune the search.

## Whole-launch service policy and shape-held-out check

`ServiceExecutionCostPolicy` prices local scalar/collective/private-access
work, a continuous subgroup-demand saturation factor, whole-launch global
payload and per-worker global access. Its coefficients and capacity belong to
the caller's typed profile; they are neither queried occupancy limits nor
measured physical traffic. An explicit profile with unavailable access facts
fails rather than mixing fallback units. The complete returned kernel score
drives both the C++ width solver and optional model-only staged/JIT selection.

The [frozen whole-launch policy study](../../performance/tile/reductions.md#whole-launch-policy-shape-held-out-gains-and-small-case-failures)
records the calibration boundary and held-out cases, including small-shape
regressions. It does not justify changing the default. Subsequent
[width/reuse ablation and tail repair](../../performance/tile/reductions.md#tail-packs-a-structural-repair-after-width-reuse-ablation)
are separate experiments, not additional calibration labels or composable gains.
