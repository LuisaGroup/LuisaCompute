# Execution mapping: constraints, cost model, and search

Status: the Metal FP32 matrix realization described in sections 4--6 is
implemented. The target-independent formulation and later search strategies
describe the extension contract, not features already available on every target.

The planner chooses an implementation of the existing execution structure. It
does not introduce a second programming language, infer a new meaning for
`parallel`, or let a fast candidate override memory and numerical semantics.

```{figure} ../_static/tile/execution-planner.svg
:alt: Semantic facts and target capabilities define legal candidates. A cost model and search rank them; realization is verified again before JIT. Correctness-checked measurements calibrate ranking, never legality.
:width: 100%

Legality and profitability are separate. Search only chooses realizations the
compiler can actually emit; measurements close the loop on ranking.
```

## 1. What is being solved?

For a captured TileIR program, a plan is a set of choices over five existing
relationships:

| Choice | Meaning | Example |
|---|---|---|
| Execution binding `B` | logical ancestor coordinates to target participants | program to group, child coordinates to subgroup/worker |
| Distribution `D` | logical values to participant/local-slot correspondence | several matrix fragments owned by one subgroup |
| Atom `A` | legal realization of a semantic operation | FP32 matrix instruction or ordered scalar contraction |
| Resource assignment `R` | materialization, resource instance, address, and lifetime | retained fragment, private array, shared staging buffer |
| Schedule `Theta` | temporal order, overlap, and versions | serial recurrence, software-prefetch window, asynchronous protocol |

These are compiler records and typed maps, not five new DSL objects. In
particular, choosing shared memory does not create an execution hierarchy.
Multiple resources at the same logical scope can have different mappings.

Conceptually:

~~~text
minimize    predicted_time(B, D, A, R, Theta; workload, target, calibration)
subject to  semantic_constraints AND target_constraints AND emitter_support
~~~

Shape-specialized JIT makes many extents constant. A new tile shape or pipeline
window may simply recapture the C++ definition; there is no one-capture-only
restriction. A planner can search multiple physical mappings of one capture,
while an outer tuner searches captures. Neither mechanism substitutes for the
other.

The first implementation lives in the optional `tile/bridge/tirx` module. It
extracts facts from the shared structural TIRx export, before assigning physical
workers, and checks the actual operation body. The numerical solver itself
needs no TVM types or JIT. Later, a TileIR analysis can produce the same facts
before export. Native Metal/CUDA code generation still belongs to the backend;
external IR integration remains a bridge. No MLIR dependency is introduced.

## 2. Hard constraints are not costs

The following failures cannot be compensated by a better score:

- **Execution:** preserve logical domains and ancestor projections; respect
  target containment and explicit scope/participant constraints. `parallel`
  already promises independent SIMT instances. We prove the *mapping* covers
  those instances, not that the user meant something else by `parallel`.
- **Distribution:** exact coverage, no accidental duplicates or holes, legal
  replication, and complete/converged participants for collective atoms.
- **Values and effects:** preserve reaching definitions, simultaneous carried
  updates, aliasing, visible snapshots, and explicit `Memory.store` effects.
- **Arithmetic:** preserve the selected type/precision/reassociation contract.
  A target name alone does not authorize reduced-precision multiplication.
- **Resources:** required allocation sizes and actual target limits; compatible
  owner/instance/access maps. Resource kinds are not treated as a total order.
- **Time:** dependences, loop distances, complete barrier participation, and
  nonoverlapping lifetimes of reused versions. A source stage cut is not proof
  of an asynchronous engine or a one-cycle initiation interval.
- **Implementation:** the selected layout, instruction, copy protocol, and
  binding must have a supported emitter. Unknown capabilities fail closed.

Facts carry stable operation identities during one compilation; labels are
diagnostic only. A plan is consumed by the same compilation and the emitter
rechecks its preconditions. A cached plan needs a structural specialization key,
not pointer values. Transformation invalidates affected analyses/plans.

Register pressure illustrates the distinction: an actual compiled register
allocation or a documented architectural limit can inform a capacity check.
A count of live fragment scalars is only a pressure proxy. It must not be
reported as a measured register count or used to claim a particular occupancy.

## 3. Cost model architecture

### 3.1 Predict work and bottlenecks before predicting time

Extract features from the *realization*, not just source FLOPs:

| Feature family | Required information |
|---|---|
| Instruction work | atom issues, scalar/vector work, conversion and address work |
| Memory traffic | transfers by resource path, transaction utilization, reuse, layout conversion |
| Communication | collective/shuffle work, barriers, handoffs, participant count |
| Live state | fragment scalars, compiler-reported registers/spills, shared bytes and versions |
| Parallelism | programs, participant shape, active tails, resident groups when known |
| Temporal structure | recurrence distances, pipeline fill/drain, engine contention |
| Host overhead | dispatch, cold JIT, cache lookup; reported separately from warm execution |

For a later calibrated model, a region's service-time estimate should account
for shared bottlenecks and only overlap independent work. A possible hierarchy
is:

~~~text
engine demand = work on that engine / calibrated effective service rate
steady II     >= max(recurrence bound, each shared-engine demand bound)
pipeline time = fill + (iterations - 1) * II + drain
kernel time   = launch + resource-constrained execution of group waves
~~~

These are bounds/estimates, not an unconditional equality between runtime and
`max(compute, memory)`. Serialized stages must add; two stages using the same
engine compete; an unavailable async copy cannot overlap by assumption. The
number of groups and partial last wave matter for small shapes. CPU task/SIMD
and GPU group/subgroup realizations share this feature contract but need
different target parameters and realized-work models.

### 3.2 Calibration and uncertainty

Use nonnegative target-specific priors as a bootstrap. Then measure
correctness-checked JIT candidates, retain the raw samples, and fit ranking or
time predictions. Keep cold compilation, allocation/upload, warm dispatch, and
device execution separate. The TileIR/PyTorch benchmark measures warm amortized
host-wall time, including dispatch; it is not a GPU-event timer. The separate
native MPP/MPS experiment also records Metal command-buffer GPU intervals.
Those include batch dispatch and synchronization, not only arithmetic time.

The calibration key includes the device/architecture, driver, compiler and
bridge revisions, numerical policy, and timing method. A model version belongs
in the tuning-cache key. Source dimensions, layouts/strides, and specialized
configuration also belong there. A hard-coded coefficient inferred from one
1024-square GEMM is not a portable hardware model.

Evaluate ranking on held-out square, rectangular, ragged, small, and large
shapes. Useful metrics are top-choice regret against the measured candidate
set, top-K coverage, and search/compile budget, not just training error. Keep a
known-correct fallback and remeasure the selected winner independently. Report
unmeasured/uncertain choices as such; a noisy minimum is not a stable win.

### 3.3 Concrete contract for the calibrated model

The proposed general model is a **resource-constrained execution model**, not
a sum of costs assigned to TileIR opcode names. Its inputs have three layers:

~~~text
semantic facts       candidate realization           calibrated target
domains / effects -> participant and address maps -> rates / latencies
dependences          live intervals / engine uses     allocation granularity
numerical contract   actual copies and instructions   measurement uncertainty
~~~

The candidate supplies a region DAG, with edges for dependence, publication,
and resource-version reuse. Nodes describe emitted work on the target's actual
engines. A copy emitted as ordinary worker loads/stores consumes worker issue
capacity as well as its memory path; it is not assigned an imaginary DMA
engine. These are derived analysis records, not another public IR or DSL.

For each region, retain the following quantities before collapsing them to a
score:

| Quantity | Derivation and use |
|---|---|
| `work[r,e]` | instructions/transactions issued by region `r` to engine `e`, including repeated and masked work |
| `service[r,e]` | `work[r,e] / rate[e, context]`; context includes active participants, layout, and concurrent demand |
| `latency[r]` | producer-to-consumer delay for the emitted protocol, separately from reciprocal throughput |
| `live[r,q]` | simultaneously live, allocation-rounded state in resource `q`; a sum of all source allocations is only a conservative upper bound |
| `availability[r]` | legal participants, ready predecessors, and available versions; determines which service can overlap |

Address maps, not only byte counts, determine transaction utilization. A/B
reuse, a transpose, and a different execution distribution can change traffic
and address work even when logical element count is identical. Unknown cache
reuse or bank behavior gets an uncertainty flag or conservative estimate, not
a fabricated exact transaction count.

The evaluator then performs a small deterministic resource-constrained
schedule estimate. Serial dependences add latency. Concurrent nodes compete
for shared engines. Pipelines use recurrence distances and version lifetimes
to calculate a feasible initiation interval, fill, and drain. For a periodic
schedule with start offsets `s`, dependence `u -> v` at distance `d`, and
producer latency `ell[u]`, the timing constraint is:

~~~text
s[v] + d * II >= s[u] + ell[u]
~~~

Engine capacity and buffer-version reuse impose additional constraints. The
recurrence/engine lower bounds alone do not construct a feasible schedule;
the evaluator must also honor the proposed placement and emitted synchronization.

Whole-device time uses the number of programs and feasible group residency.
For a target whose scheduling-unit limits are known, an initial residency
bound is the minimum of its group, thread, subgroup, shared-storage, and
register-allocation bounds, using target allocation granularity. A missing
compiled register count remains unknown; fragment scalars cannot fill it in
as an exact fact. Occupancy is an input to effective service rates, not an
independent multiplicative speedup. A simple wave estimate is useful only for
homogeneous groups; ragged groups and heterogeneous regions require separate
work classes or a small scheduling simulation. CPU has its own task/SIMD
realization and rates, rather than reusing GPU occupancy equations.

The output should contain a time estimate or interval, its feature breakdown,
the dominant predicted bottleneck, and unsupported/uncertain features. The
current `PlanCost` is only the bootstrap relative-work summary in section 5;
this richer calibrated evaluator is not yet implemented.

Calibration should independently excite identifiable terms: copy/layout
sweeps, atom throughput with different reuse and live state, barriers and
handoffs, and varying program counts. End-to-end kernels then validate the
composition. Fit effective rates on one set and test ranking on held-out
shapes, tile configurations, and operator families. A single GEMM trace cannot
identify copy throughput, occupancy, and issue latency simultaneously.

### 3.4 Diagnose the model and the solver separately

Keep three costs distinct for a measured finite candidate set `C`:

~~~text
p_model  = argmin predicted_time(p), p in C
p_search = the candidate returned under the search budget
p_oracle = argmin measured_time(p),  p in C

search gap   = predicted_time(p_search) - predicted_time(p_model)
model regret = measured_time(p_model) / measured_time(p_oracle) - 1
~~~

The first quantity diagnoses search only when the model optimum is known
(or replaced by a documented lower bound). The second diagnoses ranking only
within the measured set, subject to measurement uncertainty. It does not
establish a hardware optimum. A solver can have zero search gap and substantial
model regret. That is precisely why the small exhaustive baseline remains
valuable after introducing a more complicated search algorithm.

For an uncalibrated or uncertain model, measure a diverse shortlist including
the incumbent/reference realization, not only near-identical descendants of
the model's favorite. Exploration measurements and final winner remeasurement
have separate budgets. Never average an invalid result into the training set
or silently replace a failed final validation with its earlier search timing.

Equal modeled cost is not an equivalence proof between realizations. The
[M1 Max equal-score layout experiment](../../scripts/benchmark/tile_torch/results/m1-max-20260903-layout-tie.md)
swaps the subgroup/local-fragment rectangle while keeping all reported work,
resource, and synchronization features identical. The alternative is about
3.7% slower on 1024-cubed across four rounds. The default tie order happens to
win that comparison; this does not make the prior a calibrated layout model.
A measurement-oriented shortlist must preserve structural diversity even when
the exact single-incumbent solver can legitimately collapse score/resource
ties. Generalizing dominance requires observationally relevant boundary
features, not just changing the numerical tie-breaking order.

The same report includes a separate lane/value interchange for cooperative
copies. It improves the large-square paired medians by about 1% but regresses
small shapes. The bootstrap work/resource features are again unchanged. Both
experimental default changes were reverted; neither is a calibrated universal
layout policy. A future shortlist and plan fingerprint must retain the actual
copy participant/local-value map, not just a maximum batch size.

### 3.5 External libraries are performance targets, not solver candidates

The benchmark now has separate direct-library GEMM baselines:
[Accelerate cblas_sgemm](https://developer.apple.com/documentation/accelerate/blas-library)
on CPU and [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
on Metal, alongside eager PyTorch. The native system-library executable does
not link TileIR or TVM and cannot silently replace a candidate's lowering.
See the [measurement protocol](../../scripts/benchmark/tile_torch/README.md#direct-blas-and-mps-gemm-baselines).

These comparisons answer a different question from model regret: how far is
our best *tested realization* from an optimized external implementation of
the same operation? A library timing is not a proved lower bound and not a
sample of our execution-mapping family. It must not be inserted into the
cost model as if it were another reachable mapping. In particular:

- Calibrate model features on our own legal realizations and controlled
  microbenchmarks; evaluate ranking on held-out shapes.
- Keep library, PyTorch and Tile API/submission costs visible. Batched host
  wall times do not by themselves identify pure device compute time.
- If all measured candidates remain far behind a library, inspect the
  realization family, memory movement and pipeline before enlarging the
  solver budget. Integer programming or annealing cannot select an absent
  microkernel, layout or overlap protocol.

### 3.6 Native MPP experiment: operation scope is not launch size

The native `benchmark_tile_mpp` experiment tests a second atom family before
adding it to TileIR lowering. It uses Apple's
[MPP tensor operations](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4),
not an MPS library call. Its `tensor_inline` arguments are ordinary buffers
plus layouts, so the experiment does not justify a new frontend resource type.
The result can remain a backend cooperative tensor until its explicit store.

There are two distinct, tunable execution maps:

~~~text
whole-group collective                 independent subgroup cohort
group                                  group
  all G subgroups -> one BM x BN MMA      (r, c) subgroup -> one TM x TN MMA
                                         BM = Cm * TM, BN = Cn * TN
                                         G = Cm * Cn

logical output coordinate:
  m = (group_m * Cm + r) * TM + i
  n = (group_n * Cn + c) * TN + j

independent memory-view composition:
  A: (m, k) -> base_A + m * stride_A + k
  B: (k, n) -> base_B + k * stride_B + n
  C: (m, n) -> base_C + m * stride_C + n
~~~

These maps describe an execution nest followed by three different resource
layouts. No buffer determines the hierarchy. In particular, four subgroups
computing a collective `64x64` tile are not interchangeable with four
independent `32x32` operations, even though both produce 4096 elements with
128 threads. Their synchronization, input reuse, private layout and compiler
implementation can differ. Total FLOPs, thread count and accumulator bytes
cannot distinguish them in a cost model.

The local macOS 26.5 MPP headers constrain an operation to one SIMD group or
all SIMD groups in its threadgroup. For this FP32 family, descriptor M/N must
be multiples of eight, with at least one a multiple of sixteen; a static K
must be a multiple of sixteen. A dynamic K supports tails. The native probe
checks these constraints and the compiled pipeline's thread limit. Static
tensor slices are used only for proved interior tiles; tail groups use
bounded dynamic views. An inactive cohort member may skip work only when the
operation is subgroup-scoped and its entire subgroup is inactive.

**Integration contract (not an implemented TileIR-to-MPP pass yet):**

1. **Planner:** enumerate the atom implementation, operation participation
   scope, outer group/cohort factorization, operand-view maps and temporary
   materializations together. Respect explicit execution and memory bindings;
   an `exec::group` constraint is not permission to merge logical groups into
   subgroups. Broader tile shapes can use ordinary recapture/JIT.
2. **Legality:** direct operand-view forwarding requires a load/effect proof.
   A Tile load is a snapshot; replacing it with an MPP read is illegal if an
   intervening aliasing store can change the value. Explicit manual stores,
   user barriers, arithmetic policy and accumulator initialization remain
   observable. Recognize semantic bodies, never kernel names.
3. **Cost model:** key samples by atom family, operation scope, cohort map,
   dtype/precision, shape, strides, bounds mode, compiler and device. Account
   for active/inactive groups, full versus edge operations, K extent, buffer
   reuse and output materialization. For an opaque MPP operation, unknown
   internal registers, barriers and copy stages stay unknown; do not invent
   native-8x8 issue counts or port its coefficients to another atom family.
4. **Solver:** exhaustive enumeration is sufficient for the small tested
   family. Apply integer coverage/resource constraints first, shortlist with
   a model, then JIT and validate. The current experiment uses measured GPU
   batch time as an empirical cost and retains every rejected/failed candidate.
   Frozen independent replay evaluates a selected plan, not a search minimum.
5. **Emitter:** `TileIR -> TIRx` retains typed operations, regions, effects and
   views. A supported Metal realization lowers its chosen atom to MPP and
   constructs inline tensors from buffer/layout expressions. Until that
   emitter and its capability checks exist, an MPP timing cannot win the
   production `plan_group` search. Native 8x8 and scalar fallbacks remain valid.

The benchmark runner is
[`compare_mpp.py`](../../scripts/benchmark/tile_torch/compare_mpp.py). It separates
search from six-order replay and records host and GPU batch times. Inline
tensors use a tracked classic Metal queue, as does MPS. The optional tensor
handle probe uses a Metal 4 queue with explicit dispatch barriers and commit
feedback; Metal 4 does not supply automatic resource hazard tracking. API
differences are recorded, not attributed to shader arithmetic.

## 4. Implemented matrix mapping family

The current planner targets a proved Metal group-level FP32 MMA with a
complete 32-lane subgroup and an 8x8 atom. Let:

~~~text
U = M / 8, V = N / 8, Q = K / 8
G = threads_per_group / 32
gm * gn = G
gm * rm = U
gn * rn = V
~~~

All quantities are positive integers. `gm, gn` distribute the atom grid between
subgroups; `rm, rn` specify the resident atom rectangle inside each subgroup.
The remaining candidate is the one-atom-at-a-time reference, which supports
uniform subgroup job tails. An incompatible exact thread constraint uses the
checked scalar reference instead of partially participating matrix operations.

For subgroup ordinal `s` and local fragment ordinal `f`:

~~~text
atom_m = floor(s / gn) * rm + floor(f / rn)
atom_n = (s mod gn) * rn + (f mod rn)

s = floor(atom_m / rm) * gn + floor(atom_n / rn)
f = (atom_m mod rm) * rn + (atom_n mod rn)
~~~

The factor equalities prove exact coverage and the two directions agree on the
finite domain. This is a mixed-radix layout, not a separate distribution
algebra. The forward map exports a native C++ TIRx `TileLayout` with shard
extents `(gm, rm, gn, rn)` and physical contributions
`(gn*warpid, rn*m, warpid, m)`. The inverse is a core Tile `IndexMap`, lowered
to native coordinate arithmetic. TIRx regrouping may remove unit-extent shards;
zero offsets retain both named output coordinates in degenerate cases.

Here native `m` names a *fragment ordinal*, not a byte address. An 8x8 atom's
internal lane/register layout remains its hardware contract. The access path
is still composition:

~~~text
(ancestor program, subgroup, local fragment, atom operand coordinate)
  -> logical tile coordinate -> view coordinate -> resource address
~~~

`A`, `B`, and the accumulator therefore share the execution plan without
sharing a memory layout. No `mma_team` or new frontend scope is required.

### Realization changes

The contraction loop surrounds the resident fragment rectangle. Per K atom,
each subgroup loads `rm` A fragments and `rn` B fragments, then performs
`rm*rn` matrix updates. A/B inputs are reused rather than reloaded once per
output fragment.

A closed recurrence can additionally retain the accumulator across iterations:

~~~text
before loop: load initial C into native fragments
loop:        load A/B; update fragments
after loop:  publish final fragments to C
~~~

Promotion requires one directly recognized MMA and its full `D -> C` carry
update, one local D allocation, and no other observation of C or D in that
loop body. Reading the old accumulator into another carry, an explicit memory
effect that observes it, or an unsupported annotated/control-flow body prevents
promotion. In particular, C must not also be an A/B multiplicand: the next
iteration would otherwise read stale shared C while the updated value exists
only in native fragments. A `break`, `continue`, or `return` between the MMA
and its carry update also prevents promotion. Literal initialization,
zero-iteration behavior, transposes, tails, and the nonresident fallback have
regressions. This baseline eliminates D but retains initial/final C storage.

This recognition currently applies to closed flat serial loops, including the
window-1 lowering. Window-2 software-pipeline prologue/steady/drain structure
generally retains the reference storage behavior. Joint pipeline/residency
planning is future work, not a promise made by this transformation.

### Direct accumulator output

The resource plan can also remove C when its complete lifetime is recognized:

~~~text
source:    fill C; [load A/B -> MMA -> yield C]*; ...; store output <- C
realized:  fill CF; [load A/B -> update CF]*;     ...; matrix-store output <- CF
~~~

Here CF is native fragment state, not a new DSL value kind. The global write
stays at the original output statement, including when intervening code reads
the old output. The transformation does not infer that different external
buffers cannot alias, nor does it move a global write across another effect.

Eligibility requires an unannotated compiler-owned C allocation, one complete
FP32 literal fill, the closed recurrence, and exactly one complete C-to-global
copy. A whole-group use audit rejects any other observation or pointer escape;
`SeqStmt` grouping is not treated as a lifetime boundary. An explicit manual
resource annotation prevents allocation removal. The output must have a
positive compact affine row/column-major projection. Every output guard and
every destination coordinate must be proved valid over the ancestor execution
domains and the local matrix domain. The query uses TVMx's native C++
`StmtSimplify` pass; unknown bounds keep the shared/guarded reference path.

`PlannerOptions::direct_accumulator_store` independently disables this choice.
`MatrixWorkload::has_direct_output` is an analysis fact, not a profitability
hint. The selected `MatrixDistribution` reports whether it uses that proof.
Both C and D can then be removed from the shared budget. Padded/offset and
transposed destinations, ragged output fallback, an extra accumulator consumer,
and a read of old output before the sink have full-output numerical tests.
MMA-operand aliasing and interrupted carry updates are native-IR counterexamples,
so incidental frontend value copies cannot conceal an unsafe residency proof.
The pinned TVMx LLVM emitter rejects native `Break`/`Continue`; CPU tests check
that specific rejection, while Metal executes those two control-flow oracles.

## 5. Implemented relative-work model

`ExecutionCostModel` is an inspectable prior, **not a nanosecond predictor**.
For an MMA executed `E` times, and a proved resident recurrence of length `L`:

~~~text
matrix issues I = U * V * Q * E

input transfers = G * (rm + rn) * Q * E       [rectangle]
                = 2 * U * V * Q * E          [reference]

accumulator transfers = 2 * U * V * E        [nonresident]
                      = 2 * U * V * E / L    [proved resident]
                      = 0                    [proved direct output]

direct global stores  = U * V * E / L        [proved direct output]

live fragment scalars/lane R = 2 * (rm*rn + rm + rn)
                            = 6              [reference]

work     = I * matrix_issue + transfers * shared_fragment_transfer
pressure = max(1, R / preferred_fragment_scalars_per_lane)
parallel = max(1, min(preferred_subgroups, U*V) / G)
score    = sum(work * pressure * parallel)
           + independent_elements * independent_element
           + G * subgroup_setup
~~~

Transfers count subgroup fragment operations, not unique DRAM bytes. Fragment
scalar counts are live logical state, not compiler register allocations.
Direct global stores are reported separately. They are not free: the original
logical independent-element work still prices the output conservatively. It
also still includes an elided literal fill; this bootstrap does not yet assign
calibrated prices to different output protocols.
Ordinary independent-element work is recorded but constant across this first
mapping family; it does not yet predict the effect of thread count on copy
throughput. The group count is recorded as a workload fact but is not yet used
to model whole-device occupancy. The pressure/parallelism factors are explicit
heuristics with replaceable coefficients. General bank-conflict, spill,
repartition, launch, and pipeline-overlap models are not implemented here.

For a 32x64 output tile, K tile 32, and 32 temporal iterations, one legal
128-thread plan uses a 1x4 subgroup grid and 4x2 local fragments. It changes the
accounting as follows (these are derived work counts, not timing claims). The
last column additionally requires the literal-fill/full-output proof above:

| Quantity | One-atom reference | 4x2 resident, shared output | 4x2 resident, direct output |
|---|---:|---:|---:|
| Matrix atom issues | 4096 | 4096 | 4096 |
| A/B fragment transfers | 8192 | 3072 | 3072 |
| C/D shared fragment transfers | 2048 | 64 | 0 |
| Direct global fragment stores | 0 | 0 | 32 |
| Live fragment scalars/lane | 6 | 28 | 28 |
| Compact shared allocation | 28 KiB | 20 KiB | 12 KiB |

The tradeoff is explicit: fewer repeated transfers and less shared storage,
but more live per-subgroup state. A model must eventually price both using
target evidence. Counting static matrix call sites in generated source is not
counting dynamic instruction work.

### Cooperative copy batching

`PlannerOptions::max_copy_batch` optionally groups up to 16 independent values
per worker: emit their reads/computation into native TIRx bindings, then their
stores. One is the default reference sequence. This is an instruction-level
parallelism option, not an asynchronous transfer or a vector-alignment claim.
It does not change the worker-to-element map, move a stage, or remove a fence.

The emitter only batches independent domains writing a compiler-owned shared
temporary, with no destination read/modify/write, conditional store, or opaque
effect. Short-circuit bounded loads retain their predicates. Only full worker
chunks are batched; the remainder uses the original guarded path. Reports
include the requested maximum and the number of actually batched operations.
The benchmark/replay driver preserves the option as `--copy-batch N`.

This choice is currently explicit, not included in the matrix model's search
or calibrated score. Four-round same-binary measurements at 64x64x32 tiles show
substantial improvements on two ragged GEMMs, but essentially no improvement
on 512-cubed or 1024-cubed. No universal speedup or default change follows from
those observations. The [copy-plan report](../../scripts/benchmark/tile_torch/results/m1-max-20260903-copy-plan.md)
keeps all shapes, timing distributions, and correctness checks.

### Dependence-aware group synchronization

After resource/distribution realization, the bridge can coalesce its own group
barriers across independent effects. For example, two input copies to distinct
shared buffers can publish together before their matrix consumer:

~~~text
before:  copy A -> As; fence; copy B -> Bs; fence; MMA; fence
after:   copy A -> As;        copy B -> Bs; fence; MMA; fence
~~~

No copy, memory effect, or participant mapping moves. The first fence is
unnecessary only if its removal leaves every publication/order dependence
covered. At a candidate cut, let `P` be all effects since the last **retained**
fence and `S` the next segment. Coalescing requires neither side to be opaque,
`W(P)` disjoint from `R(S)` and `W(S)`, and `R(P)` disjoint from `W(S)`.
The implementation accumulates P after a removal; checking just the adjacent
operation would incorrectly accept `write A; unrelated B; read A`.

Only the mapper's fresh unplaced shared allocations have distinct alias
classes. All external global buffers share one conservative class, even when
their parameter identities or constness differ. Unknown storage, calls,
explicit synchronization, and control exits remain hard boundaries. The pass
recognizes compiler fences by their emission-local IR identity, not an opcode
or external symbol name; an explicit identical-looking native barrier stays.
The last fence of a sequential region is retained for the loop backedge or
enclosing consumer. Barriers never move across loops/branches, and their full
shared-plus-device fence semantics are unchanged. Resource reuse added by a
later transformation must invalidate/recheck this synchronization plan.

`PlannerOptions::coalesce_group_barriers` disables this pass independently;
disabling the planner also retains the reference fences. `GroupPlan` reports
`group_barrier_sites_before/after`: static sites, not dynamic executions or a
latency estimate. These post-realization facts do not yet contribute calibrated
barrier costs to the bootstrap score. CPU/Metal regressions cover nonadjacent
dependencies, non-subgroup-multiple worker counts, aliased global parameters,
and aliased output read on the next pipeline iteration. A Metal native-IR test
also distinguishes an explicit barrier from compiler-owned barriers.

The [M1 Max synchronization report](../../scripts/benchmark/tile_torch/results/m1-max-20260903-barrier-plan.md)
holds the tile, worker count, fragment layout, and resource realization fixed.
Four counterbalanced rounds show modest, shape-dependent gains, including one
1024-cubed regression. Fewer barrier sites are not a proportional latency model
and do not explain the remaining large-square gap to PyTorch.

## 6. Implemented solver: enumeration plus Pareto dynamic programming

For the currently small target thread bound, enumerate every complete-subgroup
thread count (or honor one exact requested count). For each MMA enumerate
integer subgroup factorizations, derive local factors by division, and reject
nonexact coverage or excess compiler fragment/code-size budget before scoring.
The one-atom reference is always included. Eligible rectangular candidates use
accumulator residency unless that optimization is disabled. When the additional
direct-output proof is present and enabled, they also eliminate C and its shared
transfers. The current model prefers this choice; independent measurements must
still validate its real runtime effect.

Several operations share one group's resource limit, so independent greedy
selection is insufficient. At a fixed thread count, dynamic programming keeps
a frontier over:

~~~text
(selected realizations, summed score, released shared bytes)
~~~

A partial choice dominates another only if its score is no larger **and** its
released bytes are no smaller. Remaining operation alternatives contribute
independently to these quantities, so dominated states cannot be required by
an optimal completion. Reject final combinations exceeding shared capacity,
then select the lowest-score surviving combination across thread counts.

This is exact for the supplied additive model and enumerated family. It does
not prove a global runtime optimum, and the dominance rule must change if later
costs include interactions or overlapping live ranges. Frontier state must then
carry those interactions instead of discarding them. The current tests include
a case where a slower resident realization is the *only* capacity-feasible
choice, both for one operation and for two coupled operations.

The solver has no mutable global state or timing calls. Changing coefficients
cannot make an illegal map legal. `CompileOptions::planner` controls the model,
exact thread constraint, residency choice, and fragment search budget;
`CompilationResult::plans()` reports the selected distributions, resource
footprint, work counts, and search counters. Benchmark JSON includes these
reports under `execution_plans`.

Thread capacity and search budget are separate: the reference launch width of
256 is not a hardware cap. An automatic search exceeding
`max_thread_candidates` returns a diagnostic instead of silently searching a
prefix and claiming exactness. An exact thread request needs only one width;
subgroup divisors are enumerated in square-root time, with deterministic order.

## 7. When to use integer programming, beam search, or annealing

Do not make the rest of the compiler depend on one search algorithm. The
interface is: enumerate/propose supported realizations, check constraints,
extract features, score, and return a verified incumbent with diagnostics.

| Search | Appropriate use | Boundary |
|---|---|---|
| Enumeration + Pareto DP | small factored space and additive shared-resource coupling | current implementation; exact only within its stated family/model |
| MILP / CP-SAT | many discrete choices with linearizable capacity, assignment, dependence and schedule constraints | proposed; solver optimality concerns the encoded model |
| Beam search | hierarchically composed candidates with cheap partial bounds | proposed; retain diverse resource/layout states, not only one cheap prefix |
| Simulated annealing | large irregular neighborhood and a nonlinear calibrated score | proposed; finite budget, deterministic seed, verified incumbent |
| JIT measurement | rank uncertain finalists and refresh stale calibration | existing outer tuner; not a semantic verifier |

For example, precomputed alternatives permit binary variables `x[o,j]` with
`sum_j x[o,j] = 1`. One-hot thread choices constrain which alternatives are
compatible. Shared capacity is:

~~~text
original_shared_bytes - sum(x[o,j] * released_bytes[o,j]) <= target_capacity
~~~

A precomputed additive objective makes this a small multiple-choice resource
problem. There is no need to import an integer-programming runtime to solve
today's case. For pipeline scheduling, CP-SAT can represent placement and
dependence/disjunctive resource constraints; effective latencies and overlap
assumptions still require validation. Products of shape variables can be
pre-enumerated into legal alternatives instead of pretending they are linear.

An optional CP-SAT adapter must also state its integer time unit and rounding
policy. Its encoded optimum can differ from the floating-point model after
quantization. The [CP-SAT interface](https://developers.google.com/optimization/cp/cp_solver)
distinguishes a feasible incumbent from an optimal solution; an adapter must
preserve that distinction. Interval and nonoverlap constraints, as illustrated
by the [job-shop scheduling model](https://developers.google.com/optimization/scheduling/job_shop),
are a useful starting point for exclusive engines. Shared-throughput engines
need cumulative demand or a validated service model, not a fictitious
one-operation-at-a-time constraint.

For annealing, proposed moves can change subgroup factorization, resident
fragment shape, a supported resource placement, or a pipeline window. Validate
or repair the whole affected map/resource region before evaluating it. Accept
a worse *legal* candidate with a temperature-dependent probability, keep the
best legal incumbent separately, record the seed/budget, and independently JIT
measure finalists. Annealing cannot discover an emitter the compiler lacks.

Moves should operate on constructive layout operations and legal alternatives:
factor transfer between adjacent hierarchy levels, interchange of independent
axes, or replacement of one supported materialization/protocol. Arbitrarily
changing entries of a layout matrix usually leaves the valid realization
space. A repair must propagate into dependent address maps, resource sizes,
and schedules; otherwise rejecting the move is safer. Simple annealing can use
`min(1, exp(-(new_score - old_score) / temperature))` for symmetric proposals.
Repair or asymmetric proposal distributions change the sampling behavior;
without the corresponding correction, this is a search heuristic, not a
claim of a particular stationary distribution or convergence guarantee.

### Compositional search contract

The execution hierarchy suggests bottom-up decomposition, but a locally cheap
child is not necessarily globally cheap. Its parent needs a boundary summary
of live input/output distributions, resource demand, dependence/engine demand,
and explicit binding constraints. Only candidates with compatible interfaces
can be compared or composed; a cheaper score alone does not dominate a
different layout that saves a parent's conversion. The current two-dimensional
Pareto rule is valid only for the narrower additive problem in section 6.

Keep the candidate generator, verifier, evaluator, search strategy, and
measurement driver separately replaceable. A future search result should
report:

- The best verified incumbent and diverse finalists, plus the realization and
  model versions needed to reproduce them.
- Status: exact within the encoded family, feasible under budget, no candidate
  found, or proved infeasible. A timeout is not an infeasibility proof.
- Search-space bounds, rejected-constraint counts, seed where relevant,
  evaluations/compilations/measurements consumed, and a valid lower bound only
  when one is actually available.

Use deterministic evaluation-count budgets in regression tests. A wall-clock
budget is also useful interactively, but its explored set need not be
reproducible. Compiler and JIT measurement budgets are independent: a million
cheap model evaluations do not authorize a million device compilations.

This supports a hybrid strategy: discrete constraints select feasible binding
and resource alternatives, a nonlinear evaluator ranks complete plans, and
local/beam search expands the shortlist before measured selection. An integer
solver with a surrogate objective supplies candidates and bounds for that
surrogate; it does not certify optimality for an unrelated nonlinear model.

The intended progression is therefore:

~~~text
verified realization family
  -> transparent prior and exact small-space baseline
  -> measured calibration + top-K JIT selection
  -> larger compositional search where measurements justify it
~~~

This separation has useful precedents. Halide's
[2019 autoscheduler](https://halide-lang.org/papers/autoscheduler2019.html)
combines beam search over schedules with derived features and a learned cost
model. TVM's [MetaSchedule architecture](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)
separates space generation, search strategy, cost model, builder/runner, and a
measurement database. These inform the modular boundary; neither demonstrates
that a particular search algorithm will be best for our realization family.
MetaSchedule integration would need an explicit compatible schedule/measurement
adapter; it is not automatically supplied by emitting ordinary TIRx statements.

## 8. Validation and remaining work

Current regression coverage includes all thread-count choices up to the target
test bound; multiple rectangular shapes; exact atom coverage; native TIRx
forward/core-IndexMap inverse agreement; invalid coefficients and constraints;
shared-capacity tradeoffs; generated fragment reuse/residency; and CPU/physical
Metal numerical execution. Intermediate accumulator observers retain the
original storage behavior. Native matrix eligibility still checks capability,
arithmetic policy, the actual typed body, and operand layouts.

Performance validation uses multiple shapes, full-output FP64 references, a
frozen pre-planner binary/library bundle, counterbalanced repeated comparisons,
and PyTorch measured separately. An improvement over our old lowering must not
be described as an improvement over PyTorch.

Remaining structural work includes broader synchronization planning, more
selective materialization, layout-aware cooperative copies, combined software
pipeline/residency planning, general nested hierarchy binding, CPU task/SIMD and
storage planning, calibrated target models, and additional atom families. The
first planner removes specific repeated fragment traffic; it does not yet solve
all execution-to-hardware mappings or establish PyTorch-level performance.
