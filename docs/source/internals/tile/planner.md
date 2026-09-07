# Execution mapping: constraints, cost model, and search

This page owns the target-independent planning problem, hard constraints,
calibration discipline and search interface. Implemented realization families
have separate references: [Metal matrices](matrix.md), [Metal reductions](reductions.md),
[TIRx CPU](cpu.md), and [backend cost policies](cost-policy.md).
The general resource-constrained model and later search strategies are extension
contracts, not features already available on every target.

The independent [XIR/SIMD CPU planner](xir.md) now searches root
axis order and Runtime worker packing using a separate relative-work model.
It does not reuse GPU occupancy equations or claim that Tile partitioning,
cache blocking and physical pipelining are already implemented.

The planner chooses an implementation of the existing execution structure. It
does not introduce a second programming language, infer a new meaning for
`parallel`, or let a fast candidate override memory and numerical semantics.

```{contents} On this page
:local:
:depth: 2
```

```{figure} ../../../_static/tile/execution-planner.svg
:alt: Semantic facts and target capabilities define legal candidates. A cost model and search rank them; realization is verified again before JIT. Correctness-checked measurements calibrate ranking, never legality.
:width: 100%

Legality and profitability are separate. Search only chooses realizations the
compiler can actually emit; measurements close the loop on ranking.
```

## What is being solved?

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

### Formal finite optimization problem

This is the proposed general formulation; the current relative-work solvers
implement narrower projections of it. Fix a specialization `p`, target `T`,
numerical/semantic assumptions `Gamma`, and bounded emitter vocabulary `K`.
The [execution calculus](calculus.md) defines legal refinement witnesses.
Let `C(P,p,T,K)` be candidate witnesses, not arbitrary integer schedules:

```text
x = (region partition/fusion, tau, B, D, A, R, Theta, protocol)
minimize    F_theta(x) = predicted makespan of the emitted execution DAG
subject to  Gamma |- P =>[x] Emit(x)
            target containment/access/capacity constraints
            exact contribution coverage and reaching-value preservation
            available code/protocol emitter for every selected component
```

Fusion/fission changes the physical partition of the existing region tree;
it does not change the primitive meanings. Matrix, reduction and pointwise
families are subsets of `C`, not kernel-name cases in the objective. A
candidate that the bridge cannot express is absent, even if the abstract
hardware might support it. Generating a larger family is different from
retuning the cost coefficients on a small family.

For a finite constraint encoding, enumerate alternatives `j` for region/atom
`o` and choose binary variables `x[o,j]` with `sum_j x[o,j]=1`. An alternative
includes its participant factors, boundary distributions, storage and emitted
work, not just thread count. Incompatible neighboring alternatives are
forbidden; a supported conversion can be another explicit alternative with
its own cost, lifetime and synchronization. A fusion candidate covers a set
of source operations; exact-cover constraints prevent selecting overlapping
realizations or leaving an operation uncovered.

The selected alternative expands to events `u` with start `s[u]`, modeled
duration `d[u]`, resource demand `r[u,q]`, and dependence edges. An asynchronous
atom has separate issue/completion events; a consumer cannot start merely
because the producer was issued. For a periodic region:

```text
s[v] + distance(u,v)*II >= s[u] + latency(u,v)
sum_{u,k: s[u]+k*II <= t < s[u]+k*II+d[u]} r[u,q] <= capacity[q]
sum_{v: birth[v] <= t < last_completion[v]} rounded_storage[v,q] <= storage_capacity[q]
```

The resource inequalities are per physical resource instance, not one global
pool. Nonperiodic regions use one event copy. Shared-bandwidth resources use
calibrated cumulative demand rather than pretending every operation exclusively
owns the entire memory engine. Synchronization/protocol feasibility is part of
the refinement check, not a latency coefficient.

An edge-only DAG model is insufficient for an in-order issuing context:
blocking on a completion can prevent that same context from issuing otherwise
independent work. A realization must include those issue restrictions and any
cross-participant transfers. Joint scheduling, warp assignment and liveness
already have a close precedent in
[Twill](related-work.md#twill-the-nearest-joint-cost-solver-comparison);
this formulation must be evaluated against it, not presented as an original
joint-scheduling result.

For a version ring of size `V`, version `k` and `k+V` must not overlap in live
time in the same slot; producer completion precedes reads and final consumer
completion precedes reuse. For uniform steady-state lifetimes, this yields
`V*II >= live_duration`, using half-open intervals. Irregular lifetimes require
their actual interval constraints. Extra buffering is therefore coupled to
mapping capacity: a second pipeline slot may exclude a cooperative group that
fits with ordered stages. Pipeline depth is not an independent positive-speedup
factor.

Readiness and reuse are different relations. A copy's completion permits its
consumers to start; only the necessary consumers' final completions permit
overwriting the same storage. Copy elimination, partitioning and fusion must
rewrite both relations. A textual last use is insufficient when it issues
an asynchronous operation that still reads the buffer. The
[Cypress](related-work.md#cypress-the-closest-execution-resource-separation)
and [Tawa](related-work.md#tawa-compiler-internal-channels-with-operational-semantics)
comparisons motivate this requirement. These are analysis/plan records, not
public event handles.

Useful conditional lower bounds are critical-path latency, total demand over
each resource's service capacity, and recurrence bounds:

```text
II >= max_cycle(sum edge_latencies / sum iteration_distances)
```

A positive-latency zero-distance cycle is infeasible. These are lower bounds
for the encoded work/service assumptions, not assertions that hardware
achieves them. Missing cache traffic, compiled register counts and overlap
behavior remain unknown. A continuous or integer surrogate's optimality
certificate says nothing about an unrelated evaluator or real elapsed time.

`F_theta` estimates the completion time of all required sinks, including
explicit transfers and dispatch boundaries inside the plan. Pure GPU time
and end-to-end dispatch are different objectives: the latter additionally
models host work and submission/completion edges. Cold JIT is a separate
objective or a stated amortization constraint, never silently mixed in.

### Conditional bounds and search certificates

The algebra supports a bounded **relative** completeness argument: if domain
and hierarchy depth, map construction size, atom/protocol catalog, buffer
counts, and schedule horizon/time quantum are finite, enumerating their typed
compositions and checking the constraints visits every expressible plan in
that set. This is a specification-level enumeration result, not a claim that
the present generator does so, or that unrestricted rewrites terminate.

For the current additive subproblem, Pareto pruning is sound only for plans
with the **same boundary interface** and future-independent resource/cost
summaries. Different output layouts, live values, ready times or communication
requirements may reverse the parent's preference. Equal shape or fewer bytes
does not establish dominance. Beam/annealing search can retain an incumbent;
only an exact finite search/solver with a valid bound can report an optimality
gap for the encoded model. Unknown legality, a timeout and proven infeasibility
must remain distinct results.

There is one useful conditional performance statement. If a cost model has
a *uniformly established* multiplicative error `epsilon < 1` for every plan
in `C`, and `x_hat` is its exact minimum, then:

```text
(1-epsilon)*time(x) <= F_theta(x) <= (1+epsilon)*time(x), for all x in C
time(x_hat) <= (1+epsilon)/(1-epsilon) * min_{x in C} time(x)
```

The argument compares the model minimum to the measured optimum through the
two error bounds. With additive model search gap `delta`, add
`delta/(1-epsilon)` to the upper bound. A test-set mean error, one favorable
trace or a per-candidate confidence interval is **not** that uniform premise.
We have not established such an error bound for Metal or SIMD. Current
performance claims must continue to use held-out measurements and regret.

### Program traversal is a mapping choice, not a memory scope

The TIRx bridge now supports an **opt-in bounded rectangular traversal** for
explicit Metal group programs. `PlannerOptions::program_order_rows` and
`program_order_columns` default to `1,1`; they are positive JIT constraints,
not measured cache sizes or automatically selected cost-model outputs. The
last two axes of the original `parallel` domain form the program grid. Earlier
axes remain separate batches, and the launch contains exactly the original
number of programs, including partial final rectangles.

```text
physical program ordinal -- bounded bijection --> logical ancestor coordinates
                                                       |
                                      existing worker/local distribution
                                                       |
                                      existing per-resource address maps
```

Structural export retains the original axis extents in typed native TIRx
metadata. Resource, recurrence and matrix analyses run on the original logical
domain; only then does realization substitute the permuted program ordinal.
No operation-name matching, shared-memory placement, worker redistribution or
pipeline-stage reordering follows from this choice. In particular, several
memory resources used by the same program keep their independent layouts.

For each row band, the map concatenates its bounded column rectangles, then
enumerates each rectangle row-major. Those rectangles partition the grid,
and each inner enumeration is bijective, so their concatenation has neither
holes nor duplicates. Signed-64-bit extent/product checks precede expression
construction; a final band uses its actual height and width. One-row traversal
or a rectangle spanning all columns reduces to the original row-major map.
This is a coordinate permutation, **not a guarantee of GPU scheduling order**.

An exact nondefault request rejects unsupported targets/bindings, disabled
planning, missing or inconsistent axis metadata, and rank-one programs.
This initial emitter supports explicit Metal groups only; CPU, automatic
pointwise fusion, reduction-program packing and native MPP do not silently
inherit it. Stencil-like pipelines, local reductions and staged/view matrices
exercise the same map in tests. Performance generalization is a separate
question: the current cost policy does not price cross-program reuse or
traversal address arithmetic. Keep traversal explicit until a frozen policy
passes multi-shape and multi-operator validation.

## Hard constraints are not costs

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

## Cost model architecture

### Predict work and bottlenecks before predicting time

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

### Calibration and uncertainty

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

### Concrete contract for the calibrated model


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
current `PlanCost` implements the deterministic relative-work bootstrap above;
it does **not** yet implement this richer calibrated time/interval evaluator,
hardware-counter feedback, or uncertainty propagation.

Calibration should independently excite identifiable terms: copy/layout
sweeps, atom throughput with different reuse and live state, barriers and
handoffs, and varying program counts. End-to-end kernels then validate the
composition. Fit effective rates on one set and test ranking on held-out
shapes, tile configurations, and operator families. A single GEMM trace cannot
identify copy throughput, occupancy, and issue latency simultaneously.

### Diagnose the model and the solver separately

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

Equal modeled cost is not an equivalence proof. The
{download}`equal-score layout experiment <../../../../scripts/benchmark/tile_torch/results/m1-max-20260903-layout-tie.md>`
and {download}`structural emitter experiments <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-tirx-structure.md>`
are counterexamples to collapsing structurally different candidates only
because their bootstrap features tie. The
[MPP v1/v2 study](../../performance/tile/results.md#balanced-metal-evidence-mpp-cost-v2-closes-this-gemm-cohort)
separately records the concurrency-model correction and its residual in-cohort
regret. Those measured tables have one owner in the performance report;
neither an equal score nor an in-cohort improvement proves held-out ranking.

The outer Staged/JIT benchmark now jointly enumerates requested block shapes,
pipeline windows, group widths, and cooperative-copy batches. The product has
an explicit compilation budget and includes rejected candidates in its report.
Selection is followed by a fresh capture/JIT of all winning settings; frozen
replay uses the recorded native configuration, not an old score. This is an
implemented measurement-based outer tuner, separate from the inner planner's
still-uncalibrated analytic ranking. MPS/Torch remain external baselines and
cannot win this candidate search by replacing Tile lowering.

### External libraries are performance targets, not solver candidates

The benchmark now has separate direct-library GEMM baselines:
[Accelerate cblas_sgemm](https://developer.apple.com/documentation/accelerate/blas-library)
on CPU and [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
on Metal, alongside eager PyTorch. The native system-library executable does
not link TileIR or TVM and cannot silently replace a candidate's lowering.
See the direct BLAS and MPS GEMM baseline section in the
{download}`measurement protocol <../../../../scripts/benchmark/tile_torch/README.md>`.

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

#### When a library routine is a legal target atom

An independent library executable is not a schedule candidate. A provider
call can nevertheless become part of our realization family when the target
backend owns that choice and proves an exact semantic contract. These two
roles must remain visibly different:

| Role | Enters Tile compilation? | Purpose |
|---|---|---|
| Direct CBLAS/MPS benchmark | No | External performance and correctness control |
| CBLAS/vDSP/vForce target atom | Yes, behind an explicit policy | Reachable implementation of a proved TileIR/TIRx contract |

The current CPU contract path is deliberately narrow. The TileIR exporter
matches typed SSA/dataflow, attaches a versioned contract, and the CPU pass
rechecks the transformed TIRx body, ABI, layout, aliasing and numerical policy.
Only then may it emit an external call. External symbol strings are used at
the C/TVM provider ABI boundary; they are not used to identify TileIR or TIRx
operations. Reference loops remain the semantic control, while an explicit
provider request with no supported realization fails closed.

A provider atom still needs a cost model. Its features include call/packing
overhead, layout conversion, synchronization, problem size, target library
version and thread policy. The current CBLAS and Accelerate choices are explicit
compile options, not an automatic claim that the provider wins every shape.
The saved direct-library replay measures exactly why: a 32³ GEMM is dominated
by wrapper overhead even though large matrices approach direct CBLAS.

### Implemented Metal reduction family: execution before storage

The [Metal reduction reference](reductions.md) owns subgroup collectives,
cooperating-program packing, worker ownership, resource bounds and fallback
rules. Its finite solver searches only maps supported by that emitter; the
[backend policy](cost-policy.md) scores them independently of legality.
Historical schedules and A/B results belong in [reduction measurements](../../performance/tile/reductions.md).

### Implemented shared-Tile materialization choice

[Shared SSA](ir.md#shared-ssa-preserves-a-resource-planning-choice) preserves
one logical definition without forcing storage. Retain, recompute and materialize
are target choices. [Metal ownership and stripe budgets](reductions.md#logical-tile-materialization-to-physical-worker-storage)
and [CPU realization](cpu.md) have different constraints; measurements are
maintained in [the materialization comparison](../../performance/tile/reductions.md#fused-residual-layernorm-and-materialization-choice).

### Automatic GPU element grids

An independent Tile element domain must not accidentally become serial just
because it lives inside a logical program. The bridge now admits a second
automatic physical map, after immutable-input forwarding:

```text
Logical coordinates:       program p × local element e
Old worker realization:    thread p; serial e
Fused realization:         i = p * tile_volume + linear(e)
                           block = i / threads; worker = i % threads
```

This is a coordinate factorization of execution, not a new memory level or a
kernel-specific DSL primitive. The inverse reconstructs the original program
and local coordinates before evaluating the original guarded load/store.
Consequently ragged tiles, negative input origins and zero-fill retain their
semantics. A nondivisible physical grid gets an additional tail predicate.

The current family requires one automatic static root and one perfect static
element nest ending in a single compact-global-buffer store. Inputs must be
proved immutable under `noalias`; each nontrivial local axis must independently
appear with unit coefficient in a distinct output coordinate. Opaque effects,
overlapping read/write snapshots, custom output strides/layouts, unproved
allocations, multiple final effects and any explicit execution binding decline
this family. One additional admitted form is a sequence of compiler-owned,
versioned pure-Tile producers followed by that same final element nest:

```text
Tile SSA:       u = A + B; v = f(u); output = g(u, v)
Reference:     full local u[] -> full local v[] -> final element loop
Mapped point:  let u = A[i] + B[i]; let v = f(u); output[i] = g(u, v)
                    one definition, all consumers share its scalar value
```

This is same-domain scalarization, not arbitrary recomputation. All producer
domains must have identical static extents; minima and axis variables are
normalized. Every compiler-local read and unique defining write must address
the current point exactly. Producers must dominate consumers and every
allocation must have a proved definition. Manual memory, unmarked producers,
transposed/neighbor reads, conditional or repeated writes, changed inputs,
cross-stage boundaries and domain mismatches retain the reference mapping.
The number of scalar definitions is bounded at 64. This preserves expensive
multi-consumer expressions once per element while eliminating full Tile arrays.
The root's inter-program independence comes from `parallel`, not a new user
proof obligation. `fuse_gpu_elementwise=false` retains the old mapping.

The first implementation has one linear ordering and a bounded default launch
width (at most 256, subject to target capacity). It is not yet a general
layout-permutation/coalescing solver. Explicit `threads_per_group` is checked
against the actual target capacity. Plans report
`elementwise_elements_per_program` and `elementwise_scalar_temporaries`; they
do not pretend this is an MMA or a reduction. An exact row-reduction mapping
request cannot be bypassed by this family. Four-shape M1 Max add and shared-SSA
GELU A/B evidence is linked from the status report.

## When to use integer programming, beam search, or annealing

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
Pareto rule is valid only for the [narrower additive matrix problem](matrix.md#implemented-solver-enumeration-plus-pareto-dynamic-programming).

The proposed compiler-internal interface makes that summary concrete:

```{table} What crosses a candidate boundary
:class: design-table
:name: candidate-boundary-interface

| Boundary information | What the parent must preserve or realize |
|---|---|
| Logical value/effect identities and active domains | Coverage, reaching values, alias/effect order and contribution multiplicity |
| Entry/exit distributions and replicas | Matching consumer ownership, or an explicit supported conversion |
| Numerical and participant contracts | Fold/tree permissions, convergence, binding and participation obligations |
| Ready/release relations and resident versions | Safe publication, overlap, last-reader release and physical slot reuse |
| Resource/engine profile and exposed scheduling choices | Peak per-instance storage, service contention and compatible issue/completion behavior |
```

A child candidate packages this interface with its checked realization and
cost features. These are analysis/plan records, not new public Nest or Memory
entities. If a parent may reschedule a child, the summary must retain the
corresponding event/resource profile or a parametric schedule; a single scalar
duration does not summarize every overlap choice. Proven-independent additive
children can use a cheaper summary.

Composition has an explicit proof obligation: if each child refines its source
region, their interfaces agree (or a verified adapter connects them), and the
parent satisfies cross-boundary effects, resource capacities and protocols,
then the combined plan must refine the composed source. This is a **theorem to
establish for the declared fragment**, not a proof already provided by the
current Pareto solver. Cost optimality needs further assumptions and does not
follow from semantic compositionality.

In particular, do not prune different entry/exit layouts using only local
latency and bytes. A parent may save a conversion, keep a value resident or
overlap work differently. Compare dominance only for an equivalent interface
and a future-independent summary; otherwise retain distinct candidates. Do not
use `T(parent) = sum T(child)` for an overlapping resource-contending pipeline.

Representation is itself a candidate choice. Original, materialized and
fragment-resident forms should compete when each has a supported emitter;
successfully lowering the first form is not evidence that it is cheapest.
The current initializer-materialization fallback is a useful coverage mechanism,
not this joint representation search. Similarly, a composed reduction needs
fold length, load/distribution and collective features, not merely an output
element count. XIR needs within-Tile packetization candidates before its cost
policy can choose them. These gaps motivate expanding candidate families before
replacing the search algorithm.

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

Rejections must distinguish semantic denial, target/resource infeasibility,
missing emitter, unknown legality and a legal candidate losing on estimated
cost. All should retain the responsible region/constraint and evidence.
Otherwise a new cost coefficient can appear to fix a coverage failure that
the evaluator never had a chance to consider.

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

## Validation and remaining work

Current regression coverage includes all thread-count choices up to the target
test bound; multiple rectangular shapes; exact atom coverage; native TIRx
forward/core-IndexMap inverse agreement; invalid coefficients and constraints;
shared-capacity tradeoffs; generated fragment reuse/residency; and CPU/physical
Metal numerical execution. Intermediate accumulator observers retain the
original storage behavior. Native matrix eligibility still checks capability,
arithmetic policy, the actual typed body, and operand layouts.

Performance validation uses multiple shapes, full-output FP64 references,
frozen binary/library fingerprints, counterbalanced repeated comparisons, and
PyTorch measured separately. The v2 Metal replay establishes parity for its
eight GEMMs; the CPU provider replays establish parity or better for their
eligible FP32 GEMM/sum/softmax cohorts. Neither result covers arbitrary
dtypes/operators or the portable XIR/reference realization. Improvements over
old lowering are reported separately from external comparisons.

Remaining structural work includes broader synchronization planning, a
calibrated materialization model with traffic/expression-depth/spill features,
layout-aware cooperative copies, combined software
pipeline/residency planning, general nested hierarchy binding, CPU task/SIMD,
cache/packing and provider break-even planning, calibrated target models, and
additional atom families. The current planners and provider proofs solve
specific realization gaps; they do not yet solve all execution-to-hardware
mappings or establish cross-target, cross-operator PyTorch-level performance.
