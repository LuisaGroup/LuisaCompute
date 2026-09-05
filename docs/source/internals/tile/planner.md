# Execution mapping: constraints, cost model, and search

Status: the Metal FP32 matrix realization described in sections 4--6, the
Metal FP32 row-program family in section 3.7, and the bounded shared-Tile
materialization choice in section 3.8 are implemented. The target-independent
formulation and later search strategies describe the extension contract, not
features already available on every target.

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

The optional TIRx MPP path now has a separate read-only snapshot-forwarding
candidate family. Its order is `effect/bounds proof -> snapshot forwarding ->
pipeline planning -> group resource/geometry planning -> typed MPP codegen`.
Full-K global views therefore do not consume imaginary shared A/B allocations.
Contract v2 also distinguishes an overwriting MPP multiply from
multiply-accumulate. The bridge selects it only for a reassociable standalone
positive-zero MMA or a closed positive-zero direct-output recurrence with one
static iteration; nonzero/negative-zero C, multiple K steps and observable
carry state retain accumulation. Mode is therefore a derived legal-realization
feature, not a name-based peephole or a user-visible DSL primitive.

After these proofs, the current bridge selects a separately versioned
`metal_mpp_memory_v2` bootstrap model instead of scoring MPP candidates with
the SIMD-group reference formula. For `G` participating subgroups, `P` logical
programs and target-profile subgroup capacity `Q`, it computes:

~~~text
issue = weighted(MMA issues, MPP operations, fragment reads, A/B footprints,
                 accumulator initialization, output traffic, aspect terms)
program_score = issue * state_pressure / G
              + independent_elements * element_weight / G
              + group_setup
waves        = max(1, P * G / Q)
kernel_score = program_score * waves
~~~

The division by `G` is important: disjoint subgroup tensor operations are a
concurrent program critical path, not serial work. The outer wave factor then
prices whole-device subgroup demand once. `Q=512` is currently an M1-class
replaceable prior, not queried occupancy, and fractional waves are deliberately
a smooth ranking heuristic. All quantities and the selected score are emitted
in the compilation/benchmark report.

```{figure} ../../../_static/tile/mpp-cost-model.svg
:alt: Metal MPP planning separates semantic and target facts, hard legality, target-specific cost features, bounded exact search, and authoritative staged JIT measurement.
:width: 100%

Cost-model v2 narrows a legal candidate set; complete-output validation and measured selection remain the authority.
```

Legality remains independent of this score. Exact coverage, target thread and
shared-memory limits, fragment/code-size bounds, and MPP's requirement that
each local matrix have M or N divisible by 16 are hard rejections. A coefficient
cannot rescue an illegal descriptor. The benchmark keeps the old TIRx and
staged MPP controls and selects/replays forwarding candidates separately. A
measured choice belongs to the specialized configuration, not to a hard-coded
shape dispatch in the bridge.

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
{download}`M1 Max equal-score layout experiment <../../../../scripts/benchmark/tile_torch/results/m1-max-20260903-layout-tie.md>`
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

The {download}`subsequent structural experiments <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-tirx-structure.md>`
also reject blanket contraction unrolling, direct global fragment loading,
shared-row padding, and double staging as universal defaults. Similar output
ownership to the MPP cohort did not reproduce its performance with the 8x8
atom family. These measurements constrain proposed defaults; they do not
identify a specific hardware bottleneck from elapsed time alone.

The {download}`MPP cost-model study <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-replay/notes.md>`
records the correction rather than hiding the failed model. On the same
8-shape × 45-candidate calibration cohort, v1 summed work from concurrent
subgroups and systematically overvalued narrow threadgroups. Replacing that
with subgroup critical-path work plus an outer wave prior changes the measured
finite-set regret as follows:

| Metric | MPP cost v1 | MPP cost v2 |
|---|---:|---:|
| Mean top-choice regret | 74.18% | 8.82% |
| Median top-choice regret | 43.05% | 2.59% |
| Maximum top-choice regret | 239.58% | 34.37% |
| Exact measured winner | 1 / 8 | 4 / 8 |

These are short 3-sample, 10 ms **in-cohort** search measurements. They show
that the concurrency term fixed a structural ranking error; they do not show
held-out generalization or a hardware optimum. Residual regret on 512³,
1024×128×256 and 513×257×129 still calls for calibrated cache/layout, edge and
launch features. The final frozen replay recompiles and remeasures the selected
v2 schedules independently of these search timings.

The outer Staged/JIT benchmark now jointly enumerates requested block shapes,
pipeline windows, group widths, and cooperative-copy batches. The product has
an explicit compilation budget and includes rejected candidates in its report.
Selection is followed by a fresh capture/JIT of all winning settings; frozen
replay uses the recorded native configuration, not an old score. This is an
implemented measurement-based outer tuner, separate from the inner planner's
still-uncalibrated analytic ranking. MPS/Torch remain external baselines and
cannot win this candidate search by replacing Tile lowering.

### 3.5 External libraries are performance targets, not solver candidates

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

**Integration contract:**

The backend-local native realization and an optional independent TIRx MPP
emitter now exist; see [Tile Runtime](runtime.md). TIRx MPP contract
v2 has separate typed `D=A*B` and `D=A*B+C` operations and fails closed on
mixed allocation modes. `plan_group` now has a separate MPP basis, enumerates
legal thread widths and exact rectangular subgroup factorizations, and ranks
them with target-specific realization features. The generic Machine TileIR
transform and calibrated time/uncertainty model described here are not yet
implemented; the native Metal emitter also does not yet consume this TIRx-local
planner.

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
3. **Cost model:** key samples by atom family, overwrite/accumulate mode,
   operation scope, cohort map, dtype/precision, shape, strides, bounds mode,
   compiler and device. Account
   for active/inactive groups, full versus edge operations, K extent, buffer
   reuse and output materialization. For an opaque MPP operation, unknown
   internal registers, barriers and copy stages stay unknown; do not invent
   native-8x8 issue counts or port its coefficients to another atom family.
4. **Solver:** exhaustive enumeration is sufficient for the small tested
   family. Apply integer coverage/resource constraints first, shortlist with
   a model, then JIT and validate. The standalone native probe records GPU and
   host batch time; the TIRx outer tuner ranks synchronized host-wall samples.
   Both retain every rejected/failed candidate. Frozen independent replay
   evaluates a selected plan, not a search minimum.
5. **Emitter:** `TileIR -> TIRx` retains typed operations, regions, effects and
   views. A supported Metal realization lowers its chosen atom to MPP and
   constructs inline tensors from buffer/layout expressions. Explicit TIRx MPP
   compilation now invokes the MPP legality/cost basis in `plan_group`;
   automatic cross-family selection among MPP, SIMD-group and scalar atoms is
   still absent. The independent native emitter has its own capability gate.

The benchmark runner is
[`compare_mpp.py`](../../../../scripts/benchmark/tile_torch/compare_mpp.py). It separates
search from six-order replay and records host and GPU batch times. Inline
tensors use a tracked classic Metal queue, as does MPS. The optional tensor
handle probe uses a Metal 4 queue with explicit dispatch barriers and commit
feedback; Metal 4 does not supply automatic resource hazard tracking. API
differences are recorded, not attributed to shader arithmetic.

### 3.7 Implemented Metal reduction family: execution before storage

The TIRx bridge now implements a separate, opt-in planner for structurally
proved FP32 add/max/min row programs. It is not an MMA plan with different
coefficients and does not reuse MPP's opaque resource assumptions. The complete
formal mapping, ownership proof, tests and evidence are in
[TIRx Metal reductions](reductions.md).

For `1 <= S <= min(32, target_max_threads/32)` SIMD groups per logical program,
`W=32S` workers stripe
every reduction and independent element domain. `S=1` may pack several source
`parallel` programs into one threadgroup; `S>1` assigns one program per group
and combines subgroup results through a proved shared partial array. A reused
compiler-owned logical Tile becomes `ceil_div(N,W)` private values per worker
only when affine analysis proves every access has that worker's distributed
element coordinate. Thus resource layout is derived from execution ownership;
it does not define the source hierarchy.

Guarded indirect reads require a second correspondence. If an immutable input
snapshot is consumed as `tile[label]`, path-sensitive view analysis may retain
the lazy bounds/fill condition and substitute a direct Tensor read. Otherwise
a whole-program audit requires every access to any distributed nonscalar local
buffer to equal the current worker-owner coordinate. An unknown proof declines
the subgroup candidate; it never leaves a worker reading another worker's
logical element from its own private allocation.

After hard target/effect/alias/control constraints, the finite v1 score is:

~~~text
rounds(S) = sum[d] ceil_div(independent_domain[d], 32S)
          + sum[r] ceil_div(reduction_extent[r], 32S)

score(S, P) = scalar_round_cost * rounds(S)
         + collective_cost * reduction_count * S
         + group_setup_cost / P
~~~

The default coefficients `1, 2, 16` are abstract M1-class priors, not measured
nanoseconds or occupancy. Each independent domain is rounded separately so
two softmax passes cannot share a fictitious tail round. Exhaustive enumeration
is exact for the admitted `(S, P)` family: up to eight packed-program choices
at `S=1`, plus every target-legal single-program width up to 32 subgroups.
An insufficient width-search budget fails explicitly, never silently truncates
the family. A nonzero `threads_per_group` is an exact
constraint; `run.py --tune-group-threads` recaptures/JITs each concrete width,
validates it, and independently recompiles the winner. Measurement calibrates
ranking but can never override legality.

The original 24-case sum/softmax/RMSNorm/LayerNorm/cross-entropy/residual-
LayerNorm reports select one, two, four and eight groups as widths grow, check
every output, and are faster than eager Torch MPS in all saved rows. Balanced
same-binary A/B replays attribute 21.19×--49.87× RMSNorm and
14.04×--75.54× LayerNorm/cross-entropy improvements to this mapping family.
The residual-LayerNorm A/B separately attributes up to 1.421× to preserving
and compacting shared Tile SSA instead of recomputing it. Sum and softmax use
preallocated Torch outputs; the functional normalization/loss comparisons
include returned-output allocation, while native A/B comparisons do not.
These facts validate the need for a structural execution and resource plan;
they do not establish the coefficient prior on held-out devices or a
production LLM operator suite.

### 3.8 Implemented shared-Tile materialization choice

Use count is a semantic/dataflow fact, not a placement decision. Structural
lowering now preserves every pure Tile SSA definition with multiple consumers
by default. Target planning can subsequently retain it, assign it a physical
resource, or prove that recomputation is preferable. The explicit
`EXPENSIVE_ONLY` lowering candidate preserves shared transcendental results but
recomputes cheap arithmetic, matching the earlier policy.

For the Metal row-program family, a preserved logical Tile can become a
worker-private stripe only after the element access proves the same affine
owner as the active worker. Candidate `S` is also subject to:

~~~text
stripe_scalars(S) = sum[t in materialized Tiles]
                    ceil_div(elements(t), 32*S)

stripe_scalars(S) <= max_reduction_striped_scalars_per_worker
~~~

The default bound is 64 scalars. It is an explicit compiler-created
software-state budget, not a measured register limit. At residual LayerNorm
width 4096, 32- and 64-thread candidates would require 256 and 128 scalars per
worker and fail before code generation; 128 and 256 threads require 64 and 32
and remain legal.

```{figure} ../../../_static/tile/shared-tile-planning.svg
:alt: Structural Tile SSA sharing is preserved while a target planner chooses recomputation or bounded physical materialization.
:width: 100%

The logical definition survives export; the target owns its resource choice.
```

The existing v1 row score counts scalar rounds, collectives and group setup.
It does not yet count duplicated global reads, expression depth, local stripe
traffic or measured spills. Consequently it chose `EXPENSIVE_ONLY` for the
four diagnostic residual-LayerNorm shapes and incurred 6.82%, 1.80%, 37.51%
and 43.66% measured regret. The finite staged/JIT search independently
captured, compiled and validated both policies, then selected `PRESERVE` on
Metal. The equivalent CPU search selected `EXPENSIVE_ONLY` for every shape.
That cross-target split is the intended architecture: shared SSA remains in
the semantic IR, while resource/recomputation policy is target-specific.

### 3.9 Automatic GPU element grids

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

### 3.10 Backend-owned execution cost policy

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

The
{download}`target-width replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-reduction-width-validation/notes.md>`
is a concrete test of that boundary. Expanding the six-width measured
subfamily beyond the restricted {32,128,256} reference yields 1.051×,
1.141× and 1.101× no-counter GPU gains for 1024×4096 sum, softmax and
RMSNorm, with fixed V=4/P=1/U=1. But two other search winners are slower
in all four independent pairs. The same-plan controls also expose variation
that a minimum-only selection policy cannot distinguish from improvement.
The six measurements are a subset of all newly legal subgroup widths, not
an exhaustive hardware optimum or default-policy comparison.

Consequently, measured selection should retain an incumbent and use a
separate acceptance phase before promoting a winner. The next backend policy
must relate physical group/subgroup demand, pack/tail code shape and
memory/issue/collective service; useful lane work alone is insufficient.
Both GPU execution and E2E dispatch should be reported, with the selection
objective explicit. This experiment changes no default scoring coefficients
and provides no held-out model calibration. The current small candidate
family still favors exact enumeration, not a more complex search algorithm.

The optional `cache_reduction_inputs` extends the materialization choice to
proved immutable inputs reused across distinct consumer domains. It is a
bridge/planner policy, not a Tile DSL memory declaration. It retains only
compiler snapshots admitted by the immutable-view audit; the existing
same-worker audit and cumulative private-stripe budget still decide legality.
An exact cache request rejects cross-worker gather or excess live allocation
instead of silently falling back. Same-domain `x*x` alone is not reuse.

The
{download}`fixed-width cache/reload replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-input-cache-validation/notes.md>`
holds W=512/V=4/U=1/P=1 and validates all 25 cases in four paired rounds.
1024×4096 softmax/RMSNorm/LayerNorm gain 1.378×/1.265×/1.221× GPU
throughput, with positive E2E gains. Ten identical-source controls and three
smaller changed-source cases with mixed individual GPU pairs are retained.
This is a materialization A/B, not width tuning or held-out model fitting.

The default score ranks 4096-column RMSNorm's cached plan as more expensive
(72 versus 64), and softmax/LayerNorm as 120 versus 112: a private copy adds
scalar rounds, but the model has no access-resource/traffic feature to reward
eliminated input reloads. Using that score to prune cache candidates would
miss these measured wins. Distinct global/private service, unique same-phase
loads, private live state and whole-device demand belong in backend policy
features; proof and budget checks remain bridge-owned. No fitted coefficient
or cache default is installed from this finite cohort.

The next implementation checkpoint adds **resource-sensitive payload access
demand** to `ReductionCandidate` and the realized `GroupPlan`. Each has an
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
The {download}`access-demand checkpoint
<../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-access-demand-validation/notes.md>`
records implementation tests and the subsequent 12-case joint search. It
retains 101 valid trials, 19 rejections and 12 freshly JITed winners. A frozen
four-round replay then compares the best joint candidate against the best
measured reload width from the same five-width family, at fixed V=4/U=1/P=1.
All 192 replay outputs pass. At 1024×4096 softmax/RMSNorm/LayerNorm gain
1.200×/1.214×/1.234× GPU throughput and 1.199×/1.221×/1.248× E2E throughput,
with every pair positive. Seven changed-source cases improve in every GPU
pair, four have mixed pairs, and one unchanged-source control is retained.

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

#### Whole-launch service policy and shape-held-out check

`ServiceExecutionCostPolicy` prices local scalar/collective/private-access
work, a continuous subgroup-demand saturation factor, whole-launch global
payload and per-worker global access. Its coefficients and capacity belong to
the caller's typed profile; they are neither queried occupancy limits nor
measured physical traffic. An explicit profile with unavailable access facts
fails rather than mixing fallback units. The complete returned kernel score
drives both the C++ width solver and optional model-only staged/JIT selection.

The first six-coefficient nonnegative fit was frozen in `47314e616` before
measuring softmax/RMSNorm/LayerNorm at 37×1537, 256×3072, 768×6144 and
64×12289. It uses no kernel-name or per-shape winner table. The independent
audit reconstructs all 32 candidate widths and confirms that input caching
was selected by model score, not new timing labels. At 768×6144 the four-round
replay gains 1.360×/1.287×/1.231× GPU and 1.372×/1.280×/1.233× E2E throughput
over the legacy automatic width/reload plan, with all pairs positive.

The same holdout rejects promotion to default: 37×1537 softmax and LayerNorm
regress in every GPU/E2E-throughput pair, and small RMSNorm GPU is mixed.
All three small plans change W=192/reload to W=416/cache, so that comparison
does not isolate width versus reuse. The subsequent fixed 2×2 ablation finds
the wider mapping slower at fixed reuse; the worker-pack tail repair then
improves small-case E2E throughput at fixed plans. Neither result establishes
a new optimum or justifies fitting noisy GPU labels. The
[reduction measurements](../../performance/tile/reductions.md#tail-packs-a-structural-repair-after-width-reuse-ablation)
retain both follow-ups and their controls. See the original
{download}`full service-policy evidence <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-service-policy-validation/notes.md>`
for all 288 validated outputs, separate GPU/E2E scopes and unchanged artifacts.

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

## 5. Implemented relative-work models

`ExecutionCostModel` contains two separately reported bases. The selected
`GroupPlan::cost_basis` prevents MPP features from being mislabeled as
SIMD-group work. Both are inspectable priors, **not nanosecond predictors**.

### 5.1 SIMD-group reference basis

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
In this reference basis, ordinary independent-element work is recorded but
constant across the first mapping family; it does not yet predict the effect
of thread count on copy throughput. Logical program count uses a coarse
preferred-program wave prior. The pressure/parallelism factors are explicit
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

### 5.2 Metal MPP memory v2 basis

MPP memory-input operations do not have the reference realization's shared A/B
fragment copies. The v2 feature vector instead records:

| Feature | Meaning and limitation |
|---|---|
| `matrix_issues` | Logical 8×8 result atoms × K atoms × executions; target issue work, not a source call-site count |
| `metal_mpp_operations` | Participating subgroup tensor operations × executions |
| `memory_fragment_reads` | Per-subgroup A/B memory fragment requests; separate from unique footprints |
| `lhs/rhs_footprint_fragments` | Unique logical A/B footprints with asymmetric reuse priors, not claimed DRAM transactions |
| `accumulator_initializations` | Absent only when the overwrite proof selects `D=A*B` |
| output/shared transfers | Derived from direct-output and persistent-accumulator proofs |
| Tile/local aspect terms | Target-versioned rectangle priors; not layout-equivalence claims |
| `fragment_scalars_per_lane` | Opaque-output logical live state, not measured registers |
| `independent_elements` | Scalar address/guard/store work outside the matrix atom |

For one group realization, the code computes:

~~~text
issue = sum(feature[i] * coefficient[i])
state_pressure = max(1, fragment_scalars_per_lane / preferred_fragment_state)
score = issue * state_pressure / subgroups
      + independent_elements * element_weight / subgroups
      + group_setup
waves = max(1, logical_programs * subgroups / concurrent_subgroup_prior)
kernel_score = score * waves
~~~

The solver compares `kernel_score` across thread widths. The subgroup division
models disjoint MPP operations on the program critical path; `waves` prices
outer machine demand once. The default concurrent-subgroup prior is 512 for the
tested M1-class profile. It is deliberately replaceable and fractional—neither
a target query nor a claim about physical residency boundaries.

The {download}`v1→v2 study <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-mpp-cost-v2-search/notes.md>`
shows why both terms matter and retains every invalid candidate. The v2 score
reduces in-cohort mean regret from 74.18% to 8.82%, but the 34.37% maximum miss
and absence of held-out data keep measured Staged/JIT ranking authoritative.

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
those observations. The {download}`copy-plan report <../../../../scripts/benchmark/tile_torch/results/m1-max-20260903-copy-plan.md>`
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
In the general effect analysis, the last fence of a sequential region is retained for the loop backedge or
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

The {download}`M1 Max synchronization report <../../../../scripts/benchmark/tile_torch/results/m1-max-20260903-barrier-plan.md>`
holds the tile, worker count, fragment layout, and resource realization fixed.
Four counterbalanced rounds show modest, shape-dependent gains, including one
1024-cubed regression. Fewer barrier sites are not a proportional latency model
and do not explain the remaining large-square gap to PyTorch.

### Independent subgroup programs: legality is not profitability

There is now a stricter whole-group proof for the opt-in TIRx MPP read-only
view realization. The matrix mapper supplies emission-local statement
identities for private cooperative-tensor initialization, synchronous MPP
steps and a verified partitioned output. The view analysis separately proves
the exact A/B parameter identities immutable under the caller's noalias
contract. A scope name, zero shared-memory usage or `metal_mpp=true` alone is
not enough.

The entire group must contain only these operations, uniform constant-bound
serial loops, no-ops and compiler-owned fences, followed by exactly one output
store outside all loops. A second global effect, a post-store consumer, an
output on a loop backedge, shared storage, branches, escapes, explicit fences
or unknown statements rejects the proof. Full subgroup participation and the
nonoverlapping output rectangles come from the matrix distribution verifier.
No instruction or memory access moves when the fences are removed.

~~~text
realized body + immutable-input / partition / participation facts
                             |
                   whole-group isolation proof
                    /                       \
                 fails                     succeeds
                   |                          |
       general conservative coalescing   legal choices: retain | elide
                                              |
                                  explicit planner/JIT choice
                                  (retain is the default)
~~~

`GroupPlan::independent_subgroups` records the proof independently of the
profitability choice. `PlannerOptions::elide_independent_subgroup_barriers`
is default-off and also requires the planner and `coalesce_group_barriers`
enabled. Selecting it never supplies missing proof facts. The fixed-geometry
{download}`A/B replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-subgroup-sync-replay/results.md>`
is a counterexample to pricing each removed fence as a guaranteed benefit:
512³ got slower in all four rounds, despite six interior cases having no
cross-subgroup communication. Apple's [MPP programming guide, §2.3.4](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
also describes periodic group barriers as a cache-working-set tuning tool.
That is a possible mechanism, not a measured diagnosis of this M1 result.

Read-only snapshot forwarding now reaches a fixed point. Axis relabeling can
produce `input -> snapshot -> relabeled snapshot -> MMA`; each round rechecks
whole-function effects, bounds, dominance and nonescape, then removes at least
one unique compiler allocation. This finite iteration handles anonymous shape
axes without asking the DSL author to reuse axis objects merely to avoid
storage. It does not remove manual `Memory` or weaken any snapshot contract.

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

### LLVM compiler-temporary storage realization

`PlannerOptions::max_cpu_stack_bytes` is an opt-in storage budget (0–65536
bytes, default zero), independent of the logical execution hierarchy. It does
not change the C++ DSL, tile shape, loop order, arithmetic, input/output layout,
or CPU task binding. Disabling the planner retains the workspace path too.

~~~text
TileIR compiler temporaries / explicit Memory
                  |
     TIRx flatten + vectorize + unroll
                  |
  allocation identity + escape/placement audit
                  |
       compact local scalar storage?
       static positive extent, nonescaping?
       no manual/unknown allocation contract?
                  |
     cumulative aligned payload <= budget
          /                         \
        yes                          no
   LLVM stack allocation       TVM workspace allocation
   (may become registers)      (original mechanism)
~~~

The pass runs immediately before host builtin lowering, after transforms that
can expand storage or duplicate definitions. It charges each eligible static
allocation with 16-byte padding; branch copies are summed, not assumed to share
lifetimes. Vector-typed, dynamic, strided, custom-layout, aliased, raw-pointer,
address-taken, and manually materialized resources are not candidates. Explicit
Memory remains explicit even without a resource-class argument; its marker
survives the common passes until this audit. A 68-byte payload therefore needs
80 budget bytes; two such buffers need 160, not two independent 80-byte limits.
This bounds **newly planned temporary payload**, not the total thread stack:
LLVM spills, host wrappers, and user/TVM-owned stack objects are separate.

The pinned TVM host pass otherwise sends `local` allocations through
`TVMBackendAllocWorkspace`, even for small static tiles. This plan uses TVM's
native `disable_lower_builtin` allocation annotation to retain `AllocBuffer`
for LLVM; it neither rewrites generated LLVM nor replaces a kernel with BLAS.
The ordinary TIRx realization remains selectable with budget zero. Removing
these allocation calls is not by itself a CPU GEMM microkernel: operand packing,
multi-row register reuse, temporal accumulator residency, and task/cache
partitioning still need independent models and measurements.

The {download}`first CPU storage replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-stack-replay/notes.md>`
improves paired medians over the previous lowering but also has per-round
regressions. The {download}`direct Torch/BLAS follow-up <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-stack-system/notes.md>`
still shows an approximately 11× 1024³ gap. This is a legal realization
candidate, not evidence of a solved CPU mapping/cost model.

### Cartesian CPU register packs

`PlannerOptions::max_cpu_vector_lanes` bounds the logical scalar count in a
Cartesian pack: 16 (default), 32, 64, or 128. Non-default values require LLVM
and automatic vectorization. This is a compiler/code-size budget, **not** a
hardware SIMD width, measured register count, or a new DSL hierarchy. Disabling
the planner retains the existing single-row automatic pack.

For the two innermost independent axes, choose a power-of-two column width
`W <= 16` and row count `R`, with `R*W <= budget`. The full-pack coordinate map
is `(m_min + rm*R + r, n_min + cn*W + lane)`, where `0 <= r < R` and
`0 <= lane < W`. Column tails cover only complete rows; row tails cover all
columns. These domains are disjoint and cover the original rectangle. All
outer axes and each element's temporal recurrence are retained.

~~~text
single-row packing                   Cartesian packing (R=4, W=16)

for row                              for row_pack
  initialize C[row, vector]             initialize C[4 rows, vectors]
  for k                                for k
    load B[k, vector]                    load B[k, vector]  <-- shared SSA value
    update C[row, vector]                update row 0 vector
                                         update row 1 vector
                                         update row 2 vector
                                         update row 3 vector
~~~

This is unroll-and-jam of independent element instances, not tensorization.
Only sequences, rectangular serial loops with element-invariant bounds/steps,
and element stores are distributed. Allocations, definitions, control flow,
unknown annotations, and lane-dependent temporal bounds retain the single-row
fallback. The semantic independent-element contract permits the interchange;
an MMA annotation alone does not. Arithmetic expressions and serial K order are
unchanged, including when reassociation is forbidden.

Rows remain separate contiguous TIRx vectors inside the common temporal loop.
The first prototype instead flattened row/column coordinates into a single
64-lane vector; the pinned TIRx emitter expanded the irregular address vector
into scalar loads/stores and regressed the 512³ probe. That implementation was
removed. The current emission exposes shared B loads to ordinary LLVM CSE,
without rewriting LLVM text or calling a GEMM library. It is still an opt-in
candidate: code size, tail work, packing traffic, and register pressure must be
measured, independently of the stack-allocation budget.

The {download}`four-round fixed-geometry replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-cartesian-replay/notes.md>`
improves seven paired median ratios but regresses 32³; the default remains 16.
The {download}`six-order CPU/Torch/BLAS comparison <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-cartesian-system/notes.md>`
still shows about an 8× 1024³ gap. Larger packs alone are not a complete CPU
mapping model.

### CPU immutable-input expressions

`CompileOptions::forward_readonly_tile_loads` also admits an independent LLVM
candidate. It stays default-off and is separate from stack placement and
Cartesian packing. The existing MPP policy remains strict: a cooperative
memory-input atom requires an unconditionally valid address.

For CPU scalar/SIMD consumers, a padded snapshot can instead become the same
**lazy guarded expression** at its use site. If the original copy is
`T[i] = if_then_else(G(i), A[F(i)], fill(i))`, a later `T[j]` may become that
expression with `i := j`, only after all of these checks:

- The entire invocation satisfies `noalias`; the external source has no
  stores or escapes, and no unknown effect can invalidate that fact.
- Every memory read in the address, guard, and fill expression is immutable
  too. Read-only A alone does not make `A[index_buffer[i]]` a stable snapshot.
- The unique complete initialization dominates every use, each consumer
  index is proved in the temporary's domain, and the guard implies a valid
  source address. An unknown proof keeps materialization.
- The temporary is compiler-owned, compact FP32 storage with no explicit
  Memory annotation, alias, or observed storage identity.

~~~text
Tile load snapshot
        |
  effects + dominance + bounds + resource constraints
        |
        +-- unknown / mutable / manual ---> retain snapshot
        |
        +-- proved immutable
                  |
          +-------+--------------------+
          |                            |
     LLVM consumer                cooperative MPP input
     retain lazy guard/fill       require unconditional address
          |                            |
     SIMD + storage planning      MPP resource + geometry planning
~~~

This does not turn a padded Tile into an unguarded pointer view. Bounds and
fill semantics survive substitution; ordered MMA keeps the same K recurrence.
The proof still rejects unguarded memory-dependent consumer indices, including
some mathematically bounded expressions outside its fragment. A pure lazy
branch whose path condition proves every temporary bound can now forward a
guarded gather without dropping its source bounds/fill expression. Unknown
cases remain correct snapshots, not promises of complete indirect-access
optimization.

Legality is not profitability. Forwarding trades copy/workspace traffic for
direct global strides and repeated predicates; compact packing may still win
through cache reuse or vectorization. Candidate selection must measure that
tradeoff at fixed geometry, thread request, stack budget, and pack budget.

The {download}`four-round forwarding A/B <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-views-replay/notes.md>`
confirms staging-array removal and 1.45–1.76× paired median improvements for
five regular shapes. In that historical revision both ragged GEMMs kept the same snapshot code;
their timing differences are not evidence of forwarding's guard cost.
The {download}`six-order CPU/Torch/BLAS comparison <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-views-system/notes.md>`
measures 5.541 ms versus 0.982/0.989 ms at 1024³. The cost model must record
actual realization and fallback coverage, not merely the requested switch.

### Full-vector guard specialization

Ragged immutable views exposed two separate lowering problems. First, the
native arithmetic proof did not always recognize `G => G` when coordinates
contained mixed-radix division/remainder. The forwarding audit now also
accepts bounds that are structurally present as conjuncts of the original
guard. Association/order and extra masks do not matter; an OR arm is never
treated as an assumption. All existing immutability/dominance checks remain.

Second, **legally forwarded does not mean efficiently vectorized**. The
{download}`retained failed experiment <../../../../scripts/benchmark/tile_torch/results/m1-max-20260904-cpu-guard-plan/notes.md>`
eliminated A/B storage but emitted only scalar input loads and scalar FMAs
for both ragged benchmark shapes. A lazy per-lane bounds check had scalarized
the contraction. More aggressive copy elimination alone was much slower.

Automatic CPU packing now proposes one full-vector fast path. Let `p` be
the available enclosing coordinates, `l` a lane of a width-`W` pack, and `G`
the selected lane-dependent guard conjuncts. Its sufficient precondition is

`U(p) = AND(g(p, l) for g in G for l in [0, W))`.

~~~text
lazy guarded input expression
              |
    compute U for this pack
              |
       +------+------+
       |             |
     U=true        U=false
       |             |
  SIMD fast arm    original guarded arm
  selected g=true same padding and bounds
       |             |
       +------+------+
              |
     same ordered recurrence
~~~

The fast arm replaces only those established Boolean facts. Scalar guards,
including the K coordinate and row bounds, stay in the expression. The slow
arm is the original loop, not an unchecked pointer access. Full and tail
arms do not execute together, and no FP zero products or recurrence steps
are discarded. This is a transformation of independent element domains,
not an MMA-only rule or an additional DSL primitive.

The same proof now handles lane-dependent statement guards around stores.
This matters even for plain ragged elementwise kernels: a predicated scalar
store inside a vector loop can force LLVM to scalarize an otherwise contiguous
full pack. A statement guard is versioned only when it has no `else` effect and
every lane-dependent condition contributes a proved pack predicate. Scalar
conditions remain in both arms. The 17×257 add control consequently fell from
the earlier approximately 2.84 µs observation to about 0.42 µs without any
provider call or change to its bounds semantics.

The precondition uses **every lane**, not merely endpoints: masks can have
interior holes. Candidate packs have 4–16 lanes and at most eight distinct
guard leaves; they receive one binary version, not a decision tree per
predicate. Only pure integer conditions already evaluated unconditionally
for every lane are selected. Memory reads, dynamic divisors, variables
defined inside the pack, predicated-read address expressions, unknown
effects, nested lazy arms, and possibly empty inner loops cannot supply
speculated facts. Native SSA renewal and statement simplification run before
vector lowering. Explicit vector scopes and Metal/MPP selection are unchanged.

Versioning increases code size and may increase JIT time; it does not prove
profitability. The planner must compare actual snapshot, lazy scalarized, and
full-vector realizations at fixed geometry. The guarded-view and automatic
packing options remain independently opt-in. Tests check actual allocation
removal and vector FMA emission, plus nonzero padding, interior masks,
negative offsets, nonzero loop minima, dynamic/zero-trip recurrences, and
untaken expressions that would overflow or divide by zero if speculated.

The {download}`fixed-geometry, frozen-binary A/B <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-guards-replay/notes.md>`
measures 1.279×/1.818× paired median improvements for the two ragged GEMMs,
with all four pairs improving for each. The six regular shapes have unchanged
LLVM instructions and remain no-op timing controls. The independent
{download}`six-order library comparison <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-guards-system/notes.md>`
still measures 5.919 ms Tile versus 1.021/1.028 ms Torch/BLAS at 1024³; this
repair is not the end of CPU execution/target planning.

### CPU root launch-cost guard

Logical `parallel` still means independent instances. It does not require the
CPU target to pay a thread-pool launch for every tiny automatic root. The
current planner uses a transparent bootstrap rule:

~~~text
explicit worker scope                       -> parallel (hard constraint)
planner disabled                            -> parallel (reference policy)
automatic root, extent >= 64                -> parallel
automatic root, extent < 64, expensive body -> parallel
automatic root, extent < 64, cheap body     -> serial
~~~

The body audit recognizes transcendental TIRx operations by typed `Op`
identity. Unknown external and packed calls are conservatively expensive.
Known synchronous vDSP reductions are treated as cheap so several short rows
do not pay a host launch only to call a small array routine. This is a target
scheduling choice, not a proof of `parallel` independence and not a new DSL
scope. The value 64 is a replaceable prior; a calibrated model should use
task count, per-task work, provider overhead, thread-pool state and cache
footprint. Tests cover the boundary, a transcendental body, explicit worker
binding and the planner-disabled reference path.

### Shared Tile SSA, target materialization, and CPU provider atoms

Lazy Tile expressions are valuable for fusion, but structural export must not
erase a shared SSA boundary by cloning its producer. The default exporter now
preserves every pure multi-consumer Tile as one compiler-owned logical
materialization. This is independent of memory scope and creates a target
resource candidate, not a user-visible `Memory`. `EXPENSIVE_ONLY` remains an
explicit diagnostic/JIT alternative that preserves `exp`, `log`, `sqrt` and
`tanh` while allowing cheap arithmetic to fuse into consumers. Only the exact
shared FP32 `exp` expression currently carries a versioned provider contract,
and the target pass re-proves that expression rather than trusting the generic
materialization annotation.

The CPU provider proof is two-level:

1. TileIR structure proves semantic intent. A reduction must be one rank-one
   FP32 recurrence with exactly one extracted element, one typed add/max/min,
   one yielded carry, and the corresponding `0`, `-inf`, or `+inf` identity.
   Whole GEMM requires the complete proved `C=A*B` dataflow contract.
2. The target pass revalidates the transformed TIRx. Array math requires a
   static compact zero-based map or contiguous final-dimension recurrence;
   CBLAS requires three compact rank-two FP32 parameters with matching extents
   and `noalias`. A stale annotation cannot rescue a mismatching body.

```{figure} ../../../_static/tile/tirx-realization-pipeline.svg
:alt: Structural TileIR export creates versioned proof contracts; target-specific passes revalidate them before choosing portable, CPU-provider, or Metal matrix atoms.
:width: 100%

The second proof firewall lets common passes evolve without turning annotations
or diagnostic names into unchecked rewrite authority.
```

`CpuMatrixBackend::CBLAS` replaces the eligible whole function with one
registered `tvm.contrib.cblas.matmul` packed call. It does not decompose the
execution hierarchy mechanically or call CBLAS from ordinary elementwise
code. `CpuMathBackend::ACCELERATE` maps proved reductions to synchronous vDSP
and the shared exp map to vForce. Known wrapper pointers are nonescaping, so
the bounded compiler-temporary stack planner may still retain their compact
scratch storage.

The math option explicitly permits provider semantics: vDSP may reorder FP32
reductions, while vForce has documented denormal and exception differences.
Reference remains the default. On a build/target without the provider, an
explicit request fails. On a supported target, an unrecognized local pattern
stays in reference TIRx; it is never approximately matched by an opcode label.

The {download}`array-math replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-accelerate-ops-replay/notes.md>`
reports 2.71--6.12× paired gains for row sums and 2.10--5.46× for softmax,
while add controls remain approximately 1×. The
{download}`CBLAS replay <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-cblas-v2-replay/notes.md>`
reports all eight shapes, direct CBLAS and eager Torch in all six execution
orders. The
{download}`CPU residual-LayerNorm search <../../../../scripts/benchmark/tile_torch/results/m1-max-20260905-cpu-residual-layernorm-materialization-search/notes.md>`
selects recomputation for all four shapes while retaining structural SSA and
holding its native LLVM/input-view/vectorization/64 KiB stack policies fixed.
These results validate two reachable provider families and one target-specific
materialization choice; they do not imply that direct XIR or the portable
reference loops have reached library parity.

## 8. Validation and remaining work

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
