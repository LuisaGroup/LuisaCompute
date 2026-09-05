# Metal matrix planning and realization

This reference describes the TIRx bridge's bounded SIMD-group and MPP mapping families, work models and exact finite solver. The independent native Metal emitter has its own capability boundary; it does not automatically share these plans. Measurements belong in [compiler-route results](../../performance/tile/results.md).

```{contents} On this page
:local:
:depth: 2
```

## Native MPP experiment: operation scope is not launch size

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

## MPP cost model boundary

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

## Bounded K views avoid nominal padding storage

The optional `metal-mpp-bounded-k-v1.patch`, applied after MPP memory v2,
adds a separately checked capability without changing the DSL or TileIR.
It distinguishes the **logical** contraction tile BK from the **physical**
input interval `actual_k = min(BK, source_k - origin_k)`. M/N still have to
be fully in bounds; this is not yet a general masked tensor atom.

~~~text
logical A[BM, BK] / B[BK, BN]
              |
 immutable + noalias + dominance
              |
 canonical zero fill + unit K map
              |
equal positive actual_k; full M/N
              |
 A[BM, actual_k] / B[actual_k, BN]
              |
     MPP / pipeline / store

missing proof -> strict snapshots
~~~

The matcher derives this from native expression identities and enclosing
execution domains, not kernel names. It omits only a common **zero×zero**
suffix; a one-sided mask could multiply zero by an unmasked value and is
not interchangeable. A/B transposes retain their descriptor flags and
physical leading strides. The backend receives a positive signed actual-K
extent in both inline tensors; the nominal M/N/K and cooperative accumulator
contract remain unchanged.

Guarded forwarding is transactional. Every reassociable MMA in the candidate
must still receive a verified atom plan; otherwise the compiler retries the
original strict path, including its usual resource errors. Missing TVM
capability, M/N tails, additional masks, nonzero fills and unequal A/B K
intervals do not create a scalar fallback disguised as a successful MPP view
plan. Existing noalias, mutation, escape, manual-memory and recurrence gates
remain in force.

For BM=128, BN=32, BK=1024, this can remove the nominal 640 KiB A/B staging
requirement even when the final contraction is short. The existing resource
solver sees the resulting physical allocations; its work score still charges
nominal K conservatively. This extends legality, not the cost model's accuracy
or a claim of MPS parity. M/N edge atoms and automatic physical K retiming
remain separate work. The patch ABI and build order are documented in
[the TVM patch README](https://github.com/LuisaGroup/LuisaCompute/blob/codex/tile-programming-design/src/tile/bridge/tirx/patches/README.md).

## Physical program traversal remains a candidate

Program-grid traversal is another execution-layout choice, independent of
memory layout. It composes before the already planned subgroup/local map:

```text
physical program ordinal
          |
  bounded grid permutation
          |
logical program coordinate
          |
  subgroup + local coordinate
          |
     operand address map
```

`parallel` instances already have independent execution semantics. A candidate
permutation needs an in-domain bijection, not a new dependence proof. It must
preserve every program exactly once, including partial grid rectangles, and
does not promise the hardware scheduler's actual execution order. Memory
views and nested scope/resource ownership remain unchanged.

The standalone MPP benchmark can now enumerate bounded row/column rectangles
with no padded programs. It checks uint32 launch arithmetic before allocation;
small/tail GPU outputs and host enumeration validate the mapping. **The
production TIRx/native planner does not yet select these traversals.** TIRx's
cooperative mapper already uses a linear physical grid; the hand benchmark's
legacy-2D versus linear comparison is not a new bridge optimization.

The [two-order diagnostics](../../performance/tile/results.md#k-partition-and-program-walks-diagnostics-not-new-defaults)
reject a generic preference for larger row groups. For a 128×32 group output,
a 4×1 program stripe is a 512×32 output region, not a square reuse region.
Square-region candidates were also inconclusive under substantial timing
variation. A future backend policy should consider program aspect, operand
access maps and K partition jointly; the current per-group MPP footprint model
does not measure cross-group cache reuse or actual concurrent occupancy.
Keep those as candidate/model extensions pending stable held-out evidence,
without adding a DSL primitive or assigning a hardware memory level to a nest.

## Implemented matrix mapping family

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

## Implemented relative-work models

`ExecutionCostModel` contains two separately reported bases. The selected
`GroupPlan::cost_basis` prevents MPP features from being mislabeled as
SIMD-group work. Both are inspectable priors, **not nanosecond predictors**.

### SIMD-group reference basis

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

### Metal MPP memory v2 basis

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
profitability choice. The `elide_independent_subgroup_barriers` option in
`PlannerOptions` is default-off and also requires the planner and `coalesce_group_barriers`
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

## Implemented solver: enumeration plus Pareto dynamic programming

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
