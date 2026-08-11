# SIMD scheduler formal model

Status: executable small-step specification and bounded refinement audit.

This document defines the control-flow and uniformity contract of the SIMD CPU
backend independently of LLVM instruction selection. It is intentionally
precise about the boundary of the claim: the topology may be any reachable,
well-formed reducible CFG after XIR destructuring, while instructions,
resources, barriers, and device-library functions are supported only when
listed in `SIMD_BACKEND_DESIGN.md`.

The C++ reference transition system is `schedule/cohort_scheduler.h`. The
bounded exhaustive audit is `tests/unit/simd/test_scheduler_model.cpp`.
LLVM refinement fixtures are in
`tests/unit/simd/test_llvm_schedule_codegen.cpp`.

## 1. Static objects and assumptions

Let:

- `W` be the logical packet width and `L = {0, ..., W - 1}` the lane set;
- `G = (B, E, entry)` be the reachable CFG;
- `succ(b)` be the ordered executable successors of block `b`;
- `mu_T(s)` be the immediate post-dominator of a varying split or indexed
  branch `s` over terminating executions, when it exists in `B`;
- `Loops` be the natural loops discovered from dominators;
- `back(e)` name the natural loop whose header is targeted by back-edge `e`;
- `loops(b)` be the natural loops containing `b`.

The source obligations are:

1. every reachable block has exactly one supported terminator;
2. every PHI has exactly one incoming value for every CFG predecessor;
3. removing all natural-loop back-edges makes the graph acyclic;
4. every cycle therefore has a dominating natural-loop header;
5. structured XIR has already been destructured;
6. block barriers are excluded until cooperative-block scheduling is enabled.

Condition 3 is the operational reducibility check in `xir_to_schedule.cpp`.
Irreducible input is rejected before Schedule IR is constructed.

The subscript `T` is significant for natural loops. The repository's default
post-dominator analysis conservatively treats a reachable cycle as a possible
infinite maximal execution. SIMD scheduling instead requests
`account_for_infinite_paths = false`: a back-edge is not a virtual exit when
computing the rendezvous of lanes that eventually leave the loop. This does
not assert termination for every runtime input. If a lane loops forever, a
packet waiting for it may also run forever; the progress theorem below is
explicitly conditional on scalar-lane termination.

`ControlEdge::joins` and `ConvergencePoint::parent` are static hints for the
common dominance-nested case. They are not the authority for a dynamic
arrival. A block can be reached from both inside and outside a convergence
scope, so the runtime token and the destination block decide whether an
arrival occurs.

## 2. Uniformity lattice

The value-class lattice is

```text
warp_uniform <= cohort_uniform <= varying
```

where moving right loses information. `mask` and `token` are scheduler types,
not members of this data-value lattice.

The semantic predicates are:

- `warp_uniform(v)`: for all live lanes `i, j`, every dynamic observation of
  `v` in the packet has `v[i] = v[j]`;
- `cohort_uniform(v, A)`: for all `i, j` in the currently executing cohort
  mask `A`, `v[i] = v[j]`;
- `varying(v)`: no equality fact is assumed.

Seeds:

- kernel value and resource parameters, constants, block ID, kernel ID,
  block size, dispatch size, and warp size are `warp_uniform`;
- lane ID, thread ID, dispatch ID, and raster state are `varying`;
- callable parameters are conservatively `varying` until interprocedural
  specialization proves more.

For a pure instruction `r = f(x_0, ..., x_n)`, the transfer function is the
least upper bound of the operand classes. The worklist is monotone: a value can
only move from `warp_uniform` to `cohort_uniform` to `varying`.

For a PHI `p`:

- identical incoming values preserve the incoming class;
- a recurrent PHI has a minimum class of `cohort_uniform`, because its value
  may change between loop epochs;
- a distinct PHI selected by a warp-uniform control path may remain
  `warp_uniform`;
- a distinct PHI selected by a cohort-uniform or varying path is `varying`
  after suspension, because different cohorts may fill different lanes of its
  state slot.

The lattice has height three. With dependency edges indexed once, fixpoint
construction terminates after at most two degradations per value and has
`O(I + U + B + E)` work, where `I`, `U`, `B`, and `E` are instruction,
operand-use, block, and CFG-edge counts.

Soundness obligation: if analysis reports `warp_uniform`, scalar storage must
be observationally equivalent to a lane vector. If it reports
`cohort_uniform`, scalar evaluation is valid only while the current cohort is
owned by the caller; a suspension spill is lane-wise. A splat is introduced
only when a varying consumer requires `<W x T>`.

## 3. Dynamic state

A machine state is

```text
Sigma = (live, runnable, pc, token, epoch, frames, state, result)
```

with:

- `live[l]`: lane `l` has not returned;
- `runnable[l]`: lane `l` is queued rather than parked;
- `pc[l] in B`: the next static block for lane `l`;
- `token[l]`: zero or a one-based dynamic convergence-frame index;
- `epoch[l, q]`: the natural-loop epoch of lane `l` in loop `q`;
- `state[l, v]`: lane-wise suspension storage for Schedule values;
- `result[l]`: the optional scalar kernel return used by test fixtures.

Each active frame `f` contains

```text
frame[f] = (static_id, target, parent, expected, arrived)
```

where `expected` and `arrived` are lane masks, `target` is `mu_T(s)`, and
`parent` is the dynamic token that was current when `f` was allocated. The
parent is dynamic because the same static block may be reached through several
control contexts.

Frame capacity is bounded by `W`, independent of static CFG size. Allocation
occurs only for a nontrivial partition with at least two nonempty successor
masks, so it replaces one lane-owning leaf of the dynamic cohort forest with
at least two leaves. A forest over at most `W` lanes has fewer than `W`
branching nodes. Frame reuse adds no node, and return immediately retires every
frame whose effective expected set becomes satisfied. The LLVM state therefore
reserves `W` slots and traps only on an invariant violation.

For a block `b`, define the scheduling key of lane `l` as

```text
key_b(l) = (pc[l], token[l], {epoch[l, q] | q in loops(b)})
```

The scheduler may choose any nonempty equivalence class of runnable lanes with
the same key. Scheduling policy changes only which class is selected first.

## 4. Small-step transition relation

Let `A` be the selected cohort mask.

### 4.1 Pure block execution

Every supported pure instruction is evaluated pointwise for lanes in `A`.
Warp-uniform and cohort-uniform values may use scalar LLVM values according to
Section 2. Side effects are predicated by `A`.

### 4.2 Conditional and indexed branch partition

For a conditional selector `c`:

```text
A_true  = {l in A | c[l]}
A_false = A \ A_true
```

For ordered, unique switch labels `k_0 ... k_n`:

```text
A_i       = {l in A | selector[l] = k_i}
A_default = A \ union_i A_i
```

The successor masks are pairwise disjoint and their union is exactly `A`.
For a scalar warp/cohort-uniform selector, at most one successor mask is
nonempty and LLVM emits a scalar branch or `switch`.

If at least two successor masks are nonempty and `mu_T(s)` exists, allocate a
frame with `expected = A`, `arrived = empty`, `target = mu_T(s)`, and
`parent = token[A]`; then set `token[l]` to that frame for all `l in A`.
Allocation is lazy. Re-entering the same split while its frame is current may
reuse the frame, which is how a loop exit gate spans several epochs.

Using `mu_T` is required for loop refinement. For a header whose lanes have
different trip counts, the first cohort split allocates an exit frame; early
exiting lanes park, continuing lanes reuse that frame on later iterations, and
the block after the loop cannot execute until all still-live expected lanes
have exited. The same rule handles several loop exits that later share a real
merge. Treating the back-edge as a virtual exit would omit this frame and let a
post-loop collective execute once per exiting cohort.

### 4.3 Edge routing

For an edge `e = (u, v)` and flow mask `M`:

1. apply all PHI assignments in parallel under `M`;
2. if `back(e) = q`, increment `epoch[l, q]` once for every `l in M`;
3. perform dynamic target arrival for `v`;
4. set `pc[l] = v` and make every released/non-parked lane runnable.

Dynamic target arrival examines the top frame of `M`. If its `target` is not
`v`, the edge is an ordinary continuation. If its `target` is `v`:

```text
arrived[f] := arrived[f] union M
park M
R := arrived[f]
complete iff arrived[f] = expected[f] intersect live
```

On completion, deactivate `f`, set `token[l] = parent[f]` for `l in R`, and
continue the same target-arrival rule with `R`. The cascade is bounded by `W`
because a `W`-lane packet owns at most `W` live frames.

This dynamic rule is essential. Consider an acyclic graph where split `S`
reaches shared block `T`, `T` also has an entry that bypasses `S`, and both
paths later enter merge `M`. `S` does not dominate `T`, so a dominator-only
edge annotation cannot know whether `T -> M` carries `S`'s token. Comparing
the current frame target with `M` handles both contexts correctly.

### 4.4 Return

For every `l in A`, write its return value, clear `live[l]` and
`runnable[l]`, and remove it from the effective expected set of every frame.
Any frame satisfying

```text
arrived[f] = expected[f] intersect live
```

is released. Its arrived mask is restored to the frame's parent token and is
then routed through the same dynamic target-arrival cascade as an ordinary
edge before the target becomes runnable. This extra step matters when a child
and one or more ancestors share a target: directly resuming the target would
bypass the parent gate. A return can therefore neither leave a gate waiting
for a dead lane nor expose a partially reconverged cohort at the target.

## 5. Inductive invariants

The following invariants must hold after initialization and after every
quiescent transition boundary:

1. **Lane ownership.** Every live lane is in exactly one ready cohort or is
   parked at exactly one gate. A lane held by the currently executing cohort
   is in neither set until the transition finishes.
2. **Ready partition.** Ready masks are nonempty, pairwise disjoint, subsets
   of `live`, and unique by continuation key. Their union is `runnable`.
3. **Parked partition.** Gate-arrived masks are pairwise disjoint subsets of
   `live`; their union is the parked mask and is disjoint from `runnable`.
4. **Gate containment.** `arrived[f]` is a subset of `expected[f]`.
5. **Token validity.** Every nonzero live token names an active frame. Frame
   parents form a finite forest and allocation never creates a cycle.
6. **Cohort coherence.** All lanes in an executing cohort have the same PC,
   dynamic token, and relevant loop epochs.
7. **Branch conservation.** Branch/switch successor masks are a disjoint
   partition of the incoming cohort mask.
8. **Epoch monotonicity.** A lane's loop epoch changes only on that loop's
   natural back-edge and increases exactly once per traversal.
9. **PHI fidelity.** Each lane receives exactly the incoming value associated
   with the edge it traversed, before it becomes visible in the target block.
10. **Uniformity soundness.** A scalar value is used only in a dynamic scope
    satisfying its lattice predicate from Section 2.

`CohortScheduler::invariants_hold()` checks invariants 2--5.
`quiescent_invariants_hold()` additionally checks invariant 1.
The lowering verifier checks the static portions of 7--9. LLVM JIT fixtures
check representative refinements of 6--10.

## 6. Safety and progress claims

Under the static assumptions and invariants above:

- **lane safety:** each lane follows exactly one scalar CFG edge at every
  terminator and executes no instruction after return;
- **reconvergence safety:** a block that is a live frame target cannot execute
  for a member lane until every still-live expected lane has arrived;
- **collective instance safety:** lanes with different tokens or relevant loop
  epochs cannot be combined, while lanes released from the same gate execute
  the target as one cohort;
- **scheduler-policy independence:** for race-free programs, changing the
  order in which ready equivalence classes execute does not change per-lane
  results or collective participant sets;
- **conditional progress:** if every scalar lane execution terminates and no
  unsupported barrier is present, the packet cannot finish with live lanes and
  no runnable cohort; such a state is a lowering/runtime invariant failure.

The proof argument is induction over the small-step relation. Pure execution
preserves lane ownership. Branch partition preserves masks by construction.
Edge routing changes only the selected lanes, and dynamic arrival transfers a
mask atomically from ready ownership to exactly one gate or back to ready
ownership. Return monotonically shrinks `live` and all effective expected
masks. Reducibility assigns every cyclic transition a natural-loop header and
epoch, preventing distinct dynamic iterations from being identified by the
scheduler key.

This is a formal specification and proof outline, not a mechanized unbounded
proof in a theorem prover. The executable audit below is deliberately retained
as a regression oracle for the implementation refinement.

## 7. Bounded exhaustive audit

`test_simd_scheduler_model` fixes `W = 3` and exhausts:

- all seven nonempty active-lane masks;
- six input states for every active lane, for 342 initial states total;
- every legal choice of the next ready cohort, not only depth-first and
  largest-cohort policies;
- nested conditional/N-way divergence, a shared block entered from inside and
  outside an inner convergence scope, early return, cascading convergence,
  partial warps, and lane-dependent loop trip counts.

At every transition it checks the executable invariants. At every terminal
state it compares all lane results with an independent scalar interpreter.
The current fixture explores 4,782 complete scheduler interleavings and 47,764
small-step transitions.

The bounded audit is complemented by permanent LLVM regressions for:

- uniform scalar `switch` without eager lane broadcast;
- varying N-way switch masks and common reconvergence;
- switch-in-loop multiple exits and early returns;
- multiple natural-loop back-edges;
- different per-lane loop trip counts followed by a collective, including a
  loop with two distinct exits that share the collective merge;
- nested convergence, every reachable five-block forward topology (122
  graphs), 96 larger generated forward CFGs, and a 96-block JIT CFG;
- the non-dominating shared-entry counterexample described in Section 4.3,
  with a warp collective at the merge;
- a return that completes an inner gate sharing its target with the parent,
  with a warp collective proving that the released lanes cascade through both
  gates before the target executes;
- partial and full packets at widths 1, 4, 8, and 16 where applicable.

Every future counterexample found by formal audit must first be added as a
regression that fails on the old implementation, then fixed in the transition
refinement.

## 8. Mapping to implementation

| Model object | Schedule/LLVM implementation |
|---|---|
| `live` | `live.mask` |
| ready/runnable mask | `runnable.mask` |
| `pc[l]` | `lane.pc` |
| `token[l]` | `lane.convergence.token` |
| frame active/static ID/parent | `frame.active`, `frame.static.id`, `frame.parent.token` |
| frame expected/arrived | `frame.expected`, `frame.arrived` |
| loop epochs | `loop.epoch.*` |
| terminating-execution post-dominance | `PostDomTreeOptions::account_for_infinite_paths = false` |
| dynamic target arrival/cascade | `_arrive_at_convergence_target`, `_cascade_at_convergence_target` |
| branch partition | `_emit_terminator` split/switch lowering |
| PHI edge transfer | `_apply_assignments` before `_route_edge` |
| scalar-uniform storage | `LLVMValueLayout` plus `WarpUniformity` |

The generated LLVM remains target-independent fixed-vector IR. Machine ISA
selection, legalization, register allocation, and scheduling remain LLVM's
responsibility.
