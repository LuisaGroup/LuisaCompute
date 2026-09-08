# Execution calculus and compositional refinement

This page specifies the **proposed formal foundation**, not a completed
verified compiler. It connects the [language's execution nests](../../tile/execution.md),
[layout algebra](../../tile/layouts.md), [mutable TileIR](ir.md), and
[optimization formulation](planner.md#formal-finite-optimization-problem).
The bounded compiler families and their tests are implementations of parts of
this model. General scope fusion, proof certificates, distributed execution,
and the complete cost evaluator below remain extensions.

```{contents} On this page
:local:
:depth: 2
```

## Scope of the claims

There are four different questions; none implies the others:

```{table} Distinct correctness and optimization claims
:class: design-table

| Claim | Precise scope |
|---|---|
| Representation closure | Typed maps/relations and region composition remain representable after composition |
| Soundness | An admitted transformation preserves observable behavior under the source contracts and numerical policy |
| Relative mapping completeness | Every mapping in a declared, bounded map/atom/protocol vocabulary is representable and enumerable |
| Search optimality | The returned plan minimizes the stated objective within that encoded candidate set |
```

We do **not** claim a complete optimizer for arbitrary C++, every equivalent
algorithm, every layout with efficient code generation, or actual hardware
runtime. An exhaustive finite table establishes expressibility, not an
efficient normal form, useful search space, or publication-worthy result.

The first tractable fragment specializes extents and strides at JIT time and
uses finite integer domains, pure index expressions, typed SSA, explicit
memory effects, and registered arithmetic/atom contracts. For symbolic
reasoning, Presburger-definable sets and relations provide closure under
composition, projection and restriction; constant division/modulo encode
mixed-radix factors. The [isl manual](https://libisl.sourceforge.io/user.html)
provides an established representation and operations for integer sets and
relations, not a required compiler dependency. Products of two unrestricted
runtime parameters are outside that core. Fixed-width bit-linear layouts use
a separate GF(2)/bit-vector procedure; arbitrary data-dependent indices and
opaque calls keep explicit, weaker contracts. A solver timeout means unknown,
not false.

## Program, events, and contracts

The region grammar is:

```text
P ::= op | P ; P | if predicate then P else P
    | parallel(D, P) | serial(D, P) | pipeline(D, Pi, P)
    | reduce(R, g, A, P)
```

Sequence and conditionals are ordinary region structure, not additional
execution primitives. A function body can contain several sibling scopes;
it is an ordered region tree, not necessarily a single perfect loop nest.
Within a spatial path, prefixes and ancestor projections form the nest
described by the language. Temporal/reduction axes remain separately typed.

For specialization parameters `p`, define a semantic instance domain
`Omega_p` as the tagged union of operation occurrences and their active
spatial, temporal, and local element coordinates. Tags distinguish two
identically shaped operations. An operation may be expanded into a reference
scalar program for reasoning, without forcing the compiler to scalarize its IR.

`Gamma` records source assumptions, selected numerical semantics, and target
capabilities. The reference program determines value/effect labels and
required ordering, including reaching definitions, temporal carries and
observable memory order. A load creates a value snapshot; later stores do not
retroactively change it. Values assigned at the end of a temporal iteration
are updated simultaneously, including swaps.

For ordinary memory effects, the `parallel` contract is noninterference between
distinct active child instances. In a simple read/write model it implies:

```text
for i != j: W(i) intersect (R(j) union W(j)) = empty
```

Read-only sharing is permitted. This is **not** global argument `noalias`:
`C[i]` may be loaded and then overwritten through an aliased output in the
same instance. Nor does it remove ordering between two operations inside
that instance. Explicit atomic/collective extensions would carry their own
effect contracts; arbitrary colliding stores are not implicitly reductions.

The compiler assumes this language contract for valid inputs. It checks a
new distribution's address coverage, communication and storage reuse; it does
not run dependence analysis to ask whether the user really meant `parallel`.
If a transformation changes operation boundaries, its newly crossed effects
still need analysis. Contract provenance survives lowering: semantic facts,
derived facts and unchecked invocation preconditions are different things.

## Strength is a product order, not an enum order

Fix the same events, effect labels, arithmetic semantics and assumptions.
Ordering constraints form a partial order by inclusion of their transitive
closures:

```text
H1 <=order H2  iff  closure(H1) subset closure(H2)
then: linear_extensions(H2) subset linear_extensions(H1)
```

An unordered independent domain admits more schedules than one with added
order; a total order is a maximal restriction. Two partial pipeline orders may
be incomparable. Adding order must still produce an acyclic, implementable
schedule. This relation is **not** permission to erase observable edges from
a sequential program.

Assumptions form a separate implication order. `parallel` has weak ordering
but a strong noninterference precondition. Serial execution does not require
that precondition. Thus a valid `parallel` can be serialized; a `serial`
recurrence cannot be parallelized merely by changing its constructor.

Reduction is an additional algebraic dimension, not a position between
`parallel` and `serial`. For contributions `R` and output groups `G`:

```text
g : R -> G
fiber(o) = [r0, ..., rn-1] in source contribution order
fold_left(o)  = update_left(...update_left(seed[o], r0)..., rn-1)
fold_right(o) = update_right(r0, ...update_right(rn-1, seed[o])...)
```

The fiber uses active coordinates in lexicographic source-axis order, not
memory or lane order. Left/right folds specify different update structures and
operand orientations; right fold is not simply the unchanged left update run
backwards. Empty fibers return the incoming seed. Masks omit contributions;
physical replicas never create additional semantic occurrences.

**The source default is `unordered_tree`.** For a compatible reducer,
it permits merge trees and contribution permutations, including order-dependent
floating-point results. Ordered trees and strict folds are explicit restrictions.
These choices define sets of admitted computations; they do not oblige the
backend to use a tree. A serial realization can be in that set too. The typed
local policy is implemented; the standalone Metal candidate flag now controls
availability only. This does not make the general reducer-law calculus below
an implemented decision procedure.

For an exact left-reference tree rewrite, a sufficient contract is:

```text
update_left(s, r) = merge(s, lift(r))
merge is associative; identity is a two-sided identity
result[o] = merge(seed[o], ordered_merge(identity, map(lift, fiber(o))))
```

A right-reference rewrite instead has `update_right(r, s) = merge(lift(r), s)`
and the seed on the right. These sufficient monoid laws are not prerequisites
for a strict fold. Nor can an arbitrary `State x Elem -> State` update be
treated as a `State x State -> State` merge. Default/tree policies without a
compatible merge must be diagnosed, not silently assigned an invented algebra.

For `y[m] = reduce_k f(m,k)`, `R = M x K` and `g(m,k)=m`: distinct `m`
groups are independent; contributions along `k` combine. Several reduction
axes or a non-coordinate grouping map use the same definition. A pure recurrence
without a merge may use an explicit fold; general ordered effects use `serial`.
A scan also
exposes prefix results and cannot be replaced by a final-result reduction.

For exact equivalence, associativity permits changing parentheses while retaining
contribution order. Arbitrary permutation additionally requires commutativity or an
explicit permutation-invariant contract. A worker-striped fold can change
order, so associativity alone is insufficient for an exact rewrite. Floating-point
add is not an exact monoid: the default tree policy authorizes its changed
grouping/order without claiming such a proof. FMA behavior, precision and
exceptional-value handling remain separate requirements. `parallel` never
supplies reduction permission. See the
[language contract](../../tile/values.md#reference-fold-and-permitted-regrouping)
for the policy boundary and edge cases.

## Typed mapping witness

A plan uses the existing layout algebra in five different roles:

```{table} Typed components of a mapping witness
:class: design-table

| Map/record | Domain and meaning |
|---|---|
| Execution remap `tau` | active new logical occurrences to original occurrences |
| Binding `beta` | logical prefixes to target participant prefixes plus virtual/temporal context |
| Distribution `delta_v` | participant/local-slot occurrences to logical elements of SSA value `v` |
| Resource maps `mu_s`, `addr_s` | logical owner/version to resource instance, then local coordinate to address |
| Schedule/protocol `Theta` | realized events to issue order/time, completion and communication |
```

These are compiler records, not five new public DSL entities. A target prefix
means a controllable launch/cooperation coordinate, not an invented mapping to
an undisclosed physical GPU core. Multiple logical programs may share a group;
their virtual context and allocation-instance identities must remain distinct.
Memory capabilities are an access relation, not another copy of the execution
containment order.

For a bijective remap and each semantically observed old cut `j`, retain the
existing prefix-factorization condition:

```text
pi_j o tau = tau_j o pi'_rho(j)
```

An identity-defining cut needs a bijective `tau_j` on active prefixes. Crossing
that boundary requires an explicit re-anchoring/re-homing refinement, not just
equal flattened cardinalities. Binding retains compatible prefix projections,
including virtual and temporal context. Resource-specific address composition
then satisfies, for each logical access:

```text
logical element = delta_v(participant, local_slot)
logical access  = access_s(logical_context, logical element)
physical access = (mu_s(owner, version), addr_s(logical access))
```

There is no unique map from hierarchy to memory: `access_s`, materialization
and `addr_s` are independently chosen for every resource. A/B/accumulator may
therefore share participants while having unrelated layouts and lifetimes.
Copies/repartitioning implement mismatched producer and consumer distributions.

```{figure} ../../../_static/tile/execution-resources-calculus.svg
:alt: Ancestor execution coordinates compose with each resource's local access and address map; one execution hierarchy can access several independently mapped resources.
:width: 100%

Execution determines participants and context. Each resource independently
supplies its access and address maps; there is no single hierarchy-to-memory
function shared by every buffer.
```

### Observations, not syntax, determine preserved cuts

A lexical region boundary is not automatically a hardware boundary, allocation,
barrier, or persistent semantic identity. If no value, resource, effect or
collective contract observes a cut, a pass may remove or factor it while
preserving its logical coordinates. An observed owner or participant identity
must remain distinguishable, possibly through virtual context rather than a
dedicated hardware level.

Explicit bindings constrain admissible plans; they do not require one physical
level for every unbound lexical nest. The prefix law above is a sufficient
check for identity-preserving remaps, not the only admissible transformation.
Re-homing is a separate refinement with resource/effect obligations. Rejecting
all remaps that fail that simple law would make the compiler unnecessarily
rigid. Unknown legality should retain a conservative plan, not redefine the
source.

### Bridge compatibility is a relation check

For a value, represent its realized distribution as a relation:

```text
L_v subset LogicalElement x (ParticipantContext, LocalSlot)
L_v = {(e, p, s) | delta_v(p, s) = e, and (p, s) is active}
```

One logical element may have multiple physical copies; inactive slots need
not denote an element. An adapter to a logical-to-physical storage interface
uses this relation, not a functional inverse that loses replicas. For each
resource/version context it must preserve the same active pairs, offset,
resource instance and participant meaning. The TIRx shard/replica/offset
formulation is a relevant destination representation; see the
[versioned comparison](related-work.md#tirx-a-compatible-lower-boundary-not-our-automatic-planner).

This is an extension obligation for adapters, not a claim that every
`LayoutCorr` already lowers to every bridge. A plan outside a destination's
supported normal forms requires an available explicit realization, another
candidate, or a diagnostic. It must not silently discard replication or
substitute a convenient flattened layout.

### Established rectangular task product

For independent rectangular work, use the task construction formalized in
[Hidet, Section 5.1](https://arxiv.org/abs/2210.09603), rather than inventing a
second spatial/temporal algebra. Let `f: worker -> ordered task list` have
shape `d1` and `n1` workers; let `h` have shape `d2` and `n2` workers:

```text
(f tensor h)(w) = [x*d2 + y | x in f(w div n2), y in h(w mod n2)]
shape(f tensor h) = d1*d2; workers(f tensor h) = n1*n2
```

Products on coordinates are componentwise. This product is associative, not
commutative. `spatial` contributes worker factors; `repeat` contributes ordered
local tasks. This is an internal construction, not additional DSL primitives.
It builds a binding/schedule witness, whereas `tau` remaps semantic occurrences.
Observed prefixes, cross-task effects and asynchronous protocols still need
their separate contracts; a rectangular product alone does not prove them.

For replication or collectives, use `LayoutCorr` rather than a fictitious
functional inverse. One semantic contribution may have several storage
replicas, but a reduction consumes it exactly once. A collective atom refines
a whole semantic subgraph, so it is not modeled as a bijection between scalar
FLOPs and machine instructions. Its contract connects input/output values,
participant convergence, effects, and numerical policy.

## Constructive rules and proof obligations

Write `Gamma |- P =>[w] P'` for a transformation carrying witness `w`.
The primitive rewrite basis is small; compound schedules compose it:

```{table} Refinement rules and their additional obligations
:class: design-table

| Rule | Construction | Additional obligation |
|---|---|---|
| Reindex | identity, product, composition, permutation, unit insertion/removal | active-domain bijection and observed-cut factorization |
| Split/fuse | `i = q*b + r`, `0 <= r < b`, `q*b+r < N` | positive factor, exact active coverage, integer representability |
| Bind/serialize | assign independent instances to participant and local time factors | exact coverage, preserved identity, target containment |
| Repartition/materialize | compose distribution and resource maps, insert transfers | each use observes its reaching value; copies, publication and lifetimes |
| Retiming/versioning | move issue times and select version slots | all dependence distances, completions and reuse hazards preserved |
| Reduction factorization | partition each grouping fiber and merge partial states | contribution accounting, merge/order/numerical laws |
| Atom refinement | replace a matching semantic subgraph | target atom's value, layout, convergence, effect and arithmetic contract |
| Scope fusion/fission | combine or split sibling region implementations | preserved cross-region dependences and valid synchronization |
```

Some useful laws have short proof arguments in this restricted model:

1. **Reindex soundness.** A bijection gives exactly one new occurrence for each
   old one. Pulling labels, accesses and required edges back through it leaves
   the labeled event graph isomorphic. Prefix factorization preserves every
   referenced owner and value identity. Thus observations are unchanged.
2. **Composition.** Two such remaps compose by ordinary map composition;
   bijections and the commuting prefix equations compose. Contextual
   refinement rules compose only when their effect and numerical contracts
   are preserved. An arbitrary per-kernel `allclose` tolerance is not a
   transitive contextual equivalence and cannot replace that condition.
3. **Reduction factoring.** Partition a fiber into disjoint contribution
   subsets with no omissions. For a commutative monoid, folding partials equals
   folding the original set. For a noncommutative monoid, require contiguous,
   order-preserving pieces and an ordered merge. Floating-point realization
   requires its separate admitted numerical semantics.
4. **Resource remapping.** If every read is supplied with the same logical
   version, publication precedes use, and simultaneously live distinct values
   do not collide without a permitted alias, changing physical placement does
   not change observations. Capacity and target access checks establish that
   this placement can actually be emitted; they do not follow from `parallel`.

These are proof sketches, not mechanically checked theorems about the current
C++ implementation or TVM/LLVM. The
{download}`finite reference tests <../../../../scripts/test_tile_execution_calculus.py>`
check task-product associativity, exact coverage, prefix preservation, fusion,
reduction ordering and version lifetimes. They exercise examples and
counterexamples, not arbitrary-domain soundness or the production emitters.

```sh
python3 scripts/test_tile_execution_calculus.py
```

## Multiple scopes and fusion

Sibling scopes have sequential composition semantics. Initially `A ; B` makes
the observations of A precede B; analysis may remove only observationally
redundant ordering. This is distinct from choosing an implementation of an
already-independent `parallel` domain.

For producer instances `A(i)` and consumer instances `B(j)`, derive the
cross-region dependence relation `Dep_AB`. It includes RAW, WAR, WAW, state,
opaque-effect and protocol constraints, not only tensor shapes. A simple
sufficient rule for vertical pointwise fusion is:

```text
parallel_D(A) ; parallel_D(B)  =>  parallel_D(A ; B)
when Dep_AB subset {(i,i) | i in D}, with compatible ownership and effects
```

There is no new proof that A's own instances or B's own instances are
independent. The diagonal check concerns **new interleavings between A and B**.
Different domains can first be matched through a typed map; equality of their
extents is neither necessary nor sufficient.

```text
Pointwise dependence                 Cross-instance dependence
A(0) -> B(0)                         A(0) -----> B(1)
A(1) -> B(1)                         A(1) -----> B(0)
each pair can become one worker      per-worker fusion loses a required edge
```

For example, `tmp[i]=f(x[i]); out[i]=g(tmp[i])` admits pairwise fusion.
`out[i]=tmp[(i+1)%N]` does not admit that mapping even though both original
parallel scopes are individually independent. A shared group with a complete
barrier might handle the second case; otherwise retain a launch boundary.
Likewise, a scalar consumer of a global reduction must wait for every required
contribution. It may fuse into a final reduction stage, not execute once per
arbitrary producer worker.

Ordered time has a different sufficient rule:

```text
serial_i(A(i)) ; serial_j(B(j)) => serial_i(A(i); B(i))
when domains align and every cross edge A(i)->B(j) satisfies i <= j
```

This rule assumes the same increasing total order and no additional crossed
effects; more general cases use event-order preservation directly. Pipeline
fusion replaces these inequalities with dependence-distance and version
constraints. Horizontal fusion of independent siblings is a tagged disjoint
union, not pointwise zipping. Fusion is optional: saved launches/traffic may
lose to increased live state, barriers, reduced parallelism or a slower atom.

## Single source kernel versus physical execution plan

A source kernel is one invocation with a region tree and explicit effects.
It need not always become one physical kernel launch. A general compiled plan
contains supported code artifacts plus their launch/copy/completion DAG and
temporary lifetimes. Fusing nodes produces one device kernel only when the
target supplies the necessary within-kernel synchronization.

An ordinary GPU threadgroup barrier is not a device-wide phase boundary.
Unless a supported cooperative launch/protocol provides that boundary, global
dependences require separate launches. Busy-waiting resident groups while
other groups have not been scheduled is not a valid universal implementation.
The current Tile Runtime is a single-artifact path; multi-artifact execution
is an explicit future Runtime extension, not a feature inferred from the DSL.

The same formulation extends to multiple devices without new semantic nest
primitives. Binding adds device/rank coordinates; resource capabilities say
which buffers are local, remotely accessible, or require transfers. A crossing
edge becomes a supported copy/collective protocol with participants, matching
epochs, completion and visibility. Ordinary buffer pointers do not acquire
remote accessibility. The cost policy supplies topology-specific service and
latency; the solver also checks link contention and version capacity.

Fine-grained streaming partitions a dependence relation into ready subsets
and overlaps their producer/consumer epochs. Global reductions select a
merge topology consistent with the arithmetic policy. Neither distribution
nor pipelining invents associativity, ignores deterministic reduction order,
nor removes deadlock/progress requirements. Multi-node failures/retries and
distributed allocation ownership require a separate Runtime contract; no
current single-device proof establishes those properties.

## Validation, implementation and research obligations

Always validate IR ownership/use lists, types, dominance, state flow and
explicit target constraints. Pass-local checks validate the new map/protocol;
failure means another implementation or a diagnostic. Optional debug checks
can test invocation bounds, disjointness contracts, complete outputs and
canaries without making such checks an optimization prerequisite. A future
race/contract instrumentation mode must report its checked domain; sampling
does not prove absence of races. The current verifier is structural, and the
Runtime has bounded argument/range checks, not a universal race validator.

The current automatic Metal group admission, singleton-axis projection and
pipeline capacity reservation exercise reindex/bind/resource rules. Numerical
tests cover permuted program coordinates, leading/interleaved unit axes,
same-instance input/output aliases, explicit worker bindings, disabled
planning, arithmetic permission, and multi-phase attention states. They do not
establish general sibling fusion or a joint planner for every intra-group
reduction/distribution alternative. Limited collective emitters are not that
general composition result.

To support a paper, the deliverables should be: a small reference interpreter;
formal judgments and proofs for a stated fragment; a typed transformation
witness checked independently of search; backend refinements; and held-out
multi-operator experiments with mapping/communication/materialization
ablations. Compare model regret, search gap, compile budget, GPU timing and
dispatch separately. A theorem about map coverage is not a speedup theorem.

### Closest prior work and the open contribution

The [focused literature survey](related-work.md) compares Hidet's composed
task mappings, Cypress's task/resource model, Hexcute's layout/atom synthesis,
verified scheduling rewrites, concurrency algebras and constrained mapping
optimization. Execution-oriented tensor programming is not an unexplored
alternative to layout algebra. We should reuse established components and
test the combined effect/resource/time model against these close precedents.

The research hypothesis is **compositional, prefix-aware refinement of an
execution-first SSA program, coupled to a resource/time optimization problem
and reusable backend policies**. Novelty, useful relative completeness, and
cross-target performance still need to be demonstrated against these systems.
