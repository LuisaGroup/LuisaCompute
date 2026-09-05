(execution-structure-first)=
# Execution Nests and coordinates

These are language contracts and design extensions, not a promise of complete target support. Use the [executable examples](kernels.md) for current C++ spelling and [implementation coverage](../performance/tile/implementation.md) for admitted lowering.

```{contents} On this page
:local:
:depth: 2
```

## The semantic skeleton

This is not an algorithm IR followed by an optional bag of schedule commands.
The semantic object is:

~~~text
TileProgram = (
  SpatialNest E,
  ExecRegions R(E),
  Dataflow G,
  MemoryObjects M,
  Layouts L,
  ReductionAlgebras A,
  TemporalPolicies Pi,
  Constraints C)
~~~

`E` is the open logical spatial hierarchy. `R(E)` is the structured temporal
and algebraic tree of sequences, branches, loops, pipelines, reductions, and
explicit descents, with each region anchored at a prefix cut of `E`. Together
they are the execution structure. `G` contains tile SSA and effects, `M`
contains only stable addressable identities, `L` contains maps and
correspondences, `A` contains typed reducer contracts, `Pi` contains pipeline
policies, and `C` contains guards and still-open scheduling choices.

Function arguments, tensor extents, and host configuration may of course be
read before the outer `parallel`, because they can determine its extents. But every
semantic tile operation belongs to an execution region from the moment it is
captured. Its anchor is already known; only its descendant frontier, value
distribution, target binding, and realization may remain open. There is no
free-floating tile operation that a backend later drops into an anonymous loop
nest.

This does **not** mean hardware first. `parallel`, `serial`, `pipeline`, and
`reduce` declare logical execution structure, not blocks, warps, lanes, cores,
threads, or vectors. Hardware mapping remains a late `ExecBinding`, while a
ReductionPlacement separately factors algebraic axes into participant and
serial coordinates. It means the program's concurrency, ordering, and
aggregation skeleton is explicit before computation placement or physical
binding is solved.

Compiler-created levels are equally first class. A pass that splits or fuses
the nest creates real `ExecLevel` objects and an explicit remapping proof;
Scheduled TileIR never hides extra semantic loops solely inside backend codegen.

## The precise relationship to Halide

The design keeps Halide's most important separation: where a value is computed
and where it is stored are independent decisions. In Halide, a
[`Func`](https://halide-lang.org/docs/api/generated/Func.html) is a pipeline
stage and the schedule places its production and storage into a resulting loop
nest with mechanisms such as `compute_at` and `store_at`; the official
[scheduling tutorial](https://halide-lang.org/docs/tutorial/lesson_08_scheduling_2.html)
shows that loop nest as the concrete meaning of a schedule.

The difference here is the direction of construction:

| Question | Halide model | This Tile DSL |
|---|---|---|
| Semantic center | `Func` definitions and updates | `ExecNest` plus anchored TileIR regions |
| Execution structure | Concretized by scheduling the pipeline | Present in Candidate TileIR before tile operations are scheduled |
| Computation placement | `compute_root` / `compute_at` | Region anchor plus inferred or constrained frontier |
| Storage placement | `store_at`, `store_in`, folding | Owner prefix, then separate resource and address bindings |
| Target parallelism | Schedule transforms such as vectorization and GPU tiling | Prefix-compatible `ExecBinding` layouts |
| Temporal overlap | Loop/storage scheduling and folded storage | First-class issue schedule, dependence graph, and version map |
| Distributed tile value | Expressed through the lowered loop/vector organization | Explicit `Distribution` map or correspondence |

So the analogy is useful but exact: Halide is algorithm definitions plus a
schedule that produces an execution structure; this DSL starts with a logical,
transformable execution structure and hangs dataflow, storage, distribution,
and time from it. That extra indirection is what permits the same tile program
to be rebound to a GPU hierarchy, CPU teams and vectors, or accelerator cores
without making any of those machines the language ontology.

## Dimensions and spaces

The language has no predefined `axis::feature`, `axis::channel`, `axis::query`,
or even `axis::k`. Such names would turn neural-network conventions into the
type system. The primitive is a function-local dimension symbol:

~~~text
Dim       = (identity, optional_diagnostic_label)
DimFactor = (Dim, IndexSet)
Space     = ordered/nested product of DimFactor
~~~

`identity` is fresh and stable within a function. The optional label is printed
in diagnostics only; equality and mapping never inspect its spelling. Extent
belongs to a particular factor/domain rather than the symbol, so the same `m`
dimension may describe a full tensor extent in one space and a block extent in
another.

The public C++ forms are local and open-ended:

~~~cpp
auto [m, n, reduction] = dims("m", "n", "reduction");
auto unnamed = dim();

auto A_mk = A.with_dims(m, reduction);
auto B_kn = B.with_dims(reduction, n);
auto C_mn = C.with_dims(m, n);

auto feature = x.dim<1>(); // obtain an existing dimension by position
~~~

`with_dims` is a checked identity retyping of a view, not a memory operation.
It is needed when separate arguments must share dimension identity; dimensions
already inherited by a subview or Tile need no rebinding. Positional access is
only a way to obtain a `Dim` handle. All reduction, broadcasting, MMA, view,
and layout rules then compare handles and typed maps, never application-domain
words.

Dimension identities are part of the logical type. Two dimensions of the same
extent are not interchangeable without an explicit permutation, reindex, or
retyping operation.

A non-rectangular domain is an `IndexSet`:

~~~text
IndexSet = {x in Space | predicate(x, parameters)}
~~~

The common case has predicate `true`. Keeping a predicate on the domain is
important for ragged tiles and active participant sets, but a view's
out-of-bounds behavior remains a separate policy.

## A small, complete structured-region calculus

The surface has four core structured-region constructors. They deliberately
share one C++ range-for capture shape, but only `parallel` extends the spatial
owner hierarchy:

~~~cpp
for (auto &nest : parallel(grid_shape)) {
    for (auto &subnest : nest.parallel(
             subnest_shape,
             exec::warp)) {
        for (auto &k : subnest.pipeline(k_domain, pipeline_policy{...})) {
            k.stage("produce");
            // Producer operations.

            k.stage("consume");
            // Consumer operations.
        }

        for (auto &i : subnest.serial(tail_domain)) {
            // Strictly ordered temporal iteration.
        }

        auto sum = zeros<f32>(output_shape);
        for (auto &elem_nest : subnest.reduce(reduction_domain)) {
            sum += contribution(elem_nest.index());
        }
    }
}
~~~

The outer free function attaches to the implicit kernel root. Every member call
names the semantic parent of a child, without becoming a builder prefix for
ordinary operations. `nest`, `subnest`, `k`, and `i` are user-chosen variable
names for non-copyable scope handles, not predefined hardware roles. The
canonical examples deliberately reserve `tile` for data `Tile<T, R>` and tiled
tensor views, so an execution node cannot be mistaken for a memory level.

Execution coordinates are explicit: `nest.index(axis)` projects a named logical
axis, and `nest.index()` is available when that nest's own domain has rank one.
These are logical iteration coordinates, not memory accesses or hardware thread
IDs. `Nest` has no `operator[]`; the `A[origin, shape]` syntax is reserved for
loading a Tile from a tensor view.

| Constructor | Structural meaning | Ordering guarantee | Changes spatial participant prefix? |
|---|---|---|---|
| `parallel(domain, exec_policy?)` | Independent logical child instances; policy may constrain its logical map or target scope | No order between distinct active children | Yes |
| `serial(domain, order?)` | Repeated temporal child region | Total order | No |
| `pipeline(domain, policy)` | Repeated producer/consumer stage graph | Dependence partial order; iterations may overlap | No |
| `reduce(domain, contract?, policy?)` | Algebraic fold region whose outer Tile states are inferred from body updates | Reducer order; may become a tree, loop, or hybrid | No |

Independence is a **semantic contract**, not a hint asking the compiler to
rediscover it through alias or dependence analysis. `parallel` instances may
be serialized, interleaved, or packed into SIMD lanes without imposing a
relative order. Required coordination must be explicit in the program's
operations/contracts; choosing a fortunate serial order cannot supply it.
The same principle applies to independent element domains of Tile operations.
What lowering must check is its own realization: coordinate coverage, layout
and resource constraints, storage reuse, and target capabilities. It must not
reinterpret an inner `serial`/reduction recurrence as another independent axis.

```{figure} ../../_static/tile/nest-calculus.svg
:alt: Parallel extends space, serial and pipeline extend time, and reduce introduces an algebraic fold domain.
:width: 100%

The four regions separate independent space, ordered time, pipelined time, and
associative aggregation without naming hardware.
```

The second `parallel` argument is optional expert control, not part of the
logical domain. Omitting it leaves binding open. Common concise constants are
`exec::cluster`, `exec::block`, `exec::warp` (`exec::subgroup` is its portable
alias), `exec::thread`, `exec::core`, and `exec::vector`. A target maps each
supported constant to its execution topology and rejects an unavailable one;
an extensible catalog token remains available for unusual accelerators.
`exec::layout(map)` separately constrains the logical child map. These
constraints are retained in Candidate TileIR and participate in variant guards
and legality diagnostics.

Target execution scopes form a containment **partial order**, not the numeric
order of the convenience constants. Let `contains(h_parent, h_child)` mean that
one instance of `h_parent` owns the relevant instances of `h_child`. For every
pair of bound logical levels `Ej` and `Ek`, `j < k`, the solver must prove:

~~~text
contains(bind(Ej), bind(Ek))

target_parent_j^k o bind_map_k
  = bind_map_j o logical_parent_j^k
~~~

Thus `block -> warp -> thread` can be legal, `warp -> block` is not, and two
incomparable scopes on a heterogeneous accelerator cannot be nested merely
because their enum constants have an order. Unbound intermediate levels inherit
upper and lower constraints from their nearest bound ancestors and descendants.
Several logical levels may bind to the same hardware scope only when their
combined coordinate map factors inside that scope and satisfies cardinality,
uniqueness, and convergence constraints. Diagnostics cite both conflicting
nest sites and the missing target-topology relation.

This policy binds **execution**, not storage. It does not force every value in
the body to materialize at that target scope. Virtual Tile SSA edges are still
planned independently, and distinct explicit Memory objects declared under the
same `subnest` may bind to different resource kinds. A value-materialization or
Memory-resource constraint therefore targets that value/object in the schedule;
it is never a blanket field of `parallel`.

```{figure} ../../_static/tile/binding-relations.svg
:alt: Nested logical execution bindings are checked against a target containment poset, while memory resources use an independent capability relation.
:width: 100%

Execution ancestry has an order. Memory resources remain independent classes
connected to execution by topology and operation-specific accessibility.
```

`k.stage(optional_name)` is not another nest or a runtime operation. It
advances the frontend capture cursor to a new logical phase in that `k`
pipeline. The first call begins stage zero; each later call closes the preceding
segment and begins the next. The resulting pipeline owns an ordered list of
stage subregions. The name is an optional compile-time label; identity is local
to the owning pipeline. The iteration coordinate still comes from `k.index()`.

`reduce` is a nest-like region, but not a new execution or memory level. Its
range-for value is a non-copyable scope handle with `index()` and `coord()`, just
like the other three regions; it never changes type into an accumulator or an
input Tile. The compiler infers reduction states from outer `Tile` variables
updated by the body. A reducer contract lets scheduling factor the reduction
coordinate into spatial participants, serial steps, and a merge tree. There is
no public loop-result accessor. [Reduction semantics](values.md#reduction-is-a-structured-algebraic-region) defines the contract.

This set is representationally complete for the structured static-control
kernel domain targeted here. Spatial products and hierarchy factor into
`parallel`; a temporal total order is `serial`; a periodic partial order with
finite producer/consumer phases and fixed-distance loop-carried edges is
`pipeline`; and an order-relaxed fold with a stated algebra is `reduce`. Any
finite acyclic phase graph can be topologically staged, while the IR retains the
actual dependence DAG rather than mistaking textual order for an all-to-all
barrier. Dynamic `$if`, `$switch`, and `$while` remain ordinary Luisa
control-flow regions, not execution-nest kinds. Truly dynamic task creation or
host-visible queues would be a separate future facility, not silently
overloaded onto `parallel`.

Several tempting constructs are therefore derived rather than primitive:

- source block order already represents a sequence of statements;
- split, fuse, reorder, unroll, vectorize, and software-threading are execution
  transforms or policies;
- block, warp, lane, core, SIMD, and tensor-core placement are target bindings;
- elementwise operators lift through one generic scalar region; `map` is its
  custom-lambda surface;
- general einsum/contraction, convolution, scan, gather, scatter, copy, sort,
  and Top-K begin as library compositions; `matmul` is short syntax over the
  semantic `mma` value operation, and expression
  `reduce(value, axes, reducer)` is shorthand for a reduction region;
- synchronization is one abstract effect when semantically explicit; inferred
  barriers, events, and async protocol tokens are scheduled realizations or
  results of target atoms, not new region kinds.

Formally, every lexical point has a typed execution context:

~~~text
C = (P, T)
P = spatial participant prefix
T = temporal path, including serial/pipeline iteration and current stage
R = optional lexical reduction path; never a resource-owner prefix

parallel : (P, T) x I -> (P', T)
serial   : (P, T) x I -> (P, T')
pipeline : (P, T) x I -> (P, T') plus a local StageGraph
stage(s) : (P, T')     -> (P, T' ▷ s)
reduce   : (P, T) x (R, Reducer?) -> (P, T), with body context R
           and captured outer Tile updates as inferred states
~~~

This typing is the important decoupling: memory ownership and execution
binding project from `P`; liveness, recurrence, pipeline issue time, and
version selection project from `T`; reduction legality and factorization
project from `R`. A pipeline can reorganize time without inventing a memory
level, a reduction can move between space and time without becoming a memory
owner, and a spatial hierarchy can be rebound without changing either
producer/consumer or reduction semantics.

All four ranges use the same one-pass capture technique as Luisa's
`dynamic_range`. They do **not** enumerate logical coordinates on the host:

~~~text
begin / dereference : create the typed region op, push it, return Scope&
increment           : close captures, pop the region, mark iterator done
compare to sentinel : true only before that single capture pass
destructor           : close an open region during host stack unwinding
~~~

Thus braces, variable lifetime, and indentation accurately show the execution
tree, while extents remain staged properties of generated code. A native C++
`break` or `continue` only affects capture and is diagnosed; device control
flow uses the ordinary DSL constructs. A generation check likewise diagnoses
an escaped scope reference.

The spatial constructor has a concise default and an explicit coordinate-map
form:

~~~cpp
for (auto &child : parent.parallel(child_shape)) { /* append axes */ }

for (auto &child : parent.parallel(child_shape, child_exec_layout)) {
    /* reshape, permute, or swizzle logical child coordinates */
}
~~~

For parent spatial prefix `P`, raw child index set `I`, and child prefix `P'`,
the second form supplies a typed nest map:

~~~text
ParallelMap : P x I -> P'
parent_projection o ParallelMap = project_P
~~~

The default is `ParallelMap(p, i) = append(p, i)`. The preservation equation
makes ancestry mechanically checkable. The analogous temporal maps for
`serial` and `pipeline` preserve `P` and append or reorder only temporal axes.

These maps describe **execution coordinates only**. They do not choose which
tensor elements a value holds, how A/B/C are indexed, where a temporary is
allocated, or which hardware entity runs it. Those remain independent
`Distribution`, `Access`/`ViewMap`, `AddressMap`, and `ExecBinding` maps.

Each handle distinguishes its local coordinate from its complete typed path:

~~~text
subnest.index() = coordinate in I_subnest
subnest.coord() = ParallelMap(nest.coord(), subnest.index())

k.index()       = coordinate in I_pipeline
k.coord()       = append_time(subnest.coord(), k.index())

r.index()       = coordinate in R_reduce
r.coord()       = append_reduction(subnest.coord(), r.index())

leaf.index()    = coordinate in I_leaf
leaf.coord()    = LeafMap(subnest.coord(), leaf.index())
~~~

A local index is staged and readable only inside its lexical range:

~~~cpp
for (auto &subnest : nest.parallel(subnest_shape)) {
    auto s = subnest.index();

    for (auto &leaf : subnest.parallel(exec::infer)) {
        auto w = leaf.index();
        auto parent = subnest.coord();
        auto full = leaf.coord();
        // full is derived from (nest.coord(), s, w).
    }
}
~~~

Entering `parallel` changes the default spatial frontier; entering `serial` or
`pipeline` changes temporal context but does not automatically clone a value.
An operation's anchor is inferred from operand identities, free execution
coordinates, effects, and destination. A value that depends on
`subnest.index()` requires at least the subnest anchor; `per(subnest,
expression)` is the explicit form when no such dependence makes that intent
evident. Assigning an ancestor-owned value keeps the ancestor anchor and
creates a collective region update. At region close, layout analysis must prove
that child fragments form an exact, non-conflicting result, or an explicit
reduction/atomic combiner must resolve overlap. This is what makes direct
`acc = mma(a, b, acc)` concise without making ownership depend on C++ spelling
accidents.

## Execution is a nest, not yet a layout

For levels `E0 ... Ed`, the logical execution hierarchy is:

~~~text
E = E0 ▷ E1 ▷ ... ▷ Ed
Pj = Prefix(E, j) = E0 x ... x Ej
Fiber(E, a, b) = E(a+1) x ... x Eb
~~~

`▷` preserves level boundaries in addition to forming the product. The
boundaries define ownership, visibility, operation placement, and legal
ancestor projections.

Execution hierarchy and nested layouts are related, with an important type
distinction:

- the hierarchy is the **domain** and its factorization;
- a layout is a **map from that domain** to another coordinate space.

Without a codomain there is no layout function yet. Once a hierarchy is bound
to a target, the binding is a prefix-compatible family of layouts:

~~~text
beta_j : Pj -> TargetPrefix_j x SerialAxes_j

for j < k:
  target_parent_j^k o beta_k = beta_j o pi_j^k
~~~

Here `pi_j^k` is logical ancestor projection and `target_parent_j^k` is the
corresponding target ancestor projection. The deepest `beta_d` is the full
`ExecBinding`; retaining the compatible prefix maps prevents a later
transformation from losing hierarchy ownership information.

The maps may split, fuse, permute, serialize, vectorize, or elide logical axes.
Consequently no hierarchy constructor is named after a physical parallelism
mechanism.

## Logical anchor and execution frontier

A collective operation needs two depths, even though its normal surface only
mentions the second:

~~~text
anchor(op)   = a                  one logical instance per coordinate in Pa
frontier(op) = b, where a <= b    operation is distributed through level b
Active(op) is a subset of Pb      execution events that actually participate

participants(op, p) =
  {q in Fiber(E, a, b) | (p, q) in Active(op)}
~~~

The frontier defaults to the innermost enclosing `parallel` range. The anchor is
the least legal prefix satisfying operand/value identity, explicit execution-
coordinate dependencies, effects, and assignment constraints. `serial` and
`pipeline` add temporal context without changing that spatial frontier:

~~~cpp
for (auto &nest : parallel(grid)) {
    auto acc = zeros<f32>(output_shape); // anchor(acc) = nest

    for (auto &subnest : nest.parallel(subnest_shape)) {
        for (auto &k : subnest.pipeline(iterations, policy)) {
            auto a = load_a(k.index());
            auto b = load_b(k.index());

            acc = mma(a, b, acc);
            // anchor(mma) = nest, frontier(mma) = subnest
        }
    }
}
~~~

The existing identity of `acc` constrains the MMA update to remain one
logical update per `nest`; the nested scope says how deeply that update is
distributed. It does not duplicate a whole result per `subnest`. Conversely,
an expression depending on `subnest.index()`, or an explicit
`per(subnest, expression)`, creates a distinct logical value per subnest.

```{figure} ../../_static/tile/anchor-frontier.svg
:alt: One ancestor-anchored value updated collectively through a deeper spatial frontier.
:width: 100%

Anchor counts logical identities; frontier and `Active(op)` describe how those
identities are collectively realized.
```

The base DSL therefore needs no expression-level `.at(level)`. Operation
placement is visible in lexical execution structure and value ownership. Atom
contracts and the target catalog may infer opaque local participants and slots
beneath the visible frontier; expert scheduling constraints can pin them
without contaminating the semantic expression syntax.

The frontier is a maximum hierarchy depth, not a claim that every coordinate
through that depth executes the operation. `Active(op)` is an `IndexSet`; its
default is the full `Pb`, while predicates, atom contracts, or a pipeline plan
may infer a subset such as copy producers and MMA consumers. This fact need not
appear as `accessed_by(role)` in ordinary C++. A role becomes a named execution
level only when it has stable logical coordinate, nesting, or ownership meaning;
a target-specific engine assignment remains a late scheduling choice.

Fine-grained SIMT-like code simply nests again:

~~~cpp
for (auto &leaf : subnest.parallel(exec::infer)) {
    auto index = leaf.index();
    output(index).store(activation(input(index).load()));
}
~~~

This is a controlled descent into a sub-hierarchy, not the declaration of a new
memory scope. Ancestor-owned memories remain accessible. Updating an ancestor
value is allowed only through the checked collective-assembly rule above; no
operation silently escapes to an ancestor or invents a merge.

## Execution transform calculus

An execution transform is a typed reparameterization, not an opaque mutation
of a schedule. Let `E` be the old nest and `E'` the transformed nest. The pass
constructs:

~~~text
tau : E' -> E
~~~

`tau` maps each active new execution coordinate to the logical coordinate whose
work it performs in the old program. A divisible split, fuse, coalesce, or
permutation is a bijection. A split with a tail uses an `IndexSet` predicate.
If a transform needs replication or another non-functional relation, it uses a
`LayoutCorr` and must separately prove the legality of duplicated effects.

Level boundaries carry semantics, so equality of flattened cardinality is not
enough. For every old prefix cut `j` referenced by an operation anchor,
frontier, memory owner, or convergence rule, a hierarchy-preserving transform
must find a new cut `rho(j)` and a prefix map `tau_j` such that:

~~~text
tau_j : P'_{rho(j)} -> P_j

pi_j^d o tau = tau_j o pi'_{rho(j)}^{d'}
~~~

The equation says that selecting an old ancestor after remapping must depend
only on the corresponding new ancestor, never on hidden descendants. If it
does not factor, the pass may not silently cross that cut. It must reject the
transform or explicitly re-anchor the operation, re-home the memory, and insert
any required distribution or synchronization while proving equivalence.

Factorization is necessary but not always sufficient. A cut that defines
logical identity, such as a value anchor or memory owner, requires `tau_j` to
be bijective on its active prefix set. Otherwise several new prefixes could
claim the same old value or allocation (or one old identity could disappear).
Replication, merging, or re-homing is possible only as an explicit semantic
rewrite with its own proof.

For the common bijective case, every affected map is rewritten mechanically:

~~~text
anchor'   = rho(anchor)
frontier' = rho(frontier)

Access' = Access o (tau_frontier x id)
beta'_{rho(j)} = beta_j o tau_j
Instance'_s = Instance_s o tau_owner(s)

Distribution'
  = (inverse(tau_anchor) x id)
      o Distribution
      o (tau_frontier x id)
~~~

The same rule applies to atom operands, view maps that mention execution axes,
and dependence coordinates. For guarded or relational transforms, TileIR uses
the corresponding `IndexSet` restriction or correspondence composition instead
of inventing an inverse.

The spatial transform basis is deliberately small:

- `split` and `fuse` use mixed-radix maps;
- `permute` uses an axis permutation, subject to prefix-cut preservation;
- `reshape` and `coalesce` require a proved cardinality-preserving map;
- `tile` is derived from split plus permutation;
- insertion or removal of a unit level uses the identity isomorphism.

Serialization, vectorization, and machine-thread assignment do not rewrite the
logical hierarchy; they are choices inside `ExecBinding`. Pipelining does not
rewrite it either; it adds the temporal coordinate and issue policy described
in [pipeline scheduling](pipeline.md). Keeping these three operations distinct is what makes the system
execution-structure first without becoming hardware-structure first.

```{figure} ../../_static/tile/execution-transform.svg
:alt: A legal split below a memory-owner cut and an illegal silent permutation across that cut.
:width: 100%

Transforms compose typed remaps and must preserve every semantically observed
prefix cut.
```
