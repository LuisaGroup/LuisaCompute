# Luisa Tile DSL: A From-Scratch Design

- Status: architecture proposal and syntax contract, revision 13
- Compatibility with the removed prototype: none
- Primary workloads: GEMM, attention, convolution, normalization, quantized,
  sparse, grouped, and persistent neural-network kernels

## 1. The design in one page

The central architectural choice is **execution-structure first**. The logical
execution nest is part of the program before tile operations are placed or a
target is chosen; it is not a loop nest reconstructed at the end from an
algorithm-only graph. Computation, distributed values, memory ownership, and
temporal pipelines are all attached to this structure through typed maps and
constraints.

The language has three first-class dimensions:

1. **Execution** is a spatial `parallel` hierarchy, temporal `serial` and
   producer/consumer `pipeline` structure, plus algebraic `reduce` regions whose
   domains may be scheduled across space and time.
2. **Data** is tile SSA, views, and explicit addressable memory.
3. **Mapping** is one typed layout algebra used for execution binding,
   value distribution, view indexing, memory addressing, and atom operands.

They meet through checked composition; none is encoded inside another.

```{figure} ../_static/tile/execution-first-overview.svg
:alt: Execution structure as the skeleton connecting dataflow, memory, layouts, pipelines, scheduling, and lowering.
:width: 100%

Execution is declared first; data, resources, mappings, and target choices attach
to it through typed relationships.
```

The canonical C++ shape is:

~~~cpp
auto make_gemm(GemmConfig cfg) {
    return tile_kernel([=](TensorView<const bf16, 2> A,
                           TensorView<const bf16, 2> B,
                           TensorView<bf16, 2> C) {
        auto [m, n, reduction] = dims("m", "n", "reduction");
        auto A_mk = A.with_dims(m, reduction);
        auto B_kn = B.with_dims(reduction, n);
        auto C_mn = C.with_dims(m, n);

        auto M = A_mk.extent(m);
        auto N = B_kn.extent(n);
        auto K = A_mk.extent(reduction);

        for (auto &nest : parallel(
                 shape(ceil_div(M, cfg.block_m),
                       ceil_div(N, cfg.block_n)))) {
            auto [tm, tn] = nest.index();
            auto m0 = tm * cfg.block_m;
            auto n0 = tn * cfg.block_n;
            auto acc = zeros<f32>(
                shape(m(cfg.block_m), n(cfg.block_n)));

            for (auto &subnest : nest.parallel(shape(cfg.groups))) {
                for (auto &k : subnest.pipeline(
                         range(0, K, cfg.block_k),
                         pipeline_policy{
                             .max_in_flight = cfg.max_in_flight,
                             .initiation_interval = 1})) {
                    auto k0 = k.index();

                    k.stage("load");
                    auto a = A_mk.tile(
                                  coord(m0, k0),
                                  shape(cfg.block_m, cfg.block_k),
                                  bounds::zero)
                                 .load();
                    auto b = B_kn.tile(
                                  coord(k0, n0),
                                  shape(cfg.block_k, cfg.block_n),
                                  bounds::zero)
                                 .load();

                    k.stage("compute");
                    acc = mma(a, b, acc);
                }
            }

            for (auto &leaf : nest.parallel(exec::infer)) {
                auto out = cast<bf16>(maximum(acc, 0.0f));
                C_mn.tile(
                     coord(m0, n0),
                     shape(cfg.block_m, cfg.block_n),
                     bounds::predicate)
                    .store(out);
            }
        }
    });
}
~~~

Important properties of this surface:

- There is no explicit builder parameter. A scoped current builder records the
  program, as in the Luisa SIMT DSL.
- `for (auto &nest : parallel(...))` and
  `for (auto &subnest : nest.parallel(...))` make spatial hierarchy and lexical
  scope the same visible C++ structure. They do not say block, warp, lane,
  SIMD, or vector.
- The outer spatial nest supplies an anchor context; the innermost enclosing
  `parallel` supplies the default frontier. Data dependencies, explicit child
  coordinates, and the assigned value constrain the final anchor. Thus
  assigning nest-owned `acc` inside `subnest` is one logical tile update
  distributed through that child level, without `.at(...)` noise.
- `parallel`, `serial`, `pipeline`, and `reduce` are the complete core
  structured-region vocabulary. They share one range-for capture protocol but
  have distinct spatial, temporal, and algebraic semantics.
- `for (auto &k : subnest.pipeline(iterations, policy))` creates one temporal
  producer/consumer pipeline per `subnest`; `k.index()` is the logical
  iteration coordinate.
- Data/effect dependences infer a producer/consumer stage graph. Optional
  `k.stage("name")` cursor cuts partition the nearest enclosing pipeline body
  into subregions that can later bind to distinct participant subsets or
  engines.
- The loop body directly assigns `acc`. Capture analysis constructs the hidden
  loop-carried SSA edge.
- A load produces a Tile SSA value. A cross-stage use already gives the resource
  planner enough information to keep, materialize, version, fuse, or recompute
  it; ordinary pipeline staging does not require explicit `Memory`.
- Addressable memory is explicit only when stable storage identity is part of
  the intended schedule.
- Ordinary host configuration values create ordinary JIT variants. A symbolic
  staging language is not required for autotuning.

Kernel arguments belong in the lambda signature, exactly as they do in Luisa's
SIMT DSL. `tile_kernel(lambda)` retains the C++ definition; it does **not** invoke
it before concrete argument metadata is available. On a JIT specialization,
the frontend creates parameters in signature order, attaches the concrete
shape/stride metadata to their proxies, and invokes the lambda to build one
candidate TileIR. `with_dims` gives positional dimensions kernel-local semantic
identities. No `input`, `output`, `GemmSpec`, or symbolic integer language is
needed in the body.

`TensorView<const T, R>` optionally prohibits writes through that parameter.
`TensorView<T, R>` permits reads and writes; it does not promise that the
parameter is an output. Actual read/write effects come from IR uses, not a
second direction declaration. Constness does not imply non-aliasing.

The signature adapter is implemented for dense `TensorView` parameters. Its
low-level `definition.capture(tensor_shape(...), ...)` entry lets IR tests and
compiler bridges provide metadata without a runtime device adapter. It runs
the definition afresh for each call. Every executable Tile DSL test and POC
uses this signature entry; the former `define`/`input`/`output`/`inout` helpers
and their `Buffer` proxy have been removed, with no compatibility surface.
The runtime `device.jit` adapter, arbitrary strided views, and scalar signature
parameters remain separate implementation work.

## 2. Non-negotiable separations

The following are different IR concepts even when a backend eventually folds
them into one instruction:

| Concept | Question | TileIR representation |
|---|---|---|
| Logical value | What tensor elements exist? | Tile SSA value and logical space |
| Execution hierarchy | Which nested program instance invokes and participates? | `ExecNest`, anchor, and frontier |
| Execution binding | How do logical instances map to a machine? | `LayoutMap<Exec, TargetExec>` |
| Value distribution | Which participant and local slot hold an element? | `LayoutCorr<PhysicalSlot, LogicalCoord>` |
| View | Which coordinates of an existing object are named? | `ViewMap` plus bounds predicate |
| Memory | Which addressable object has stable identity? | `Memory` and owner prefix |
| Reduction | Which semantic contributions merge under which laws? | reduction domain, grouping map, and reducer contract |
| Materialization | Which virtual SSA edges require storage? | compiler-owned materialization and version plan |
| Resource instance | Which per-program physical allocation is selected? | `InstanceMap` |
| Address | Which byte inside that allocation is selected? | `AddressMap` |
| Pipeline | When is an operation issued and which version is live? | schedule and version maps |
| Atom | Which participant and slot feed an instruction operand? | operand `LayoutMap` contracts |

A register fragment is normally a distributed SSA value, not a memory scope.
A logical hierarchy level is not a hardware scope. A view is not an allocation.
A pipeline stage is not automatically a ring-buffer index. A reduction tree is
not automatically a warp collective, and a cross-stage value is not
automatically a user-declared `Memory`.

### 2.1 Surface convenience is not primitive proliferation

Three extension layers must stay distinct:

| Layer | Purpose | Examples |
|---|---|---|
| C++ Tile library | Pleasant reusable algorithms that expand to core regions and SSA | GELU, matmul/einsum, convolution, softmax, Top-K, sort, scan |
| TileIR core primitive | Irreducible semantics, effect, or transformation contract that cannot be reconstructed | parallel, serial, pipeline, reduce, scalar region, `mma`, view, load/store/atomic/sync |
| Target atom | Proved replacement for a matched core subgraph on one capability set | target MMA implementation, async copy, shuffle reduce, sort network, tensor-memory operation |

A library call may remain as an in-memory call temporarily for compile-time and
diagnostic quality, but it must have a target-independent expansion into the
core. It is not a new semantic entity merely because the C++ spelling is short.

Adding a target feature normally adds an atom contract and a rewrite pattern,
not a frontend or Candidate TileIR primitive. A proposed core primitive is
accepted only if at least one of these is true:

1. its observable semantics or effect cannot be represented by composition;
2. expanding it would lose information required to prove legality;
3. it introduces an irreducible synchronization, convergence, or resource
   protocol shared by more than one target family.

Faster matching, prettier syntax, or one vendor instruction is insufficient.
If later evidence justifies promotion, the library definition remains the
reference semantics and differential oracle for the new primitive or atom.

`mma(a, b, c)` is the deliberate exception to the usual "keep tensor
operators in the library" rule. Expanding it immediately into multiply and
reduce loses the fused accumulation contract, accumulator precision and
rounding policy, and the cooperative operand-layout relationship needed to
prove legal use of matrix hardware across GPU and accelerator families. The
Candidate operation is portable semantics; a target MMA instruction remains a
Machine TileIR atom that must refine those semantics and satisfy concrete
participant and layout contracts.

The admission test applies to region kinds too. The current four survive for
different reasons; there is no fifth catch-all execution entity:

| Region | Information that would be lost by expanding it away |
|---|---|
| `parallel` | Independence, logical participant product, and the spatial prefix used by ownership and distribution |
| `serial` | The canonical counted iteration domain and mandatory recurrence order; it is the one structured loop primitive rather than a second generic loop hierarchy |
| `pipeline` | Finite stage segmentation, dependence distances, and permission to overlap iterations; flattening to `serial` destroys the intended scheduling search space |
| `reduce` | Identity, update/merge laws, grouping, accuracy policy, and permission to reassociate; a serial fold is only one legal lowering |

Distribution is a map, not a region. A memory resource is an addressable
object, not a region. A pipeline stage is a segment of one pipeline, not an
execution level. These negative decisions are as important as the four positive
ones.

The initial reference library has explicit desugarings:

| Library surface | Core expansion |
|---|---|
| lifted `+`, `exp`, `select`, GELU | generic scalar SSA region over a dimension-identity join |
| `reduce(x, dimensions, r)` | captured reduce region with generated state update |
| `matmul(a, b)` | infer conventional trailing matrix dimensions, create a zero result, then `mma(a, b, zero)` |
| general `einsum` | reindex/broadcast, elementwise multiply, reduce; a schedule may retile it into `mma` |
| convolution / pooling / normalization | views plus elementwise and reduce regions |
| `gather` | value-computed view index plus load |
| `scatter` | value-computed index plus store or atomic effect |
| `copy` | load/store edge, optionally recognized as a transfer atom |
| `topk<K>` | indexed Tile plus bounded merge-and-truncate reducer |
| `sort` / `merge_sorted` | compare/select/reindex networks or structured radix/merge library |
| `scan` / histogram | parallel/serial/reduce regions plus indexed effects |

This table is a test obligation: the target-independent expansion must run in
the TileIR interpreter, and every atom replacement is checked against it.

```{figure} ../_static/tile/primitive-layers.svg
:alt: Rich Tile library calls expand into a minimal TileIR core, while target atoms replace only proved equivalent subgraphs.
:width: 100%

Surface convenience, semantic primitives, and hardware atoms evolve at
different rates; keeping them layered prevents permanent IR bloat.
```

## 3. Execution structure first

### 3.1 The semantic skeleton

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

### 3.2 The precise relationship to Halide

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

### 3.3 Dimensions and spaces

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

### 3.4 A small, complete structured-region calculus

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

| Constructor | Structural meaning | Ordering guarantee | Changes spatial participant prefix? |
|---|---|---|---|
| `parallel(domain, exec_policy?)` | Independent logical child instances; policy may constrain its logical map or target scope | No order between distinct active children | Yes |
| `serial(domain, order?)` | Repeated temporal child region | Total order | No |
| `pipeline(domain, policy)` | Repeated producer/consumer stage graph | Dependence partial order; iterations may overlap | No |
| `reduce(domain, contract?, policy?)` | Algebraic fold region whose outer Tile states are inferred from body updates | Reducer order; may become a tree, loop, or hybrid | No |

```{figure} ../_static/tile/nest-calculus.svg
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

```{figure} ../_static/tile/binding-relations.svg
:alt: Nested logical execution bindings are checked against a target containment poset, while memory resources use an independent capability relation.
:width: 100%

Execution ancestry has an order. Memory resources remain independent classes
connected to execution by topology and operation-specific accessibility.
```

`k.stage(optional_name)` is not a fourth nest or a standalone TileIR opcode. It
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
no public loop-result accessor. Section 5.3 defines the exact semantics.

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

### 3.5 Execution is a nest, not yet a layout

For levels `E0 ... Ed`, the logical execution hierarchy is:

~~~text
E = E0 ▷ E1 ▷ ... ▷ Ed
Pj = Prefix(E, j) = E0 x ... x Ej
Fiber(E, a, b) = E(a+1) x ... x Eb
~~~

`▷` preserves level boundaries in addition to forming the product. The
boundaries define ownership, visibility, operation placement, and legal
ancestor projections.

The user's intuition that execution hierarchy is a layout nest is almost
right, with one important type distinction:

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

### 3.6 Logical anchor and execution frontier

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

```{figure} ../_static/tile/anchor-frontier.svg
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
    output[index] = activation(input[index]);
}
~~~

This is a controlled descent into a sub-hierarchy, not the declaration of a new
memory scope. Ancestor-owned memories remain accessible. Updating an ancestor
value is allowed only through the checked collective-assembly rule above; no
operation silently escapes to an ancestor or invents a merge.

### 3.7 Execution transform calculus

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
in Section 7. Keeping these three operations distinct is what makes the system
execution-structure first without becoming hardware-structure first.

```{figure} ../_static/tile/execution-transform.svg
:alt: A legal split below a memory-owner cut and an illegal silent permutation across that cut.
:width: 100%

Transforms compose typed remaps and must preserve every semantically observed
prefix cut.
```

## 4. The canonical layout algebra

### 4.1 Decision

The spatial core adopts the semantics of the production-proven
[CuTe layout algebra](https://docs.nvidia.com/cutlass/4.3.0/media/docs/cpp/cute/02_layout_algebra.html):
hierarchical shapes, hierarchical strides, composition, products, and divides.
TileIR adds types for named domain/codomain axes, explicit replica fibers, and
an F2-linear node for lossless Triton import. It does not replace CuTe's
arithmetic with an unrelated algebra.

Triton's
[LinearLayout](https://github.com/triton-lang/triton/blob/main/include/triton/Tools/LinearLayout.h)
is valuable for hardware-to-tensor distributions, but its own contract is a
power-of-two F2-linear subset and explicitly does not cover every non-power-of-
two or padded layout. It is therefore an importable normal form, not the only
canonical representation.

### 4.2 Typed map

Every layout has a domain and codomain:

~~~text
LayoutMap<D, C> : D -> C
~~~

`D` and `C` are `IndexSet`s, so the function is total on its declared domain.
The proof-friendly grammar is:

~~~text
Map := Identity
     | MixedRadix(shape, stride)
     | BitLinear(basis)
     | Translate(offset, Map)
     | Compose(Map, Map)
     | Product(Map...)
     | Project(axes)
     | Permute(axes)
     | Recast(unit, Map)
     | Transform(registered_bijection, Map)
     | Piecewise(disjoint predicates, Map...)
     | IndexExpr(pure integer expression DAG)
     | FiniteTable(domain, values)
~~~

`MixedRadix` is the CuTe-compatible base. For digitized coordinates `di` and
possibly vector-valued named strides `si`:

~~~text
MixedRadix(x) = offset + sum_i di(x) * si
~~~

Nested shapes preserve the digit factorization. A scalar stride produces a
linear index; vector strides produce named physical coordinates.

`BitLinear` maps input bits to output bits over GF(2), directly representing a
Triton `LinearLayout`. `Transform` admits registered transforms with declared
proof hooks, such as XOR swizzles. It corresponds to CuTe's
[ComposedLayout](https://github.com/NVIDIA/cutlass/blob/main/include/cute/layout_composed.hpp)
when an outer transform cannot be folded into an ordinary shape/stride layout.

`IndexExpr` is not an arbitrary C++ callback. It is an effect-free integer IR
over axis variables and declared parameters. Its analyzable subset contains
affine arithmetic, constant floor division/modulo, comparisons, and bitwise
operations; expressions outside a proof domain remain executable but carry
weaker facts. `FiniteTable` is the completeness escape hatch for a static
finite tile. It is legal but deliberately expensive and is never the preferred
generated form.

The C++ layer does not expose these node classes. Common layouts use concise
constructors and composition; an expert escape hatch captures a pure index
lambda:

~~~cpp
auto row_major = layout(shape(M, N), stride(N, 1));

auto fragment = index_map(
    domain(subtile, worker, local(4)),
    codomain(shape(BM, BN)),
    [](auto subtile_id, auto worker_id, auto slot) {
        return coord(/* symbolic integer expressions */);
    });
~~~

The lambda is executed once by the layout builder with symbolic axis values.
It cannot load data or perform effects. The result is normalized into
`MixedRadix`, `BitLinear`, `Piecewise`, or a registered transform when
possible, and otherwise remains an `IndexExpr` node.

Maps alone are not enough to describe both common layout orientations. Triton
normally maps a hardware location to a logical tensor element, whereas TIRx
maps a logical element to a set of physical locations. We represent the latter
without inventing another arithmetic algebra, using a correspondence whose two
legs are ordinary maps:

~~~text
LayoutCorr<A, B; F> = (left : F -> A, right : F -> B)
~~~

`F` is an explicit finite occurrence space. A function `f : A -> B` embeds as
`F=A, left=id, right=f`. A set-valued placement with replicas embeds as
`F=A x Replica`, `left=project_A`, and `right=placement`. Swapping the legs
reorients a correspondence without pretending a many-to-one map has a unique
inverse.

This small span representation is internal. The normal C++ surface still sees
layouts, not relations or category-theory terminology.

### 4.3 Algebraic operators

The primitive operators are typed composition and product:

~~~text
f : A -> B, g : B -> C        => g o f : A -> C
f : A -> B, g : C -> D        => f x g : A x C -> B x D
~~~

Projection, permutation, nesting, and translation are ordinary maps. CuTe-like
`coalesce`, `complement`, `logical_divide`, `logical_product`, `zipped_divide`,
and `tiled_product` are derived constructors with verifier obligations rather
than unrelated IR nodes.

A functional inverse is not a total algebraic operator. `try_inverse(map)`
returns a map and a proof only on the proven image; `preimage(map, point)` may
instead return a fiber. Reorienting a correspondence merely swaps its legs and
is always representable, but does not make either leg functional. APIs that
require a unique owner demand an injectivity proof rather than quietly choosing
one occurrence.

Composition is always closed in the IR because a layout remains an expression
DAG when no flatter normal form exists. Optimization may normalize it; the IR
never rejects a valid composition merely because flattening failed.

Correspondence product is the product of both legs. Correspondence composition
joins the two occurrence spaces on their shared coordinates and remains a DAG
until a pass needs an enumerated form. A code-generating memory access must
eventually be proved total and single-valued from its event domain; a placement
query is allowed to remain one-to-many.

### 4.4 Replication and non-injectivity

Replication is represented by an explicit occurrence/fiber axis:

~~~text
F = Logical x Replica
logical   : F -> Logical        = project_Logical
placement : F -> Physical       = shard(Logical) + replica(Replica) + offset
~~~

The resulting `LayoutCorr<Logical, Physical; F>` is exactly the set-valued map
needed for replicated placement. A `Replica` axis may denote broadcast,
multicast, or an ownership choice. Every copy has a coordinate identity, while
ordinary non-injective maps still represent cases where several physical slots
hold the same logical value without requiring a special node.

```{figure} ../_static/tile/layout-correspondence.svg
:alt: A layout correspondence with an occurrence fiber representing two physical replicas per logical element.
:width: 100%

The occurrence space makes replication explicit and avoids assuming a unique
inverse for a many-to-one placement.
```

### 4.5 Bounds and data-dependent indexing

The declared `IndexSet` determines where a map is defined. Access behavior
outside a view is not hidden inside the layout. A view carries a separate
validity predicate and policy:

~~~text
View = (domain, map, valid(domain))
~~~

`bounds::zero`, `bounds::predicate`, and `bounds::assume` are policies for the
invalid part of the domain.

Gather, scatter, indirection, and data-dependent indices are semantic index
operations feeding a layout. Treating a memory load as a layout node would
destroy most useful equivalence and invertibility reasoning.

### 4.6 What “complete” means

No useful compiler can honestly claim a complete decision procedure for every
symbolic user-defined index function. The contract is therefore explicit:

1. The grammar above is representationally closed under its operators.
2. Every static finite mapping is representable through `FiniteTable`.
3. Every static finite relation is representable by a finite occurrence space
   and two `FiniteTable` legs.
4. Mixed-radix affine maps, F2-linear maps, and registered transforms have
   exact proof procedures for equality, image, preimage, injectivity,
   surjectivity, alignment, and cardinality where their symbolic constraints
   are decidable.
5. Small static domains may use exhaustive proof as a fallback.
6. A staged `IndexExpr` may be lowered as computation, but it must
   be normalized or accompanied by guards before a pass that requires an
   inverse or a bijection.
7. Backend pattern coverage and autotuning search coverage are never called
   algebraic completeness.

This is stronger and more honest than promising that every C++ callable is a
fully analyzable layout.

### 4.7 Compatibility embeddings

The important systems embed as follows:

| Source | TileIR embedding |
|---|---|
| CuTe `Layout<Shape, Stride>` | `MixedRadix` with the same nested shape and stride |
| CuTe composed/swizzled layout | `Compose` or registered `Transform` |
| Triton `LinearLayout` | hardware-to-logical `BitLinear` map and its correspondence |
| TileLang fragment | correspondence between logical axes and thread/index axes |
| TVM TIRx layout | `Logical x Replica -> named Physical` correspondence plus translation |

TileLang permits general forward functions for a fragment. Such a function is
in the regular subset only when it normalizes to this grammar; otherwise it is
an explicit `IndexExpr` with correspondingly weaker proofs. The direction of a
source system's public API is preserved on import; no lossy inversion is used.

### 4.8 Proof discipline and algebra laws

Every nontrivial analysis returns one of `proved`, `disproved`, or `unknown`,
optionally with a guard and witness. `unknown` is never treated as success by a
legality pass. This lets the implementation start small without baking unsound
assumptions into the IR.

The interpreter and normalizer must preserve these observable laws:

~~~text
id o f = f = f o id
h o (g o f) = (h o g) o f
(f1 x f2)(x1, x2) = (f1(x1), f2(x2))
converse(converse(R)) = R
converse(S o R) = converse(R) o converse(S)
~~~

Normal forms are specialized rather than universal:

- mixed-radix maps coalesce with CuTe-compatible divisibility rules;
- `BitLinear` uses GF(2) row reduction for rank, image, kernel, and inverse;
- registered transforms provide their own exact evaluator and proof hooks;
- correspondence simplification removes redundant occurrence axes only when
  both legs remain equivalent;
- small finite maps and correspondences are checked exhaustively;
- everything else remains a composed DAG with conservative facts.

Interop tests compare semantics, not printer syntax:

~~~text
eval(import_cute(L), x) = eval_cute(L, x)
eval(import_triton(L), hardware_slot) = eval_triton(L, hardware_slot)
placements(import_tirx(L), x) = placements_tirx(L, x)
semantics(export(import(L))) = semantics(L)  when export is supported
~~~

For a static domain these are exhaustive. For a symbolic domain they are
proved under emitted shape/extent guards and supplemented with randomized
differential tests.

## 5. Value distribution is a layout, not another algebra

Let a tile SSA value `v` be produced by an operation anchored at `a`, with a
participant frontier `b`. One logical value instance exists per coordinate in
`Pa`; its pieces are carried by an active `Carrier(v)` index set through `Pb`:

~~~text
Carrier(v) is a subset of Pb
Distribution(v) : Carrier(v) x LocalSlot(v) -> Pa x LogicalCoord(v)

owner_preserving:
  project_Pa(Distribution(v)(e, slot)) = project_Pa(e)
~~~

This is an ordinary `LayoutMap`. Viewed in the opposite direction, its
`LayoutCorr` says where each logical element is placed and naturally exposes
replicas. “Distribution” remains useful terminology, but it does not get a
second set of arithmetic or composition rules.

An operation use may have its own access map:

~~~text
Events(use) is a subset of P_frontier(op)
Access(use) : Events(use) x LocalAccess(use) -> ViewCoord(use)
~~~

Producer and consumer maps need not match. If the solver cannot fuse their
correspondence into an atom or memory operation, it inserts an explicit
`Repartition` value. A repartition changes placement, not logical tensor
identity; it is therefore visible in TileIR and costable by the tuner.

An instruction atom uses the same algebra:

~~~text
AtomOperand : ParticipantCoord x OperandSlot -> OperandLogicalCoord
~~~

MMA, async copy, vector ALU, reduction, and accelerator instructions are
target-provided contracts over layouts, types, effects, and capabilities.

### 5.1 Elementwise operators lift directly to tiles

Pure scalar operators are rank-polymorphically lifted to `Tile` values. Common
arithmetic, comparison, selection, casts, and math functions therefore read as
ordinary expressions:

~~~cpp
auto y = gelu(x + bias) + residual;
auto p = exp(scores - row_max);
auto finite = select(mask, p, 0.0f);
~~~

For scalar function `f` and named logical output coordinate `q`:

~~~text
Elementwise_f(v0, ..., vn)[q]
  = f(v0[project_0(q)], ..., vn[project_n(q)])
~~~

The output dimension set is the compatible identity join of the operands. A
missing size-one dimension broadcasts; introducing or resolving any other
ambiguous dimension requires explicit `broadcast`, `reshape`, or `reindex`.
Diagnostic labels never participate in this decision. Physical
distribution is not part of elementwise semantics. If operand distributions
cannot feed one selected atom, scheduling inserts and costs a `Repartition`.

`map(values..., scalar_lambda)` remains the escape hatch for a custom pure
scalar expression, not mandatory ceremony around every elementwise operator.
Both forms become one elementwise op with a scalar SSA region in TileIR, so
fusion sees the complete expression DAG rather than an opaque C++ callback.

### 5.2 MMA is a portable value primitive

Matrix multiply-accumulate is the one tensor-shaped arithmetic operation in
the Candidate core:

~~~cpp
acc = mma(a, b, acc);
~~~

All three operands and the result are ordinary Tile SSA values. Shared logical
dimension identities determine the roles without a positional reduction-axis
argument. If the metavariables `Batch`, `M`, `K`, and `N` denote arbitrary
dimension products and

~~~text
A : Batch x M x K
B : Batch x K x N
C : Batch x M x N
~~~

then `mma(A, B, C)` returns a value in `Batch x M x N`. Batch dimensions occur
in all three operands, contraction dimensions occur in `A` and `B` but not `C`,
and the two free dimension sets are checked against `C`. Unrelated argument
dimensions must first be related with `with_dims` or an explicit `reindex`; the
operation never guesses from labels, extents, strides, or physical layouts.

Its portable reference meaning is:

~~~text
mma(A, B, C)[b, m, n]
  = accumulate_policy(C[b, m, n],
      { convert(A[b, m, k]) * convert(B[b, k, n]) | k in K })
~~~

The operation carries input conversion, accumulator element type, contraction
axes, reassociation permission, and accuracy/rounding policy. Those are
semantic constraints rather than a promise to emit a hardware instruction. A
CPU may lower it to vector FMAs; a GPU may select one or more target MMA atoms;
an unsupported exact policy may use the reference loop. In every case the
Machine realization must refine the Candidate contract.

The spelling deliberately includes `C`. `mma(a, b, acc)` is one fused update,
so `acc += mma(a, b)` would ambiguously add the accumulator twice. Common
matrix multiplication remains library sugar:

~~~cpp
auto c = matmul(a, b); // equivalent to mma(a, b, zeros(result_shape))
~~~

The two-argument library overload uses the conventional
`[..., M, K] x [..., K, N]` interpretation. A general call either supplies the
local contraction `Dim` set or uses `einsum`; this convenience rule is not part
of the core `mma` semantics.

General einsum/contraction remains a library composition over reindex,
elementwise arithmetic, and `reduce`; scheduling may tile a compatible
expansion into `mma`. There is still no GEMM, convolution, or attention opcode.
The semantic `mma` op preserves cross-target arithmetic and layout-legality
information; a target MMA atom specifies concrete participants, operand slots,
instruction shapes, and resource protocol.

### 5.3 Reduction is a structured algebraic region

A reduction uses the same invariant as every other range-for construct: the
loop variable is the current region scope, never a data value or accumulator
proxy:

~~~cpp
auto columns = x.dim<1>();
auto sum = zeros<f32>(x.shape().without(columns));

for (auto &column : nest.reduce(x.domain(columns))) {
    sum += x.at(column);
}

use(sum); // the outer Tile identity now denotes the reduced result
~~~

`column` is a non-copyable `ReduceScope`. `column.index()` is its staged
reduction coordinate and `column.coord()` is the full lexical coordinate.
`x.domain(columns)` is a typed `IndexSet`; `x.at(column)` is a pure Tile
projection implemented by reindexing. Neither operation chooses participants,
memory, or a collective.

The primitive region signature is deliberately source-free and state-free:

~~~text
nest.reduce(domain, contract?, policy?)

domain      typed semantic contribution coordinates
contract    optional identity/merge/type/reassociation contract
policy      optional accuracy/order constraints, never a warp count
~~~

The body is captured once. On region close, the frontend finds every `Tile`
defined outside the region and updated inside it. Those values become the
region's incoming block arguments and outgoing results. Locals remain locals;
stores and atomics remain effects; neither can accidentally become a reduction
state. There is no public `result()`, `yield`, accumulator proxy, or state list.

The update graph supplies information that the old spelling duplicated. A
canonical `state += contribution`, `state = maximum(state, contribution)`, or
corresponding minimum/logical form selects its registered reducer contract.
`state = mma(a, b, state)` is likewise recognized as an additive contribution
whose lift is the policy-governed product reduction; this is what makes a tap
or block domain around MMA a valid algebraic region.
Several independently recognized states form the product reducer, so one region
may compute sum and maximum together. Recognition of floating-point addition
does not by itself permit arbitrary reassociation; the math policy still decides
that. A custom contract can be supplied after the domain:

~~~cpp
auto stats = scalar(WelfordState<f32>::identity());
for (auto &feature :
     nest.reduce(range(0, features), welford)) {
    auto value = X[coord(row, feature.index())];
    stats = welford_push(stats, value);
}
~~~

If an update is not recognized and no compatible contract is present, capture
rejects the region and recommends `serial`; it never silently removes the
algebraic promise. If several custom states are needed, they should be one
explicit product/aggregate state with one typed product contract.

An external view may use the scope as an input to its ordinary view map, which
supports the spelling proposed for tiled streaming reductions:

~~~cpp
auto acc = zeros<f32>(output_shape);
for (auto &elem_nest : nest.reduce(reduction_tiles)) {
    auto x = X.tile(elem_nest, x_tile_map, bounds::zero).load();
    acc += reduce(x, x.dim<1>(), add);
}
~~~

Here `x_tile_map : ParentCoord x ReductionCoord x LocalCoord -> XCoord` is a
normal typed `ViewMap`. `elem_nest` contributes coordinates, not a memory level
or address. Writing the origin explicitly with `elem_nest.index()` is exactly
equivalent.

Common pure cases keep a short expression form:

~~~cpp
auto columns = x.dim<1>();
auto sum = reduce(x, columns, add);
auto max = reduce(x, columns, maximum);
~~~

These are library shorthand for a reduction region with a generated domain,
state, and update. The reducer argument remains here because there is no body
from which to infer it. In the region form, `sum` was merely the user's outer
Tile variable and `add` was the merge-law contract; neither belongs in the
range binding or needs to be repeated when the update graph is canonical.

After frontend variable promotion, the region has ordinary SSA plumbing rather
than a serialized reducer opcode:

~~~text
%sum1 = reduce.region domain(%columns)
          init(%sum0) reducer(@add) {
  ^update(%state, %reduction_coord):
    %elem = tile.project %x at %reduction_coord
    %next = add %state, %elem
    yield %next
}
~~~

The state, coordinate, contribution-producing operations, grouping projection,
and body are independently inspectable and rewritable. A pass can fuse a
producer, split the reduction domain, change its placement, or replace the body
with a target atom without losing the original reducer contract.

Formally the region declares a reduction coordinate space `R`. Each inferred
state has a result coordinate space `G` and a set of semantic contribution
occurrences `Omega`, with `iteration : Omega -> R` and
`group : Omega -> G`. For `x.domain(axes)`, `group` is the familiar projection
that removes those axes. For a general map-reduce body it is inferred from the
shape/index map of the state update. Given reducer monoid
`(S, identity, merge)` and contribution `lift(omega)`:

~~~text
result[g] = merge(incoming[g],
                  merge_all(identity,
                            { lift(omega) | group(omega) = g }))
~~~

For a domain occurrence `omega`, the captured body computes
`update(state, omega)` using arbitrary pure contribution-producing operations.
Parallel reassociation is legal only when the reducer contract proves or
explicitly promises the homomorphism:

~~~text
update(s, omega) = merge(s, lift(omega))
merge(merge(a, b), c) = merge(a, merge(b, c))
merge(identity, a) = a = merge(a, identity)
~~~

Built-in add, maximum, minimum, logical reducers, and deterministic argmax are
recognized update shapes with registered contracts. Welford and other custom
states supply the same typed contract explicitly or through a registered
library update. If the laws are unavailable, `reduce` is ill-formed; the honest
ordered spelling is `serial`. Floating-point reassociation and deterministic
tree shape remain explicit math/policy choices rather than accidental backend
behavior.

Storage replicas are not semantic contributions. A domain generated by the
expression-reduce library contains each logical Tile element once even if its
`LayoutCorr` has several physical occurrences. An explicit region domain
likewise creates one semantic contribution per domain coordinate on purpose.
This distinction prevents a broadcast or replicated fragment from silently
multiplying a sum.

`reduce` introduces a lexical reduction domain `R`, not a resource-owning
spatial prefix. Scheduling factors it with ordinary layout maps:

~~~text
ReductionPlacement : R -> ParticipantFiber x LocalSerialStep
MergePlan          : PartialStateOccurrences -> ResultStateOccurrences
~~~

It may therefore become a lane shuffle tree, a shared-memory tree, a SIMD
horizontal operation, a serial loop, or a spatial/temporal hybrid without
changing the source. The remaining logical axes determine the result shape;
their distribution may stay sharded or acquire explicit replica fibers.

```{figure} ../_static/tile/reduction-model.svg
:alt: A reduction region groups semantic contributions while its schedule independently maps the reduction domain to participants and serial steps.
:width: 100%

The reduction domain and monoid define meaning. Distribution and the target
catalog decide the physical collective.
```

A custom state reduction remains compact. Welford normalization, for example,
uses one state instead of hard-coding a two-pass warp algorithm:

~~~cpp
auto feature = x.dim<1>();
auto stats = full<WelfordState<f32>>(row_shape, welford_identity);
for (auto &item : nest.reduce(x.domain(feature), welford)) {
    stats = welford_push(stats, x.at(item));
}
auto y = (x - stats.mean) * rsqrt(stats.variance() + epsilon);
~~~

An ordinary `tile_kernel` still cannot assume a device-wide barrier between
independent root instances. A whole-tensor loss either chooses a proved
single-dispatch realization (for example a supported cooperative collective or
an explicitly permitted atomic result), writes per-root partials and launches a
second reduction kernel, or lives in a future multi-dispatch `tile_program`.
The frontend never disguises an illegal global synchronization as a Tile op.

### 5.4 Ordering and selection stay logical Tile operations

Full sorting is not forced into the reduction abstraction. The Tile library
defines `sort` as a logical permutation along named axes with a pure total-order
contract:

~~~cpp
auto feature = x.dim<1>();
auto sorted = sort(x,
                   feature,
                   descending,
                   stable_ties(),
                   nan::last);
~~~

It returns values and, when requested, the logical source permutation. Its
reference expansion uses core compare/select, reindex, reduction, and structured
regions; no core SortOp is required. The meaning does not mention a sorting
network, radix digit, lane exchange, shared memory, or merge pass. Distribution
analysis may keep the axis local, repartition it, or reject a requested
single-kernel schedule that cannot communicate across the required participant
scope. A target may replace a proved equivalent expansion with a sort atom.

Fixed-size Top-K *does* have a useful reduction algebra. For a deterministic
total key `(valid, value, original_index)`, define:

~~~text
State_K             = sorted sequence of at most K candidates
merge_K(a, b)       = take_K(sort(a union b))
identity_K          = empty sequence

merge_K(merge_K(a, b), c) = merge_K(a, merge_K(b, c))
~~~

Discarding everything below the current K-th candidate cannot affect any later
Top-K merge, so this is associative under the stated total order. The general
region form is therefore ordinary `reduce` over an indexed Tile:

~~~cpp
auto best = topk_identity<f32, K>(descending,
                                  tie::lowest_index,
                                  nan::last);
for (auto &feature : nest.reduce(
         x.domain(x.dim<1>()),
         topk_merge<K>)) {
    auto item = indexed_value(x.at(feature), feature.index());
    best = topk_insert<K>(best, item);
}
~~~

`topk<K>(x, axis, order...)` is a library shorthand for that region, not a
second execution model or primitive. `K` is normally an ordinary host/JIT
specialization because it changes the result type and Tile shape. A runtime `k`
uses a statically bounded `KMax` result plus a staged valid count; it never
creates a dynamically sized register fragment.

A large full sort is a launch graph: block-local `sort`, followed by
`merge_sorted` passes, or radix histogram + `scan` + `scatter` passes. Those
pieces are ordinary Tile operations nested in `parallel`, `serial`, and
`pipeline`; global pass boundaries remain explicit for the same reason as a
whole-tensor reduction.

## 6. Views, values, and addressable memory

### 6.1 Three surface objects

| Surface category | Meaning |
|---|---|
| `TensorView<T, R>` / subview | Addressable projection of external storage |
| `Tile<T, R>` | Staged value variable, promoted to tile SSA |
| `Memory<T, R>` | Explicit addressable temporary with stable identity |

Ordinary loads and operations return `Tile`. The compiler may realize a tile
in registers, an on-chip allocation, tensor memory, a vector object, or no
materialized object at all.

This virtual SSA path is the default even across pipeline stages. A
`MaterializationPlan` chooses independently for every live edge whether to keep
it virtual, fuse it, recompute it, or create an internal addressable object. For
a materialized edge it also chooses storage layout, resource class, lifetime,
version count, and synchronization. Those are compiler decisions constrained by
target capabilities and tuning, not mandatory source annotations.

`Memory` is an expert-only semantic escape hatch. It is requested only when the
programmer intentionally needs stable address identity: a mailbox, explicit
alias, persistent mutable state, pinned storage layout or swizzle, a manual
producer/consumer protocol, or exact buffer reuse. Ordinary async staging and
cross-stage dataflow are not reasons to declare one. A separate schedule may
bind an explicit `Memory` to a target resource, but the C++ type itself still
does not mean register, shared memory, SRAM, or any vendor address space.

### 6.2 Memory ownership

An abstract memory object `s` declares:

~~~text
owner(s)       = an execution prefix depth
StorageSpace(s)
AddressLayout(s)
~~~

`memory<T>(layout, resource_constraint?)` creates one abstract allocation for
each instance of the nearest enclosing spatial `parallel` prefix. The kernel
root is the fallback owner. Omitting the second argument leaves physical
placement open; common concise constraints are `mem::private_`, `mem::shared`,
`mem::cluster`, `mem::global`, and `mem::tensor`:

~~~cpp
auto As = memory<bf16>(a_layout, mem::shared);
auto Bs = memory<bf16>(b_layout); // resource class inferred
~~~

These are resource-class constraints, not C++ address spaces and not a memory
order. Register-like storage, shared SRAM, tensor memory, caches, and global
memory are generally incomparable resources rather than a lattice. A target
may reject an unsupported class or resolve an alias to its native resource. Its
catalog describes each physical resource by instance topology,
`can_access(exec_scope, resource, operation)`, capacity/alignment, coherence,
supported operations, and synchronization rules. The accessibility relation is
general rather than assumed monotone: specialized engines and operand ports may
break any imagined hierarchy. Special targets may add catalog-backed classes
without extending TileIR.

The lexical declaration still determines the logical owner. A resource
constraint does not change how many abstract Memory objects exist; it restricts
where their instances and allocation slices may be realized. Descendants can
access ancestor-owned memory only when the target's explicit access relation
accepts their bound scope, operation, and resource, and the required
ordering/coherence protocol is satisfied.

Ownership follows the lexical declaration site, so it is visible without a
redundant owner argument. To give a temporary to an outer prefix, declare it in
that outer scope and capture it below. A declaration may still occur lexically
inside a `serial`, `pipeline`, or `reduce` region: those regions do not become
memory levels, so the nearest spatial ancestor remains the owner while the
temporal/algebraic declaration point contributes lifetime information and the
planner derives required versions.

Multiple memories at the same hierarchy level are ordinary siblings:

~~~cpp
auto As = memory<bf16>(a_layout, mem::shared);
auto Bs = memory<bf16>(b_layout, mem::tensor);
auto Scratch = memory<f32>(scratch_layout); // inferred independently
~~~

This is why memory must not be encoded as execution children.

The declaration gives `s` stable logical identity. Each write additionally
defines a hidden `MemoryState<s>` token, and each read consumes the reaching
state. The frontend builds those tokens with MemorySSA after structured capture.
A software pipeline may map simultaneously live states to several physical
versions without changing the identity or alias class of `s`.

### 6.3 The execution-to-memory equation

For a use whose operation has frontier `b`, let:

~~~text
e       in Events(use), subset Pb   participant execution coordinate
u       in LocalAccess(use)         local access coordinate
t                                      logical time/iteration
pi_o^b(e)                             projection from Pb to owner(s)

ExecBinding_b     : Pb -> TargetExecAxes x SerialAxes
ResourceTopology : TargetExecAxes x SerialAxes -> ResourceInstanceCoord
Instance_s        : P_owner(s) -> ResourceInstanceCoord
Base_s            : P_owner(s) -> AllocationBaseCoord
Access_use        : Pb x LocalAccess -> ViewCoord
ViewMap           : ViewCoord -> StorageCoord
Version_s         : (MemoryState<s>, t) -> VersionCoord
Address_s         : VersionCoord x StorageCoord -> ByteOffset
~~~

`Instance_s` exists when execution binding and target topology factor through
the owner prefix:

~~~text
ResourceTopology o ExecBinding_b
    = Instance_s o pi_o^b
~~~

In other words, descendant coordinates may not change the selected instance of
an ancestor-owned resource. The verifier proves this factorization; failure
means the requested ownership/binding pair is illegal rather than ambiguous.
Each active access must additionally satisfy the target's non-algebraic
capability relation:

~~~text
can_access(target_scope(e), resource_class(s), operation(use))
~~~

This relation is deliberately not derived from a supposed ordering between
memory kinds.

The complete physical access is:

~~~text
Physical(use, e, u, t) =
  ( Instance_s(pi_o^b(e)),
    Base_s(pi_o^b(e)) +
      Address_s(Version_s(reaching_state(use), t),
                ViewMap(Access_use(e, u))) )
~~~

This is the precise version of “ancestor execution hierarchy plus local memory
access maps to the layout algebra.” Every arrow is a typed layout map; the
final pair selects a resource instance and a byte inside it. Unlike a general
placement query, this composed access must be proved total and single-valued
for every active event.

```{figure} ../_static/tile/execution-to-memory.svg
:alt: Separate maps select a resource instance and a byte offset for each execution event.
:width: 100%

Spatial ownership selects an allocation instance; access, view, temporal
version, and address maps select a byte inside it.
```

Visibility, coherence, barriers, and legal cross-instance access are not
spatial layouts. They are target capabilities and effect-verification rules.
Keeping them separate prevents a clever layout from pretending an illegal
memory access is legal.

## 7. Pipeline is a temporal producer/consumer nest

Pipeline belongs to Execution because it organizes when one logical spatial
sub-hierarchy runs. It is not another memory hierarchy. Its natural parent is
therefore visible in the same syntax as any other nest:

~~~cpp
for (auto &nest : parallel(grid_shape)) {
    for (auto &subnest : nest.parallel(subnest_shape)) {
        for (auto &k : subnest.pipeline(iteration_space, policy)) {
            k.stage("produce");
            // Producer operations for k.index().

            k.stage("consume");
            // Consumer operations for k.index().
        }
    }
}
~~~

There is one logical pipeline instance per `subnest` coordinate. Moving the
pipeline outside that `parallel` changes the semantics to one pipeline shared
by the parent nest; it is not merely formatting. The pipeline loop adds a
temporal coordinate but does not deepen the spatial execution frontier.

Like `parallel` and `serial`, the C++ range executes its body exactly once
during capture. Dereference creates `PipelineOp`, pushes its body, and returns a
non-copyable iteration handle; `k.index()` is its staged coordinate. Increment
closes the region after discovering carried values. A multidimensional domain
returns a tuple-like `index()`. Generated code executes all logical iterations
through a prologue, steady state, and epilogue chosen by scheduling.

### 7.1 Stage boundaries are lexical

A pipeline is a repeated producer/consumer graph, not a loop with only an `II`
annotation:

~~~text
Pipeline = (IterationSet I,
            StageSet S,
            StageOrder <S,
            Dependences D,
            Policy)

Dependence edge = (producer_stage, consumer_stage,
                   iteration_distance, value_or_effect)
~~~

`k.stage()` is a frontend cursor on the pipeline iteration handle, not an
executable marker operation. The first call begins stage zero and each
subsequent call ends the current source segment and begins the next.
`k.stage("load")` adds an optional compile-time name to the new segment. Capture
turns the segments into ordered child regions of the single pipeline operation;
internally each has identity `(PipelineId, ordinal)`. The name is a stable
diagnostic and scheduling label, not global identity. Consequently unrelated
pipelines cannot accidentally interleave their stage namespaces. A later fusion
pass may combine pipelines only by constructing a new graph and proving
dependence and resource equivalence.

Stage ordinals are not iteration coordinates, cycle numbers, memory versions,
execution levels, or hardware warp IDs. Source order supplies the default
same-iteration producer-before-consumer order; SSA, MemorySSA, and effect
analysis record the precise edges, including edges that skip phases and
loop-carried edges with positive iteration distance. Ordering a pair of stages
does not itself invent a whole-hierarchy barrier.

The compiler may infer all stage membership. `k.stage(optional_name)` is only
the explicit surface for pinning a cut when the programmer wants that
producer/consumer structure to be part of Candidate TileIR:

~~~cpp
for (auto &k : subnest.pipeline(k_tiles, policy)) {
    auto k0 = k.index();

    k.stage("load");
    auto a = A.tile(a_origin(k0), a_shape, bounds::zero).load();
    auto b = B.tile(b_origin(k0), b_shape, bounds::zero).load();

    k.stage("compute");
    acc = mma(a, b, acc);
}
~~~

`a` and `b` are virtual Tile SSA, not source-declared buffers. A GPU schedule
may map the stage-active operations to disjoint participant subsets and replace
the matched library expansion with copy/MMA atoms; a CPU schedule may use tasks
or serialize it. If an engine handoff requires addressable storage, the planner
inserts internal materialization, MemorySSA, versions, events, and barriers. No
target-specific role or mandatory staging buffer leaks into the kernel.

When `stage()` is absent, all stage membership is open for inference. Once a
pipeline contains an explicit cut, every effectful or tile operation in its
immediate body must follow the first cut. Cursor calls themselves must occur
unconditionally at that body level, so Candidate TileIR has a static stage
graph; stage contents may contain ordinary structured control flow and child
execution regions. Pure iteration-index expressions may precede the first cut
and be used by several stages.

Because a cursor call does not open a C++ block, `a` and `b` remain naturally
visible after the cut. Their cross-stage SSA edges are explicit in TileIR. This
is why a cursor cut is preferable to a stage-specific C++ brace scope.

```{figure} ../_static/tile/pipeline-stage-flow.svg
:alt: C++ stage cursor cuts become pipeline subregions and a dependence graph whose participant and engine bindings are chosen later.
:width: 100%

The source fixes producer/consumer cuts while leaving communication,
participant subsets, engines, and physical realization open.
```

The pipeline iteration coordinate and stage identity remain orthogonal:
`k.index() == 0` means the first logical iteration. It is useful for a true
prologue special case, but it does not select an execution role.

### 7.2 Scheduling and versioning

For a dependence `e = (sp, sc, distance, payload)`, a legal schedule satisfies:

~~~text
II >= 1
Issue(stage s, iteration i) = i * II + theta(s)
Issue(op in s, i) = Issue(s, i) + delta(op)

Issue(sc, i + distance) >= Issue(sp, i) + latency(e)

Schedule(op, i)
  = (Issue(op, i), anchor(op), frontier(op), Active(op), engine(op))

Version(materialized edge or MemoryState, i) -> VersionCoord
~~~

`theta(s)` places a logical stage in the modulo schedule and `delta(op)` orders
operations within it. `max_in_flight` bounds the scheduling window; it is not
the number of logical stage segments and is not blindly copied to every
buffer depth. `initiation_interval` belongs in `pipeline_policy` because it is
a primary temporal constraint.

The compiler derives:

- prologue, steady state, and epilogue;
- async-copy and compute issue points;
- per-edge live version count;
- storage ring indices for materialized edges;
- barriers, events, waits, and fence scopes;
- resource pressure and legality.

Only a materialized edge's `VersionCoord` enters its `AddressMap`. Pure SSA
edges do not acquire fictitious memory versions.

```{figure} ../_static/tile/pipeline-timeline.svg
:alt: Three pipeline iterations overlap across load, compute, and store engines while their memory versions remain live.
:width: 100%

The scheduler overlaps stage instances subject to dependence latency; storage
version count follows liveness rather than source stage count.
```

## 8. Direct assignment and hidden SSA plumbing

### 8.1 Surface rule

Tile variables use Luisa-style staged assignment:

~~~cpp
for (auto &nest : parallel(grid)) {
    auto acc = zeros<f32>(shape(BM, BN));

    for (auto &subnest : nest.parallel(subnest_shape)) {
        for (auto &k : subnest.pipeline(range(0, K, BK), policy)) {
            auto k0 = k.index();
            auto a = load_a(k0);
            auto b = load_b(k0);

            acc = mma(a, b, acc);
        }
    }

    use(acc);
}
~~~

There is no public yield object and no public loop-result accessor.

To make this well-defined C++, a named `Tile` object owns one frontend
variable identity:

- initialization creates that identity with an initial definition;
- move construction transfers it, so return-value optimization does not create
  accidental variables;
- copy construction creates a new snapshot variable, never an alias;
- `operator=` writes a new definition to the existing identity;
- compound assignment is exactly a read, lifted scalar/Tile operation, and
  write to that identity; reduction capture later checks the resulting update
  graph against its inferred or explicit reducer contract;
- operation parameters consume expression views or references and therefore do
  not trigger copy construction.

This gives `auto next = acc;` value-snapshot semantics and `acc = next;`
mutation semantics. Explicit aliasing is not part of the `Tile` surface.
Every definition of one variable must have the same element type, logical
shape, and inferred initialization anchor. An assignment may occur in a descendant nest,
but it does not silently change that anchor. Its execution frontier and
distribution may change; canonicalization inserts an explicit repartition when
the change is not free.

Execution/region scopes and data `Tile<T, R>` are different types. The canonical
examples name handles `nest`, `subnest`, `leaf`, `k`, `feature`, or `tap`, reserve
`tile` for data, and write every range variable as `auto &` so the distinction
stays visible. In particular, dereferencing a reduce range returns
`ReduceScope &`, never a state/value tuple.

### 8.2 Capture algorithm

Every surface `Tile` declaration owns a staged variable identity. Reading it
records the current definition; assignment records a new definition. When a
structured region closes, its builder compares the incoming and outgoing
definition environments.

For a pipeline or loop it computes two related sets:

~~~text
loop_carried  = read_before_definition_inside
              intersect written_inside

region_result = written_inside
              intersect live_after
~~~

For an execution nest it also computes:

~~~text
ancestor_update = written_inside
                intersect declared_in_ancestor_scope

Assembly : ChildPrefix x ChildFragment -> AncestorLogicalCoord
~~~

For a reduction region it computes:

~~~text
reduction_state = written_inside
                intersect declared_outside_region

next_s = Update_s(incoming_s, reduction_coord, captured_values)
~~~

Every `reduction_state` must read its incoming definition on every contributing
path and match a known update/merge homomorphism or an explicit contract.
Several valid states are combined as one product reducer. A write-only outer
Tile is not a reduction; a disjoint per-coordinate assembly belongs in
`parallel`. An unproved recurrence belongs in `serial`.

An `ancestor_update` is legal only when `Assembly` proves an exact disjoint
cover, proves that replicas agree, or names an explicit associative combiner.
Memory effects instead use MemorySSA, alias, and synchronization rules. This
turns a potentially racy-looking C++ assignment into a checked collective IR
operation rather than assuming last-writer-wins behavior.

These are control-flow data-flow sets, not textual scans:
`read_before_definition_inside` means a read not dominated by an in-region
definition on every reaching path. Nested conditionals and early exits are
therefore handled by the same definite-assignment analysis.

A recurrence may be loop-carried even when its final value is dead after the
loop; only `region_result` becomes externally visible SSA. Values read but not
written are captures, and values written without a prior read need no initial
operand unless control-flow merging requires one.

The frontend temporarily permits variable reads/writes. The first mandatory
canonicalization pass promotes them to region arguments and SSA results.

The pipeline and nested collective update above become conceptually:

~~~text
%acc1 = exec.parallel %subnest_shape init(%acc0) {
  ^subnest(%subnest_coord, %acc_fragment):
    %next = exec.pipeline range(...) init(%acc_fragment) {
      ^body(%k0, %acc_in):
        %a = load ...
        %b = load ...
        %updated = tile.mma %a, %b, %acc_in
        yield %updated
    }
    yield %next
} assemble(exact_cover)
~~~

After the region closes, the surface handle `acc` denotes `%acc1`.

Conditional assignment constructs merge values. A verifier rejects a value
that is not definitely assigned on every required path, or carries the old
definition when that is the declared semantics.

### 8.3 Addressable storage uses explicit effects

Assignment is reserved for staged Tile definitions. An explicit `Memory`
exposes only effectful `store` and `load` operations:

~~~cpp
acc = f(acc);                    // Tile SSA definition
As.store(input_view.load());     // View -> Tile -> Memory
As.store(value);                 // Tile -> Memory
auto staged = As.load();         // Memory -> Tile
~~~

The receiver makes the addressed object unambiguous, and every addressable
read or write is visible at the call site. `Memory = Tile` and `Memory = View`
are both ill-formed; there is no assignment shortcut and no implicit load from
a view. A `TensorView` or subview must first produce a Tile with `.load()`, and
`Memory::store` accepts only such a Tile value.

Each `Memory::store` defines a new hidden `MemoryState` token; `Memory::load`
consumes the reaching state and returns Tile SSA. The explicit View load and
Memory store are separate Candidate effects so alias and bounds analyses remain
honest. A schedule may nevertheless prove them equivalent to one direct or
asynchronous transfer and replace the pair with an atom. Thus source semantics
stay simple without preventing zero-intermediate-buffer lowering.

## 9. C++ staging and JIT

### 9.1 One scoped builder, no builder prefixes

`tile_kernel(lambda)` retains a typed C++ definition. The JIT/capture entry,
not the wrapper's construction, installs a thread-local scoped builder,
creates the lambda parameters in ABI order, invokes the lambda once for the
current concrete specialization, verifies stack discipline, and restores the
previous builder. Free operations find the builder from their operands or the
current scope. Constructing the wrapper emits no IR and reads no tensor data.

~~~text
typed lambda + ordinary C++ configuration captures
                     |
concrete argument metadata + target
                     |
           create signature parameters
                     |
            execute lambda on the host
                     |
          one concrete candidate TileIR
                     |
                 lower / JIT
~~~

The scoped builder is construction machinery, not the IR. TileIR owns all
regions and values after construction.

### 9.2 Ordinary configuration creates variants

Configuration is ordinary host C++:

~~~cpp
for (auto cfg : candidates) {
    auto executable = device.jit(make_gemm(cfg), A, B, C);
    benchmark(executable, A, B, C);
}
~~~

The runtime arguments provide resource metadata, not tensor contents to be
captured as constants. Each uncached candidate executes the C++ definition and
JITs its resulting TileIR; users need not introduce a staged template type or
manually invoke a specialization API. The cache key contains:

- target and feature set;
- canonical candidate TileIR hash, including the effects of host configuration;
- explicit argument specialization guards;
- compiler and ABI revision.

The object representation of an arbitrary C++ lambda is never a cache key:
padding, pointer captures, and object identity are not stable program semantics.
The baseline can simply recapture each candidate and reuse code generation
after hashing its canonical IR. Avoiding that frontend work is an optional
optimization requiring an explicit stable definition/configuration identity.

For this concrete-specialization path, `TensorView::extent` returns ordinary
host integer metadata during capture. Shape/stride changes select a different
guarded variant; input data values do not. There is no `SymInt` type or
requirement to first capture a universal symbolic kernel. A future explicitly
runtime-varying extent path must use ordinary scalar parameters and preserve
the same signature/ABI separation.

A future symbolic family representation may avoid repeated frontend work for
very large searches, but it must lower to the same concrete candidate IR. It
is an optimization rather than a semantic prerequisite.

### 9.3 Target schedules

The portable kernel may leave execution binding, value distributions, atom
selection, and realization variables open. A target schedule or solver fills
them under explicit guards.

Expert constraints may pin only what is necessary:

~~~text
bind logical execution prefix to target axes
require a particular atom family
place a Memory object in a compatible resource class
require alignment, vector width, swizzle, or pipeline engine
~~~

Constraints are IR objects with source locations, not arbitrary callbacks
hidden from cache keys.

## 10. TileIR as a thin but transformable IR

### 10.1 In-memory structure

TileIR borrows the useful structural properties of LLVM IR and Luisa XIR,
without importing MLIR:

- `Module`, `Function`, `Region`, `Block`, `Operation`;
- explicit `ExecStructure`, stable `ExecLevel` identities, and prefix cuts;
- typed SSA `Value` and explicit `Use` lists;
- operations own zero or more regions;
- interned immutable `Type`, `Dim`, `Space`, `IndexSet`, `LayoutMap`,
  `LayoutCorr`, and attributes;
- typed `ExecRemap` objects carrying old/new structures, active domains,
  prefix factorizations, and transform provenance;
- stable source locations and diagnostic provenance;
- explicit successors and block arguments;
- verified parent/ownership relationships;
- an `IRRewriter` for insertion, replacement, region movement, erasure, and
  transactional execution-structure remapping;
- an `AnalysisManager` with dependency declarations and invalidation;
- interfaces for effects, layout inference, tiling, atoms, and target export.

Serialization is one consumer of this model. It is not the in-memory design
and does not define transformability.

#### Ownership and mutation contract

The mutable spine follows Luisa XIR's managed intrusive-list model, adapted for
multi-result operations:

~~~text
Module
└─ FunctionList                         managed intrusive, ordered
   └─ Function
      └─ Region                        single owner
         └─ BlockList                  managed intrusive, ordered
            └─ Block
               └─ OperationList        managed intrusive, ordered
                  └─ Operation
                     ├─ operand slots  vector<ManagedPtr<Use>>, ordered
                     │                    │
                     │                    └── intrusive link ──> Value::UseList
                     ├─ result Values  single owner, stable address
                     └─ child Regions  single owner
~~~

`Function`, `Block`, and `Operation` need stable identity plus constant-time
detach, insertion, and movement, so their ordered parent sequences are doubly
linked managed intrusive lists. `Use` needs stable identity and constant-time
unlink but no semantic order in the defining value's user set, so it is a
managed intrusive *forward* node. The ordered operand slots retain a managed
reference to each `Use`; the defining `Value::UseList` retains another reference
only while the user operation is linked into TileIR.

This linkedness rule is part of IR semantics:

- `Operation::remove_self()` first detaches all operand uses and returns a
  managed ownership handle;
- insertion restores the parent pointer and reattaches those same `Use`
  identities;
- `set_operand` moves one `Use` between value lists in constant time;
- replace-all-uses walks only the old value's use list, never the module;
- erasure is legal only when every result has no linked uses, then automatically
  removes all incoming use-def edges;
- a detached operation is outside the IR and therefore does not participate in
  liveness, use counts, verification, or analyses until reinserted.

The verifier checks both relations for every operand: its logical
`Use::value()` and the physical identity of the `Value::UseList` that owns its
intrusive link. It also checks ordered parent membership, parent pointers,
result definitions, unique IDs, and lexical dominance. `IRRewriter` wraps these
mutations and invalidates cached analyses.

Not every object is intrusive. `Region` and result `Value` have exactly one
structural owner and never need sibling splicing; immutable types, dimensions,
layouts, and attributes are shared/interned values. Keeping those simple avoids
turning TileIR into a general object graph while preserving the operations that
real transformation passes need.

### 10.2 Minimal operations

The Candidate semantic core needs only the following operation families:

| Family | Only primitive form |
|---|---|
| Control | function, call, return, block argument, branch, conditional, region yield |
| Structured execution | `parallel`; `serial` as the canonical counted loop; `pipeline` with ordered stage subregions and dependence edges |
| Algebraic | `reduce` with logical/index domain, state update, reducer contract, and grouping map |
| Pure values | constant, tuple/aggregate, generic scalar SSA region lifted over dimension identities, reindex, semantic `mma(a, b, c)` |
| Addressable effects | `view(base, domain, index_map, validity)`, explicit `memory`, load, store, atomic, abstract sync |

This is an operation inventory, not a list of every IR class. Dimensions, layouts,
anchor/frontier constraints, bindings, reducer laws, source locations, and
remap proofs are immutable attributes or analysis/proof objects. Frontend Tile
variables are capture bookkeeping and are promoted immediately to block
arguments and SSA values. `k.stage()` creates pipeline subregion boundaries;
there is no StageMarkerOp. `subview`, reshape, transpose, broadcast, slicing,
padding, and swizzle are constructors for the one view/index-map form, not
opcodes. `Repartition` is an explicit Scheduled TileIR realization record and
cost boundary, not Candidate value semantics.

Machine TileIR adds only one parameterized operation form:

~~~text
atom.call(catalog_id, operands, layout/effect/protocol attributes)
    -> values and protocol tokens
~~~

Target MMA instructions, asynchronous transfer, shuffle collective,
barrier/event, sort network, and tensor-memory instructions are catalog entries
of that form. The Candidate `mma` operation is not itself one of these catalog
entries: atom selection replaces it with a target-specific realization only
after proving type, arithmetic-policy, participant, and operand-layout
compatibility. Protocol tokens are ordinary SSA results; resource binding and
capability guards are attributes and verified constraints. A new target
instruction therefore does not require a new C++ construct, TileIR opcode,
visitor method, or serializer enum.

The Candidate core deliberately has no dedicated GEMM, convolution, softmax,
normalization, Top-K, sort, scan, gather, scatter, or copy opcode. Their library
definitions compose the rows above: `matmul` is zero initialization plus
`mma`; general einsum is reindex + elementwise multiply + reduce; gather is a
load whose index map is a value; scatter is an indexed store or atomic; and
copy is a load/store dataflow edge. Scheduled or Machine TileIR may replace a
proved subgraph with a registered target atom referenced by a stable ID and
verified attributes.

### 10.3 Forms and invariants

One IR data structure has progressively stronger verified forms:

1. **Candidate TileIR**: logical hierarchy and semantic operations; some
   anchors, frontiers, layouts, bindings, distributions, and realizations may
   be variables.
2. **Scheduled TileIR**: execution binding, pipeline schedule, distributions,
   memory plans, transformed execution structure, and guards are concrete.
3. **Machine TileIR**: atom calls, explicit realized transfers/synchronization,
   and addresses are legal for one target.

Forms are verifier states, not three unrelated object models.

### 10.4 Essential analyses

- dominance and post-dominance;
- liveness across regions and pipeline iterations;
- memory effects and alias sets;
- layout-map and correspondence equivalence, image/preimage, and proof obligations;
- distribution compatibility and repartition cost;
- reduction grouping, reducer-law, accuracy/order, and placement legality;
- ordering-contract, logical permutation, Top-K state, and scan legality;
- execution-prefix ownership and visibility;
- execution-remap equivalence and prefix-factorization legality;
- resource pressure and occupancy;
- dependence distance and modulo scheduling;
- guard implication and variant coverage.

Analyses live in side tables keyed by stable IR identities. Transform passes do
not stuff transient conclusions into serialization fields.

## 11. Compiler bridges and native backends

### 11.1 Boundary

TileIR is the source of truth. TVM is an optional lowering bridge used to
bootstrap scheduling, legalization, and code generation. TVM types never
appear in the C++ DSL API, TileIR core headers, cache ABI, or runtime ABI.

Interoperability code lives below `luisa/tile/bridge/`: `bridge/tirx` owns the
TVM dependency and a future `bridge/xir` owns TileIR-to-Luisa-XIR translation.
The TIRx bridge constructs `tvm::tirx::PrimFunc`, layouts, expressions, and
statements directly through TVM's public C++ API. TVMScript text is useful for
debug printing and differential tests, but source generation or Python parsing
is not a compiler boundary.

Native hardware lowering is not another bridge. A backend such as Metal or
CUDA implements a common Tile compiler `DeviceExtension` alongside its other
backend services. The portable runtime asks the active device for that
extension; unsupported devices may instead select an installed compiler bridge.
This keeps external compiler interop, native backend code generation, and
hardware target identity as three distinct concepts.

### 11.2 Layout bridge

TVM TIRx models a layout as named physical-axis coordinates with shard,
replica, and offset components; see the official
[TIRx layout model](https://tvm.apache.org/docs/tirx/layout.html). The TileIR
embedding preserves that exact set-valued direction. For logical space `X`,
replica space `R`, and named physical space `P`:

~~~text
F = X x R
left(x, r)  = x
right(x, r) = D(x) + replica(r) + O

TIRx Layout X -> Set(P)
  = LayoutCorr<X, P; F>(left, right)
~~~

TIRx logical axes become `X`, its replica iters become `R`, named hardware or
memory axes become `P`, its shard body becomes `D`, and its offset becomes
`Translate`. This is an exact embedding, including replication.

A Triton-style distribution arrives in the other direction as a map
`PhysicalSlot -> Logical`. TileIR views it as a correspondence and swaps the
two legs when an exporter needs logical-to-physical placement. No unique
inverse is assumed. Export to a TIRx `TileLayout` is structural when the
reoriented correspondence factors as shard plus replica plus offset. Registered
swizzles export as `ComposeLayout` when supported. Other layouts lower to
explicit TIRx index computation or a legal materialization; they are never
silently approximated.

### 11.3 Execution bridge

The open logical hierarchy remains in TileIR until target binding. Current
[TIRx execution scopes](https://tvm.apache.org/docs/tirx/api/execution.html)
name a fixed set of GPU-like scopes, so they cannot be the canonical model for
an open hierarchy.

After `ExecBinding` is concrete, the exporter maps target axes to TIRx scopes
or TIRx loop/thread-binding constructs. Serial and vector axes remain explicit.

There is one shared structural exporter, not a complete lowerer per backend.
The changing component is the execution schedule: a CPU plan may map a logical
parallel prefix to a task loop and SIMD suffix, while a GPU plan may map the
same prefix through an affine split to grid, threadgroup, subgroup, and worker
coordinates. Target-specific resource selection and intrinsic dispatch happen
after this binding. Schematically:

~~~text
Candidate TileIR (logical execution tree)
                 |
          target/autotuned ExecBinding
                 v
Scheduled TileIR (physical scopes + index maps)
                 |
       shared structural TIRx exporter
                 v
       target-specific TVM code generation
~~~

The current scalar bootstrap implements only the default root case. It leaves
logical `parallel` as marked serial TIRx during structural export, then maps the
outermost marked region to LLVM `kParallel` or to a Metal/CUDA-style
`blockIdx.x * threads + threadIdx.x` grid with a tail predicate. Unbound nested
parallel regions remain serial until a real per-nest `ExecBinding` plan is
available. This fallback is deliberately internal rather than a public
`CPU_THREADS`/`GPU_GRID` compile option.

### 11.4 Pipeline and memory bridge

Scheduled TileIR exports:

| TileIR | TVM destination |
|---|---|
| memory declaration | `alloc_buffer` or target memory object |
| logical/physical correspondence | TIRx `TileLayout` when factorizable |
| view/address map | buffer layout or explicit TIRx index map |
| execution binding | TIRx scope IDs or TIRx thread/loop bindings |
| pipeline schedule | software-pipeline annotations or explicit staged loops |
| async copy/token | target-supported async operation and dependence |
| semantic tile op | TIRx tile op when available, otherwise decomposed scalar/vector TIRx |

The initial bridge is one-way. Round-trip import is not required and must not
weaken TileIR invariants.

The bridge's native compiler driver mirrors the TIRx pass pipeline in C++ and
dispatches `target.build.<kind>` through TVM's C++ registry. It partitions host
and device `PrimFunc`s by their bound target, finalizes each partition, and
imports generated device modules into the host runtime module. A scalar
`PrimFunc` compile-and-execute test is the minimum ABI smoke test; Python is not
loaded even for pass orchestration.

The bootstrap implementation already lowers static-JIT-specialized view
arguments, scalar constants and typed elementwise opcodes, view loads/stores,
and `parallel`/`serial`/`pipeline`/`reduce` regions with inferred scalar carried
state. Unsupported Tile, MMA, and explicit-memory forms fail closed rather than
falling through a name-based dispatch. The compiler distinguishes ordinary
TIRx statements (`STANDARD`) from programs containing native TIRx
`TilePrimitive` calls (`TILE`), because only the latter require `LowerTIRx`.
Both paths run `LowerTIRxOpaque` before host/device splitting so thread-binding
loops become device regions. Buffer `noalias` is an explicit caller contract
and defaults off until TileIR carries enough alias metadata to prove it.

### 11.5 Bootstrap lowering path

~~~text
C++ capture
  -> Candidate TileIR
  -> value-to-SSA and structural verification
  -> layout/distribution inference
  -> execution binding and guarded variant selection
  -> pipeline/resource planning
  -> Scheduled TileIR
  -> native C++ TVM TIRx
  -> existing TVM target lowering
~~~

Later native passes can replace any segment:

~~~text
Scheduled TileIR
  -> target atom legalization
  -> Machine TileIR
  -> Luisa XIR / LLVM / native backend IR
~~~

The bridge is therefore scaffolding, not an architectural dependency trap.

## 12. Target catalog

A target plugin provides data, not frontend syntax:

- target execution axes and legal bindings;
- the target-scope containment poset and parent projections;
- legal region anchors, execution frontiers, and convergence rules;
- resource kinds, instance topology, capacity, bank geometry, visibility, and
  coherence;
- copy, MMA, vector, reduction, and synchronization atoms;
- atom operand/result layouts and type constraints;
- engines, events, barriers, and pipeline capabilities;
- cost hooks and legality predicates.

The same logical hierarchy can target GPU blocks and warps, CPU thread teams
and vectors, an accelerator core hierarchy, or a simulator. The verifier
rejects a schedule whose resource-instance or visibility maps are illegal.

## 13. Required verifier invariants

Before backend export, at minimum verify:

1. Every layout composition has matching domain/codomain spaces.
2. Every `ParallelMap` or temporal child map is total on its active domain,
   preserves the required parent projection, and has no escaped `ExecScope`
   handle.
3. Every pair of nested execution bindings respects the target-scope
   containment poset and commutes with logical and target ancestor projection;
   same-scope factorization is capacity- and convergence-safe.
4. Every semantic operation belongs to an execution region and references
   existing, ordered prefix cuts.
5. Every accepted execution remap preserves its active logical domain and
   factors through all affected anchor, frontier, ownership, and convergence
   cuts, or carries an explicit equivalent rewrite.
6. Every operation has `anchor <= frontier`, uses only legal ancestor data,
   and satisfies the selected atom's participant contract.
7. Every distributed value covers the required logical domain exactly, unless
   replication or masking is explicit.
8. Every ancestor value updated inside a child nest has a proved exact
   assembly, agreeing replication, or explicit combiner.
9. Every reduction has a total grouping map, a type-correct identity/update/
   merge contract, counts semantic contributions rather than storage replicas,
   and uses only reassociations allowed by its policy.
10. Every sort/selection comparator supplies a deterministic total-order policy
   for all represented values, including ties, invalid padding, and NaNs; every
   emitted permutation is total and single-valued on its active domain.
11. Every Memory resource constraint resolves to a target class whose instance
   topology and explicit access relation are compatible with its lexical owner,
   all bound accessing execution scopes, and the performed operations.
12. Every memory access resolves to one legal resource instance, allocation
   slice, and in-range address under its predicate.
13. Descendants access ancestor-owned memory only through valid visibility and
   synchronization rules.
14. Explicit pipeline cursor calls form unconditional top-level segment cuts
   with unique optional names; dependences respect issue time, iteration
   distance, and synchronization scope.
15. Ring-buffer versions cannot alias while simultaneously live.
16. Atom operand layouts match the selected atom contract or an explicit
   repartition is present.
17. Specialization guards imply all static shape, alignment, and capability
   assumptions.
18. Every candidate family has a legal fallback or explicitly rejects the
   unsupported input region.

## 14. Minimal implementation plan

### Phase A: algebra and IR skeleton

- Dim, Space, ExecNest, ParallelMap/TemporalMap, and interned LayoutMap DAG;
- stable ExecLevel/prefix identities and typed ExecRemap transactions;
- IndexSet, LayoutCorr, mixed-radix, composition, product, projection,
  permutation, translation;
- Region/Block/Operation/Value/Use and rewriter;
- parser/printer only for debugging and tests;
- tri-state proof API, layout normalization, algebra-law property tests, and
  exhaustive finite verifier.

### Phase B: elegant C++ capture

- scoped builder stack;
- `tile_kernel`, `TensorView`, `Tile`, `Memory`, and MemorySSA state capture;
- one-pass range-for `parallel` / `ExecScope::parallel` / `serial` / `pipeline`
  / `reduce`, stage cursor cuts lowered to pipeline subregions,
  inferred outer reduction states/contracts, view/load/store, lifted
  elementwise ops, semantic `mma`, and expression-reduce shorthand;
- a header-only reference Tile library for matmul/einsum, convolution, softmax,
  normalization, Top-K, sort, scan, gather/scatter, and copy;
- direct assignment capture with specified copy/move semantics;
- structured control flow and immediate value-to-SSA promotion.

### Phase C: scheduling core

- distribution variables as LayoutMaps;
- reduction-domain factorization, reducer-contract verification, and collective
  realization selection;
- prefix-preserving split/fuse/permute/reshape and execution binding;
- explicit repartition;
- pipeline dependence graph, version analysis, and memory planning;
- target catalog interfaces.

### Phase D: TVM bootstrap

- Scheduled TileIR to native TIRx layout and execution export;
- GEMM, convolution, softmax, attention, loss reduction, Top-K, sort,
  elementwise, stencil, and copy coverage;
- differential layout/address tests against the TileIR interpreter;
- JIT cache and straightforward multi-variant autotuning.

### Phase E: native optimization

- target atom selection and precise cost models;
- native pipeline/resource passes;
- direct lowering to Luisa/XIR/LLVM/native paths where it pays off;
- persistent, sparse, and architecture-specific expert features.

## 15. Final decisions

- The language is execution-structure first: an open logical `ExecNest` and
  anchored regions are the semantic skeleton, not a loop nest invented by a
  late schedule or backend.
- The C++ hierarchy is written as nested one-pass range-for scopes, so
  parentage, lifetime, and local-to-prefix coordinate derivation are visible in
  source.
- `parallel`, `serial`, `pipeline`, and algebraic `reduce` are the complete core
  structured-region kinds. Only `parallel` extends the spatial owner hierarchy;
  a reduction domain may be mapped across space and time by scheduling.
- Every range-for binding is a scope handle. `reduce(domain, contract?)` infers
  its outer Tile states and built-in merge contracts from direct updates;
  custom algebraic states provide only the otherwise-unprovable contract.
- Public convenience does not imply a core IR entity. Neural-network,
  collective, ordering, and copy APIs are reference libraries over the minimal
  core; hardware acceleration normally adds a proved target atom, not syntax.
  `mma(a, b, c)` is the one admitted tensor arithmetic primitive because
  decomposition would discard its fused accumulation, precision, and operand-
  layout legality contract; concrete MMA instructions remain target atoms.
- Halide's separation of computation and storage placement is retained, but
  both are expressed against that pre-existing execution structure.
- Execution transforms are typed layout remaps whose prefix-cut preservation
  is proved before dependent operations or memories are rewritten.
- Execution binding is a layout map; hardware names are late target data.
- A `parallel` region may carry a concise `exec::block/warp/thread/...`
  constraint. Nested bindings are verified against the target's containment
  poset and ancestor projections, never enum ordinal values.
- A value's declaration scope constrains its logical anchor; the innermost
  `parallel` supplies the default spatial frontier, and ancestor updates require
  a proved assembly or explicit combiner.
- Distribution is a typed layout map/correspondence, not a separate algebra.
- Scalar pure operators lift directly to logical Tiles; `map` is only the custom
  scalar-region escape hatch, and physical repartition remains explicit in IR.
- Logical dimensions are fresh function-local `Dim` identities. Labels are
  diagnostics only; there is no predefined neural-network axis vocabulary.
- A reducer contract, semantic contribution domain, and grouping projection
  define reduction meaning. Physical replicas never count twice, and common
  expression reductions are shorthand for the same nest-like region.
- Fixed-size Top-K uses a merge-and-truncate reducer under an explicit total
  order. Full sort remains a logical Tile permutation and decomposes into
  visible multi-pass structure when it cannot be realized in one target scope.
- The layout core is CuTe-derived mixed-radix algebra with composition closure,
  typed dimensions, explicit correspondence fibers, F2-linear import, pure index
  expressions, and a finite fallback.
- Memory is an explicit sibling resource owned by an execution prefix, never a
  child execution level. It is an expert escape hatch for stable address
  identity; ordinary Tile materialization is compiler-planned.
- Explicit Memory uses `memory.store(tile)` and `memory.load()` effects.
  Assignment remains exclusively Tile SSA syntax; `Memory = Tile`,
  `Memory = View`, and implicit view loads are ill-formed.
- Optional `mem::shared/private_/tensor/...` tags constrain resource class but
  do not form a memory hierarchy; target legality is a general
  execution-scope/resource/operation capability relation.
- Ancestor execution plus participant-local access reaches physical memory
  through the single typed composition in Section 6.3.
- Pipeline is a temporal producer/consumer nest. `k.stage(optional_name)` marks
  source cuts inside its local stage namespace; memory versioning is derived
  only for materialized edges.
- Direct assignment is the C++ surface; region results exist only inside IR.
- Ordinary repeated JIT of ordinary host configurations is the baseline
  autotuning model.
- TileIR is thin but fully transformable, with SSA use-def, owned regions,
  rewriters, analyses, and verification.
- TVM is a replaceable lowering backend, not the semantic owner.
- MLIR is not required.

The accompanying [GEMM sketch](tile_programming_poc.cpp) and
[kernel gallery](tile_programming_poc_kernels.md) exercise the proposed syntax.
