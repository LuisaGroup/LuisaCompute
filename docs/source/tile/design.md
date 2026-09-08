# Tile language and layout

This is the overview of the execution-first language contract. Detailed topics
have their own references: [execution](execution.md), [layouts](layouts.md),
[operations and reduction](values.md), [memory](memory.md),
[pipelines](pipeline.md), and [staging/JIT](staging.md).

The executable core and proposed extensions are distinguished below. A syntax
sketch in a formal reference is not necessarily an implemented C++ overload;
the [kernel examples](kernels.md) are included from the actual test definitions.

For a first kernel, start with the [programming guide](index.md) and
[executable examples](kernels.md). These references own the language
contract; [lowering coverage](../performance/tile/implementation.md) and
[compiler internals](../internals/tile/index.md) separately describe what is
implemented. A proposed layout or scheduling contract is not a performance claim.

```{contents} On this page
:local:
:depth: 1
```

## The design in one page

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

```{figure} ../../_static/tile/execution-first-overview.svg
:alt: Execution structure as the skeleton connecting dataflow, memory, layouts, pipelines, scheduling, and lowering.
:width: 100%

Execution is declared first; data, resources, mappings, and target choices attach
to it through typed relationships.
```

The canonical executable C++ spelling is included from the same source as
the C++20/C++23 capture tests. FP32 is used here to match the current tested
native bridge; additional element types do not change the access convention.

```{literalinclude} tile_programming_poc.cpp
:language: cpp
:start-at: struct GemmConfig
:end-before: // Read equivalence
```

Important properties of this surface:

- There is no explicit builder parameter. A scoped current builder records the
  program, as in the Luisa SIMT DSL.
- `for (auto &nest : parallel(...))` and
  `for (auto &subnest : nest.parallel(...))` make spatial hierarchy and lexical
  scope the same visible C++ structure. They do not say block, warp, lane,
  SIMD, or vector.
- The outer spatial nest supplies an anchor context; the innermost enclosing
  `parallel` supplies the default frontier. Data dependencies, explicit child
  coordinates, and the assigned value constrain the final anchor. Distributed
  child updates require an exact-cover proof; the current capture rejects
  outer-value mutation in `parallel` until that analysis is implemented.
  `serial`, `pipeline`, and `reduce` already infer Scalar and Tile carried state.
- `outer.index(...)` resolves against that Nest's own ancestor path, never
  against an active descendant. Nested positional shapes may reuse dimension
  identities without changing an explicitly named parent's coordinate.
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
candidate TileIR. `axis(name, extent)` gives logical dimensions kernel-local
identities; `shape(axis...)` relates operands without mutating the external
views. Numeric `shape(extents...)` supplies positional dimensions. No `input`,
`output`, `GemmSpec`, or symbolic integer language is needed in the body.

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
The opt-in `luisa/tile/runtime.h` adapter can compile a captured kernel through
`tile::compile(device, kernel)` and dispatch it on an ordinary Runtime Stream.
Automatic argument-shape capture at invocation, arbitrary strided views, and
scalar signature parameters remain separate implementation work.

## Non-negotiable separations

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

### Surface convenience is not primitive proliferation

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
| lifted `+`, `exp`, `ite`, GELU | generic scalar SSA region over a dimension-identity join |
| `reduce(x, dimensions, r)` | captured reduce region with generated state update |
| `matmul(a, b)` | infer conventional trailing matrix dimensions, create a zero result, then `mma(a, b, zero)` |
| general `einsum` | reindex/broadcast, elementwise multiply, reduce; a schedule may retile it into `mma` |
| convolution / pooling / normalization | views plus elementwise and reduce regions |
| `gather` | value-computed view index plus load |
| `scatter` | value-computed index plus store or atomic effect |
| `copy` | load/store edge, optionally recognized as a transfer atom |
| `topk<K>` | indexed Tile plus bounded merge-and-truncate reducer |
| `sort` / `merge_sorted` | compare/ite/reindex networks or structured radix/merge library |
| `scan` / histogram | parallel/serial/reduce regions plus indexed effects |

This table is a test obligation: the target-independent expansion must run in
the TileIR interpreter, and every atom replacement is checked against it.

```{figure} ../../_static/tile/primitive-layers.svg
:alt: Rich Tile library calls expand into a minimal TileIR core, while target atoms replace only proved equivalent subgraphs.
:width: 100%

Surface convenience, semantic primitives, and hardware atoms evolve at
different rates; keeping them layered prevents permanent IR bloat.
```

## Compiler references

[TileIR and capture](../internals/tile/ir.md), [TIRx export](../internals/tile/lowering.md),
and [planning](../internals/tile/planner.md) are implementation references, not
additional parts of the C++ surface. The [architecture decisions](../internals/tile/decisions.md)
retain the original design checklist; current work and measured results live in
[implementation coverage](../performance/tile/implementation.md) and [performance](../performance/tile/index.md).
