(views-values-and-addressable-memory)=
# Views and explicit memory

These are language contracts and design extensions, not a promise of complete target support. Use the [executable examples](kernels.md) for current C++ spelling and [implementation coverage](../performance/tile/implementation.md) for admitted lowering.

```{contents} On this page
:local:
:depth: 2
```

## Three surface objects

| Surface category | Meaning |
|---|---|
| `TensorView<T, R>` / subview | Addressable projection of external storage |
| `Tile<T>` | Staged value variable, promoted to tile SSA; rank is in its IndexSpace |
| `Memory<T>` | Explicit addressable temporary with stable identity and an IndexSpace |

Ordinary loads and operations return `Tile`. The compiler may realize a tile
in registers, an on-chip allocation, tensor memory, a vector object, or no
materialized object at all.

This virtual SSA path is the default even across pipeline stages. A
`MaterializationPlan` chooses independently for every live edge whether to keep
it virtual, fuse it, recompute it, or create an internal addressable object. For
a materialized edge it also chooses storage layout, resource class, lifetime,
version count, and synchronization. Those are compiler decisions constrained by
target capabilities and tuning, not mandatory source annotations.

The current TIRx bridge demonstrates this boundary without adding syntax. It
preserves every pure multi-consumer Tile definition by default, leaving a
target pass free to compact it, retain it or inline it later. An explicit
`EXPENSIVE_ONLY` candidate preserves shared `exp`, `log`, `sqrt` and `tanh`
but recomputes cheap arithmetic. A versioned exp contract may select a checked
CPU array-math atom only after the target pass revalidates the exact
expression. These are target-planning policies, not language semantics and not
a reason for the programmer to declare `Memory`.

`Memory` is an expert-only semantic escape hatch. It is requested only when the
programmer intentionally needs stable address identity: a mailbox, explicit
alias, persistent mutable state, pinned storage layout or swizzle, a manual
producer/consumer protocol, or exact buffer reuse. Ordinary async staging and
cross-stage dataflow are not reasons to declare one. A separate schedule may
bind an explicit `Memory` to a target resource, but the C++ type itself still
does not mean register, shared memory, SRAM, or any vendor address space.

## Memory ownership

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
state. The frontend builds those tokens during structured capture and resolves
their loop-carried definitions as each temporal region closes.
A software pipeline may map simultaneously live states to several physical
versions without changing the identity or alias class of `s`.

### Implemented explicit-Memory path

The entry points are `memory<T>(shape, resource = mem::auto_)` for planned dense
storage and `memory<T>(index_map, resource = mem::auto_)` for an explicit local
address layout. Common strided layouts use `layout(shape, stride(...))`, with
strides in elements. `Memory` is move-constructible but
neither copyable nor assignable; helpers can borrow it by reference. Only
`store(Tile<T>)` mutates it, and `load()` returns an immutable Tile SSA snapshot.
Reading before a definite store is rejected, including the zero-iteration case
where initialization only occurs inside a loop. There is no implicit zero fill.

An explicit layout maps the logical Tile element space to the allocation's
local storage space. Padding may leave unused storage; transposes and composed
XOR swizzles may reorder it. Neither changes the logical Tile shape, the
allocation identity, the MemorySSA chain, or the declaration's owner:

~~~text
owner coordinate p ---- execution/resource binding ---- instance + base(p)
logical element u ---- local IndexMap L --------------- storage indices L(u)

address(p, u) = base(p) + sizeof(T) * linearize_codomain(L(u))

same nest
  |-- As : [m,k] -- padded row strides --> one allocation
  `-- Bs : [k,n] -- column strides ------> another allocation
~~~

TileIR stores this map as a typed, replaceable property of `memory.alloc`, not
as a different Memory type or an opaque string attribute. A pass can replace
the map while retaining all logical users and state tokens. Whole-Tile
storage requires an injective map: a broadcast/non-injective map is valid
general layout algebra, but cannot realize arbitrary independent Memory
elements. Padding is included in physical capacity accounting. A load copies
the mapped elements back into a dense logical Tile snapshot; unused padding
is neither read nor treated as initialized data.

An empty logical domain has no address events, so unreachable map arithmetic
is not evaluated. After checking execution/resource constraints, native
lowering removes unused zero-sized plain allocations. It rejects a zero-sized
allocation with surviving load/store uses rather than leaving a dangling
buffer. An empty allocation never bypasses an explicit resource constraint.
Known zero factors are recognized before multiplying extents, even when the
remaining product overflows or contains dynamic factors. Layout safety does
not imply feasible storage: native Memory also checks signed-64-bit byte
addressing, followed by the target's separate resource-capacity constraints.

~~~text
parallel instance (logical owner)
  |-- Memory A ---- state A0 --store--> A1 --load--> Tile snapshot
  |-- Memory B ---- state B0 --store--> B1
  |
  `-- pipeline / serial / reduce
        state inputs A1, B1
        store/load effects; state outputs A2, B2
        (tokens are inferred; no user-written result()/state plumbing)

Later A.store(...) changes A's state, not the already loaded snapshot.
Child parallel instances can read visible ancestor resources. Whole-object
ancestor writes from child parallel instances are not independent and fail.
~~~

Memory states use the existing structured-operation operands, block arguments,
results, and yields. They are not castable data, allocations, or runtime
counters. The verifier checks unique reaching states, memory identity through
loop carries, definite initialization, and lexical dominance. Reusing an old
state after a store or swapping same-typed states between resources is invalid.
Fine-grained subviews and disjoint parallel writes need range-aware MemorySSA;
they are not implemented by pretending each whole-Memory write is independent.

Native TIRx export retains hard resource constraints on allocations. CPU worker
memory currently uses local storage; Metal group memory uses shared storage and
descendant-worker memory uses private storage. Unsupported constraints, such as
group-owned private memory or worker-owned shared memory without a slicing
plan, fail rather than silently changing the logical owner. Shared capacity
includes manual and compiler-generated temporaries. Load snapshots are
materialized conservatively, with cooperative copies and uniform barriers.
MemoryState tokens themselves disappear only after verification into ordered
TIRx effects. The native bridge now software-pipelines safe producer/consumer
cuts with iteration-local buffer versions; recurrences through outer Memory
remain ordered unless a legal cut can be proved. Hardware-asynchronous copy,
range-aware parallel writes, and global/cluster/tensor allocations remain
planning work. The [software-prefetch reference](../internals/tile/lowering.md#implemented-native-software-prefetch-path) states the implemented scheduling boundary.

The manual GEMM spelling is compiled by both C++20 and C++23 capture tests:

```{literalinclude} tile_programming_poc.cpp
:language: cpp
:start-at: // Explicit Memory is optional
:end-before: }// namespace
```

## The execution-to-memory equation

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

```{figure} ../../_static/tile/execution-to-memory.svg
:alt: Separate maps select a resource instance and a byte offset for each execution event.
:width: 100%

Spatial ownership selects an allocation instance; access, view, temporal
version, and address maps select a byte inside it.
```

Visibility, coherence, barriers, and legal cross-instance access are not
spatial layouts. They are target capabilities and effect-verification rules.
Keeping them separate prevents a clever layout from pretending an illegal
memory access is legal.
