# TileIR representation and capture

This page owns in-memory IR, capture dataflow, ownership and verification. The mutable SSA core is implemented; the full analysis inventory, region-form pipeline stages and general Scheduled/Machine forms below are extension contracts, not an inventory of finished passes. See [implementation coverage](../../performance/tile/implementation.md).

```{contents} On this page
:local:
:depth: 2
```

## TileIR as a thin but transformable IR

### In-memory structure

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

### Minimal operations

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

### Forms and invariants

One IR data structure has progressively stronger verified forms:

1. **Candidate TileIR**: logical hierarchy and semantic operations; some
   anchors, frontiers, layouts, bindings, distributions, and realizations may
   be variables.
2. **Scheduled TileIR**: execution binding, pipeline schedule, distributions,
   memory plans, transformed execution structure, and guards are concrete.
3. **Machine TileIR**: atom calls, explicit realized transfers/synchronization,
   and addresses are legal for one target.

Forms are verifier states, not three unrelated object models.

### Essential analyses

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

## Capture algorithm

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

The current straight-line capture uses a temporary forwarding definition for
each live C++ variable at temporal-region entry. At region exit, a mutated
variable resolves to its own body argument; an unchanged variable resolves to
its incoming definition. The forwarding definitions are then removed, so the
stored TileIR remains SSA without an additional public operation kind:

~~~text
before loop:  a -> %initial, b -> %initial, snapshot -> %initial
inside loop:  a -> %a_in,    b -> %b_in,    snapshot -> %initial
~~~

Sharing an initial definition does not merge variable identities. In
particular, `auto old_a = a; ...; b = old_a;` yields `%a_in`, not `%initial`.
The builder installs the yield before resolving forwarding definitions, and
also rewrites live C++ handles before erasing them. The implementation
conservatively carries every mutated incoming variable; later liveness and
canonicalization can eliminate redundant state. CPU and Metal tests cover
Scalar and Tile snapshots, zero/one/many iterations, and nested pipelines.

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

## Shared SSA preserves a resource-planning choice

A Tile SSA definition with several consumers is one logical value. That fact
must survive canonicalization and structural bridge export because erasing it
by cloning the producer is irreversible. It does **not** mean that the source
declared an addressable allocation or that every backend must store the value.

For a pure definition `%v = f(...)`, target planning may choose among:

~~~text
recompute(%v, use)     inline f at selected consumers
retain(%v)             keep it in distributed registers/fragments
materialize(%v, R)     assign a compiler-owned resource, layout and lifetime
~~~

The choices are equivalent only after checking purity, effects, aliasing,
layout correspondence, active participants and the target arithmetic policy.
`materialize` additionally needs an ownership map, address map, lifetime,
capacity proof and legal target access. A later pass may still inline a
preserved pure definition; it cannot reconstruct sharing that structural
lowering already destroyed.

```{figure} ../../../_static/tile/shared-tile-planning.svg
:alt: A shared semantic Tile value remains one SSA definition while target planning chooses recomputation or bounded materialization.
:width: 100%

Sharing is semantic information. Storage, placement and recomputation are
target-dependent resource decisions.
```

The implemented TIRx bridge therefore preserves every pure multi-consumer
Tile by default. Its `EXPENSIVE_ONLY` mode is an explicit diagnostic/JIT
candidate that keeps shared transcendentals but recomputes cheap arithmetic.
The Metal row-program mapper can realize a preserved value as a bounded
worker-private stripe after proving ownership; LLVM may instead profit from
recomputation and fusion. Both modes keep the same Candidate TileIR semantics.

This rule also explains why compiler-created materialization is not surfaced
as `memory<T>(...)`. Manual `Memory` states that stable addressable identity is
part of the requested schedule. It cannot be silently recomputed and every
write remains an explicit effect.

## Required verifier invariants

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
