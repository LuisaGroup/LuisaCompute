# Tile operations and reductions

These are language contracts and design extensions, not a promise of complete target support. Use the [executable examples](kernels.md) for current C++ spelling and [implementation coverage](../performance/tile/implementation.md) for admitted lowering.

```{contents} On this page
:local:
:depth: 2
```

## Elementwise operators lift directly to tiles

Pure scalar operators are rank-polymorphically lifted to `Tile` values. Common
arithmetic, comparison, selection, casts, and math functions therefore read as
ordinary expressions:

~~~cpp
auto y = gelu(x + bias) + residual;
auto p = exp(scores - row_max);
auto finite = ite(mask, p, 0.0f);
~~~

`ite(condition, true_value, false_value)` means if-then-else, using the same name
and argument order as Luisa's SIMT DSL. It supports Scalar and Tile values,
including broadcast conditions and operands. There is no Tile DSL `select`
alias: Luisa's existing `select(false_value, true_value, condition)` uses a
different order. `ite` selects already-computed values; it is not lazy control
flow and does not by itself guard an unsafe load in either operand.

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

## MMA is a portable value primitive

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

The current typed operation records input/accumulator element types,
contraction dimensions, and `MmaPolicy::allow_reassociation`. Inputs are
converted to the accumulator type by the reference realization. MMA permits
reassociation by default, but does not permit silently reducing input precision
(for example, substituting TF32 for FP32). Ordered accumulation is explicit:

~~~cpp
acc = mma(a, b, acc); // target may choose a compatible cooperative matrix atom
acc = mma(a, b, acc, {.allow_reassociation = false}); // preserve contraction order
~~~

The policy is stored on the TileIR operation and survives lowering; it is not
just a frontend scheduling hint. Disabling reassociation retains the reference
K order, not a promise to disable target FMA fusion or obtain bit-identical
results on different devices. More detailed accuracy/rounding policies are
design extensions, not implemented public options. A CPU may use vector FMAs;
a GPU may select matrix atoms only when their numerical and layout contracts
refine this operation. Otherwise the reference loop remains available.

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

## Reduction is a structured algebraic region

**Current surface versus proposed contract.** `Nest::reduce(IndexSpace, policy)`
and ordinary carried assignment are implemented. The default is
`reduction::unordered_tree`; each REDUCE operation stores a typed
`ReductionPolicy`. TIRx and XIR preserve explicit order restrictions, and backend
switches control candidate availability only. Custom lift/merge contracts,
their general validation and the richer domain conveniences below remain design
extensions. Unrecognized bodies currently retain a serial realization; they
are not silently granted an invented parallel merge.

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

`column` is a non-copyable Nest handle, just like the other range-for bindings.
Its index identifies a contribution coordinate, not a memory element or worker.
The `domain`/`coord` conveniences in this schematic surface do not prescribe
hardware binding.
`x.domain(columns)` is a typed `IndexSet`; `x.at(column)` is a pure Tile
projection implemented by reindexing. Neither operation chooses participants,
memory, or a collective.

The proposed region signature is deliberately source-free and state-free:

~~~text
nest.reduce(domain, contract?, policy?)

domain      typed semantic contribution coordinates
contract    optional lift/merge/type/law contract for tree realization
policy      reference fold and permitted regrouping, never a warp count
~~~

The body is captured once. On region close, the frontend finds every `Tile`
defined outside the region and updated inside it. Those values become the
region's incoming block arguments and outgoing results. Locals remain locals;
stores and atomics remain effects; neither can accidentally become a reduction
state. There is no public `result()`, `yield`, accumulator proxy, or state list.

In the proposed analysis, the update graph supplies information that an
explicit accumulator list would duplicate. A
canonical `state += contribution`, `state = maximum(state, contribution)`, or
corresponding minimum/logical form identifies a possible reducer contract.
`state = mma(a, b, state)` still has its own inner contraction policy; recognizing
this update does not authorize reordering the outer block/tap fold. A tree
candidate needs a compatible lift/merge and numerical contract for that fold.
Independent states may form a product reducer with permissions checked for each
component; coupled updates need a joint contract. Recognition of floating-point addition
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

An unrecognized pure update can use an explicitly selected strict fold.
The default tree policy still needs a compatible merge contract; if none is
available, diagnose the missing contract and suggest a fold policy rather than
silently inventing a merge. Programs with observable intermediate states or
general ordered effects use `serial` or a suitable scan/effect contract.

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

### Reference fold and permitted regrouping

The following presets are implemented C++ API. They extend the existing
`reduce` primitive rather than introducing four kinds of nest.
For the source contribution sequence `x0, x1, x2` and incoming state `z`:

```{table} Reduction policies: default freedom and explicit restrictions
:class: design-table
:name: reduction-policies

| Policy | Reference / permission | Required contract |
|---|---|---|
| `reduction::unordered_tree` (default) | Regroup and permute contributions; seed exactly once | Compatible lift/merge |
| `reduction::ordered_tree` | Regroup; preserve leaf order and the seed's reference position | Compatible lift/merge |
| `reduction::fold_left` | `op(op(op(z, x0), x1), x2)` | No merge law required; preserve this update chain |
| `reduction::fold_right` | `op(x0, op(x1, op(x2, z)))` | No merge law required; preserve this update chain and operand orientation |
```

**The source default is `unordered_tree`.** Ordinary `nest.reduce(domain)`
authorizes a compatible reducer's tree regrouping and contribution permutation,
including changed FP32-add rounding; users need not add a fast-math switch to
obtain that permission. `ordered_tree`, `fold_left` and `fold_right` are explicit
restrictions. A tree policy permits a serial realization too: it describes a
set of allowed computations, not a mandatory hardware algorithm.

This default does not authorize a different accumulation dtype, approximate
transcendentals, FMA contraction or ignoring NaNs; those remain independent
arithmetic contracts. A strict floating-point operation mode does not retract
the reduction's regrouping permission. Conversely, an explicit fold restriction
must survive every backend and tuning configuration. Current striped Metal
collectives and CPU array reductions require `unordered_tree`; ordered trees
conservatively retain a source-ordered recurrence. XIR/SIMD currently retains
serial reductions for all four policies. The Metal and SIMD Runtime factories
disable kernel-wide fast-math when any reduction requires order, because their
current final compiler interfaces cannot express a local arithmetic override.
This does not turn unrelated unordered reductions into explicit folds.
The TIRx LLVM target likewise cannot override order with global fast-math
flags. TVM's own Metal runtime requires the optional precise-math extension;
without it, ordered reductions fail compilation explicitly. Luisa's Metal
Runtime uses its own compiler options and needs no such TVM extension.

For order-sensitive policies, multidimensional source order is lexicographic
in the `IndexSpace` axis order over active coordinates; physical layout, lane
number and storage replication cannot redefine it. A policy is not an
optimization-strength integer: left and right folds generally compute different
functions. Mathematical laws and policy-granted relaxation are distinct
evidence. FP32 addition does not have exact associativity.

The ordinary assignment spelling remains visible in both directions:

~~~cpp
// x is an existing Tile over the axis k.
auto left = Scalar<float>{0.0f};
for (auto &r : nest.reduce(shape(k), reduction::fold_left)) {
    left = left - x.at(r);
}
auto right = Scalar<float>{0.0f};
for (auto &r : nest.reduce(shape(k), reduction::fold_right)) {
    right = x.at(r) - right;
}
~~~

Right fold visits contributions in reverse source order and uses the stated
element/state operand orientation. The policy does **not** silently change
`state - element` into `element - state`; doing only a reversed left-fold
traversal is not the same contract. For `x = [1, 2, 3]` these examples yield
`left = -6`, `right = 2`. Heterogeneous folds must type-check the appropriate
`State x Elem -> State` or `Elem x State -> State` update.

### Grouping, merge legality and edge cases

Formally the region declares a reduction coordinate space `R`. Each inferred
state has a result coordinate space `G` and a set of semantic contribution
occurrences `Omega`, with `iteration : Omega -> R` and
`group : Omega -> G`. For `x.domain(axes)`, `group` is the familiar projection
that removes those axes. For a general map-reduce body it is inferred from the
shape/index map of the state update. Each grouping fiber has an ordered list
`omega0, ..., omegaN-1`. A strict fold only requires a typed update and seed;
it does not require an identity or a parallel merge. For a tree-enabled
left-reference reducer with a lawful monoid `(S, identity, merge)` and
contribution `lift(omega)`, one admitted realization is:

~~~text
result[g] = merge(incoming[g],
                  ordered_merge(identity,
                                [lift(omega0), ..., lift(omegaN-1)]))
~~~

For a domain occurrence `omega`, the captured body computes
`update(state, omega)` using arbitrary pure contribution-producing operations.
For an exact ordered-tree rewrite, sufficient laws are:

~~~text
update(s, omega) = merge(s, lift(omega))
merge(merge(a, b), c) = merge(a, merge(b, c))
merge(identity, a) = a = merge(a, identity)
~~~

Without exact laws, the tree policy must admit the changed results for that
compatible reducer. This is how the default FP32 sum allows tree rounding
without falsely asserting exact associativity. A law declaration and numerical
permission remain separate evidence, even when they enable the same candidate.

For a right-reference update, the lift/merge relation instead has the element
on the left: `update(omega, s) = merge(lift(omega), s)`, with the seed at the
right boundary. A generic `State x Elem -> State` update does not automatically
provide `State x State -> State` merge.

The fiber is an ordered contribution sequence unless the contract also grants
permutation. Associativity permits different parentheses, not an arbitrary
permutation: preserving a noncommutative reducer's source sequence requires
contiguous partial sequences and an order-preserving merge. Striped worker
assignment generally changes order
and needs the stronger permission. For example, combining `[a,c]` and `[b,d]`
as `(a op c) op (b op d)` reorders leaves. An independent output-group direction
and a reduction direction are separate factors of one domain, as formalized in the
[execution calculus](../internals/tile/calculus.md#strength-is-a-product-order-not-an-enum-order).

The intended library catalog supplies conditional contracts for built-in
updates and custom states such as Welford. The current implementation is
narrower; body recognition is not a proof of arbitrary custom laws. Strict
folds remain valid without merge laws. Floating-point reassociation, precision,
FMA behavior and determinism are separate policy dimensions. A fixed tree can
be reproducible within one configuration without being bit-identical across
different JIT configurations or devices.

The contract must also define:

- **Seed versus identity:** incoming state is included once, in its specified
  position; it is not copied into every worker's partial. Empty fibers return
  that seed. Masked-out occurrences are skipped; padded physical lanes may use
  an identity only when it is valid under the selected arithmetic semantics,
  otherwise partials carry validity.
- **Exceptional values:** overflow, NaNs, signed zero, ties and argmax ordering
  are part of the operation contract, not guessed from an intrinsic name.
- **Local permissions:** the language/kernel default resolves at capture; a stricter
  local policy is never widened by a backend or tuner. Independent state
  components may need separate trees or a common stricter realization.
- **Proof boundary:** builtin facts, derived conditions and trusted custom
  law declarations retain their provenance. Random testing can falsify a law,
  not prove it for an arbitrary function.

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

With the corresponding permission and merge contract, it may become a lane
shuffle tree, a shared-memory tree, a SIMD horizontal operation or a hybrid.
A strict fold preserves its chain while independent output fibers may still
execute in parallel. The remaining logical axes determine the result shape;
their distribution may stay sharded or acquire explicit replica fibers.

```{figure} ../../_static/tile/reduction-model.svg
:alt: A reduction region groups semantic contributions while its schedule independently maps the reduction domain to participants and serial steps.
:width: 100%

Grouping and the selected fold/tree contract define meaning. A compatible merge
and the permitted transformations open distribution and collective choices;
missing merge laws do not invalidate an explicitly selected strict fold.
```

The first proof-driven realization of this factoring is now implemented for
Metal FP32 add/max/min row programs. It maps a logical reduction to one or more
SIMD groups and derives worker-private/shared storage from the selected owner
map; see [the generated SIMD-group intrinsic path](../internals/tile/reductions.md#warp-and-simd-group-intrinsics-in-the-generated-code). In that bounded
implementation, `metal_subgroup_reductions` only enables the candidate family;
the per-operation policy supplies tree-order permission. The Metal Runtime
enables this family automatically when it owns capability/noalias validation
and the caller has not supplied an explicit TIRx configuration. Standalone
TIRx compilation still requires those target contracts. A richer per-reducer accuracy,
determinism, NaN and signed-zero policy remains part of this language design,
not a feature already exposed by the current C++ surface.

The same execution/resource separation is observable for an indirect gather.
A distributed definition of logical `Tile[N]` does not make the other
elements of each worker's physical private array valid. The current Metal
realization either forwards a guarded immutable Tensor view or proves every
remaining local access has the current worker owner; otherwise it falls back.
This is a backend proof over the general `gather` operation, not a new
loss-specific primitive or source memory level.

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

## Ordering and selection stay logical Tile operations

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
reference expansion uses core comparison/ite, reindex, reduction, and structured
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
