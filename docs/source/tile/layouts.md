# Layouts and value distribution

These are language contracts and design extensions, not a promise of complete target support. Use the [executable examples](kernels.md) for current C++ spelling and [implementation coverage](../performance/tile/implementation.md) for admitted lowering.

```{contents} On this page
:local:
:depth: 2
```

## The canonical layout algebra

### Decision

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

### Typed map

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

### Algebraic operators

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

### Replication and non-injectivity

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

```{figure} ../../_static/tile/layout-correspondence.svg
:alt: A layout correspondence with an occurrence fiber representing two physical replicas per logical element.
:width: 100%

The occurrence space makes replication explicit and avoids assuming a unique
inverse for a many-to-one placement.
```

### Bounds and data-dependent indexing

The declared `IndexSet` determines where a map is defined. Access behavior
outside a view is not hidden inside the layout. A view carries a separate
validity predicate and policy:

~~~text
View = (domain, map, valid(domain))
~~~

`bounds::zero`, `bounds::predicate`, and `bounds::assume` are policies for the
invalid part of the domain.

The shorthand **Tensor = Buffer + Layout** describes this non-owning view:
the buffer supplies resource identity and storage, while the layout describes
the indexed domain and its mapping into that storage. More explicitly,
`TensorView = (buffer, domain, address_map, validity, bounds_policy)`. Validity
and the policy remain explicit view metadata; they are not side effects hidden
inside the layout algebra. Slicing composes the view map and intersects bounds;
it does not allocate, copy, or choose an execution scope.

For example, a backend can express the same view as a Metal `tensor_inline`
constructed from an ordinary buffer pointer, extents, and strides. This does
not require a new TileIR resource kind or a tensor-handle kernel ABI. A Metal
`cooperative_tensor` is different: it is one possible realization of a Tile SSA
value in participant-private storage, not the meaning of a frontend tensor view.

Gather, scatter, indirection, and data-dependent indices are semantic index
operations feeding a layout. Treating a memory load as a layout node would
destroy most useful equivalence and invertibility reasoning.

### What “complete” means

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

### Compatibility embeddings

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

### Proof discipline and algebra laws

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

The implemented expression DAG currently contains constant/coordinate leaves,
integer add/subtract/multiply, floor division/modulo, XOR/AND, and logical
bit-pattern shifts. Typed inspection is available to analyses and bridges;
composition substitutes DAG nodes and does not serialize a callback. Native
TIRx export lowers these operations directly as 64-bit index arithmetic.
In particular, a logical right shift uses an unsigned bit pattern, not a
signed arithmetic shift. These expressions are not mislabeled as TIRx
`IndexMap`/`IterSumExpr` nodes when they fall outside that class's inverse
analysis contract. The factored shard/replica subset separately uses native
`TileLayout`; see the [TIRx layout bridge](../internals/tile/lowering.md#layout-bridge).

The implemented `IndexMap::prove()` returns `LayoutProof`, with separate
`PROVEN`, `DISPROVEN`, or `UNKNOWN` facts for totality, bounds, injectivity,
and surjectivity. A total map's checked `apply()` must succeed throughout its
domain and land inside the declared codomain. `analyze_finite()` remains an
independent exhaustive oracle; passing zero to `prove()` disables that fallback.

~~~text
IndexExpr DAG
   |-- checked affine normalization --> exact range + injectivity conditions
   |-- structural GF(2) normalization --> bit-image range + exact matrix rank
   |-- concrete invalid point / colliding pair --> disproof
   `-- unresolved --> exhaustive finite fallback, within its budget
                        |
                        `-- still unknown: preserve Candidate IR,
                                          reject native Memory realization
~~~

For static boxes, the affine normalizer retains `b + A*x` only if every
original subexpression has a representable signed-64-bit range. Cancellation
never hides an overflowing intermediate or a division by zero. Constant and
coordinate leaves, add/subtract, and multiplication by a domain-constant
expression are supported. This normal form is analysis state, not another
layout type or a restriction on the stored DAG.

Injectivity uses two exact sufficient conditions. Given equal outputs, write
`delta = x - y`; each unresolved coordinate satisfies `|delta_i| <= n_i - 1`.
If a row has

~~~text
|A[j,i]| > sum(k != i, unresolved) |A[j,k]| * (n_k - 1),
~~~

then any nonzero `delta_i` is too large to be canceled by the others, so
`delta_i = 0`. Repeat this argument to recover mixed-radix coordinates,
including ordinary dense, padded, reversed, and transposed strided layouts.
Full column rank of the remaining matrix modulo the prime `2^31 - 1` is
another sufficient proof: a nonzero modular minor is a nonzero integer minor,
so the remaining delta is zero over the rationals too. A modular rank failure
is **unknown**, not a proof of aliasing. Pigeonhole arguments, zero columns,
and evaluated GCD-derived colliding pairs provide disproofs. Known cardinality
and injectivity determine surjectivity where possible.

The second implemented normalizer uses the same bit-basis algebra described
by [Triton's LinearLayout](https://github.com/triton-lang/triton/blob/main/include/triton/Tools/LinearLayout.h),
extended with an XOR translation:

~~~text
f(x) = c XOR XOR(input_bit[i] * basis[i])

typed expression DAG --> prove GF(2) structure --> exact binary matrix
                                                   |-- rank --> injectivity
                                                   `-- affine image --> bounds
~~~

It derives the basis structurally from XOR, constant masks, constant logical
shifts, carry-free addition, nonnegative power-of-two division/modulo/multiplication
(with overflow checks), and Boolean-times-constant expressions. Every original
subtree must be safe before cancellation or masking. In particular, a
bit-pattern shift may discard high bits, whereas integer multiplication must
not overflow. Evaluating only zero and basis vectors cannot establish
linearity: a nonlinear expression can agree at those points and still alias.
The normal form is transient analysis state and introduces no Triton/MLIR
dependency or new DSL entity.

The matrix spans every input and output coordinate, including maps with more
than 64 total bits. For a static prefix box, each input axis uses
`ceil(log2(extent))` bits. The normalizer works on the enclosing power-of-two
box; an exact XOR-basis maximum proves output bounds without assuming that all
individually possible bits can occur together. An invalid origin or unit-bit
point disproves safety on the actual domain. An invalid envelope point alone
does not disprove a ragged domain and may leave its bounds unknown.

After bounds are proved, full column rank is equivalent to injectivity even
on ragged prefix boxes. For `[0,n)` with `b = ceil(log2(n))`, every b-bit delta
is an XOR of two valid coordinates: use `(delta,0)` when `delta<n`; otherwise
use `(2^(b-1), delta XOR 2^(b-1))`. Thus a nonzero matrix-kernel vector always
gives an actual colliding pair. On full power-of-two domains the affine image
has exactly `2^rank` points, which also decides surjectivity. On ragged domains,
surjectivity uses conservative cardinality facts or the finite fallback.

This is not the full mixed-radix complement/inversion algebra described by
[CuTe](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html).
General image/preimage construction, inverse APIs, mixed arithmetic/bitwise
normal forms, and symbolic constraints remain separate work. The finite
fallback budget is 1,048,576 logical points, but proved affine and bit-linear
layouts are not subject to that cap. Unsupported nonlinear or symbolic maps
may remain unknown. Exhaustive small-box tests cross-check every affirmative
or negative fact against the independent evaluator, including all 3-by-3
GF(2) matrices and XOR offsets over full/ragged domains; none of the safety
conditions rely on sampled success.

## Value distribution is a layout, not another algebra

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
