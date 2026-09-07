# Assignment, staging and JIT

These are language contracts and design extensions, not a promise of complete target support. Use the [executable examples](kernels.md) for current C++ spelling and [implementation coverage](../performance/tile/implementation.md) for admitted lowering.

The `device.jit(...)` loop below is architectural pseudocode. Current capture and Runtime entry points are documented in [the overview](design.md) and [Runtime integration](../internals/tile/runtime.md).

```{contents} On this page
:local:
:depth: 2
```

## Direct assignment and hidden SSA plumbing

### Surface rule

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

### Addressable storage uses explicit effects

Assignment is reserved for staged Scalar and Tile definitions. Addressable
storage, including `Memory`, views, and scalar element references, exposes
only explicit `store` and `load` effects:

~~~cpp
acc = f(acc);                    // Tile SSA definition
As.store(input_view.load());     // View -> Tile -> Memory
As.store(value);                 // Tile -> Memory
auto staged = As.load();         // Memory -> Tile
auto x = A[coord(m0, k0), shape(m, k)]; // TensorView -> Tile
C(coord(m0, n0), shape(m, n)).store(value);
~~~

The receiver makes the addressed object unambiguous, and every addressable
read or write is visible at the call site. `Memory = Tile` and `Memory = View`
are both ill-formed. The access convention is:

| Surface | Meaning |
|---|---|
| `A(origin, shape)` | `MemoryRef`, no memory effect |
| `A.tile(origin, shape)` | The identical `MemoryRef` |
| `A[origin, shape]` | `A(origin, shape).load()`, a Tile SSA value |
| `C(origin, shape).store(value)` | The explicit memory write |

`origin` is `coord(...)`; the optional third argument is a bounds policy.
`bounds::zero` is the default for every spelling: zero-fill reads and masked
tail stores. `bounds::assume` promises valid addresses. A nonzero read fallback
is explicit: `A(origin, shape).load(fallback)`.

C++23 uses native multidimensional `operator[]`. In C++20, `operator,` is
overloaded only for the DSL coordinate/shape/bounds combinations and packages
one typed selection for `operator[]`. Both paths enter the same load builder;
the language version cannot change the access, bounds, or SSA semantics.
The optional Clang comma-subscript deprecation diagnostic is suppressed only
at deliberate C++20 use sites, not globally from a public header.

There is no implicit `MemoryRef`-to-Tile conversion. `auto x = A[...]` loads
once; subsequent `x = ...` only defines a value. Tile assignment is lvalue
qualified, so `A[...] = ...` is rejected rather than becoming a discarded
value assignment or an accidental store. Manual `Memory::store` likewise
accepts only an explicitly produced Tile.

The same rule applies after scalar descent: `y(i) = value` and `y(i) = x(i)`
are ill-formed, not stores or reference rebinding. The implemented
`ElementRef` has no implicit load and deletes copy/move assignment; retaining
an address with `auto ref = y(i)` is allowed, then `ref.load()` and
`ref.store(value)` record effects. Compile-time tests enforce this distinction
while ordinary `acc = ...` and `acc += ...` still capture value definitions.

Each `Memory::store` defines a new hidden `MemoryState` token; `Memory::load`
consumes the reaching state and returns Tile SSA. The explicit View load and
Memory store are separate Candidate effects so alias and bounds analyses remain
honest. A schedule may nevertheless prove them equivalent to one direct or
asynchronous transfer and replace the pair with an atom. Thus source semantics
stay simple without preventing zero-intermediate-buffer lowering.

## C++ staging and JIT

### One scoped builder, no builder prefixes

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

### Ordinary configuration creates variants

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

### Target schedules

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
