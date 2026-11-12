---
name: tile
description: Write, capture, compile, and dispatch LuisaCompute Tile DSL kernels. Use for TensorView/Tile/Scalar values, axes and broadcasting, execution nests, reductions/MMA, explicit Memory/layouts, portable tile algorithms, runtime buffer binding, and TileIR diagnostics.
---

# Tile DSL

Use Tile for execution-first, shape-typed tensor kernels. It is separate from the ordinary SIMT DSL: express logical tiles and structured execution, then let a planner choose the physical mapping.

## Includes and model

```cpp
#include <luisa/core/mathematics.h> // ceil_div, host math
#include <luisa/tile.h>             // DSL, IR, Memory, algorithms
#include <luisa/tile/runtime.h>     // Device compile/dispatch adapter
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
namespace tile = luisa::compute::tile;
using namespace tile;
```

| Type | Meaning |
|---|---|
| `Scalar<T>` | Scalar SSA value; `nest.index()` is `Scalar<int64_t>`. |
| `Tile<T>` | Shape-typed, elementwise SSA value; not a memory reference. |
| `TensorView<T, Rank>` | External dense tensor argument. `const T` forbids stores. |
| `MemoryRef<T, Rank>` | Address of a tensor subtile; requires explicit `load()`/`store()`. |
| `Memory<T>` | Optional addressable local allocation with whole-tile access. |

`tile_kernel` records a reusable definition. `capture(tensor_shape(...))` specializes it for static argument extents and returns TileIR. Kernel parameters must be `TensorView`s; use ordinary lambda captures for host configuration. Create or recapture a specialization when configuration or extents change.

## Minimal kernel

```cpp
auto definition = tile_kernel(
    "vector_add",
    [](TensorView<const float, 1> A,
       TensorView<const float, 1> B,
       TensorView<float, 1> C) {
        auto n = axis("n", 32);
        auto blocks = ceil_div(A.extent<0>(), n.extent().constant_value());
        for (auto &block : parallel(shape(blocks))) {
            auto origin = coord(block.index() * 32);
            C(origin, shape(n)).store(A[origin, shape(n)] +
                                      B[origin, shape(n)]);
        }
    });

auto kernel = definition.capture(
    tensor_shape("A", size), tensor_shape("B", size),
    tensor_shape("C", size));
// Always check kernel.valid() before compiling.

tile::CompileOptions options; // NATIVE lowering, automatic planning
auto shader = tile::compile(device, kernel, options,
                            {.enable_fast_math = false});
// Check `shader`; on failure inspect shader.metadata().error.
stream << shader(A_buf, B_buf, C_buf).dispatch() << synchronize();
```

Runtime arguments are positional `Buffer`s or contiguous `BufferView`s with the captured element type and sufficient size. Dispatch geometry is already planned; do not supply dispatch dimensions. Buffer-view offsets are supported. Writable argument ranges must not overlap when `metadata().disjoint_writes` is set.

## Axes, shapes, and values

- Create an `Axis` once and reuse it: `auto m = axis("m", 16);`. Axis identity is not its name; two axes named `"m"` differ.
- `shape(m, n)` uses named dimensions. `shape(3, 4)` uses positional dimensions, aligned from the right for broadcasting.
- Elementwise operations broadcast scalars and Tiles with matching dimension identities and equal or singleton extents. Use `cast<T>(x)` for numeric conversion; arithmetic does not implicitly mix element types.
- `full<T>(space, value)`, `zeros<T>(space)`, `map<T>(space, fn)`, and `iota(axis)` construct Tiles. `map` is pure: no view or `Memory` effects.
- `x.at(coord(...))` extracts a scalar using Tile-local coordinates; `x.at(nest)` uses the nest's matching axes. This is value extraction, not memory access.
- `relabel(x, space)` changes dimension identities only, never extents or element order. `broadcast_to(x, space)` explicitly broadcasts.
- Use `ite(condition, if_true, if_false)` for value selection. It is not lazy control flow. Dynamic `Scalar<bool>` is not convertible to host `bool`.
- `bitcast` currently supports only `float <-> uint32_t`, preserving shape and all bits.

## Tensor access and bounds

These tile loads are equivalent:

```cpp
auto x0 = A[coord(m0, n0), shape(m, k)];       // C++23; typed C++20 adapter too
auto x1 = A(coord(m0, n0), shape(m, k)).load();
auto x2 = A.tile(coord(m0, n0), shape(m, k)).load(); // most explicit/portable
C(coord(m0, n0), shape(m, n)).store(acc);
```

Use `.tile(...).load()` in generic code when avoiding the C++20 comma-subscript adapter. Never assign to `A[...]` or a `MemoryRef`; store explicitly. Assignment to a named `Tile` only replaces that C++ handle's SSA value.

Default `bounds::zero` zero-fills out-of-bounds loads and suppresses out-of-bounds stores, making ragged tiles safe. Use `.load(fallback)` for another fill value. `bounds::assume` requires the caller to guarantee in-bounds access.

For scalar element access:

```cpp
auto value = A(i, j).load();
auto masked = A(i, j).load(in_bounds, 0.0f);
C(i, j).store(value);
```

A tile load is a snapshot: later stores to its source do not change the loaded `Tile`.

## Structured execution

Always use the range form `for (auto &nest : ...)`; it records one structured region, not a host loop over iterations. A normal host `for` instead unrolls/stages C++ during capture.

| Construct | Semantics |
|---|---|
| `parallel(space, exec::Scope::AUTOMATIC)` | Unordered independent instances; observable results use memory effects. No carried `Scalar`/`Tile` mutation. |
| `serial(space)` / `nest.serial(space)` | Ordered iterations with automatic loop-carried values. |
| `reduce(space, policy)` / `nest.reduce(...)` | Ordered or reducible contributions with automatic accumulator carry. |
| `pipeline(space, policy)` / `nest.pipeline(...)` | Ordered effects/carries with permitted stage overlap; not independent iterations. |

Reassignment of outer `Scalar` or `Tile` variables inside serial/pipeline/reduce creates implicit carries. Copies preserve variable identity and snapshots; yields are simultaneous. Parallel nests reject such carries.

Leave execution scope automatic unless a constraint is required. Explicit scopes are `DEVICE`, `GROUP`, `SUBGROUP`, `WORKER`, and `VECTOR`; nested scopes must descend in that containment order.

Pipeline example:

```cpp
for (auto &step : nest.pipeline(shape(k_tiles), {.window = 2, .interval = 1})) {
    step.stage("load");
    auto a = A.tile(coord(m0, step.index() * BK), shape(m, k)).load();
    auto b = B.tile(coord(step.index() * BK, n0), shape(k, n)).load();
    step.stage("compute");
    acc = mma(a, b, acc);
}
```

`window` bounds in-flight iterations: zero lets the planner choose, one disables overlap. It is not the number of stages. `interval` is a positive scheduling-model issue interval. Call `stage()` only directly on the active pipeline nest.

## Reductions, MMA, and algorithms

```cpp
auto row_max = reduce(x, column, maximum); // removes `column`
auto row_sum = reduce(x, column, add);
auto normalized = x / row_sum;             // reduction result broadcasts back
auto acc = mma(a, b, zeros<float>(shape(m, n)));
```

Reducers are `add`, `maximum`, and `minimum`; custom reducers provide `identity<T>()` and `operator()`. Reduction policies are `reduction::unordered_tree` (default), `ordered_tree`, `fold_left`, and `fold_right`. Ordered policies constrain contribution order; use them when bitwise/order-sensitive results matter.

`mma` contracts shared named axes absent from the accumulator. Positional rank-two Tiles use conventional `(M,K) x (K,N) -> (M,N)`. Input and accumulator types are independent. By default MMA permits reassociation; `{.allow_reassociation = false}` requests reference contraction order and may be rejected by a backend.

From `<luisa/tile/algorithms.h>`:

- `reshape(x, space)`: equal static volume; `reindex(x, space, coordinates)`: pure coordinate permutation/projection.
- `gather(x, indices, axis, fallback)`, `argmax(x, axis)`, `inclusive_sum(x, axis, policy)`.
- `sort(x, axis, descending)` and `topk(x, axis, count, largest)` return `{values, indices}`. Ordering is stable by original index and requires non-NaN values. `SortAlgorithm::PACKED_FP32` is opt-in for float32, power-of-two extents, and `N <= 2^31`.

## Explicit Memory only when needed

Use `Memory` for stable address identity, a resource constraint, or an explicit allocation-local layout—not for ordinary SSA temporaries.

```cpp
auto local = memory<float>(shape(m, k), mem::shared);
local.store(A.tile(coord(m0, k0), shape(m, k)).load()); // initialize first
Tile<float> snapshot = local.load();
```

`Memory` supports only whole-tile load/store with exactly matching element type and space. A load is a snapshot. Declare it in the owning lexical nest; parallel descendants may read initialized ancestor Memory but must not write it. Carried writes require temporal regions and must respect parallel ownership boundaries. Resources (`auto_`, `private_`, `shared`, `cluster`, `global`, `tensor`) are independent of execution scope and remain backend-qualified.

Custom storage uses element strides:

```cpp
auto padded = memory<float>(
    layout(shape(m, k), stride(k.extent().constant_value() + 1, 1)), mem::private_);
```

Layouts describe allocation-local addresses, not execution mapping. They must be total, in bounds, and injective; capture rejects proven invalid layouts.

## Compile and diagnose

- `CompileOptions::lowering`: `NATIVE` (default) or `TIRX`. Support is operation-, shape-, and backend-specific; unsupported routes fail closed rather than silently falling back.
- Nonzero `threads_per_group` is an exact constraint, not a hint. Zero selects a legal default.
- Runtime compilation defaults to `enable_fast_math=false`; local ordered-reduction/MMA policies still constrain eligible transformations.
- Check `kernel.valid()` after capture. Print `kernel.diagnostics()` and, for verifier failures, `verify(kernel.module()).diagnostics()`.
- Check `static_cast<bool>(shader)` after compilation and report `shader.metadata().error`. `metadata().arguments`, `dispatch_size`, `source`, and `realization` expose the realized ABI/route.

## Source map

Start with executable examples in `examples/compute/tile/kernels.h` and `docs/source/tile/tile_programming_poc.cpp`. Public semantics live in `include/luisa/tile/{dsl,value,memory,algorithms,runtime}.h`. Capture and automatic carries are implemented in `src/tile/dsl.cpp`; shape, purity, dominance, and MemoryState checks are in `src/tile/verifier.cpp`. Use `src/tests/unit/tile/test_tile_{dsl,values,memory,primitives,native_runtime}.cpp` as conformance references, then the `bridge/` tests for route-specific support.
