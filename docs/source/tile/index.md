# Tile programming

Tile is LuisaCompute's experimental execution-first C++ DSL for neural-network
and other tiled compute kernels. Use the [SIMT DSL](../dsl.md) when writing
individual worker computations; use Tile when the program should express
whole Tile operations and their nested execution structure. Both use the
[Runtime's buffers, devices and streams](../resources.md).

The language core is executable, but target coverage is still bounded. Read
[implementation coverage](../performance/tile/implementation.md) before assuming
that a layout or schedule represented by TileIR can be lowered on every target.

```{toctree}
:hidden:
:maxdepth: 1

kernels
design
```

## Programming model

- Declare kernel parameters in a staged C++ lambda. Ordinary host configuration
  values produce independent capture/JIT candidates.
- Build lexical execution Nests with range-for `parallel`, `serial`, `pipeline`
  and `reduce`. A Nest describes execution, not a memory level.
- Use Tile-level arithmetic and `mma`; ordinary assignment captures carried SSA
  state. Elementwise operators, normalization, convolution and attention compose
  this core instead of adding kernel-specific hierarchy objects.
- A Tensor is storage plus layout. `A(origin, shape)` names a `MemoryRef`;
  `A[origin, shape]` loads a Tile. Memory writes always use `.store(...)`.
- Let the compiler plan temporaries. Explicit `Memory` is for stable addressable
  storage or manual resource constraints, independent of execution mapping.

The [kernel examples](kernels.md) use the same source included by the C++20 and
C++23 capture tests. The [language reference](design.md) contains the full
layout algebra, numerical contracts and verifier boundaries; it also marks
features that remain design rather than implemented lowering.

## Compilation and performance

Compilation goes from captured TileIR through a backend-selected native or
bridge path, then to an ordinary Runtime shader and Stream launch. Independent
Tile kernels are supported; intra-kernel mixing with the SIMT DSL is not yet
part of this implementation.

- [Compiler and Runtime integration](../internals/tile/index.md): ownership,
  execution planning, TIRx, native Metal and XIR/SIMD.
- [Current status and performance](../performance/tile/index.md): what works,
  comparisons with Torch/MPS, timing definitions and remaining acceptance work.

Performance results belong to a particular route, cohort and measurement
method. They are not part of the source-language contract.
