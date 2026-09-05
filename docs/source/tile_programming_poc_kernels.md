# Luisa Tile DSL: Executable Kernel Gallery

These examples are the actual C++ test definitions, included directly from the
test sources so the documented syntax cannot drift into a second DSL.
All operator POCs use subtiles and Tile SSA. Scalar memory-access tests are
separate, explicitly low-level tests.

The native TVMx bridge provides a correctness/reference schedule plus guarded
optimizations. An `mma` remains an MMA operation in TileIR; its reference TIRx
realization is a contraction loop. With an explicit device-capability opt-in,
eligible FP32 Metal group operations select native SIMD-group matrix atoms.
This is not a performance-parity claim. Safe pipeline cuts can use two-window
software prefetching; hardware-asynchronous transfers and warp specialization
remain future work.

## 1. One memory-access convention

| Expression | Result / effect |
|---|---|
| `A(origin, shape)` | `MemoryRef`; no read |
| `A.tile(origin, shape)` | The same reference |
| `A[origin, shape]` | `Tile<T>`; one load |
| `A(origin, shape).load()` | The same Tile load |
| `C(origin, shape).store(value)` | Explicit write |
| `value = expression` | A new value definition, never a memory write |

`origin` is `coord(...)`. `shape(...)` accepts extents or kernel-local
`axis("diagnostic_name", extent)` values. Shared axis identities express
broadcasting and contraction; no axis name is predefined by the language.

The default `bounds::zero` fills out-of-bounds reads with zero and masks
out-of-bounds stores. `bounds::assume` is an explicit in-bounds caller contract.
For another fill value, use `A(origin, shape).load(fallback)`.

C++23 uses native multi-argument `operator[]`. In C++20 the comma operator,
overloaded only for DSL coordinates plus a shape and optional bounds policy,
forms one typed selection argument. Both paths call the identical load builder:

~~~cpp
auto x = A[coord(m0, k0), shape(m, k)];
auto y = B[coord(k0, n0), shape(k, n), bounds::zero];
C(coord(m0, n0), shape(m, n)).store(mma(x, y, acc));
~~~

Some compilers diagnose the pre-C++23 comma-subscript grammar as deprecated
even when the comma is overloaded. With warning-as-error builds, suppress only
that diagnostic in the C++20 client. No global comma overload or header-wide
diagnostic suppression is installed.

The project minimum remains C++20. CMake adds the C++23 companion test when
the toolchain supports that language level; XMake exposes it as the opt-in
`test_tile_values_cpp23` target.

## 2. Execution-first GEMM

The outer range enumerates logical output programs; the inner pipeline
enumerates K tiles. The shape of `a`, `b`, or `acc` does not create an
execution or memory hierarchy. Assignment to `acc` captures loop-carried SSA
state without `loop.result()`.

An explicit `outer.index(...)` resolves within that Nest and its ancestors,
even when written inside a child. Reusing positional axes in a nested shape
does not make the child silently replace the parent's coordinate.
Use `nest.index(axis)` for a named logical coordinate, or `nest.index()` for a
rank-one domain. `Nest` has no subscript overload; `A[origin, shape]` is only
the Tile-load syntax for tensor views.

Value selection uses `ite(condition, true_value, false_value)`, matching the
SIMT DSL's `ite`. Both Scalar and Tile operands use this order; the Tile DSL
does not expose the differently ordered `select` name.

```{literalinclude} tile_programming_poc.cpp
:language: cpp
:start-at: struct GemmConfig
:end-before: // Read equivalence
```

~~~text
parallel(output blocks)
  ├─ acc: Tile<M,N>
  └─ pipeline(K tiles)
       ├─ load:    A[M,K], B[K,N] → Tile values
       ├─ compute: mma(A, B, acc) → next acc
       └─ inferred carry         → next K iteration
  └─ C(origin, shape).store(acc)
~~~

### 2.1 Optional explicit Memory

The execution structure is unchanged. Two sibling resources are declared in
the output program and reused by the pipeline. `store` consumes a Tile and
`load` produces a snapshot; assigning a Tile never means storing to Memory.
The frontend infers each resource's loop-carried MemoryState alongside `acc`.
The example uses padded rows for As and column-major local storage for Bs;
their logical MMA operand shapes remain unchanged.

```{literalinclude} tile_programming_poc.cpp
:language: cpp
:start-at: // Explicit Memory is optional
:end-before: }// namespace
```

For the current native reference mapper, choose `exec::Scope::GROUP` with
`mem::shared` on Metal, or `exec::Scope::WORKER` with `mem::private_` on CPU.
Leaving the resource unspecified lets the mapper choose it. A resource
constraint does not change ownership; unsupported placements fail explicitly.
Use `memory<T>(layout(shape, stride(...)), resource)` for ordinary strided
storage, or pass a composed `IndexMap` for a custom address map. A whole-Tile
Memory layout must be total, in bounds, and injective. `IndexMap::prove()`
proves ordinary affine and structurally bit-linear layouts without enumerating
their elements. Gray-code/XOR swizzles, bit transposes, constant masks, and
power-of-two digit packing use exact GF(2) rank and image bounds. Remaining
maps use an exhaustive fallback of up to 1,048,576 logical points; an unknown
proof fails native realization, not structural representation. Range-aware
parallel writes and asynchronous pipeline versions are not implemented yet.

The native bridge does implement synchronous software prefetching across safe
stage cuts, with two versions for cross-phase iteration-local temporaries.
An outer mutable Memory resource is not silently renamed into an independent
per-iteration resource: cross-phase dependencies keep its execution ordered.
`test_tile_tirx_pipeline` exercises real versioned CPU/Metal execution, short
and ragged loops, multiple carried values, possible input/output aliasing,
late stores, and shared-capacity fallback. Hardware-asynchronous transfers and
warp specialization remain separate work; see design section 7.3.

`test_tile_tirx_memory` runs ragged manual GEMMs, multiple independently updated
resources, old-value snapshots, nested temporal regions, ancestor reads,
worker-private state, and resource/capacity rejection on CPU and Metal. It also
checks padded/transposed/composed-XOR layouts and logical shifts through the
sign bit, including physical allocation shapes, padding capacity, and empty
domains. LLVM also tests vector-private state and 513-by-2047 Memory snapshots
with identity, padded, and transposed layouts, above the enumeration budget.
The same large layouts export on Metal but correctly exceed its threadgroup
storage capacity; this is a resource constraint, not a failed layout proof.
Gray-code and digit-transpose Memory POCs exercise both CPU and Metal for
sizes 1, 31, 32, 257, and 1024, plus CPU execution at 2,097,152 elements. The
large case again exports on Metal and rejects its infeasible shared allocation.
Native index-expression tests compare compiled address calculations with the
TileIR evaluator. The canonical example above is independently captured
under both C++20 and C++23.

### 2.2 Optional matrix atom selection

The kernel syntax is unchanged. The native compiler's `cooperative_matrix`
option asserts device support; Metal additionally requires
`thread_warp_size=32` (Apple GPU family 7+). This is not inferred from a generic
Metal target. Eligible rank-two FP32 group MMA uses 8×8 SIMD-group matrix
instructions; other types, scopes, layouts, and shapes retain the reference
loops. Global ragged edges and transposed loaded Tiles are supported.
`mma(a, b, acc, {.allow_reassociation = false})` also keeps the ordered
contraction; the default permits reassociation without reducing input precision.

`test_tile_tirx_matrix` verifies both emitted Metal instructions and numerical
results against a double-precision oracle. The native/PyTorch benchmark records
the capability request separately from generated matrix call counts, so a
fallback is not reported as a matrix-hardware measurement.

## 3. Elementwise bias + GELU + residual

Two-dimensional blocks load once. The one-dimensional bias broadcasts by its
shared column axis; neither loads nor stores are hidden in assignment.

```{literalinclude} ../../src/tests/unit/tile/bridge/test_tirx_poc_neural.cpp
:language: cpp
:start-after: void test_bias_gelu_residual(Runtime &runtime) {
:end-before:     auto kernel = definition.capture(
```

## 4. Statistics and losses

Sum, mean, maximum, stable argmax, MSE, MAE, Huber, and binary cross entropy
compose the same Tile operations. `reduce(value, axis, add)` is a library
expression; it expands into an element region and the existing reduction nest.
The region form still yields a Nest, not a different kind of iterator element.

```{literalinclude} ../../src/tests/unit/tile/bridge/test_tirx_poc.cpp
:language: cpp
:start-after: void test_row_statistics_and_losses(Runtime &runtime) {
:end-before:     auto kernel = definition.capture(
```

A whole-tensor reduction can use one logical program; larger reductions can
write partial Tiles and invoke a second kernel. No implicit grid barrier is
invented. The current small whole-tensor test uses the former.

## 5. Softmax, LayerNorm, and RMSNorm

```{literalinclude} ../../src/tests/unit/tile/bridge/test_tirx_poc.cpp
:language: cpp
:start-after: void test_softmax_layernorm_rmsnorm(Runtime &runtime) {
:end-before:     auto kernel = definition.capture(
```

The LayerNorm POC uses the simple mean-square formulation to exercise Tile
broadcasting. Numerically robust Welford state is a library extension, not a
claim that this reference formula is suitable for every input distribution.
The same concise row-sum, softmax, RMSNorm and LayerNorm shapes now have an
opt-in proof-driven Metal SIMD-group realization without changing their DSL
source; its formal mapping, storage proof and measured limits are in
[TIRx Metal reductions](tile_tirx_reduction_report.md).

## 6. Sparse cross entropy and gradient

`gather` semantically indexes an already-loaded Tile. Its portable reference
expansion uses pure Tile extraction. A target realization may replace an
immutable input snapshot with a guarded direct Tensor read only after proving
source immutability, path bounds and distributed worker ownership; this is an
optimization, not a different source operation.

```{literalinclude} ../../src/tests/unit/tile/bridge/test_tirx_poc_neural.cpp
:language: cpp
:start-after: void test_sparse_softmax_cross_entropy(Runtime &runtime) {
:end-before:     auto kernel = definition.capture(
```

## 7. Causal FlashAttention with online softmax

Q is loaded once per query block. Each pipeline iteration loads a K/V block,
forms scores with one MMA, rescales the carried row state, and applies the
second MMA. Query/key and feature dimensions are explicit local identities.
The causal mask and key-tail mask are separate from memory bounds.

```{literalinclude} ../../src/tests/unit/tile/bridge/test_tirx_poc_neural.cpp
:language: cpp
:start-after: void test_flash_attention_online_softmax(Runtime &runtime) {
:end-before:     auto kernel = definition.capture(
```

## 8. CNN: padded, strided convolution

The input access names a complete receptive-field window. `reindex` maps
logical filter taps to that loaded window; multiplying by a weight Tile and
reducing the tap/input-channel axes is the direct-convolution library form.
A later im2col/MMA schedule need not change the memory-access syntax.

```{literalinclude} ../../src/tests/unit/tile/bridge/test_tirx_poc.cpp
:language: cpp
:start-after: void test_padded_strided_conv2d(Runtime &runtime) {
:end-before:     auto kernel = definition.capture(
```

## 9. Depthwise convolution and pooling

```{literalinclude} ../../src/tests/unit/tile/bridge/test_tirx_poc_algorithms.cpp
:language: cpp
:start-after: void test_depthwise_convolution_and_max_pool(Runtime &runtime) {
:end-before:     auto kernel = definition.capture(
```

This POC deliberately uses zero padding for both outputs. A conventional
negative-infinity-padded max pool would use an explicit fallback on its load.

## 10. Traditional filters: Sobel and median

```{literalinclude} ../../src/tests/unit/tile/bridge/test_tirx_poc_algorithms.cpp
:language: cpp
:start-after: void test_sobel_and_ordered_median(Runtime &runtime) {
:end-before:     auto kernel = definition.capture(
```

The small-window median uses the same sorting library as Top-K. There is no
median-specific primitive or execution scope.

## 11. Stable sort and Top-K

The `topk` library returns Tile values and source indices. Its current
quadratic composition orders finite values and breaks ties by source index.
A tuned network, radix sort, and a documented NaN policy remain future library
work; these are not hidden behind a claim of hardware-optimized sorting.

```{literalinclude} ../../src/tests/unit/tile/bridge/test_tirx_poc_algorithms.cpp
:language: cpp
:start-after: void test_stable_sort_and_topk(Runtime &runtime) {
:end-before:     auto kernel = definition.capture(
```

## 12. Segmented accumulation

This reference uses one program per bucket and a masked Tile reduction,
including empty buckets. It is deterministic and needs no atomics. An atomic
scatter implementation is a different schedule/library choice.

```{literalinclude} ../../src/tests/unit/tile/bridge/test_tirx_poc_algorithms.cpp
:language: cpp
:start-after: void test_segmented_accumulation(Runtime &runtime) {
:end-before:     auto kernel = definition.capture(
```

## 13. Nested temporal composition

This test combines parallel → serial → pipeline → reduction. A single outer
Tile variable is assigned inside the nested regions, exercising inferred
state at each lexical boundary.

```{literalinclude} ../../src/tests/unit/tile/bridge/test_tirx_poc_algorithms.cpp
:language: cpp
:start-after: void test_all_structured_regions(Runtime &runtime) {
:end-before:     auto kernel = definition.capture(
```

## 14. Validation boundary

- The same subscript syntax has compile/capture tests in C++20 and C++23.
- Every operator POC is registered for native TVMx CPU and, when available,
  actual Metal execution. Metal failures do not silently fall back to CPU.
- POC validation rejects scalar View loads/stores. Fine-grained scalar access
  remains available only through explicit low-level reference tests.
- Numerical checks use independent host reference formulas, including tail
  tiles, boundary padding, masks, and stable ties.
- Performance comparisons must use multiple shapes and report compile,
  transfer, cold call, and synchronized warm execution separately. Correctness
  tests do not establish tensor-core lowering or competitive performance.
