# Tile architecture decisions

## Architecture decision ledger

This table is the compact answer to the design discussion. Detailed syntax and
proofs live in the [language reference](../../tile/design.md); the
[compiler index](index.md) and [implementation coverage](../../performance/tile/implementation.md)
separate realized behavior from the remaining design.

| Question | Decision | Consequence |
|---|---|---|
| Programming model | Execution structure first, not an algorithm graph with a schedule attached afterwards | Lexical Nest structure exists before target mapping; operations inherit anchor/frontier from it |
| Core Nest vocabulary | `parallel`, `serial`, `pipeline`, `reduce` only | Elementwise, convolution, softmax, attention, Top-K and sort remain composable libraries unless irreducible target semantics justify an atom |
| C++ surface | Luisa-style staged lambda parameters, range-for Nests, ordinary carried assignment, explicit memory effects | No `GemmSpec`, public builder prefix, symbolic-integer façade, `loop.result()` or kernel-specific `mma_team` entity |
| Tensor and view | Tensor is storage plus a typed layout/view; `A[...]` loads a Tile, `A(...)` names a `MemoryRef` | Subtiles and bounds are explicit without baking execution or a memory hierarchy into Tensor |
| Layout algebra | One typed mixed-radix/index-map composition algebra for execution binding, value distribution, views, addresses and atom operands | Domain/codomain and proof obligations prevent composing unrelated coordinate spaces; representability is broader than any one emitter |
| Execution versus memory | Execution hierarchy chooses participants; resources attach independently to an owner prefix and access map | Several differently laid-out memories may serve one Nest; memory kinds are capabilities, not a fake total hierarchy |
| Pipeline | A temporal producer/consumer Nest with lexical stage cuts and dependence distances | It may organize participant specialization, overlap and versions, but a stage name alone does not promise async hardware |
| Reduction | An algebraic Nest with domain, grouping, identity/update/merge and numerical policy | Serial fold, subgroup tree, Welford and tuple states are realizations of one semantic contract, not unrelated source constructs |
| Tile SSA versus `Memory` | Preserve semantic sharing; plan retain/recompute/materialize per target. Manual `Memory` means stable addressable identity | Compiler stripes/registers/workspace do not leak into ordinary kernels; manual writes always use `.store()` |
| TileIR | Thin, typed, mutable SSA/region IR with managed intrusive ownership/use lists and analyses | It is transformable like XIR/LLVM, not a SPIR-V-style serialization schema and not an MLIR dependency |
| Backend boundary | Public `tile::compile(device, TileIR)` calls the optional backend `DeviceInterface::create_tile_kernel` factory, which selects native lowering or `tile/bridge/{tirx,xir}` | TIRx and XIR remain comparable bootstrap paths while Metal/CUDA/CPU keep target-specific bindings and atoms |
| Planning | Solve binding `B`, distribution `D`, atom `A`, resources `R` and schedule `Theta` under hard proofs, then rank | Enumeration/Pareto DP are implemented for bounded families; MILP, CP-SAT, beam or annealing are optional search engines, never legality oracles |
| Autotuning | Ordinary concrete host configurations are recaptured and JIT-compiled as a finite product | No capture-once super-kernel is required; every candidate and the fresh winner receive the full correctness oracle |
| Machine TileIR | Add it only when several backends/passes need a common scheduled atom/resource/protocol form | Current bridge-local plans remain honest stepping stones; no premature backend instruction serialization |

“Layout completeness” therefore has three separate meanings. The algebra is
closed over the admitted typed finite maps and can embed the CuTe-style
mixed-radix constructions used here; proof procedures intentionally return
unknown outside their decidable fragments; emitters support smaller target
subsets and fail closed. A complete representation never licenses an
unsupported lowering.

## Original design checklist

The following preserves the revision-17 design rationale and bootstrap order. It is historical design material, **not the current task list or a completion checklist**. Use [implementation coverage and remaining work](../../performance/tile/implementation.md) for current status.

## Minimal implementation plan

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

## Final decisions

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
  through the [execution-to-memory composition](../../tile/memory.md#the-execution-to-memory-equation).
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

The accompanying [GEMM sketch](../../tile/tile_programming_poc.cpp) and
[kernel gallery](../../tile/kernels.md) exercise the proposed syntax.
