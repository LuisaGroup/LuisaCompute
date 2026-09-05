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
