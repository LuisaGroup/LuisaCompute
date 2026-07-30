---
name: ir_pipeline
description: Legacy IR and XIR compiler pipeline, AST lowering, SSA IR, and optimization passes.
---

# IR and XIR Pipeline

Two IRs, both starting from AST (`src/ast/`), feeding into backend codegen:

| | IR (Legacy) | XIR (Preferred) |
|---|---|---|
| **Location** | `src/ir/`, `include/luisa/ir/` | `src/xir/`, `include/luisa/xir/` |
| **Impl** | Rust (`src/rust/`) | Pure C++ |
| **Serialization** | `ast2json` → Rust IR | `xir2json`/`json2xir` (yyjson) |
| **SSA** | Yes | Yes (mem2reg) |
| **Status** | Maintained (compat) | Active development |
| **Basic Blocks** | Yes | Yes |

## Pipeline Flow

```
DSL Tracing (src/dsl/) → AST (src/ast/)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              XIR (ast2xir)       IR (ast2ir → JSON → Rust FFI)
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    Backend Codegen (src/backends/<name>/)
                              │
                              ▼
                    GPU Execution (src/runtime/)
```

**Rust IR path**: `src/rust/luisa_compute_ir/` does autodiff, DCE, SSA, vectorize.
**XIR path**: Pure C++ with `ast2xir` translator, `xir2ast` round-trip, and optimization passes.

## AST → IR Translation (Legacy)

**File**: `src/ir/ast2ir.cpp`

```
AST Function → to_json() → JSON string → Rust FFI → CArc<KernelModule/CallableModule>
```

FFI: `luisa_compute_ir_ast_json_to_ir_kernel()`, `..._callable()`, `..._type()`.

Key IR classes (Rust, via C FFI): `KernelModule`, `CallableModule`, `Node`, `Instruction` (Local, Call, Phi, Loop, If, Switch, RayQuery, AdScope), `Type`, `BasicBlock`.

## AST → XIR Translation

**Files**: `src/xir/translators/ast2xir.cpp`, `xir2ast.cpp`. `AST2XIRContext` maps AST variables to XIR Values, tracks break/continue targets, handles autodiff adjoints, caches constants.

### Expression Mapping
| AST | XIR |
|---|---|
| `UnaryExpr` | `ArithmeticOp` (UNARY_MINUS, UNARY_BIT_NOT); `+x` is elided |
| `BinaryExpr` | `ArithmeticOp` (BINARY_ADD, etc.); matrix-aware; logic ops cast to bool |
| `MemberExpr` | `EXTRACT`/`SHUFFLE` or `GEP` |
| `AccessExpr` | `GEP` + `LOAD` |
| `LiteralExpr` | `Constant` |
| `ConstantExpr` | `Constant` |
| `RefExpr` | Variable lookup or `SpecialRegister` |
| `CallExpr` | `CallInst`, `ArithmeticOp`, `AtomicOp`, `Resource*Op`, `ThreadGroupOp`, `RayQuery*Op`, `Assert`/`Assume`/`Unreachable`/`RasterDiscard` |
| `CastExpr` | `CastInst` (STATIC_CAST, BITWISE_CAST) |
| `TypeIdExpr`/`StringIdExpr`/`FuncRefExpr` | Not implemented |

### Statement Mapping
| AST | XIR |
|---|---|
| `IfStmt` | `IfInst` + true/false/merge |
| `SwitchStmt` | `SwitchInst` + case/default/merge |
| `ForStmt` | `LoopInst` (prepare/body/update/merge) |
| `LoopStmt` | `SimpleLoopInst` (do-while) |
| `BreakStmt` | `BreakInst` |
| `ContinueStmt` | `ContinueInst` |
| `ReturnStmt` | `ReturnInst` |
| `AssignStmt` | `StoreInst` |
| `ExprStmt` | Expression (terminator-aware, e.g. `Unreachable`) |
| `AutoDiffStmt` | `AutodiffScopeInst` |
| `RayQueryStmt` | `RayQueryLoopInst` + `RayQueryDispatchInst` |
| `PrintStmt` | `PrintInst` |
| `DebugBreakStmt` | `DebugBreakInst` |
| `CommentStmt` | Collected as comment metadata on following instruction |

## XIR Core Architecture

### Value Hierarchy
```
Value
├── GlobalValue
│   ├── Function → FunctionDefinition → KernelFunction (entry+block size), CallableFunction, ExternalFunction
│   ├── Constant (literals)
│   ├── Undefined
│   └── SpecialRegister (SPR_ThreadID, SPR_BlockID, SPR_DispatchID, SPR_WarpLaneID, SPR_KernelID, SPR_BlockSize, SPR_WarpSize, SPR_DispatchSize, SPR_ObjectID, SPR_Barycentrics, ...)
├── FunctionScopeValue
│   ├── BasicBlock (instruction container)
│   └── Argument → ValueArgument / ReferenceArgument / ResourceArgument
└── BlockScopeValue → Instruction
    ├── TerminatorInstruction: BranchInst, ConditionalBranchInst, IfInst, SwitchInst,
    │   LoopInst, SimpleLoopInst, ReturnInst, BreakInst, ContinueInst, UnreachableInst, RasterDiscardInst
    └── Non-terminator instructions
```

### Key Classes
- **Module** (`include/luisa/xir/module.h`): container for globals, unique constants via hash
- **Function / FunctionDefinition** (`include/luisa/xir/function.h`): `ArgumentList`, `BasicBlockList`, `body_block()`, traversal orders (`PRE_ORDER`, `POST_ORDER`, ...). Traversal: `traverse_basic_blocks()`, `traverse_instructions()`
- **Argument** (`include/luisa/xir/argument.h`): `ValueArgument`, `ReferenceArgument`, `ResourceArgument`
- **BasicBlock** (`include/luisa/xir/basic_block.h`): `InstructionList`, `is_terminated()`, `terminator()`, `traverse_predecessors()`, `traverse_successors()`
- **Instruction** (`include/luisa/xir/instruction.h`): `DerivedInstruction<>`, `is_terminator()`, `control_flow_merge()`, `clone()`, `intrinsic_identifier()`
- **Value & Use** (`include/luisa/xir/value.h`, `use.h`): SSA `UseList`, `replace_all_uses_with()`, `is_lvalue()` for alloca/gep/reference args

## XIR Instruction Set

### Control Flow
| Instruction | Blocks |
|---|---|
| `IfInst` | true, false, merge |
| `SwitchInst` | cases, default, merge |
| `LoopInst` | prepare, body, update, merge |
| `SimpleLoopInst` | body, merge |
| `BranchInst` | target |
| `ConditionalBranchInst` | true, false |
| `ReturnInst` | — |
| `BreakInst` / `ContinueInst` | target (lowered before codegen) |
| `UnreachableInst` | — |

### Memory
`AllocaInst` (LOCAL/SHARED), `LoadInst`, `StoreInst`, `GEPInst`

### SSA
`PhiInst` — (block, value) pairs

### Call / Cast
`CallInst` (user/external functions), `CastInst` (`STATIC_CAST`, `BITWISE_CAST`)

### Arithmetic (`ArithmeticOp`)
- **Unary**: UNARY_MINUS, UNARY_BIT_NOT
- **Binary**: ADD, SUB, MUL, DIV, MOD, BIT_AND, BIT_OR, BIT_XOR, SHIFT_LEFT/RIGHT, ROTATE_LEFT/RIGHT, comparisons
- **Logic/Selection**: ALL, ANY, SELECT, STEP
- **Math**: ABS, MIN, MAX, CLAMP, SATURATE, LERP, SMOOTHSTEP, trig (SIN/COS/TAN/ASIN/ACOS/ATAN/ATAN2 and hyperbolic variants), exp/log families (EXP/EXP2/EXP10/LOG/LOG2/LOG10), POW, SQRT, RSQRT, FMA, COPYSIGN, CLZ, CTZ, POPCOUNT, REVERSE, ISINF, ISNAN, CEIL, FLOOR, FRACT, TRUNC, ROUND, RINT
- **Vector**: DOT, CROSS, LENGTH, LENGTH_SQUARED, NORMALIZE, FACEFORWARD, REFLECT, REDUCE_SUM/PRODUCT/MIN/MAX, OUTER_PRODUCT
- **Matrix**: MATRIX_COMP_NEG/ADD/SUB/MUL/DIV, MATRIX_LINALG_MUL, MATRIX_DETERMINANT, MATRIX_TRANSPOSE, MATRIX_INVERSE
- **Aggregate**: AGGREGATE, SHUFFLE, EXTRACT, INSERT

### Resource
`ResourceQueryOp`, `ResourceReadOp`, `ResourceWriteOp` — buffer/texture/bindless ops, ray-tracing queries, indirect dispatch, device-address loads/stores

### Atomic (`AtomicOp`)
EXCHANGE, COMPARE_EXCHANGE, FETCH_ADD/SUB/AND/OR/XOR/MIN/MAX

### Thread Group & Ray Query & Autodiff
`ThreadGroupOp` (warp, sync, SER, quad derivatives), `RayQueryLoopInst`, `RayQueryDispatchInst`, `RayQueryObjectReadInst`, `RayQueryObjectWriteInst`, `RayQueryPipelineInst`, `AutodiffScopeInst`, `AutodiffIntrinsicInst` (requires_gradient, gradient, gradient_marker, accumulate_gradient, backward, detach)

### Debug / Utility
`PrintInst`, `ClockInst`, `DebugBreakInst`, `AssertInst`, `AssumeInst`, `OutlineInst`, `RasterDiscardInst`

## XIR Optimization Passes

**Location**: `src/xir/passes/` (headers in `include/luisa/xir/passes/`)

### Core / SSA / CFG
| Pass | File | Purpose |
|---|---|---|
| DCE | `dce.cpp` | Dead instructions, unreachable blocks, dead allocas, static branch eval |
| Mem2Reg | `mem2reg.cpp` | Alloca→SSA via dominance tree/frontiers, PHI insertion |
| Dominance Tree | `dom_tree.cpp` | Immediate dominators + frontiers |
| Post-Dominance Tree | `post_dom_tree.cpp` | Post-dominance analysis |
| Early Return Elim | `early_return_elimination.cpp` | Early returns → structured control flow |
| Lower Break/Continue | `lower_break_continue.cpp` | Lower break/continue to explicit branches |
| Lower Ray Query Loop | `lower_ray_query_loop.cpp` | Ray query loop lowering |
| Lower Ray Query Loop → Loop | `lower_ray_query_loop_to_loop.cpp` | Convert ray query loops to plain loops |
| Destructure CFG | `destructure_cfg.cpp` | Flatten structured CFG to basic branches |
| Restructure CFG | `restructure_cfg.cpp` | Recover structured control flow |
| Outline | `outline.cpp` | Function outlining |
| Phi Cleanup | `phi_cleanup.cpp` | Remove trivial/duplicate PHI nodes |
| Fix Self-Referential | `fix_self_referential.cpp` | Break self-referential PHI/value cycles |
| If Conversion | `if_conversion.cpp` | Convert simple diamonds to select/min/max |

### Analysis
| Pass | File | Purpose |
|---|---|---|
| Call Graph | `call_graph.cpp` | Call-graph construction |
| Pointer Usage | `pointer_usage.cpp` | Per-field kill/touch/live analysis for pointers |
| Lexical Scope | `lex_scope_analysis.cpp` | Scope region analysis |
| Aggregate Field Bitmask | `aggregate_field_bitmask.cpp` | Bitmask tracking for aggregate fields |
| Alias Analysis | `alias_analysis.cpp` | May-/must-alias queries for memory instructions |
| Convergence Region | `convergence_region.cpp` | Divergence/convergence region analysis |
| CVP | `cvp.cpp` | Correlated value propagation via structured branches |
| Scalar Evolution | `scalar_evolution.cpp` | SCEV for loop induction variables |
| Uniformity Analysis | `uniformity_analysis.cpp` | Uniform/divergent value analysis |

### Scalar / Peephole / Global
| Pass | File | Purpose |
|---|---|---|
| Algebraic Simplify | `algebraic_simplify.cpp` | Algebraic identities (with optional fast-math) |
| Const Fold | `const_fold.cpp` | Constant folding |
| Early CSE | `early_cse.cpp` | Early common subexpression elimination |
| GVN | `gvn.cpp` | Global value numbering |
| Reassociate | `reassociate.cpp` | Reassociate expressions for CSE/folding |
| SCCP | `sccp.cpp` | Sparse conditional constant propagation |
| Simplify CFG | `simplify_cfg.cpp` | Remove empty/trivial blocks |
| Simplify Libcalls | `simplify_libcalls.cpp` | Simplify known builtin calls |
| Scalarizer | `scalarizer.cpp` | Break vector ops into scalar ops |
| Div-Rem Pairs | `div_rem_pairs.cpp` | Combine division/remainder pairs |
| Dead Arg Elim | `dead_arg_elim.cpp` | Remove unused arguments |

### Loop
| Pass | File | Purpose |
|---|---|---|
| IndVar Simplify | `indvar_simplify.cpp` | Simplify induction variables |
| LICM | `licm.cpp` | Loop-invariant code motion |
| Loop Rotation | `loop_rotation.cpp` | Rotate loops for simpler CFG |

### Memory / Local
| Pass | File | Purpose |
|---|---|---|
| SROA | `sroa.cpp` | Scalar replacement of aggregates |
| Reg2Mem | `reg2mem.cpp` | Register → memory conversion |
| Promote Ref Arg | `promote_ref_arg.cpp` | Reference argument promotion |
| Transpose GEP | `transpose_gep.cpp` | Transpose GEP through loads/stores |
| Trace GEP | `trace_gep.cpp` | GEP analysis & tracing |
| Local Load Elimination | `local_load_elimination.cpp` | Redundant load elimination |
| Local Store Forward | `local_store_forward.cpp` | Store-to-load forwarding |
| Dead Store Elimination | `dead_store_elimination.cpp` | Remove dead stores |

### AD / Interprocedural
| Pass | File | Purpose |
|---|---|---|
| Autodiff | `autodiff.cpp` | Autodiff transformations |
| Inline | `inline.cpp` | Function inlining |
| Unused Callable Removal | `unused_callable_removal.cpp` | Dead function elimination |

### Pass Pipeline Helpers
`pass_pipeline.cpp` / `include/luisa/xir/passes/pass_pipeline.h` provides `PassPipeline`, `PassReport`, and factory functions:
- `create_basic_optimization_pipeline()`
- `create_post_inline_cleanup_pipeline()`
- `create_ssa_optimization_pipeline()`
- `create_post_restructure_cleanup_pipeline()`

## Control Flow Representation

XIR uses **structured control flow** with explicit merge blocks:

```
IfInst:   condition, true_block, false_block, merge_block
LoopInst: prepare_block, body_block, update_block, merge_block
```

Design: maintains SSA, enables structured transforms, maps well to GPU shaders,
and supports PHI nodes at merges. `SwitchInst` is a first-class structured
terminator. Only the explicit `destructure_cfg` boundary maps it to raw
`IndexedBranchInst`; `restructure_cfg` reconstructs the switch and its merge.
There is no generic switch-lowering or XIR loop-unroll pass. Autodiff's private
bounded semantic expansion and SPIRV-Tools loop unrolling are separate
mechanisms with separate contracts.

## Metadata

**Headers**: `include/luisa/xir/metadata.h` plus `include/luisa/xir/metadata/{name,location,comment,curve_basis}.h`. Types: `NAME`, `LOCATION`, `COMMENT`, `CURVE_BASIS`. Applied via `MetadataListMixin`.

## JSON Serialization / Translators

- **IR (legacy)**: `ast2json` → Rust IR, C API `luisa_compute_ir_ast_json_to_ir_*`
- **XIR**: `src/xir/translators/xir2json.cpp`, `json2xir.cpp`, `xir2ast.cpp`, `xir2text.cpp` — yyjson-based module serialization and AST round-tripping, useful for cross-process transport and debugging

## Key Design Patterns

1. **Intrusive Lists** — `ManagedIntrusiveList` for node management
2. **CRTP** — `DerivedValue<>`, `DerivedInstruction<>`, `DerivedFunction<>`, `DerivedArgument<>`
3. **Visitor** — `traverse_basic_blocks()`, `traverse_instructions()`
4. **Builder** — `XIRBuilder` for instruction creation
5. **Mixin** — `MetadataListMixin`, `ControlFlowMergeMixin`, `InstructionOpMixin`, `PrintMessageMixin`
6. **Use-Def Chains** — `Use` objects track value users for SSA

## Adding a New XIR Pass

1. Create `src/xir/passes/<name>.cpp` + header `include/luisa/xir/passes/<name>.h`
2. Register in `src/xir/CMakeLists.txt`
3. Convention: accept `Module &` or `Function &`, return an info struct (often with `*_pass_run_on_module(Module *, PassReport *report = nullptr)`), use `XIRBuilder`, call `replace_all_uses_with()` for substitution
