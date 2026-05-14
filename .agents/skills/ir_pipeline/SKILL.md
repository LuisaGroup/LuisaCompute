---
name: ir_pipeline
description: IR and XIR compiler pipeline, AST translation, SSA-based IR, instruction set, optimization passes, and control flow representation in LuisaCompute
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
**XIR path**: Pure C++ with `ast2xir` translator and optimization passes.

## AST → IR Translation (Legacy)

**File**: `src/ir/ast2ir.cpp`

```
AST Function → to_json() → JSON string → Rust FFI → CArc<KernelModule/CallableModule>
```

FFI: `luisa_compute_ir_ast_json_to_ir_kernel()`, `..._callable()`, `..._type()`.

Key IR classes (Rust, via C FFI): `KernelModule`, `CallableModule`, `Node`, `Instruction` (Local, Call, Phi, Loop, If, Switch, RayQuery, AdScope), `Type`, `BasicBlock`.

## AST → XIR Translation

**File**: `src/xir/translators/ast2xir.cpp`. `AST2XIRContext` maps AST variables to XIR Values, tracks break/continue targets, handles autodiff adjoints, caches constants.

### Expression Mapping
| AST | XIR |
|---|---|
| `UnaryExpr` | `ArithmeticOp` (UNARY_MINUS, UNARY_BIT_NOT) |
| `BinaryExpr` | `ArithmeticOp` (BINARY_ADD, etc.) |
| `MemberExpr` | `EXTRACT`/`SHUFFLE` or `GEP` |
| `AccessExpr` | `GEP` + `LOAD` |
| `LiteralExpr` | `Constant` |
| `RefExpr` | Variable lookup or `SpecialRegister` |
| `CallExpr` | Various opcodes from `CallOp` |
| `CastExpr` | `CastInst` (STATIC_CAST, BITWISE_CAST) |

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
| `AutoDiffStmt` | `AutodiffScopeInst` |

## XIR Core Architecture

### Value Hierarchy
```
Value
├── GlobalValue
│   ├── Function → KernelFunction (entry+block size), CallableFunction, ExternalFunction
│   ├── Constant (literals)
│   ├── Undefined
│   └── SpecialRegister (SPR_ThreadID, SPR_BlockID, SPR_DispatchID, ...)
├── FunctionScopeValue → BasicBlock (instruction container)
└── BlockScopeValue → Instruction
    ├── TerminatorInstruction: BranchInst, ConditionalBranchInst, IfInst, SwitchInst,
    │   LoopInst, ReturnInst
    └── Non-terminator instructions
```

### Key Classes
- **Module** (`include/luisa/xir/module.h`): container for globals, unique constants via hash
- **Function** (`include/luisa/xir/function.h`): `ArgumentList`, `BasicBlockList`, `FunctionDefinition`. Traversal: `traverse_basic_blocks()`, `traverse_instructions()`
- **BasicBlock** (`include/luisa/xir/basic_block.h`): `InstructionList`, `is_terminated()`, `terminator()`, `traverse_predecessors()`, `traverse_successors()`
- **Instruction** (`include/luisa/xir/instruction.h`): `DerivedInstruction<>`, `is_terminator()`, `control_flow_merge()`, `clone()`
- **Value & Use** (`include/luisa/xir/value.h`, `use.h`): SSA `UseList`, `replace_all_uses_with()`, `is_lvalue()` for alloca/gep

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
| `BreakInst` / `ContinueInst` | target |
| `UnreachableInst` | — |

### Memory
`AllocaInst` (LOCAL/SHARED), `LoadInst`, `StoreInst`, `GEPInst`

### SSA
`PhiInst` — (block, value) pairs

### Arithmetic (`ArithmeticOp`)
- **Unary**: UNARY_MINUS, UNARY_BIT_NOT
- **Binary**: ADD, SUB, MUL, DIV, MOD, BIT_AND, BIT_OR, BIT_XOR, SHIFT_LEFT/RIGHT, ROTATE_LEFT/RIGHT, comparisons
- **Math**: ABS, MIN, MAX, CLAMP, SATURATE, LERP, SMOOTHSTEP, trig, exp, log
- **Vector/Matrix**: DOT, CROSS, NORMALIZE, MATRIX_COMP_MUL, MATRIX_LINALG_MUL, MATRIX_DETERMINANT, MATRIX_TRANSPOSE, MATRIX_INVERSE
- **Aggregate**: AGGREGATE, SHUFFLE, EXTRACT, INSERT

### Resource
`ResourceQueryOp`, `ResourceReadOp`, `ResourceWriteOp` — buffer/texture/bindless ops, ray tracing queries

### Atomic (`AtomicOp`)
EXCHANGE, COMPARE_EXCHANGE, FETCH_ADD/SUB/AND/OR/XOR/MIN/MAX

### Thread Group & Ray Query & Autodiff
`ThreadGroupOp` (warp, sync, SER, quad derivatives), `RayQueryLoopInst`, `RayQueryDispatchInst`, `RayQueryObjectReadInst`, `RayQueryObjectWriteInst`, `RayQueryPipelineInst`, `AutodiffScopeInst`, `AutodiffIntrinsicInst` (requires_gradient, gradient, accumulate_gradient, backward, detach)

## XIR Optimization Passes

**Location**: `src/xir/passes/`

### Core
| Pass | File | Purpose |
|---|---|---|
| DCE | `dce.cpp` | Dead instructions, unreachable blocks, dead allocas, static branch eval |
| Mem2Reg | `mem2reg.cpp` | Alloca→SSA via dominance tree/frontiers, PHI insertion |
| Dominance Tree | `dom_tree.cpp` | Cooper et al. 2001; immediate dominators + frontiers |
| Early Return Elim | `early_return_elimination.cpp` | Early returns → structured control flow |

### Analysis
| Pass | File |
|---|---|
| Call Graph | `call_graph.cpp` |
| Pointer Usage | `pointer_usage.cpp` |
| Lexical Scope | `lex_scope_analysis.cpp` |
| Aggregate Field Bitmask | `aggregate_field_bitmask.cpp` |

### Transformation
| Pass | File | Purpose |
|---|---|---|
| SROA | `sroa.cpp` | Scalar replacement of aggregates |
| Outline | `outline.cpp` | Function outlining |
| Autodiff | `autodiff.cpp` | Autodiff transformations |
| Lower Ray Query | `lower_ray_query_loop.cpp` | Ray query lowering |
| Reg2Mem | `reg2mem.cpp` | Register → memory conversion |
| Promote Ref Arg | `promote_ref_arg.cpp` | Reference argument promotion |
| Transpose GEP | `transpose_gep.cpp` | Transpose GEP through loads/stores |
| Trace GEP | `trace_gep.cpp` | GEP analysis & tracing |
| Local Load Elimination | `local_load_elimination.cpp` | Redundant load elimination |
| Local Store Forward | `local_store_forward.cpp` | Store-to-load forwarding |
| Unused Callable Removal | `unused_callable_removal.cpp` | Dead function elimination |

## Control Flow Representation

XIR uses **structured control flow** with explicit merge blocks:

```
IfInst:   condition, true_block, false_block, merge_block
LoopInst: prepare_block, body_block, update_block, merge_block
```

Design: maintains SSA, enables structured transforms, maps well to GPU shaders, supports PHI nodes at merges.

## Metadata

**File**: `include/luisa/xir/metadata.h`. Types: `NAME`, `LOCATION`, `COMMENT`, `CURVE_BASIS`. Applied via `MetadataListMixin`.

## JSON Serialization

- **IR (legacy)**: `ast2json` → Rust IR, C API `luisa_compute_ir_ast_json_to_ir_*`
- **XIR**: `src/xir/translators/xir2json.cpp`, `json2xir.cpp` — uses yyjson for module serialization, cross-process, debugging

## Key Design Patterns

1. **Intrusive Lists** — `ManagedIntrusiveList` for node management
2. **CRTP** — `DerivedValue<>`, `DerivedInstruction<>`, `DerivedFunction<>`
3. **Visitor** — `traverse_basic_blocks()`, `traverse_instructions()`
4. **Builder** — `XIRBuilder` for instruction creation
5. **Mixin** — `MetadataListMixin`, `ControlFlowMergeMixin`, `InstructionOpMixin`
6. **Use-Def Chains** — `Use` objects track value users for SSA

## Adding a New XIR Pass

1. Create `src/xir/passes/<name>.cpp` + header `include/luisa/xir/passes/<name>.h`
2. Register in `src/xir/CMakeLists.txt`
3. Convention: accept `Module &` or `Function &`, return `bool` (changed?), use `XIRBuilder`, call `replace_all_uses_with()` for substitution
