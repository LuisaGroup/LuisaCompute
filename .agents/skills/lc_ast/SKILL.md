---
name: lc_ast
description: Locate AST read/write usage markers and builtin CallOp usage rules in LuisaCompute. Use when investigating or modifying AST variable usage propagation, FunctionBuilder internals, or CallOp argument marking.
---

# LuisaCompute AST Usage Markers

Quick reference for how manual `FunctionBuilder` AST tracks variable read/write usage and how builtin `CallOp` calls propagate usage to their arguments.

## Usage Enum

`include/luisa/ast/usage.h`

```cpp
enum struct Usage : uint32_t {
    NONE = 0u,
    READ = 0x01u,
    WRITE = 0x02u,
    READ_WRITE = READ | WRITE
};
```

Flags accumulate via OR over a variable's lifetime.

## Two-Layer Marker Design

### 1. Per-expression cache

`include/luisa/ast/expression.h`

```cpp
class Expression {
protected:
    mutable Usage _usage{Usage::NONE};
    virtual void _mark(Usage usage) const noexcept = 0;
public:
    void mark(Usage usage) const noexcept;
    [[nodiscard]] auto usage() const noexcept { return _usage; }
};
```

`src/ast/expression.cpp`

```cpp
void Expression::mark(Usage usage) const noexcept {
    if (auto a = to_underlying(_usage), u = a | to_underlying(usage); a != u) {
        _usage = static_cast<Usage>(u);
        _mark(usage);
    }
}
```

Propagation is idempotent: it only forwards when new bits are added.

### 2. FunctionBuilder storage

`include/luisa/ast/function_builder.h`

```cpp
luisa::vector<Usage> _variable_usages;

void mark_variable_usage(uint32_t uid, Usage usage) noexcept;
[[nodiscard]] auto variable_usage(uint uid) const noexcept { return _variable_usages[uid]; }
```

`src/ast/function_builder.cpp`

```cpp
void FunctionBuilder::mark_variable_usage(uint32_t uid, Usage usage) noexcept {
    auto old_usage = to_underlying(_variable_usages[uid]);
    auto u = static_cast<Usage>(old_usage | to_underlying(usage));
    _variable_usages[uid] = u;
}

uint32_t FunctionBuilder::_next_variable_uid() noexcept {
    auto uid = static_cast<uint32_t>(_variable_usages.size());
    _variable_usages.emplace_back(Usage::NONE);
    return uid;
}
```

## RefExpr Forwarding

`src/ast/expression.cpp`

```cpp
void RefExpr::_mark(Usage usage) const noexcept {
    if (auto fb = detail::FunctionBuilder::current(); fb == builder()) {
        fb->mark_variable_usage(_variable.uid(), usage);
    }
}
```

Only marks when the current builder owns the expression, preventing stale marking across function boundaries.

## Manual API Example

```cpp
auto &cur = *FunctionBuilder::current();
auto ref = cur.reference(Type::of<float4>());
cur.mark_variable_usage(ref->variable().uid(), Usage::READ_WRITE);
```

## Builtin CallOp Usage Marking

### Builtin detection

`include/luisa/ast/op.h`

```cpp
[[nodiscard]] constexpr auto is_builtin_operation(CallOp op) noexcept {
    return op != CallOp::CUSTOM && op != CallOp::EXTERNAL;
}
```

`include/luisa/ast/expression.h`

```cpp
[[nodiscard]] auto is_builtin() const noexcept { return is_builtin_operation(_op); }
```

### CallExpr::_mark rules

`src/ast/expression.cpp`

```cpp
void CallExpr::_mark() const noexcept {
    if (is_builtin()) {
        switch (_op) {
            case CallOp::BUFFER_VOLATILE_WRITE:
            case CallOp::BUFFER_WRITE:
            case CallOp::BINDLESS_BUFFER_WRITE:
            case CallOp::BYTE_BUFFER_VOLATILE_WRITE:
            case CallOp::BYTE_BUFFER_WRITE:
            case CallOp::TEXTURE_WRITE:
            case CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM:
            case CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY:
            case CallOp::RAY_TRACING_SET_INSTANCE_OPACITY:
            case CallOp::RAY_TRACING_SET_INSTANCE_USER_ID:
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX:
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT:
            case CallOp::RAY_QUERY_COMMIT_TRIANGLE:
            case CallOp::RAY_QUERY_COMMIT_PROCEDURAL:
            case CallOp::RAY_QUERY_TERMINATE:
            case CallOp::RAY_QUERY_PROCEED:
            case CallOp::GRADIENT_MARKER:
            case CallOp::ACCUMULATE_GRADIENT:
            case CallOp::ATOMIC_EXCHANGE:
            case CallOp::ATOMIC_COMPARE_EXCHANGE:
            case CallOp::ATOMIC_FETCH_ADD:
            case CallOp::ATOMIC_FETCH_SUB:
            case CallOp::ATOMIC_FETCH_AND:
            case CallOp::ATOMIC_FETCH_OR:
            case CallOp::ATOMIC_FETCH_XOR:
            case CallOp::ATOMIC_FETCH_MIN:
            case CallOp::ATOMIC_FETCH_MAX:
            case CallOp::INDIRECT_SET_DISPATCH_KERNEL:
            case CallOp::INDIRECT_SET_DISPATCH_COUNT:
            case CallOp::COOPERATIVE_OUTER_PRODUCT_ACCUMULATE:
            case CallOp::COOPERATIVE_VECTOR_ACCUMULATE:
            case CallOp::COOPERATIVE_VECTOR_STORE:
            case CallOp::COOPERATIVE_VECTOR_WORKGROUP_STORE:
                _arguments[0]->mark(Usage::WRITE);
                for (size_t i = 1; i < _arguments.size(); i++) {
                    _arguments[i]->mark(Usage::READ);
                }
                break;
            default:
                for (auto arg : _arguments) {
                    arg->mark(Usage::READ);
                }
        }
    } else if (is_external()) {
        auto f = external();
        for (size_t i = 0; i < _arguments.size(); i++) {
            _arguments[i]->mark(f->argument_usages()[i]);
        }
    } else {
        // custom callable
        auto args = custom().arguments();
        for (size_t i = 0; i < args.size(); i++) {
            auto arg = args[i];
            _arguments[i]->mark(
                arg.is_reference() || arg.is_resource() ?
                    custom().variable_usage(arg.uid()) :
                    Usage::READ);
        }
    }
}
```

### Rule summary

- **Default builtin**: every argument marked `READ`.
- **Write-style builtins** (list above): argument 0 marked `WRITE`; remaining arguments marked `READ`.
- Atomic ops mark their target reference (argument 0) as `WRITE`; `AtomicRefNode::operate()` builds the `CallExpr` with the target as `_arguments[0]` (`src/ast/atomic_ref_node.cpp`).

## Files of Record

| Purpose | Path |
|---------|------|
| Usage enum | `include/luisa/ast/usage.h` |
| Expression base & `CallExpr` | `include/luisa/ast/expression.h` |
| `Expression::mark`, `RefExpr::_mark`, `CallExpr::_mark` | `src/ast/expression.cpp` |
| `FunctionBuilder` declaration & `_variable_usages` | `include/luisa/ast/function_builder.h` |
| `FunctionBuilder::mark_variable_usage`, `_next_variable_uid`, `call()` | `src/ast/function_builder.cpp` |
| `CallOp` enum, `is_builtin_operation`, `is_atomic_operation` | `include/luisa/ast/op.h` |
| `check_builtin_call_valid` | `src/ast/op.cpp` |
| `Function::variable_usage` exposure | `src/ast/function.cpp` |
| Atomic op construction | `src/ast/atomic_ref_node.cpp` |
| Manual AST skill doc | `.agents/skills/ast/SKILL.md` |

## Common Modifications

- **Add a new write-style builtin op**: extend the switch in `src/ast/expression.cpp` `CallExpr::_mark()` so argument 0 is `WRITE`.
- **Query usage after building**: call `Function::variable_usage(uid)` or `FunctionBuilder::variable_usage(uid)`.
- **Custom callable reference/resource args**: explicitly mark the reference variable `READ_WRITE` via `mark_variable_usage()` so callers propagate usage correctly.

---

# Appendix: Full AST C++ Structure

## File Inventory

### Headers (`include/luisa/ast/`)

| File | Main Class(es) | Description |
|------|----------------|-------------|
| `usage.h` | `Usage` (enum) | `NONE`, `READ`, `WRITE`, `READ_WRITE` flags |
| `attribute.h` | `Attribute` | Key-value pair struct for type/variable metadata |
| `variable.h` | `Variable` | Typed variable with `Tag` (LOCAL, SHARED, REFERENCE, BUFFER, TEXTURE, BINDLESS_ARRAY, ACCEL, and builtins like THREAD_ID, BLOCK_ID, DISPATCH_ID, etc.) |
| `type.h` | `Type` | Central type system: scalar types (BOOL, INT8..FLOAT64, FLOAT8), VECTOR, MATRIX, ARRAY, STRUCTURE, BUFFER, TEXTURE, BINDLESS_ARRAY, ACCEL, COOPERATIVE_VECTOR, COOPERATIVE_VECTOR_REF, COOPERATIVE_MATRIX_REF, CUSTOM. Factory methods: `of<T>()`, `array()`, `vector()`, `matrix()`, `buffer()`, `texture()`, `structure()`, `custom()`, `from(description)` |
| `type_registry.h` | `TypeDesc<T>`, macros `LUISA_STRUCT_REFLECT` | Compile-time type description generation; C++20 aggregate member counting via `member_reflect.inl.h` |
| `member_reflect.inl.h` | `count_member<T>()`, `member_reflect<T>()` | Compile-time struct reflection: counts aggregate members (up to 126) and builds `struct<align,member1,member2,...>` description strings |
| `constant_data.h` | `ConstantData`, `ConstantDecoder` | Constant data storage with type + raw bytes; `ConstantDecoder` virtual dispatch for decoding vectors, matrices, structs, arrays |
| `expression.h` | `Expression` (base), `UnaryExpr`, `BinaryExpr`, `MemberExpr`, `AccessExpr`, `LiteralExpr`, `RefExpr`, `ConstantExpr`, `CallExpr`, `CastExpr`, `TypeIDExpr`, `StringIDExpr`, `FuncRefExpr`, `CpuCustomOpExpr`, `GpuCustomOpExpr` | Full expression tree with visitor pattern; `_usage` cache, `_mark()` virtual, `traverse_subexpressions()` helper |
| `op.h` | `UnaryOp`, `BinaryOp`, `CallOp`, `CallOpSet`, `TypePromotion` | Operation enums (CallOp has 300+ entries) |
| `statement.h` | `Statement` (base), `BreakStmt`, `ContinueStmt`, `ReturnStmt`, `ScopeStmt`, `IfStmt`, `LoopStmt`, `ExprStmt`, `SwitchStmt`, `SwitchCaseStmt`, `SwitchDefaultStmt`, `AssignStmt`, `ForStmt`, `CommentStmt`, `RayQueryStmt`, `SuspendStmt`, `AutoDiffStmt`, `PrintStmt`, `DebugBreakStmt` | Full statement tree with visitor pattern; `traverse_expressions()` template helper |
| `function.h` | `Function` | Public function handle wrapping `FunctionBuilder*`. Provides access to variables, arguments, bindings, callables, block size, hash, usage queries |
| `function_builder.h` | `FunctionBuilder` | Central AST construction API. RAII scope guards, thread-local function stack. Creates expressions/literals/variables/statements. `mark_variable_usage()`, `hash()`, `duplicate()`, `sort_bindings()`, `_internalize()` |
| `external_function.h` | `ExternalFunction` | Named external function with typed argument list and per-argument `Usage` |
| `atomic_ref_node.h` | `AtomicRefNode` | Helper for building atomic operations: chains access paths and emits `CallExpr` |
| `callable_library.h` | `CallableLibrary` | Serialization/deserialization of callable function graphs to/from binary blobs |
| `ast2json.h` | `to_json()` | Convert `Type` or `Function` to JSON string for debugging |
| `interface.h` | — | Convenience include aggregating type/variable/expression/statement/function headers |

### Sources (`src/ast/`)

| File | Key Functions |
|------|---------------|
| `type.cpp` | `TypeRegistry` singleton, `TypeImpl`, `_decode()` recursive parser for type descriptions, `Type::from()`, `Type::array()`, `Type::vector()`, `Type::matrix()`, `Type::buffer()`, `Type::texture()`, `Type::structure()`, `Type::custom()`, type query predicates |
| `variable.cpp` | `Variable::hash()` |
| `expression.cpp` | `Expression::mark()`, `Expression::hash()`, all expression `_mark()`/`_compute_hash()` overrides, `CallExpr::_mark()` (builtin write-style vs read-style dispatch), `CallExpr::custom()/external()` |
| `statement.cpp` | All `Statement::_compute_hash()` overrides, `PrintStmt`/`DebugBreakStmt` constructors, default `StmtVisitor` methods |
| `function.cpp` | `Function` methods delegating to `FunctionBuilder`, binding hash functions |
| `function_builder.cpp` | `FunctionBuilder::push/pop/current`, `break_/continue_/return_/suspend_/ray_query_/autodiff_/if_/loop_/switch_/case_/default_/for_/assign`, `mark_variable_usage`, `_internalize`, `_ref`, `_builtin`, `local/shared/argument/buffer/texture/bindless_array/accel`, `literal/unary/binary/member/swizzle/access/cast/string_id/type_id/func_ref/call`, `_compute_hash`, `sort_bindings`, `_duplicate_if_necessary`, `duplicate`, `set_block_size`, `set_name` |
| `op.cpp` | `CallOpSet::Iterator`, `promote_types()`, `check_builtin_call_valid()` |
| `constant_data.cpp` | `ConstantDecoder` vector/matrix/struct/array decoding, `ConstantData::create()` with deduplication |
| `external_function.cpp` | `ExternalFunction` constructor, `_compute_hash()` |
| `atomic_ref_node.cpp` | `AtomicRefNode` construction, `access()` chaining, `operate()` builds `CallExpr` |
| `callable_library.cpp` | Full serialization/deserialization of `FunctionBuilder` graph |
| `function_duplicator.cpp` | `FunctionDuplicator` deep-copies `FunctionBuilder` graph with variable remapping; `_duplicate_if_necessary()` for leaked variables; `deduplicate_custom_callables()` |
| `ast2json.cpp` | `JSON` value type, `AST2JSON` visitor converts full AST to JSON |
| `lc_ast_pch.h` | Precompiled header |

## Class Hierarchy Overview

```
Expression (abstract)
├── UnaryExpr
├── BinaryExpr
├── MemberExpr
├── AccessExpr
├── LiteralExpr
├── RefExpr
├── ConstantExpr
├── CallExpr
├── CastExpr
├── TypeIDExpr
├── StringIDExpr
├── FuncRefExpr
├── CpuCustomOpExpr
└── GpuCustomOpExpr

Statement (abstract)
├── BreakStmt
├── ContinueStmt
├── ReturnStmt
├── ScopeStmt
├── IfStmt
├── LoopStmt
├── ExprStmt
├── SwitchStmt
├── SwitchCaseStmt
├── SwitchDefaultStmt
├── AssignStmt
├── ForStmt
├── CommentStmt
├── RayQueryStmt
├── SuspendStmt
├── AutoDiffStmt
├── PrintStmt
└── DebugBreakStmt
```

## Key Design Patterns

1. **Ownership**: `FunctionBuilder` owns all `Expression` and `Statement` objects via `unique_ptr` vectors. All raw pointers are non-owning views.

2. **Builder stack**: Thread-local `_function_stack()` enables `Expression` constructors to automatically capture their owning builder. `FunctionStackGuard` pushes/pops on definition.

3. **Expression internalization** (`_internalize()`): When a callable references a variable from an outer scope, the builder clones/captures the expression chain into the current function. Lvalue locals become reference arguments; resources become new resource arguments; builtins become new builtins; statically-evaluable expressions are recursively cloned.

4. **Usage propagation**: Two-phase: (a) `Expression::_usage` bitfield caches the aggregate usage at each expression node; (b) `RefExpr::_mark()` writes through to `FunctionBuilder::_variable_usages[uid]` for final variable-level query.

5. **CallOp semantics**: `CallOpSet` (bitset) tracks which builtins a function directly/propagatedly uses.

6. **Serialization**: `CallableLibrary` provides a custom binary serialization format for distributing callable function graphs.

7. **Duplication**: `FunctionDuplicator` creates a deep copy of a `FunctionBuilder` graph, remapping variable UIDs and hoisting leaked references.

8. **AtomicRefNode**: Chains buffer/array/structure access paths into a flat argument list for atomic `CallExpr` construction.

## Visitor Helpers

- `traverse_subexpressions(expr, enter, exit)` — walks all expression nodes recursively.
- `traverse_expressions<recurse_subexpr>(stmt, visit, enter_stmt, exit_stmt)` — walks all expressions nested in a statement tree.
- `ExprVisitor` — abstract visitor with virtual methods for each expression type.
- `StmtVisitor` — abstract visitor with virtual methods for each statement type.

## Type System Details

- `Type::from(description)` parses string descriptions like `"array<struct<16,int,float>,10>"` into interned `Type` objects.
- `TypeRegistry` (singleton) manages type pool and deduplication via `unordered_set`.
- `TypeImpl` extends `Type` with concrete storage for hash, tag, size, alignment, dimension, members, member_attributes.
- `TypeDesc<T>` maps C++ types to their string descriptions at compile time.
- `struct_member_tuple<T>` decomposes structs into `std::tuple` of member types with offset validation.

## Variable Tags

```
LOCAL | SHARED | REFERENCE | BUFFER | TEXTURE |
BINDLESS_ARRAY | ACCEL | THREAD_ID | BLOCK_ID |
DISPATCH_ID | DISPATCH_SIZE | KERNEL_ID |
WARP_LANE_COUNT | WARP_LANE_ID |
RASTER_OBJECT_ID | RASTER_BARYCENTRICS
```

---

# How to Add a New CallOp

## Overview

Adding a new `CallOp` requires changes across the AST layer, validation, each backend codegen, and optionally the DSL. Below is the complete checklist.

## Step 1: Add the enum value

**File:** `include/luisa/ast/op.h`

Append to the `CallOp` enum in the appropriate category section.

```cpp
enum struct CallOp : uint32_t {
    // ... existing ops ...
    MY_NEW_OP,
    // ...
};
```

> ⚠️ **DO NOT reorder existing values** — enum integer values are embedded in serialized function hashes and are assumed by `call_op_count`. Append your new op in the appropriate category section before `CLOCK` (the last enumerator). If you must add after `CLOCK`, update `call_op_count` and `LUISA_MAGIC_ENUM_RANGE` accordingly.

### Also update:

- **`call_op_count`** (line ~522): `static constexpr size_t call_op_count = to_underlying(CallOp::CLOCK) + 1u;` — This defines the size of the `CallOpSet` bitset. If your new op is added BEFORE `CLOCK`, `call_op_count` already covers it. If added AFTER `CLOCK`, increment this value.

- **`LUISA_MAGIC_ENUM_RANGE`** (line ~664): `LUISA_MAGIC_ENUM_RANGE(luisa::compute::CallOp, CUSTOM, CLOCK)` — Enables `to_string`/`from_string` for the range `[CUSTOM, CLOCK]`. If your new op is after `CLOCK`, extend the range to include it.

## Step 2: Update usage propagation

**File:** `src/ast/expression.cpp` — `CallExpr::_mark()`

### Read-only (default):
No change needed — the `default` case marks all args `Usage::READ`.

### Write-style (arg[0] = WRITE, rest = READ):
Add to the existing switch:
```cpp
case CallOp::MY_NEW_OP:
    _arguments[0]->mark(Usage::WRITE);
    for (size_t i = 1; i < _arguments.size(); i++) {
        _arguments[i]->mark(Usage::READ);
    }
    break;
```

### Custom usage:
Implement arbitrary logic in the switch.

## Step 3: Add validation (optional but recommended)

**File:** `src/ast/op.cpp` — `check_builtin_call_valid()`

Add a case to validate argument types and counts at AST construction time:

```cpp
case CallOp::MY_NEW_OP: {
    LUISA_ASSERT(args.size() == 2 &&
                 args[0]->type()->is_buffer() &&
                 args[1]->type()->is_uint32(),
                 "MY_NEW_OP: expected (buffer, uint32)");
    break;
}
```

## Step 4: Add helper functions for category detection (optional)

**File:** `include/luisa/ast/op.h`

If your op belongs to a new category, add a `constexpr` helper:
```cpp
[[nodiscard]] constexpr auto is_my_category_operation(CallOp op) noexcept {
    auto v = to_underlying(op);
    return v >= to_underlying(CallOp::MY_CATEGORY_START) &&
           v <= to_underlying(CallOp::MY_CATEGORY_END);
}
```

## Step 5: Update each backend codegen

Each backend has a switch on `CallOp` that emits native code or IR. Add your case to all of them:

| Backend | Codegen File(s) | Nature |
|---------|-----------------|--------|
| **CUDA** | `src/backends/cuda/cuda_codegen_ast.cpp` | Direct AST→CUDA C++ string emission |
| **Metal** | `src/backends/metal/metal_codegen_ast.cpp` | Direct AST→Metal Shading Language string emission |
| **HLSL/DX12** | `src/backends/common/hlsl/codegen_utils/function_codegen.cpp` (main CallOp dispatch), `src/backends/common/hlsl/hlsl_codegen.cpp` (AST visitor), `src/backends/dx/` (DXIL compilation) | AST→HLSL string, compiled to DXIL; no own CallOp switch in `dx/` |
| **SPIR-V (LLVM)** | `src/backends/common/spirv_llvm/llvm_state_visitor.cpp` | AST→LLVM IR → SPIR-V binary via `spirv64` target machine |
| **Vulkan** | `src/backends/vk/` + `src/backends/common/spirv/` (default) or `spirv_llvm/` (experimental) | Uses common SPIR-V codegen; no own CallOp switch |
| **XIR** (intermediate) | `src/xir/translators/ast2xir.cpp` | AST→XIR; CUDA, HIP, Vulkan, and Fallback consume XIR directly, while DX/Metal use XIR for lowering before AST codegen |
| **Fallback** | `src/backends/fallback/` + `src/xir/translators/ast2xir.cpp` | Uses XIR as input; no direct AST CallOp switch |
| **Hip/AMD** | `src/backends/hip/` + `src/xir/translators/ast2xir.cpp` | Uses XIR as input; no direct AST CallOp switch |
| **Toy C** | `src/backends/toy_c/` | Simple C output; no direct AST CallOp switch |
| **Validation** | `src/backends/validation/` | AST validation layer wrapping another backend; no own CallOp switch |

Example CUDA addition:
```cpp
case CallOp::MY_NEW_OP: {
    _scratch << "my_new_op(";
    for (auto i = 0u; i < args.size(); i++) {
        if (i) _scratch << ", ";
        emit(args[i]); // use the backend's expression emitter
    }
    _scratch << ")";
    break;
}
```

## Step 6: (Optional) Add DSL helper

If the op should be exposed via the high-level DSL, add a helper in `src/dsl/`:

```cpp
// src/dsl/something.cpp
[[nodiscard]] auto my_new_op(Expr<float> x) noexcept {
    return detail::FunctionBuilder::current()->call(...);
}
```

## Step 7: (Optional) Update XIR passes

If your op needs special handling in the XIR optimization pipeline, add it to `src/xir/passes/`.

## Full Checklist

| # | What | File(s) |
|---|------|---------|
| 1 | Add enum value | `include/luisa/ast/op.h` |
| 2 | Update `call_op_count` / `LUISA_MAGIC_ENUM_RANGE` if needed | `include/luisa/ast/op.h` |
| 3 | Add usage marking in `CallExpr::_mark()` | `src/ast/expression.cpp` |
| 4 | Add argument validation in `check_builtin_call_valid()` | `src/ast/op.cpp` |
| 5 | Add codegen for each backend | See table above |
| 6 | (Optional) Add DSL helper | `src/dsl/` |
| 7 | (Optional) Add category helper | `include/luisa/ast/op.h` |

---

# How to Add a New Expression

## Step 1: Add the expression class

**File:** `include/luisa/ast/expression.h`

1. Add a new `Tag` enum value to `Expression::Tag` (e.g., `MY_NEW_EXPR`).
2. Forward-declare the class (e.g., `class MyNewExpr;`).
3. Add a `virtual void visit(const MyNewExpr *) = 0;` to `ExprVisitor`.
4. Implement the class inheriting `Expression`:

```cpp
class LUISA_AST_API MyNewExpr final : public Expression {
    friend class CallableLibrary;

private:
    // your data members
    MyNewExpr() noexcept = default;

protected:
    void _mark(Usage) const noexcept override { /* propagate usage if needed */ }
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    MyNewExpr(/* params */) noexcept
        : Expression{Tag::MY_NEW_EXPR, type} /*, init members */ {}
    // accessors
    LUISA_EXPRESSION_COMMON()
};
```

## Step 2: Add hash computation

**File:** `src/ast/expression.cpp`

Implement `_compute_hash()`:
```cpp
uint64_t MyNewExpr::_compute_hash() const noexcept {
    return hash_combine({/* member hashes */});
}
```

## Step 3: Add to `traverse_subexpressions`

**File:** `include/luisa/ast/expression.h` (the free function at the bottom)

Add a case for `Expression::Tag::MY_NEW_EXPR` so the traversal helper works correctly.

## Step 4: Add creation method to FunctionBuilder

**File:** `include/luisa/ast/function_builder.h` (declaration) and `src/ast/function_builder.cpp` (definition)

```cpp
// in function_builder.h:
[[nodiscard]] const MyNewExpr *my_new_expr(/* params */) noexcept;

// in function_builder.cpp:
const MyNewExpr *FunctionBuilder::my_new_expr(/* params */) noexcept {
    return _create_expression<MyNewExpr>(/* params */);
}
```

## Step 5: Add serialization support (optional)

**File:** `src/ast/callable_library.cpp`

Add `ser_value` and `deser_ptr` specializations for the new expression type, plus integrate into the `Expression` base `ser_value`/`deser_value` dispatch.

## Step 6: Add codegen in each backend

Each backend that processes AST expressions directly (CUDA, Metal, HLSL, SPIR-V LLVM, LLVM/CPU) needs a `case Expression::Tag::MY_NEW_EXPR` in its visitor switch.

## Step 7: Add JSON export (optional)

**File:** `src/ast/ast2json.cpp`

Add a conversion method in `AST2JSON` and wire it into `_convert_expr()`.

---

# How to Add a New Statement

## Step 1: Add the statement class

**File:** `include/luisa/ast/statement.h`

1. Add a new `Tag` enum value to `Statement::Tag`.
2. Forward-declare (e.g., `class MyNewStmt;`).
3. Add `virtual void visit(const MyNewStmt *) = 0;` to `StmtVisitor`.
4. Implement the class inheriting `Statement`:

```cpp
class LUISA_AST_API MyNewStmt final : public Statement {
    friend class CallableLibrary;

private:
    // data members
    MyNewStmt() noexcept = default;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    MyNewStmt(/* params */) noexcept
        : Statement{Tag::MY_NEW_STMT} /*, init */ {
        // mark expression usages here
    }
    // accessors
    LUISA_STATEMENT_COMMON()
};
```

## Step 2: Add hash computation

**File:** `src/ast/statement.cpp`

```cpp
uint64_t MyNewStmt::_compute_hash() const noexcept {
    return hash_combine({/* member hashes */});
}
```

## Step 3: Add to `traverse_expressions`

**File:** `include/luisa/ast/statement.h` — add a case in the `traverse_expressions` template function.

## Step 4: Add creation method to FunctionBuilder

**File:** `include/luisa/ast/function_builder.h` / `src/ast/function_builder.cpp`

```cpp
// declaration
[[nodiscard]] MyNewStmt *my_new_stmt_(/* params */) noexcept;

// definition — use _create_and_append_statement<MyNewStmt>(...)
MyNewStmt *FunctionBuilder::my_new_stmt_(/* params */) noexcept {
    return _create_and_append_statement<MyNewStmt>(/* params */);
}
```

## Step 5: Add to AST->XIR translation (if applicable)

**File:** `src/xir/translators/ast2xir.cpp` — add a case for the new statement tag.

## Step 6: Add JSON export (optional)

**File:** `src/ast/ast2json.cpp` — add a conversion method in `_convert_stmt()`.

---

# Backend Codegen Locations Reference

## Directory Structure

```
src/backends/
├── CMakeLists.txt
├── xmake.lua
├── common/
│   ├── c_codegen/          # LLVM CPU codegen (AST→LLVM IR)
│   │   ├── codegen_visitor.cpp/h   — main visitor dispatching on Expression/Statement tags
│   │   ├── codegen_utils.cpp/h     — helper utilities
│   │   ├── builtin/                — builtin function implementations
│   │   └── lc_ccodegen_pch.h
│   ├── hlsl/               # HLSL codegen (AST→HLSL string)
│   │   ├── hlsl_codegen.cpp/h      — top-level HLSL codegen
│   │   ├── codegen_stack_data.cpp/h— per-function state
│   │   ├── codegen_utils/          — detailed sub-emitters
│   │   │   ├── function_codegen.cpp— main CallOp dispatch (huge switch)
│   │   │   ├── resource.cpp        — buffer/texture/bindless ops
│   │   │   ├── type_system.cpp     — HLSL type mapping
│   │   │   ├── cbuffer.cpp         — constant buffer layout
│   │   │   ├── constant.cpp        — constant emission
│   │   │   ├── entry_points.cpp    — kernel entry point generation
│   │   │   ├── property.cpp        — shader properties
│   │   │   └── variable.cpp        — variable declarations
│   │   └── hlsl_codegen_util.txt
│   └── spirv_llvm/         # SPIR-V via LLVM (AST→LLVM IR→SPIR-V)
│       ├── llvm_state_visitor.cpp  — main AST visitor
│       ├── llvm_codegen_result.h
│       ├── llvm_codegen_stack_data.cpp/h
│       └── llvm_codegen_utility.cpp/h
├── cuda/                  # CUDA backend
│   ├── cuda_codegen_ast.cpp/h      — AST→CUDA C++ string codegen
│   ├── cuda_codegen_xir.cpp/h      — XIR→CUDA codegen (alternative path)
│   └── llvm_codegen/               — LLVM-based CUDA codegen path
├── metal/                 # Metal (Apple GPU) backend
│   └── metal_codegen_ast.cpp/h     — AST→MSL string codegen
├── dx/                    # DirectX 12 backend (uses common HLSL codegen; no own AST switch)
├── vk/                    # Vulkan backend (uses common SPIR-V codegen; no own AST switch)
├── hip/                   # AMD HIP backend
│   └── llvm_codegen/           — own LLVM codegen (input: XIR, not direct AST)
├── fallback/              # Fallback CPU reference implementation (uses XIR)
│   └── fallback_codegen.cpp    — XIR-based LLVM codegen
├── toy_c/                 # Simple C output for debugging
└── validation/            # AST validation layer (wraps another backend)
```

## Where to Add Codegen for Each Tag Type

### For `CallOp` (the most common change):
Look for the giant `switch` on `CallOp` in each of these files:

| File | What it does |
|------|-------------|
| `src/backends/cuda/cuda_codegen_ast.cpp` | Emits CUDA C++ function call strings like `lc_buffer_read(...)` |
| `src/backends/metal/metal_codegen_ast.cpp` | Emits Metal Shading Language strings |
| `src/backends/common/hlsl/codegen_utils/function_codegen.cpp` | Emits HLSL strings (main HLSL dispatch; used by DX12) |
| `src/backends/common/spirv_llvm/llvm_state_visitor.cpp` | Generates LLVM IR that gets translated to SPIR-V (used by Vulkan) |
| `src/backends/common/c_codegen/codegen_visitor.cpp` | Generates LLVM IR for the CPU JIT backend |
| `src/xir/translators/ast2xir.cpp` | Translates AST → XIR IR; backends using XIR (CUDA XIR path, Metal XIR path, HIP, Fallback) handle CallOp through their own XIR visitors |

### For `Expression::Tag`:
Each direct-AST codegen visitor has a switch on `Expression::Tag`:

| File | Notes |
|------|-------|
| `src/backends/cuda/cuda_codegen_ast.cpp` | Full AST visitor with switch on `Expression::Tag` |
| `src/backends/metal/metal_codegen_ast.cpp` | Full AST visitor with switch on `Expression::Tag` |
| `src/backends/common/hlsl/hlsl_codegen.cpp` | Full AST visitor with switch on `Expression::Tag` (used by DX12) |
| `src/backends/common/spirv_llvm/llvm_state_visitor.cpp` | Full AST visitor with switch on `Expression::Tag` (used by Vulkan) |
| `src/backends/common/c_codegen/codegen_visitor.cpp` | Full AST visitor with switch on `Expression::Tag` (used by CPU) |
| `src/xir/translators/ast2xir.cpp` | AST→XIR translator; has its own switch |

### For `Statement::Tag`:
Same files as `Expression::Tag` — each visitor also has a `switch` on `Statement::Tag`:

| File | Notes |
|------|-------|
| `src/backends/cuda/cuda_codegen_ast.cpp` | Full AST visitor with switch on `Statement::Tag` |
| `src/backends/metal/metal_codegen_ast.cpp` | Full AST visitor with switch on `Statement::Tag` |
| `src/backends/common/hlsl/hlsl_codegen.cpp` | Full AST visitor with switch on `Statement::Tag` (used by DX12) |
| `src/backends/common/spirv_llvm/llvm_state_visitor.cpp` | Full AST visitor with switch on `Statement::Tag` (used by Vulkan) |
| `src/backends/common/c_codegen/codegen_visitor.cpp` | Full AST visitor with switch on `Statement::Tag` (used by CPU) |
| `src/xir/translators/ast2xir.cpp` | AST→XIR translator; has its own switch |
