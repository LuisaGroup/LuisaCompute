---
name: lc-ast
description: Trace and modify LuisaCompute AST usage propagation and AST-to-XIR lowering for calls, resources, raster stages, ray queries, PACK/UNPACK, opaque custom arguments, and external functions. Use when changing CallExpr marking, typed bindless aliases, raster builtins, direct ray-query operations, callable argument usage, TypeID/StringID handling, or XIR lvalue/resource conventions.
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
            case CallOp::PACK:
                _arguments[0]->mark(Usage::READ);
                _arguments[1]->mark(Usage::WRITE);
                _arguments[2]->mark(Usage::READ);
                break;
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
                arg.is_reference() || arg.is_resource() || arg.type()->is_custom() ?
                    custom().variable_usage(arg.uid()) :
                    Usage::READ);
        }
    }
}
```

### Rule summary

- **Default builtin**: every argument marked `READ`.
- **Write-style builtins** (list above): argument 0 marked `WRITE`; remaining arguments marked `READ`.
- **`PACK(value, words, offset)`**: value and offset are `READ`; the destination `buffer<uint>` is `WRITE`. Do not put `PACK` in the ordinary argument-0 write group.
- **`UNPACK(words, offset)`**: follows the default rule, so both arguments are `READ`.
- Atomic ops mark their target reference (argument 0) as `WRITE`; `AtomicRefNode::operate()` builds the `CallExpr` with the target as `_arguments[0]` (`src/ast/atomic_ref_node.cpp`).
- External calls copy each `ExternalFunction::argument_usages()` entry to the matching argument.
- Custom callable reference, resource, and opaque-custom arguments propagate the callee variable usage. Ordinary value arguments are always `READ`.

## AST-to-XIR call and ID conventions

Use `src/xir/translators/ast2xir.cpp` as the source of truth for argument form.

- Cache external declarations by `ExternalFunction::hash()`, preserve their name and return type, and accept `void` returns.
- Lower a non-resource external argument with `READ` or `NONE` usage as an XIR value.
- Lower a non-resource external argument containing `WRITE`, and every opaque custom argument, as an XIR reference. Require the call operand to be an lvalue.
- Keep resource arguments as XIR resources; do not additionally wrap them in ordinary references.
- Represent opaque custom arguments to ordinary custom callables as references even when the AST surface presents the backend handle by value.
- Lower `TypeIDExpr` to `uint64(0)` until the source Metal/CUDA paths define a stable cross-backend type-ID ABI.
- Lower `StringIDExpr` to the 64-bit `luisa::hash_value` of the string contents.

Metal4 AIR preserves these declarations as exact LLVM declarations. Every used
symbol must be defined by `ShaderOption::native_include` as compatible textual
LLVM IR or bitcode. CodeGen checks target/data layout, function ABI, address
space, calling convention, ABI attributes, and reference alignment, then links
needed definitions before O2 and LLVM-14 downgrade. Values use register ABI,
references use generic pointers, and external calls receive no hidden Luisa
state parameters. Missing or incompatible definitions fail shader creation;
Metal4 has no MSL or legacy-IR fallback. Compute and raster AOT loaders consume
their compiled archives without rerunning AST-to-XIR or preflight.

This policy belongs to the separate `metal4` backend. The original `metal`
backend remains source-MSL codegen and must not acquire a dependency on the
Metal4 LLVM/AIR pipeline.

## Preserve raster-stage identity and payloads

- An AST `Function::Tag::RASTER_STAGE` does not encode vertex versus fragment.
  Require the caller to set `AST2XIRConfig::raster_stage` and create a
  `RasterStageFunction` with that explicit role. Do not guess from argument or
  return types.
- Keep argument zero as the stage payload: `AppData` for vertex and the vertex
  return type for fragment. All later arguments form the shared host root ABI;
  do not reorder them with kernel-style binding sorting.
- Keep `Function::arguments()` zipped with the full raster
  `bound_arguments()` array. Raster bindings may contain `monostate` entries
  in-place, so `unbound_arguments()` and a bound-prefix assumption are not
  valid for this path.
- Expose `raster_object_id()`, `raster_barycentrics()`, and
  `raster_discard()` through the normal DSL. Object ID is `uint`, barycentrics
  are `float3`, and discard lowers to the XIR raster-discard terminator.
- Lower `DDX` and `DDY` to raster quad derivative XIR operations. Reject these
  operations outside a fragment-stage backend configuration rather than
  treating them as compute thread-group operations.

## Normalize bindless aliases and ray-query state

- Lower every supported `TYPED_BINDLESS_*` and
  `TYPED_UNIFORM_BINDLESS_*` query, read, or write alias to the corresponding
  ordinary XIR bindless `ResourceQueryOp`, `ResourceReadOp`, or
  `ResourceWriteOp`. Keep the original operands and result type; backends
  should not need duplicate typed or uniform opcode families.
- Require the first operand of a direct ray-query object operation to be the
  query lvalue. Pass that same lvalue to every XIR read or write emitted for
  one AST call.
- Lower direct `RAY_QUERY_PROCEED(query)` to a
  `RAY_QUERY_OBJECT_PROCEED(query)` write immediately followed by a
  `RAY_QUERY_OBJECT_IS_TERMINATED(query)` read, and return
  `UNARY_BIT_NOT` of the read. The AST operation means “a candidate is
  available,” which is the logical inverse of termination.
- Treat the four top-level `RAY_TRACING_QUERY_{ALL,ANY}` constructors, including
  their motion-blur variants, as fresh mutable state in downstream XIR passes.
  They must not be commoned or hoisted even when their operands match.

## PACK/UNPACK lowering contract

Validate `PACK` as `(packable value, buffer<uint>, uint offset) -> void` and
`UNPACK` as `(buffer<uint>, uint offset) -> packable value`. Reject resources
and opaque custom types.

AST-to-XIR wraps the packed value in a one-member Luisa structure whose
alignment is at least four bytes, bitwise-casts the complete wrapper to
`array<uint, sizeof(wrapper) / 4>`, and emits consecutive buffer writes or
reads. This wrapper makes scalar bool/byte/short values one full word and
retains Luisa padding for values such as `float3`. A four-field
`{bool, bool, bool, bool}` value occupies four bytes and must not become the
one-byte LLVM vector `<4 x i1>`; a Luisa byte4 becomes `<4 x i8>` and also
occupies four bytes. A `float3` wrapper occupies 16 bytes. Backends must
initialize padding before the bitwise cast; the Metal4 AIR path uses zero so
`PACK` never observes LLVM poison.

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
| AST-to-XIR calls, IDs, bindless aliases, ray queries, PACK/UNPACK | `src/xir/translators/ast2xir.cpp` |
| PACK/UNPACK usage regression | `src/tests/unit/xir/test_ast_pack_usage.cpp` |
| Typed bindless lowering regression | `src/tests/unit/xir/test_ast_typed_bindless_lowering.cpp` |
| Direct ray-query proceed regression | `src/tests/unit/xir/test_xir_pass_lower_ray_query_loop.cpp` |
| External lowering regression | `src/tests/unit/xir/test_ast_external_lowering.cpp` |
| Manual AST skill doc | `.agents/skills/ast/SKILL.md` |

## Common Modifications

- **Add a new write-style builtin op**: extend the switch in `src/ast/expression.cpp` `CallExpr::_mark()` so argument 0 is `WRITE`.
- **Add an op whose destination is not argument 0**: give it a dedicated case, as `PACK` does for argument 1.
- **Query usage after building**: call `Function::variable_usage(uid)` or `FunctionBuilder::variable_usage(uid)`.
- **Custom callable reference/resource/custom args**: explicitly mark the callee variable with its real usage so callers propagate it correctly.
- **Change external lowering**: update declaration form and call operand form together, then run `test_ast_external_lowering` and the `unit_xir` CTest label.
- **Change bindless aliases or direct ray queries**: preserve ordinary XIR op
  normalization and lvalue identity, then run
  `test_ast_typed_bindless_lowering` and
  `test_xir_pass_lower_ray_query_loop`.

## Broader AST changes

Use `.agents/skills/ast/SKILL.md` for manual `FunctionBuilder` construction.
When adding an expression, statement, or `CallOp`, inspect the current enum,
builder, usage marking, hashing, traversal/visitor, validation, serialization,
and every affected AST or XIR backend. Search the implementation rather than
copying a backend inventory or enum boundary into this skill.
