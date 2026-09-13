---
name: hlsl
description: HLSL code generation, StringBuilder patterns, builtin headers, and DXIL embedding.
---

# HLSL Code Generation

Codegen utilities using `vstd::StringBuilder` and formatting helpers.

## StringBuilder

```cpp
#include <luisa/vstl/string_builder.h>

vstd::StringBuilder builder;
builder.clear();
size_t sz = builder.size();
vstd::string_view view = builder.view();
char* data = builder.data();
```

### Appending
```cpp
str.append("Hello");
str += "World"sv;
str.append(' ');
str << '\n';
str.append(other_builder);
str << "func" << '(' << ")";
```

### Numbers
```cpp
vstd::to_string(42, str);        // int (faster than format)
vstd::to_string(3.14f, str);     // float
str << luisa::format("{}", 42);
str << luisa::format("{}, {}!", "Hello", "World");
str << luisa::format("{:016X}", hash);
```

### String View Literals
```cpp
using namespace std::string_view_literals;
str += "void"sv;  // no allocation
```

## Code Generation Patterns

### Function Declaration
```cpp
void GetFunctionDecl(Function func, vstd::StringBuilder &str) {
    vstd::StringBuilder data;
    if (func.return_type()) CodegenUtility::GetTypeName(*func.return_type(), data, Usage::READ);
    else data += "void"sv;
    data += " "sv;
    CodegenUtility::GetFunctionName(func, data);
    data += '(';
    for (auto &&arg : func.arguments()) {
        Usage usage = func.variable_usage(arg.uid());
        CodegenUtility::GetTypeName(*arg.type(), data, usage);
        data << ' ';
        vstd::StringBuilder varName;
        CodegenUtility::GetVariableName(func, arg, varName);
        data << varName << ',';
    }
    if (!func.arguments().empty()) data[data.size() - 1] = ')';
    else data += ')';
    str << '\n' << data;
}
```

### Template Parameters
```cpp
str << "template<"sv;
for (uint64 i = 0; i < tempIdx; ++i) {
    str << "typename T"sv; vstd::to_string(static_cast<int64_t>(i), str); str << ',';
}
if (tempIdx > 0) *(str.end() - 1) = '>';
```

### Call Expression
```cpp
str << "func_name"sv << '(';
if (!args.empty()) {
    for (size_t i = 0; i < args.size() - 1; ++i) { args[i]->accept(vis); str << ','; }
    args.back()->accept(vis);
}
str << ')';
```

### Type-Aware Generation
```cpp
if (t->is_texture() || t->is_buffer()) { str << 'T'; vstd::to_string(tempIdx++, str); }
else if (t->is_vector()) { str << "float"sv; vstd::to_string(t->dimension(), str); }
else if (t->is_matrix()) { auto n = vstd::to_string(t->dimension()); str << "_float"sv << n << 'x' << n; }
```

### RAII Wrapping
```cpp
struct CodeWrapper {
    vstd::StringBuilder *_result;
    CodeWrapper(vstd::StringBuilder *r, string_view open) : _result(r) { if (_result) *_result << open; }
    ~CodeWrapper() { if (_result) *_result << ')'; }
};
CodeWrapper wrapper{&str, "to_float4x4("};
// inner content generated; ')' appended automatically
```

### Trailing Comma Fix
```cpp
str << '(';
for (auto &&item : items) str << item << ',';
if (!items.empty()) str[str.size() - 1] = ')';
else str << ')';
```

## Iteration & Hash

```cpp
for (char c : str) {}
str.erase(str.begin() + 5);
vstd::hash<vstd::StringBuilder> hasher; size_t h = hasher(builder);
if (builder1 == builder2) {}
if (builder1 == "literal"sv) {}
```

## Best Practices

1. Use `"text"sv` literals to avoid allocations
2. `vstd::to_string()` for integers (faster than `luisa::format()`)
3. `builder.reserve(1024)` when size known
4. `operator<<` for chaining, `operator+=` for single items
5. Check `size()` before modifying last character

## HLSL Builtins

Located in `src/backends/common/hlsl/builtin/`. Access via:

```cpp
#include <backends/common/hlsl/builtin/hlsl_builtin.hpp>
auto header = lc_hlsl::get_hlsl_builtin("hlsl_header");
std::string_view code(static_cast<const char*>(header.ptr), header.size);
```

### Header Files (`.bytes`)
| Key | Description |
|---|---|
| `hlsl_header` | Main HLSL header |
| `hlsl_header_fallback` | Fallback header |
| `spv_alias` | SPIR-V aliases |
| `bindless_upload.bytes` / `bindless_upload_vk.bytes` | Bindless upload helpers |
| `accel_process.bytes` / `accel_process_vk.bytes` | Acceleration-structure processing |
| `accel_process_vk_motion.bytes` | Acceleration-structure processing with motion blur |
| `raytracing_motion_header` | Ray tracing motion blur |
| `dx_linalg` | Linear algebra utils (DXIL) |
| `vk_linalg` | Linear algebra utils (SPIR-V) |
| `raytracing_header` | Ray tracing |
| `tex2d_bindless` / `tex3d_bindless` | Bindless textures |
| `compute_quad` | Compute quad ops |
| `determinant` / `inverse` | Matrix ops |
| `indirect` | Indirect dispatch/draw |
| `resource_size` | Resource queries |
| `accel_header` | Acceleration structures |
| `copy_sign` | Sign ops |
| `bindless_common` | Bindless utils |
| `auto_diff` | Autodiff |
| `reduce` | Parallel reduction |
| `oob_runtime` / `oob_flush` | Out-of-range access detection (debug only) |

### DXIL Files (`.dxil`)
| Key | Description |
|---|---|
| `accel_process_vk.dxil` | Vulkan acceleration structures |
| `load_bdls.dxil` / `load_bdls_vk.dxil` | Bindless loading |
| `set_accel4.dxil` | Set acceleration structure |
| `bc6_encodeblock.dxil` / `bc6_trymodeg10.dxil` / `bc6_trymodele10.dxil` | BC6 compression |
| `bc7_encodeblock.dxil` / `bc7_trymode02.dxil` / `bc7_trymode137.dxil` / `bc7_trymode456.dxil` | BC7 compression |

### Adding Builtins
1. Add `.bytes` or `.dxil` to `src/backends/common/hlsl/builtin/`
2. Declare: `LC_HLSL_DECL_VARNAME(my_bytes)`
3. Register: `LC_HLSL_INSERT_VARNAME(my_bytes, "my_key")`

Build: `.hlsl` → `.bytes`, shaders → `.dxil`, embedded via `bin2obj`.

### Out-of-Range Access Detection (debug only)
An out-of-range index into a buffer, bindless array, local array, shared
array or accel instance silently removes the D3D12 device (no exception, no
log). The detector is active only for **host debug builds** (`#ifndef NDEBUG`
gate in `CodegenUtility::Codegen`) with `ShaderOption{.enable_debug_info=true}`
on the **DXIL compute path** (not SPIR-V): it sets `CodegenStackData::oob_check`,
emits `#define _LC_OOB_CHECK 1` plus the `oob_runtime`/`oob_flush` builtins, and
every generated code path is wrapped in `#ifdef _LC_OOB_CHECK`.

| Resource | Guard | Bound source |
|---|---|---|
| buffer read/write/atomic | `_bfread`/`_bfwrite` macros, `AccessChain::call_this_func` | cbuffer `_validate_N` slot (existing debug-validation ABI) |
| bindless slot | `_READ_BUFFER*` macros | cbuffer `_validate_N` slot |
| local / struct-member array | `StringStateVisitor::visit(const AccessExpr *)` | compile-time `Type::dimension()` |
| shared array | same | compile-time `Variable::type()->dimension()` |
| accel instance | `accel_header.bytes` `_LC_OOB_INST_IDX` | `GetDimensions` on the instance buffer |

HLSL has no exceptions, so "quit and return from all function calls" is a
**manually generated multiple return**: `_lc_oob_guard` records the violation
and clamps the index to 0 (so no invalid access reaches the GPU), then
`StringStateVisitor::EmitOobGuard` appends `if(_lc_oob_err){ return; }`
(`return (T)0;` for typed callables) after every statement, and `visit(const/nReturnStmt *)` materializes the returned value before checking. The kernel entry
calls `_lc_oob_exit()`, which flushes kind/index/bound/dispatch-id through the
device-printer ABI, so the host reports it via `Stream::set_log_callback`.
Covered by `src/tests/unit/runtime/test_out_of_range.cpp` (run with `dx`).

### Codegen Debug

Set env `LUISA_DUMP_SOURCE=1` to dump generated HLSL to per-shader files.  
Output: `hlsl_output_<shader_name>.hlsl` in the working directory.

**Naming priority** depends on the backend and shader path:

*DX compute / raster / save paths:*
1. `ShaderOption::name` / `fileName` — user-provided name
2. `Function::name()` — kernel/callable debug name
3. `Function::hash()` formatted as hex — fallback (e.g. `hlsl_output_a1b2c3d4.hlsl`)

*VK internal builtin compute helpers only (`ComputeShader::compile_builtin_hlsl_to_spirv`):*
1. `file_name` — user-provided / cached name
2. Generated HLSL MD5 — fallback

*VK raster (`VkRasterExt`):*
`ShaderOption::name` is always required, so the dump file uses that name.

VK user compute shaders must use native SPIR-V codegen (`LUISA_XIR_TO_SPIRV` or `LUISA_AST_LLVM_TO_SPIRV`); the Vulkan `Function` compute path must not call HLSL/DXC.

Files are written with `"wb"` (overwrite) — no need to delete old files.  
Each shader gets its own file; no more single `hlsl_output.hlsl` with appended content.
