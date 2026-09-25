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
char& ch = builder[0];      // operator[] (both const and non-const)
builder.reserve(1024);
builder.resize(64);
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
vstd::to_string(42, str); // int (faster than format)
vstd::to_string(3.14f, str); // float
vstd::string s = vstd::to_string(42); // single-arg form -> vstd::string
str << luisa::format("{}", 42);
str << luisa::format("{}, {}!", "Hello", "World");
str << luisa::format("{:016X}", hash); // same as luisa::hash_to_string(hash)
```

### String View Literals
```cpp
str += "void"sv; // no allocation
```
`<luisa/vstl/common.h>` already enables the literals: it includes `vstl/vstring.h`, which
includes `vstl/string_hash.h` - and that header has `using namespace std::literals;` at
(global) namespace scope. So generator `.cpp` files write `"..."sv` without their own
`using namespace std::string_view_literals;`.

## Code Generation Patterns

### Function Declaration
```cpp
// real source: CodegenUtility members, codegen_utils/function_codegen.cpp
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
// buffer / texture arguments become template parameters T0, T1, ... (GetFunctionDecl)
if (t->is_texture() || t->is_buffer()) { str << 'T'; vstd::to_string(tempIdx++, str); }
// GetTypeName: element type name followed by the dimension
else if (t->is_vector()) { GetTypeName(*t->element(), str, usage); vstd::to_string(t->dimension(), str); }
// a 3-row matrix is stored as float3x4 (columns are padded to 4)
else if (t->is_matrix()) { GetTypeName(*t->element(), str, usage); vstd::to_string(t->dimension(), str); str << 'x'; vstd::to_string(t->dimension() == 3 ? 4 : t->dimension(), str); }
// matrix element of a buffer/texture is wrapped: "_WrappedFloat" + n + 'x' + n,
// with n = vstd::to_string(t->element()->dimension()) (type_system.cpp, BUFFER case)
```

### RAII Wrapping
```cpp
// StringStateVisitor::Scope - the real RAII wrapper of the generator
struct Scope {
    StringStateVisitor *self;
    explicit Scope(StringStateVisitor *self);            // self->str << "{\n"sv
    ~Scope() { self->str << "}\n"sv; }
};
Scope scope{this};                                       // braces emitted automatically

// non-RAII case (balanced open/close emitted by hand, e.g. aliased struct casts):
AliasedToOrigin(expr->type(), str);
str << '(';
// ... inner content ...
str << ')';
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
// included as "../builtin/hlsl_builtin.hpp" by codegen_utils/entry_points.cpp,
// under `#ifndef LC_NO_HLSL_BUILTIN`
auto header = lc_hlsl::get_hlsl_builtin("hlsl_header"); // {void const* ptr; size_t size;}
std::string_view code(static_cast<const char*>(header.ptr), header.size);
```

Generator and backend code never calls `get_hlsl_builtin` directly - it goes through
`CodegenUtility::ReadInternalHLSLFile(name)` (`codegen_utils/entry_points.cpp`), which
returns the same bytes as a `vstd::string_view`. The DX/VK shader loaders also use it for
pre-compiled blobs: `CacheType::Internal` (`src/backends/dx/Shader/ComputeShader.cpp`)
and `SerdeType::kBuiltin` (`src/backends/vk/shader_serializer.cpp`) try the embedded
asset first, then fall back to `BinaryIO::read_internal_shader`.

With `LC_NO_HLSL_BUILTIN` defined (CMake `LUISA_COMPUTE_VULKAN_ENABLE_DXC_COMPATIBILITY=OFF`,
xmake: `lc_vk_backend_enable_dxc_compatibility` not set) `get_hlsl_builtin` is stubbed out and
returns `{}`, so no builtin is available.

### Header Files (`.bytes`)
Keys are the strings passed to `ReadInternalHLSLFile` / `get_hlsl_builtin`. The backing file is `builtin/<key>` with `.bytes` appended unless the key already carries the extension (three keys below keep `.bytes` in the lookup string), and the BC6/BC7 encoder fragments are extensionless files.

| Key | Description |
|---|---|
| `hlsl_header` | Main HLSL header |
| `hlsl_header_fallback` | Fallback header (`noRegister` path, e.g. DX without DXC) |
| `spv_alias` | SPIR-V aliases (SPIR-V route only) |
| `fallback_rtx_header` | Software ray-tracing traversal (`DeviceConfigExt::use_fallback_rtx()`) |
| `dx_linalg` | Linear algebra utils (DXIL) |
| `vk_linalg` | Linear algebra utils (SPIR-V) |
| `raytracing_header` | Ray tracing |
| `raytracing_motion_header` | Ray tracing motion blur |
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
| `bindless_upload.bytes` | Bindless upload source -> compiled to `load_bdls.dxil` |
| `accel_process.bytes` | Acceleration-structure processing source -> `set_accel4.dxil` |
| `accel_process_vk_motion.bytes` | Acceleration-structure processing with motion blur |
| `bc6_header` / `bc6_encode_block` / `bc6_trymode_g10cs` / `bc6_trymode_le10cs` | BC6 encoder HLSL fragments |
| `bc7_header` / `bc7_encode_block` / `bc7_trymode_02cs` / `bc7_trymode_137cs` / `bc7_trymode_456cs` | BC7 encoder HLSL fragments |

`builtin/bindless_upload`, `builtin/bindless_upload_vk` and `builtin/load_bdls_dxil` are
extensionless files that are *not* embedded (neither the CMake list nor the xmake globs
match them) and have no dict key - the registered sources are `bindless_upload.bytes` and
`load_bdls.dxil`.

### DXIL Files (`.dxil`)
Pre-compiled builtin shader artifacts (`ShaderSerializer` containers), keyed with their full file name.

| Key | Description |
|---|---|
| `accel_process_vk.dxil` | Vulkan acceleration structures |
| `load_bdls.dxil` / `load_bdls_vk.dxil` | Bindless loading |
| `set_accel4.dxil` | Set acceleration structure |
| `bc6_encodeblock.dxil` / `bc6_trymodeg10.dxil` / `bc6_trymodele10.dxil` | BC6 compression |
| `bc7_encodeblock.dxil` / `bc7_trymode02.dxil` / `bc7_trymode137.dxil` / `bc7_trymode456.dxil` | BC7 compression |

`*_vk.dxil` are legacy HLSL-to-SPIR-V (`dxc -spirv`) containers: the current Vulkan
builtin kernels load `.spv` artifacts embedded as `luisa_compute_vk_builtin_*`
(`src/backends/vk/builtin_kernel.cpp`), and `scripts/verify_compile_builtin.py` treats
`load_bdls_vk.dxil` as a stale v2 container.

### Adding Builtins
1. Add `.bytes` or `.dxil` to `src/backends/common/hlsl/builtin/` - e.g. `my_helper.bytes`.
   A `.bytes` asset is hand-written HLSL text that is edited in place (nothing generates it
   from a `.hlsl`).
2. Declare the symbol `luisa_embed_device_lib` / `bin2obj` produces for that file (the file
   stem with its extension turned into `_`, because of `--preserve-ext`), inside
   `extern "C" { ... }` in `builtin/hlsl_builtin.hpp`:
   `LC_HLSL_DECL_VARNAME(my_helper_bytes)`
3. Register the lookup key: `LC_HLSL_INSERT_VARNAME(my_helper_bytes, "my_helper")`
4. CMake: append `my_helper.bytes` to `LUISA_COMPUTE_HLSL_DEVICE_LIB_SOURCES` in
 `src/backends/common/hlsl/CMakeLists.txt` - the list is explicit, new files are not globbed.
5. xmake: nothing to do for `.bytes`/`.dxil` (`src/backends/common/hlsl/xmake.lua` globs
   `builtin/*.bytes` and `builtin/*.dxil`); extensionless assets need their own
   `add_files(...)` entry with `{rules = "utils.bin2obj"}`.

Embedding: xmake uses the `utils.bin2obj` rule plus `add_defines('LUISA_BIN_2_OBJ')`, so the
`_binary_<name>_start/_end` symbols are produced per object file. CMake runs the host tool
`luisa_embed_device_lib` (`utils/embed_device_lib.cpp`) with `--unsigned ... --preserve-ext`
in the `luisa-compute-backend-hlsl-embed` target to regenerate
`builtin/hlsl_builtin_embed.cpp` (gitignored). That target runs the tool twice - once to
write, once with `--check` - and `luisa-compute-backend-hlsl-embed-check` is the standalone
check-only variant; `luisa-compute-hlsl-codegen` depends on the embed target.
`--preserve-ext` is why `hlsl_header.bytes` becomes the symbol `hlsl_header_bytes`.

Regenerating a compiled `.dxil` builtin from its `.bytes` source (same compiler as the
runtime, `src/backends/tools/main.cpp`):

```text
xmake build lc_compile_builtin
xmake run lc_compile_builtin dx src/backends/common/hlsl/builtin/bindless_upload.bytes \
  src/backends/common/hlsl/builtin/load_bdls.dxil --name load_bdls.dxil --verify
xmake run lc_compile_builtin vk <input-shader> <output> # dxc -spirv, VK v10 container
xmake run lc_compile_builtin dx|vk inspect <artifact> # decode contract fields
```

`--shader-model <n>` is a packed model (`62` == `cs_6_2`, default `62`); DX runtime
selection is `kShaderModel = 65`, `kHighShaderModel = 66` (allowed warp size),
`kTensorShaderModel = 69` (cooperative ops / 8-bit, `src/backends/dx/DXApi/LCDevice.cpp`).
Dumped HLSL can also be compiled by hand: `python scripts/compile_dxil.py [source]
[--shader-model 6_5|6_6|6_9] [-E main] [--dxc <dxc.exe>] [-o out.dxil]` (dxc is
auto-searched: `D:\DirectXShaderCompiler\build\bin\dxc.exe`, then PATH, then
`bin/debug/dxc.exe`). `scripts/compile_builtin.py dx|vk <output> [--name <kernel>] [--no-build]`
is a thin wrapper that first builds `lc_compile_builtin` in release.

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
(`return (T)0;` for typed callables) after every statement of `visit(const ScopeStmt *)`, and `visit(const ReturnStmt *)` materializes the returned value before checking. The kernel entry
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

*VK compute on the HLSL-to-SPIR-V compatibility route (`ComputeShader::compile`, `src/backends/vk/compute_shader.cpp`):*
1. `file_name` — `ShaderOption::name`, else the cache name `<md5>.spv` when `enable_cache`
2. MD5 of the generated HLSL — fallback

*VK ray tracing (`RayTracingShader::compile`, `src/backends/vk/rt_shader.cpp`):*
`file_name` → `code_md5` → literal `rt_unknown`.

*VK raster (`VkRasterExt`):*
`ShaderOption::name` is always required (`LUISA_ASSERT(!option.name.empty(), "Raster shader name must not be empty.")`), so the dump file uses that name.

VK user compute normally goes through native XIR-to-SPIR-V; it falls back to this HLSL route only for features the native codegen cannot express - `native_include`, device printing, `async_copy`/pipeline commit+wait, motion blur, or software (fallback-RTX) ray tracing (`lc::vk::detail::plan_user_compute_codegen_route` in `src/backends/vk/user_compute_codegen_route.h`). `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` forbids the HLSL fallback, and `LC_NO_HLSL_BUILTIN` (no DXC compatibility) removes the HLSL route entirely - then those features must be compiled by `LUISA_XIR_TO_SPIRV` or `LUISA_AST_LLVM_TO_SPIRV`.

Per-shader dumps are written with `"wb"` (overwrite). The one exception is the VK `option.compile_only` path (`src/backends/vk/device.cpp`), which still appends everything to a single `hlsl_output.hlsl` with `"ab"`.
