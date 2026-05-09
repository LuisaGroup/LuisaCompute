---
name: hlsl
---

# HLSL Code Generation

Code generation utilities for HLSL shaders using `vstd::StringBuilder` and formatting utilities.

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
vstd::StringBuilder str;
str.append("Hello");
str += "World"sv;          // string_view literal
str.append(' ');
str << '\n';               // chain with operator<<
str.append(other_builder);
str << "func" << '(' << ")";
```

### Number Conversion

```cpp
// Integers - use vstd::to_string() for performance
vstd::to_string(42, str);       // appends "42"
vstd::to_string(uint64_val, str);

// Floats
vstd::to_string(3.14f, str);
vstd::to_string(2.718, str);
```

### luisa::format

```cpp
str << luisa::format("{}", 42);
str << luisa::format("{}, {}!", "Hello", "World");
str << luisa::format("{:016X}", hash);  // zero-padded hex
```

## String View Literals

```cpp
using namespace std::string_view_literals;
str += "void"sv;           // no allocation
```

## Code Generation Patterns

### Function Declaration

```cpp
void GetFunctionDecl(Function func, vstd::StringBuilder &str) {
    vstd::StringBuilder data;
    if (func.return_type()) {
        CodegenUtility::GetTypeName(*func.return_type(), data, Usage::READ);
    } else {
        data += "void"sv;
    }
    data += " "sv;
    GetFunctionName(func, data);
    data += '(';
    for (auto &&arg : func.arguments()) {
        GetTypeName(arg.type(), data);
        data << ' ' << arg.name() << ',';
    }
    if (!func.arguments().empty()) {
        data[data.size() - 1] = ')';
    } else {
        data += ')';
    }
    str << '\n' << data;
}
```

### Template Parameters

```cpp
str << "template<"sv;
for (uint64 i = 0; i < tempIdx; ++i) {
    str << "typename T"sv;
    vstd::to_string(static_cast<int64_t>(i), str);
    str << ',';
}
if (tempIdx > 0) {
    *(str.end() - 1) = '>';
}
```

### Call Expression

```cpp
str << "func_name"sv << '(';
if (!args.empty()) {
    for (size_t i = 0; i < args.size() - 1; ++i) {
        args[i]->accept(vis);
        str << ',';
    }
    args.back()->accept(vis);
}
str << ')';
```

### Type-Aware Generation

```cpp
if (t->is_texture() || t->is_buffer()) {
    str << 'T';
    vstd::to_string(tempIdx++, str);
} else if (t->is_vector()) {
    str << "float"sv;
    vstd::to_string(t->dimension(), str);
} else if (t->is_matrix()) {
    auto n = vstd::to_string(t->dimension());
    str << "_float"sv << n << 'x' << n;
}
```

### RAII Wrapping

```cpp
struct CodeWrapper {
    vstd::StringBuilder *_result;
    CodeWrapper(vstd::StringBuilder *r, string_view open) : _result(r) {
        if (_result) *_result << open;
    }
    ~CodeWrapper() {
        if (_result) *_result << ')';
    }
};

CodeWrapper wrapper{&str, "to_float4x4("};
// inner content generated here
// ')' appended automatically
```

### Trailing Comma Fix

```cpp
str << '(';
for (auto &&item : items) {
    str << item << ',';
}
if (!items.empty()) {
    str[str.size() - 1] = ')';
} else {
    str << ')';
}
```

## Iteration & Hash

```cpp
for (char c : str) { }
str.erase(str.begin() + 5);

vstd::hash<vstd::StringBuilder> hasher;
size_t h = hasher(builder);

if (builder1 == builder2) { }
if (builder1 == "literal"sv) { }
```

## Best Practices

1. Use `"text"sv` literals to avoid allocations
2. Use `vstd::to_string()` for integers (faster than `luisa::format()`)
3. `builder.reserve(1024)` when size is known
4. Use `operator<<` for chaining, `operator+=` for single items
5. Check `size()` before modifying last character

## HLSL Builtins

Located in `src/backends/common/hlsl/builtin/`. Access via `hlsl_builtin.hpp`:

```cpp
#include <backends/common/hlsl/builtin/hlsl_builtin.hpp>
auto header = lc_hlsl::get_hlsl_builtin("hlsl_header");
std::string_view code(static_cast<const char*>(header.ptr), header.size);
```

### Header Files (`.bytes`)

| Key | Description |
|-----|-------------|
| `hlsl_header` | Main HLSL header |
| `hlsl_header_fallback` | Fallback header |
| `work_graph` | Work graph shaders |
| `spv_alias` | SPIR-V aliases |
| `dx_linalg` | Linear algebra utils |
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

### DXIL Files (`.dxil`)

| Key | Description |
|-----|-------------|
| `accel_process_vk.dxil` | Vulkan acceleration structures |
| `load_bdls.dxil` / `load_bdls_vk.dxil` | Bindless loading |
| `set_accel4.dxil` | Set acceleration structure |
| `bc6_encodeblock.dxil` / `bc6_trymodeg10.dxil` / `bc6_trymodele10.dxil` | BC6 compression |
| `bc7_encodeblock.dxil` / `bc7_trymode02.dxil` / `bc7_trymode137.dxil` / `bc7_trymode456.dxil` | BC7 compression |

### Adding Builtins

1. Add `.bytes` or `.dxil` file to `src/backends/common/hlsl/builtin/`
2. Declare: `LC_HLSL_DECL_VARNAME(my_bytes)`
3. Register: `LC_HLSL_INSERT_VARNAME(my_bytes, "my_key")`

Build system compiles `.hlsl` → `.bytes` and shaders → `.dxil`, then embeds via `bin2obj`.

### Codegen Debug

1. set environment var `LUISA_DUMP_SOURCE`.
2. delete old `hlsl_output.hlsl` on binary dir if exists.
    - note: new codegen result append to `hlsl_output.hlsl`.
3. run program.
4. read new `hlsl_output.hlsl`.