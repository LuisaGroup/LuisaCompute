---
name: hlsl
---

# Code Generation Skill

This skill documents code generation utilities in LuisaCompute, focusing on `StringBuilder` and formatting utilities used for generating shader code.

## Overview

The code generation subsystem uses `vstd::StringBuilder` for efficient string construction when generating HLSL/Shader code. Key utilities include:

- `vstd::StringBuilder` - Efficient string builder for code generation
- `luisa::format()` - Type-safe formatting utility
- `vstd::to_string()` - Integer/float to string conversion

## StringBuilder API

### Header

```cpp
#include <luisa/vstl/string_builder.h>
```

### Basic Usage

```cpp
// Create a StringBuilder
vstd::StringBuilder builder;

// Clear the builder
builder.clear();

// Get current size
size_t sz = builder.size();

// Get string view
vstd::string_view view = builder.view();

// Access data
char* data = builder.data();
```

### Appending Content

StringBuilder supports multiple ways to append content:

```cpp
vstd::StringBuilder str;

// Append string literals (prefer string_view literals with 'sv' suffix)
str.append("Hello");
str += "World"sv;        // Using operator+= with string_view

// Append single characters
str.append(' ');
str << '\n';             // Using operator<<

// Append string_view
vstd::string_view sv = "some text";
str.append(sv);

// Append another StringBuilder
vstd::StringBuilder other;
str.append(other);

// Chain operations
str << "func" << '(' << ")";
```

### Integer to String Conversion

Use `vstd::to_string()` for efficient integer formatting without allocations:

```cpp
vstd::StringBuilder str;

// Convert integers to string
int32_t i = 42;
uint64_t u = 12345;
vstd::to_string(i, str);   // Appends "42"
vstd::to_string(u, str);   // Appends "12345"

// Template version (auto-detects type)
vstd::to_string(some_int, str);
```

### Float to String Conversion

```cpp
vstd::StringBuilder str;
float f = 3.14f;
double d = 2.718;
vstd::to_string(f, str);   // Appends "3.14"
vstd::to_string(d, str);   // Appends "2.718"
```

## luisa::format Usage

`luisa::format()` provides Python-like formatting with `{}` placeholders:

```cpp
#include <luisa/vstl/string_builder.h>  // or appropriate format header

vstd::StringBuilder str;

// Basic formatting
str << luisa::format("{}", 42);           // Appends "42"
str << luisa::format("{}", "hello");      // Appends "hello"

// Multiple arguments
str << luisa::format("{}, {}!", "Hello", "World");  // "Hello, World!"

// Type-specific formatting (hex)
str << luisa::format("{:016X}", hash);    // Zero-padded 16-char hex

// Integer formatting
uint64_t idx = 5;
str << luisa::format("T{}", idx);         // Appends "T5"
str << luisa::format("{}", idx);          // Appends "5"
```

## String View Literals

Use the `sv` suffix for efficient string literals (creates `std::string_view`):

```cpp
using namespace std::string_view_literals;

str += "void"sv;           // No allocation, stores view
str << "func"sv << '(';    // Chain with string_view literals
str.append("template"sv);
```

## Code Generation Patterns

### Pattern 1: Function Declaration Generation

```cpp
void GetFunctionDecl(Function func, vstd::StringBuilder &str) {
    vstd::StringBuilder data;
    
    // Build return type
    if (func.return_type()) {
        CodegenUtility::GetTypeName(*func.return_type(), data, Usage::READ);
    } else {
        data += "void"sv;  // Use string_view literal
    }
    
    // Function name
    data += " "sv;
    GetFunctionName(func, data);
    
    // Arguments
    data += '(';
    for (auto &&arg : func.arguments()) {
        // Process each argument
        GetTypeName(arg.type(), data);
        data << ' ';
        data << arg.name() << ',';
    }
    // Replace last comma with closing paren
    if (!func.arguments().empty()) {
        data[data.size() - 1] = ')';
    } else {
        data += ')';
    }
    
    // Output to result
    str << '\n' << data;
}
```

### Pattern 2: Template Generation

```cpp
void GenerateTemplate(vstd::StringBuilder &str) {
    uint64 tempIdx = 0;
    
    // Generate template parameters
    str << "template<"sv;
    for (uint64 i = 0; i < tempIdx; ++i) {
        str << "typename T"sv;
        vstd::to_string(static_cast<int64_t>(i), str);
        str << ',';
    }
    // Replace trailing comma with '>'
    if (tempIdx > 0) {
        *(str.end() - 1) = '>';
    }
    str << '\n';
}
```

### Pattern 3: Call Expression Generation

```cpp
void GenerateCall(vstd::StringBuilder &str, auto args, auto &vis) {
    str << "func_name"sv;
    str << '(';
    
    // Print arguments with commas
    if (!args.empty()) {
        auto last = args.size() - 1;
        for (size_t i = 0; i < last; ++i) {
            args[i]->accept(vis);  // Visit expression
            str << ',';
        }
        args.back()->accept(vis);
    }
    
    str << ')';
}
```

### Pattern 4: Type-Aware Generation with Conditionals

```cpp
void GenerateTypeSpecificCode(Type const *t, vstd::StringBuilder &str) {
    if (t->is_texture() || t->is_buffer()) {
        // Handle template types
        str << 'T';
        vstd::to_string(tempIdx++, str);
    } else if (t->is_vector()) {
        // Generate vector type name
        str << "float"sv;
        vstd::to_string(t->dimension(), str);
    } else if (t->is_matrix()) {
        // Generate matrix type with dimension
        auto dim = t->dimension();
        auto n = vstd::to_string(dim);
        str << "_float"sv << n << 'x' << n;
    }
}
```

### Pattern 5: RAII-Based Code Wrapping

```cpp
// Helper struct for automatic parentheses wrapping
struct CodeWrapper {
    vstd::StringBuilder *_result;
    
    CodeWrapper(vstd::StringBuilder *result, string_view open)
        : _result(result) {
        if (_result) {
            *_result << open;
        }
    }
    
    ~CodeWrapper() {
        if (_result) {
            *_result << ')';
        }
    }
};

// Usage
void GenerateWrappedCode(vstd::StringBuilder &str) {
    CodeWrapper wrapper{&str, "to_float4x4("};
    // ... generate inner content
    // ')' is automatically appended when wrapper goes out of scope
}
```

### Pattern 6: Modifying Last Character

```cpp
// Common pattern: replace trailing comma with closing delimiter
str << '(';
for (auto &&item : items) {
    str << item << ',';
}
if (!items.empty()) {
    str[str.size() - 1] = ')';  // Replace last comma
} else {
    str << ')';
}
```

## Best Practices

1. **Use `string_view` literals** (`"text"sv`) instead of raw strings to avoid allocations
2. **Use `vstd::to_string()`** for integers instead of `luisa::format()` for better performance
3. **Reserve capacity** when the approximate size is known: `builder.reserve(1024)`
4. **Use `operator<<`** for chaining multiple appends
5. **Use `operator+=`** when appending a single item
6. **Check `size()`** before modifying the last character

## Iteration Support

```cpp
vstd::StringBuilder str;

// Iterate over characters
for (auto it = str.begin(); it != str.end(); ++it) {
    // Process character
}

// Range-based for loop
for (char c : str) {
    // Process character
}

// Erase characters
auto it = str.begin() + 5;
str.erase(it);
```

## Hash and Compare Support

StringBuilder can be used in hash maps:

```cpp
// Hash support
vstd::hash<vstd::StringBuilder> hasher;
size_t h = hasher(builder);

// Compare support
vstd::compare<vstd::StringBuilder> comparer;
int32_t result = comparer(builder1, builder2);  // -1, 0, or 1

// Direct comparison
if (builder1 == builder2) { }
if (builder1 == "literal"sv) { }
```

## HLSL Builtin System

The HLSL backend uses a builtin system for embedding pre-compiled shader code and headers. These builtins are located in `src/backends/common/hlsl/builtin/`.

### Builtin Types

The builtin files are categorized into:

1. **Header Files (`.bytes`)** - HLSL source code headers
2. **DXIL Files (`.dxil`)** - Pre-compiled DXIL shader bytecode

### Header Files (`.bytes`)

| Key | Description |
|-----|-------------|
| `hlsl_header` | Main HLSL header with common definitions |
| `hlsl_header_fallback` | Fallback HLSL header |
| `work_graph` | Work graph shader support |
| `spv_alias` | SPIR-V alias definitions |
| `dx_linalg` | Linear algebra utilities for DirectX |
| `raytracing_header` | Ray tracing shader header |
| `tex2d_bindless` | Bindless 2D texture support |
| `tex3d_bindless` | Bindless 3D texture support |
| `compute_quad` | Compute shader quad operations |
| `determinant` | Matrix determinant computation |
| `inverse` | Matrix inverse computation |
| `indirect` | Indirect dispatch/draw support |
| `resource_size` | Resource size queries |
| `accel_header` | Acceleration structure header |
| `copy_sign` | Copy sign operations |
| `bindless_common` | Common bindless utilities |
| `auto_diff` | Automatic differentiation support |
| `reduce` | Parallel reduction operations |

### DXIL Shader Files (`.dxil`)

| Key | Description |
|-----|-------------|
| `accel_process_vk.dxil` | Vulkan acceleration structure processing |
| `load_bdls.dxil` | Bindless resource loading |
| `load_bdls_vk.dxil` | Vulkan bindless resource loading |
| `set_accel4.dxil` | Set acceleration structure (4-component) |
| `bc6_encodeblock.dxil` | BC6 texture compression - encode block |
| `bc6_trymodeg10.dxil` | BC6 compression - try G10 mode |
| `bc6_trymodele10.dxil` | BC6 compression - try LE10 mode |
| `bc7_encodeblock.dxil` | BC7 texture compression - encode block |
| `bc7_trymode02.dxil` | BC7 compression - try mode 0/2 |
| `bc7_trymode137.dxil` | BC7 compression - try mode 1/3/7 |
| `bc7_trymode456.dxil` | BC7 compression - try mode 4/5/6 |

### Using HLSL Builtins

The builtin system is accessed through `hlsl_builtin.hpp`:

```cpp
#include <backends/common/hlsl/builtin/hlsl_builtin.hpp>

// Retrieve a builtin header by key
auto header = lc_hlsl::get_hlsl_builtin("hlsl_header");
if (header.ptr && header.size > 0) {
    // Use the builtin data
    std::string_view code(static_cast<const char*>(header.ptr), header.size);
}
```

### Adding New Builtins

To add a new builtin:

1. Create your source file in `src/backends/common/hlsl/builtin/`
2. Add the corresponding `.bytes` or `.dxil` file
3. Declare it in `hlsl_builtin.hpp`:

```cpp
// Add declaration in the extern "C" block
LC_HLSL_DECL_VARNAME(my_new_bytes)
```

4. Register it in the dictionary:

```cpp
// Add in Dict constructor
LC_HLSL_INSERT_VARNAME(my_new_bytes, "my_new_key")
```

### Build System Integration

The `.bytes` files are typically generated from HLSL source files (`.hlsl`) and embedded as binary blobs. The DXIL files are pre-compiled shader bytecode.

During the build process:
- HLSL source files are compiled to `.bytes` files
- Shader source files are compiled to `.dxil` bytecode
- The `bin2obj` or similar tool converts these to object files with embedded binary data
- The `LC_HLSL_DECL_VARNAME` macro handles different symbol naming conventions between build tools
