---
name: lc_dsl
---

# LuisaCompute DSL (Domain Specific Language) Usage Guide

This skill documents the usage patterns for the LuisaCompute DSL based on test cases in `src/tests/test_dsl.cpp`, `src/tests/test_dsl_sugar.cpp`, `src/tests/test_path_tracing.cpp`, and `src/tests/test_atomic.cpp`.

## Table of Contents

1. [Overview](#overview)
2. [Headers and Namespaces](#headers-and-namespaces)
3. [Kernel Definitions](#kernel-definitions)
4. [Callable Functions](#callable-functions)
5. [Struct Definitions](#struct-definitions)
6. [Variable Declarations](#variable-declarations)
7. [Buffer Operations](#buffer-operations)
8. [Control Flow](#control-flow)
9. [Atomic Operations](#atomic-operations)
10. [Shared Memory](#shared-memory)
11. [Constants](#constants)
12. [Type Casting](#type-casting)
13. [Sugar Syntax](#sugar-syntax)
14. [Dispatch and Thread IDs](#dispatch-and-thread-ids)

---

## Overview

The LuisaCompute DSL allows you to write GPU kernels and callable functions using C++ syntax that gets compiled to GPU shader code. Key concepts:

- **Kernel**: Entry point for GPU execution (Kernel1D, Kernel2D, Kernel3D)
- **Callable**: Reusable functions that can be called from kernels
- **Var<T>**: DSL variable wrapper for type T
- **Struct Registration**: C++ structs must be registered with LUISA_STRUCT macro

---

## Headers and Namespaces

```cpp
#include <luisa/dsl/syntax.h>      // Core DSL syntax
#include <luisa/dsl/sugar.h>       // Syntactic sugar macros
#include <luisa/dsl/struct.h>      // Struct registration macros

using namespace luisa;
using namespace luisa::compute;
```

---

## Kernel Definitions

Kernels are the entry points for GPU execution. They capture variables from the host scope automatically.

### Kernel Types

```cpp
// 1D kernel - processes linear data
Kernel1D kernel1d = [](BufferVar<float> buffer, Var<uint> count) noexcept {
    auto idx = dispatch_id().x;
    buffer.write(idx, buffer.read(idx) + 1.0f);
};

// 2D kernel - processes 2D data (images, textures)
Kernel2D kernel2d = [](ImageFloat image) noexcept {
    UInt2 coord = dispatch_id().xy();
    Float4 color = image.read(coord);
    image.write(coord, color * 2.0f);
};

// 3D kernel - processes volumetric data
Kernel3D kernel3d = [](VolumeFloat volume) noexcept {
    UInt3 coord = dispatch_id().xyz();
    // Process 3D volume data
};
```

### Kernel Compilation and Dispatch

```cpp
// Compile the kernel
auto shader = device.compile(kernel1d);

// Dispatch with specified thread count
stream << shader(buffer, count).dispatch(1024u);

// For 2D kernels
stream << shader2d(image).dispatch(width, height);

// Set kernel name (for debugging)
kernel1d.function_builder()->set_name("my_kernel");
// Or using DSL sugar:
Kernel2D kernel = []() noexcept {
    set_name("my_kernel");
    // kernel body
};
```

### Block Size Configuration

```cpp
Kernel2D kernel = []() noexcept {
    set_block_size(16u, 16u, 1u);  // 16x16 threads per block
    // kernel body
};
```

---

## Callable Functions

Callables are reusable functions that can capture variables and be called from kernels.

### Basic Callable

```cpp
// Simple callable
Callable add = [](Var<int> a, Var<int> b) noexcept {
    a.set_name("a");  // Name parameters for debugging
    b.set_name("b");
    return a + b;
};

// Callable with explicit return type
Callable<float(float, float)> multiply = [](Var<float> a, Var<float> b) noexcept {
    return a * b;
};

// Template callable
Callable<int(int, int)> add = []<typename T>(Var<T> a, Var<T> b) noexcept {
    return cast<int>(a + b);
};
```

### Callable Captures

```cpp
Buffer<float> buffer = device.create_buffer<float>(1024);
Buffer<float> another_buffer = device.create_buffer<float>(1024);

// Callable captures 'buffer' automatically
Callable c1 = [&buffer](UInt a) noexcept {
    return buffer->read(a);
};

// Callable c2 captures both c1's captures AND 'another_buffer'
Callable c2 = [&c1, &another_buffer](UInt b) noexcept {
    return c1(b) + another_buffer->read(b);
};

// Kernel captures all callable captures transitively
Kernel1D kernel = [&c2] {
    auto v = c2(dispatch_x());
    // ...
};
```

### Returning Multiple Values

```cpp
// Use compose() to return multiple values as a tuple
Callable add_mul = [](Var<int> a, Var<int> b) noexcept {
    return compose(a + b, a * b);
};

// Unpack results
Kernel1D kernel = [&add_mul] {
    Var am = add_mul(3, 4);
    Var sum = am.get<0>();      // 7
    Var product = am.get<1>();  // 12
};
```

---

## Struct Definitions

C++ structs must be registered with the DSL to be used in kernels.

### Basic Struct Registration

```cpp
#include <luisa/dsl/struct.h>

struct Point3D {
    float3 v;
};

struct Material {
    float3 albedo;
    float roughness;
    float metallic;
};

// Register with LUISA_STRUCT macro
LUISA_STRUCT(Point3D, v) {};
LUISA_STRUCT(Material, albedo, roughness, metallic) {};
```

### Struct with Methods

```cpp
struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

LUISA_STRUCT(Onb, tangent, binormal, normal) {
    // Add methods that work in DSL
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};

// Usage in kernel
Kernel1D kernel = [&local_vec] {
    Var<Onb> onb = ...;
    Float3 world_vec = onb->to_world(local_vec);
};
```

### Struct with Arrays

```cpp
struct TriArray {
    int v[3];
};

struct MDArray {
    int v[2][3][4];
};

LUISA_STRUCT(TriArray, v) {};
LUISA_STRUCT(MDArray, v) {};
```

### Template Structs

```cpp
template<typename IndexType, typename ValueType>
struct KeyValuePair {
    IndexType key;
    ValueType value;
};

// Define template macros
#define LUISA_KEY_VALUE_PAIR_TEMPLATE() \
    template<typename IndexType, typename ValueType>
#define LUISA_KEY_VALUE_PAIR() KeyValuePair<IndexType, ValueType>

// Register template struct
LUISA_TEMPLATE_STRUCT(LUISA_KEY_VALUE_PAIR_TEMPLATE, LUISA_KEY_VALUE_PAIR, key, value) {};

// Usage
Kernel1D kernel = [] {
    Var<KeyValuePair<int, float>> kvp{10, 3.14f};
    Var<int> k = kvp.key;
    Var<float> v = kvp.value;
};
```

### Using Structs in Kernels

```cpp
Kernel1D kernel = [] {
    // Default construction
    Var<Point3D> p1;
    
    // Construction with initial value
    Var<Point3D> p2{make_float3(1.0f)};
    
    // Copy construction
    Var<Point3D> p3{p2};
    
    // Member access
    Var<float3> pos = p2.v;
    p2.v = make_float3(2.0f, 3.0f, 4.0f);
};
```

---

## Variable Declarations

### Var<T> Types

```cpp
// Explicit type
Var<float> f;
Var<int3> iv;
Var<float4x4> m;

// Auto type deduction
Var v = 10;                    // Becomes Var<int>
Var v = make_float3(1.0f);     // Becomes Var<float3>

// Initialization
Var<float> x = 1.0f;
Var<int3> t{make_int3(1, 2, 3)};
Var<float2> w{cast<float>(v_int), v_float};
```

### Type Aliases

```cpp
// DSL type aliases
using Float = Var<float>;
using Float2 = Var<float2>;
using Float3 = Var<float3>;
using Float4 = Var<float4>;
using Int = Var<int>;
using UInt = Var<uint>;
using UInt2 = Var<uint2>;
using UInt3 = Var<uint3>;
using Bool = Var<bool>;
```

### Literal Suffixes (dsl_literals)

```cpp
using namespace dsl_literals;

auto lx = 0._half;        // Var<half>
auto ly = 0._float;       // Var<float>
auto lz = 0_ulong2;       // Var<ulong2>
```

---

## Buffer Operations

### Basic Buffer Operations

```cpp
Kernel1D kernel = [&index, &new_value, &captured_buffer](BufferVar<float> buffer) noexcept {
    // Read from buffer
    Var<float> val = buffer.read(index);
    
    // Write to buffer
    buffer.write(index, new_value);
    
    // Read from captured buffer
    Var<float4> vec4 = captured_buffer->read(index);
};
```

### Volatile Operations

```cpp
Kernel1D kernel = [](BufferVar<float> buffer, 
                      BufferVar<int3> b0,
                      BufferVar<float4x4> b1,
                      Var<ByteBuffer> bb) noexcept {
    // Volatile read/write for untyped/coherent memory access
    b0.volatile_write(1, b0.volatile_read(0));
    b1.volatile_write(1, b1.volatile_read(0));
    bb.volatile_write(16, bb.volatile_read<float3>(1));
    bb.volatile_write(16, bb.volatile_read<float3x3>(1));
    
    // Regular volatile operations
    Var<float> val = buffer.volatile_read(index);
    buffer.volatile_write(index, val);
};
```

### Buffer Variable Types

```cpp
// In kernel parameters
Kernel1D kernel = [](BufferVar<float> buf,           // Generic buffer
                      BufferFloat float_buf,           // Typed buffer alias
                      BufferUInt uint_buf,             // Typed buffer alias
                      BufferVar<MyStruct> struct_buf   // Custom struct buffer
                      ) noexcept {
    // ...
};
```

---

## Control Flow

### If Statements

```cpp
// Basic if
if_(condition, [] {
    // true branch
});

// If-else
if_(v_int == v_int, [] {
    Var a = 0.0f;
}).else_([] {
    Var c = 2.0f;
});

// If-elif-else chain
if_(condition1, [] {
    // case 1
}).elif_(condition2, [] {
    // case 2
}).else_([] {
    // default
});
```

### Switch Statements

```cpp
switch_(123)
    .case_(1, [] {
        // case 1
    })
    .case_(2, [] {
        // case 2
    })
    .default_([] {
        // default case
    });
```

### Loops

```cpp
// While-style loop with break
loop([] {
    if_(true, break_);
});

// For loop with dynamic range
Var<int> count = 10;
for (auto v : dynamic_range(count)) {
    v_int += v;
}

// Range-based for with index
$for (i, 10u) {
    // i is Var<uint> from 0 to 9
};
```

### Conditional Operators

```cpp
// Ternary operator: ite (if-then-else)
Var vv = ite(t == 10, 1, 2);  // if t==10 then 1 else 2

// min/max
Var vvv = min(vv, 10);
Var min_val = min(xxx, 1u);
```

---

## Atomic Operations

Atomic operations provide thread-safe concurrent memory access.

### Atomic Operations on Buffers

```cpp
Kernel1D kernel = [&buffer](BufferUInt counter_buffer) noexcept {
    // fetch_add: atomically add and return old value
    Var x = buffer->atomic(3u).fetch_add(1u);
    
    // fetch_sub: atomically subtract and return old value
    buffer.atomic(0u).fetch_sub(-1.f);
    
    // fetch_max: atomically compute max and return old value
    buffer.atomic(0u).fetch_max(100u);
    
    // compare_exchange: atomic compare-and-swap
    buffer.atomic(0u).compare_exchange(expected, new_value);
};
```

### Vector Component Atomics

```cpp
Kernel1D kernel = [](BufferFloat3 buffer) noexcept {
    // Atomic on vector component
    buffer.atomic(0u).x.fetch_add(1.f);
};
```

### Matrix Element Atomics

```cpp
Kernel1D kernel = [](BufferFloat2x2 buffer) noexcept {
    // Atomic on matrix element
    buffer.atomic(0u)[1].x.fetch_add(1.f);  // [column][row]
};
```

### Nested Array Atomics

```cpp
Kernel1D kernel = [](BufferVar<std::array<std::array<float4, 3u>, 5u>> buffer) noexcept {
    buffer.atomic(0u)[1][2][3].fetch_add(1.f);
};
```

### Struct Member Atomics

```cpp
struct Something {
    uint x;
    float3 v;
};
LUISA_STRUCT(Something, x, v) {};

Kernel1D kernel = [](BufferVar<Something> buffer) noexcept {
    auto a = buffer.atomic(0u);
    a.v.x.fetch_max(1.f);
};
```

---

## Shared Memory

Shared memory is fast on-chip memory shared among threads in a block.

### Shared Memory Declaration

```cpp
Kernel1D kernel = []() noexcept {
    // Allocate shared memory
    Shared<float4> shared_floats{16};  // 16 float4 elements
    Shared<float> shared_array{256};   // 256 float elements
    
    // Read/write shared memory
    shared_floats[thread_x()] = make_float4(1.0f);
    Var<float4> val = shared_floats[thread_x()];
};
```

### Atomic Operations on Shared Memory

```cpp
Kernel1D kernel = []() noexcept {
    Shared<float> s{16};
    
    // Atomic compare-exchange on shared memory
    s.atomic(0).compare_exchange(0.f, 1.f);
    
    // Other atomic operations
    s.atomic(0).fetch_add(1.f);
};
```

---

## Constants

Constants are read-only data embedded in the shader.

### Constant Declaration

```cpp
Kernel1D kernel = []() noexcept {
    // Constant array from initializer list
    Constant float_consts = {1.0f, 2.0f};
    
    // Constant from host vector
    std::vector<int> const_vector{1, 2, 3, 4};
    Constant int_consts = const_vector;
    
    // Access constants
    Var<float> ff = float_consts.read(0);
    Var<int> iv = int_consts[index];
};
```

### Captured Constants

```cpp
// Create constant outside kernel
Constant float_consts = {1.0f, 2.0f};

Kernel1D kernel = [&float_consts]() noexcept {
    // Access captured constant
    Var<float> val = float_consts[0];
};
```

---

## Type Casting

### Explicit Casting

```cpp
Kernel1D kernel = [&buffer, &a, &b]() noexcept {
    Var<int> i = 10;
    Var<float> f = 3.14f;
    
    // Cast to different type
    Var<float> i_as_f = cast<float>(i);
    Var<int> f_as_i = cast<int>(f);
    
    // Cast in expressions
    Var<int> result = cast<int>(buffer->read(a + b));
};
```

### Method-style Casting

```cpp
Kernel1D kernel = []() noexcept {
    Var<int> i = 10;
    Var<float> f = i.cast<float>();  // Method syntax
};
```

---

## Sugar Syntax

The DSL provides syntactic sugar macros for more concise code.

### Type Sugar ($ prefix)

```cpp
#include <luisa/dsl/sugar.h>

// $ prefix creates Var<T> types
$int a;           // Same as Var<int>
$float b;         // Same as Var<float>
$float3 c;        // Same as Var<float3>
$uint2 d;         // Same as Var<uint2>

// Auto deduction with $
$ v = 10;          // Becomes $int
$ f = 1.0f;        // Becomes $float
```

### Keyword Sugar

```cpp
// $constant
$constant float_consts = {1.0f, 2.0f};
$constant int_consts = const_vector;

// $shared
$shared<float4> shared_floats{16};

// $array (local array)
$array<float, 5> array;

// $buffer type
Kernel1D kernel = &[$]($buffer<float> buf, $uint count) {
    // ...
};
```

### Control Flow Sugar

```cpp
// $if / $elif / $else
$if (w.x < 5) {
    // true branch
}
$elif (w.x > 0) {
    // elif branch  
}
$else {
    // else branch
};

// $loop / $break
$loop {
    $break;
};

// $switch / $case / $default
$switch (123) {
    $case (1) {
        // case 1
    };
    $default {
        // default
    };
};

// $for
$for (x, 1) {  // Loop from 0 to count-1
    array[x] = cast<float>(v_int);
};
```

---

## Dispatch and Thread IDs

### Dispatch IDs

```cpp
Kernel1D kernel1d = []() noexcept {
    // 1D dispatch ID
    UInt idx = dispatch_id().x;
    UInt x = dispatch_x();  // Shorthand
};

Kernel2D kernel2d = []() noexcept {
    // 2D dispatch ID
    UInt2 coord = dispatch_id().xy();
    UInt2 size = dispatch_size().xy();
};

Kernel3D kernel3d = []() noexcept {
    // 3D dispatch ID
    UInt3 coord = dispatch_id().xyz();
};
```

### Thread IDs

```cpp
Kernel1D kernel = []() noexcept {
    // Thread ID within the block
    UInt tx = thread_id().x;
    UInt tx_shorthand = thread_x();
    
    // Block ID
    UInt bx = block_id().x;
    
    // Block size
    UInt bs = block_size().x;
};
```

---

## Complete Example

```cpp
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

// Define and register a struct
struct Particle {
    float3 position;
    float3 velocity;
    float mass;
};
LUISA_STRUCT(Particle, position, velocity, mass) {};

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    
    // Create buffer
    Buffer<Particle> particles = device.create_buffer<Particle>(1024);
    
    // Define callable
    Callable update_position = []($Particle p, $float dt) noexcept {
        p.position = p.position + p.velocity * dt;
        return p;
    };
    
    // Define kernel
    Kernel1D update_kernel = [&update_position]($buffer<Particle> buf, $float dt) noexcept {
        $ idx = dispatch_x();
        $ p = buf.read(idx);
        p = update_position(p, dt);
        buf.write(idx, p);
    };
    
    // Compile and dispatch
    auto shader = device.compile(update_kernel);
    stream << shader(particles, 0.016f).dispatch(1024)
           << synchronize();
    
    return 0;
}
```

---

## Summary Table

| Feature | Syntax | Description |
|---------|--------|-------------|
| Kernel1D | `Kernel1D k = [](...) { ... };` | 1D compute kernel |
| Kernel2D | `Kernel2D k = [](...) { ... };` | 2D compute kernel |
| Kernel3D | `Kernel3D k = [](...) { ... };` | 3D compute kernel |
| Callable | `Callable c = [](...) { ... };` | Reusable function |
| Struct | `LUISA_STRUCT(Name, m1, m2) {}` | Register struct |
| Template Struct | `LUISA_TEMPLATE_STRUCT(...)` | Register template struct |
| Variable | `Var<T> v` or `$T v` | DSL variable |
| Buffer Read | `buf.read(idx)` | Read from buffer |
| Buffer Write | `buf.write(idx, val)` | Write to buffer |
| Atomic | `buf.atomic(idx).fetch_add(1)` | Atomic operation |
| Shared | `Shared<T> s{n}` | Shared memory |
| Constant | `Constant c = { ... }` | Constant data |
| Cast | `cast<T>(value)` | Type casting |
| If | `if_(cond, [] { ... })` | Conditional |
| Switch | `switch_(val).case_(...)` | Switch statement |
| Loop | `loop([] { ... })` | While-style loop |
| For | `$for (i, n) { ... }` | Range-based for |
| Dispatch ID | `dispatch_id().x` | Global thread ID |
| Thread ID | `thread_id().x` | Local thread ID |
