---
name: lc_dsl
description: LuisaCompute DSL — kernels, callables, structs, buffers, atomics, control flow, sugar syntax, and dispatch
---

# LuisaCompute DSL Usage Guide

Based on test cases in `src/tests/test_dsl.cpp`, `test_dsl_sugar.cpp`, `test_path_tracing.cpp`, `test_atomic.cpp`.

## Headers

```cpp
#include <luisa/dsl/syntax.h>   // core DSL
#include <luisa/dsl/sugar.h>    // syntactic sugar macros
#include <luisa/dsl/struct.h>   // struct registration
using namespace luisa;
using namespace luisa::compute;
```

Key concepts: **Kernel** (GPU entry, 1D/2D/3D), **Callable** (reusable function), **Var<T>** (DSL variable), **LUISA_STRUCT** (register C++ structs).

## Kernel Definitions

```cpp
Kernel1D k1d = [](BufferVar<float> buf, Var<uint> count) noexcept {
    auto idx = dispatch_id().x;
    buf.write(idx, buf.read(idx) + 1.0f);
};
Kernel2D k2d = [](ImageFloat img) noexcept {
    UInt2 coord = dispatch_id().xy();
    img.write(coord, img.read(coord) * 2.0f);
};
Kernel3D k3d = [](VolumeFloat vol) noexcept {
    UInt3 coord = dispatch_id().xyz();
};
```

### Compilation & Dispatch
```cpp
auto shader = device.compile(kernel);
stream << shader(buf, count).dispatch(1024u);            // 1D
stream << shader2d(img).dispatch(width, height);         // 2D
kernel.function_builder()->set_name("my_kernel");        // debug name
// Or inline:
Kernel2D k = []() noexcept { set_name("my_kernel"); /* ... */ };
```

### Block Size
```cpp
Kernel2D k = []() noexcept { set_block_size(16u, 16u, 1u); /* ... */ };
```

## Callable Functions

```cpp
Callable add = [](Var<int> a, Var<int> b) noexcept { a.set_name("a"); b.set_name("b"); return a + b; };
Callable<float(float, float)> mul = [](Var<float> a, Var<float> b) noexcept { return a * b; };
Callable<int(int, int)> add_t = []<typename T>(Var<T> a, Var<T> b) noexcept { return cast<int>(a + b); };
```

### Captures (transitive)
```cpp
Buffer<float> buf = device.create_buffer<float>(1024);
Callable c1 = [&buf](UInt a) noexcept { return buf->read(a); };
Callable c2 = [&c1, &another_buffer](UInt b) noexcept { return c1(b) + another_buffer->read(b); };
// Kernel captures c2 → automatically captures buf + another_buffer
Kernel1D k = [&c2] { auto v = c2(dispatch_x()); };
```

### Multiple Return Values
```cpp
Callable add_mul = [](Var<int> a, Var<int> b) noexcept { return compose(a + b, a * b); };
// Unpack: Var am = add_mul(3, 4); Var sum = am.get<0>(); Var prod = am.get<1>();
```

## Struct Definitions

```cpp
struct Point3D { float3 v; };
struct Material { float3 albedo; float roughness; float metallic; };
LUISA_STRUCT(Point3D, v) {};
LUISA_STRUCT(Material, albedo, roughness, metallic) {};
```

### With Methods
```cpp
struct Onb { float3 tangent, binormal, normal; };
LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};
// Usage: Var<Onb> onb; Float3 world = onb->to_world(local_vec);
```

### Arrays & Templates
```cpp
struct TriArray { int v[3]; };
struct MDArray { int v[2][3][4]; };
LUISA_STRUCT(TriArray, v) {};
LUISA_STRUCT(MDArray, v) {};

template<typename I, typename V>
struct KeyValuePair { I key; V value; };
#define LUISA_KEY_VALUE_PAIR_TEMPLATE() template<typename I, typename V>
#define LUISA_KEY_VALUE_PAIR() KeyValuePair<I, V>
LUISA_TEMPLATE_STRUCT(LUISA_KEY_VALUE_PAIR_TEMPLATE, LUISA_KEY_VALUE_PAIR, key, value) {};
// Usage: Var<KeyValuePair<int, float>> kvp{10, 3.14f}; Var<int> k = kvp.key;
```

### Usage in Kernels
```cpp
Var<Point3D> p1;                          // default
Var<Point3D> p2{make_float3(1.0f)};       // init
Var<Point3D> p3{p2};                      // copy
Var<float3> pos = p2.v; p2.v = make_float3(2,3,4);
```

## Variables

```cpp
Var<float> f; Var<int3> iv; Var<float4x4> m;
Var v = 10;                    // Var<int>
Var v2 = make_float3(1.0f);    // Var<float3>

// Aliases
using Float = Var<float>; using Float3 = Var<float3>; using Int = Var<int>;
using UInt = Var<uint>; using UInt2 = Var<uint2>; using Bool = Var<bool>;

// Literal suffixes
using namespace dsl_literals;
auto lx = 0._half; auto ly = 0._float; auto lz = 0_ulong2;
```

## Buffer Operations

```cpp
Kernel1D k = [](BufferVar<float> buf, BufferFloat fb, BufferUInt ub) noexcept {
    Var<float> v = buf.read(idx);
    buf.write(idx, val);
    // Volatile (coherent)
    buf.volatile_read(idx); buf.volatile_write(idx, val);
    // ByteBuffer
    bb.volatile_read<float3>(idx); bb.volatile_write(idx, val);
    bb.volatile_read<float3x3>(idx);  // matrix
    // Struct buffers
    BufferVar<MyStruct> sb; sb.read(idx).member;
};
```

## Control Flow

```cpp
// If / elif / else
if_(cond, [] { /* then */ });
if_(cond, [] {}).else_([] {});
if_(c1, [] {}).elif_(c2, [] {}).else_([] {});

// Switch
switch_(val).case_(1, [] {}).case_(2, [] {}).default_([] {});

// Loops
loop([] { if_(true, break_); });
for (auto v : dynamic_range(count)) { /* v is Var<int> */ }

// Ternary & min/max
Var vv = ite(t == 10, 1, 2);
Var vvv = min(vv, 10);
```

## Atomic Operations

```cpp
Kernel1D k = [](BufferUInt buf) noexcept {
    buf.atomic(3u).fetch_add(1u);
    buf.atomic(0u).fetch_sub(-1.f);
    buf.atomic(0u).fetch_max(100u);
    buf.atomic(0u).compare_exchange(expected, new_value);
};

// Vector component:  buf.atomic(0u).x.fetch_add(1.f);
// Matrix element:    buf.atomic(0u)[1].x.fetch_add(1.f);  // [col][row]
// Nested array:      buf.atomic(0u)[1][2][3].fetch_add(1.f);
// Struct member:     auto a = buf.atomic(0u); a.v.x.fetch_max(1.f);
```

## Shared Memory

```cpp
Kernel1D k = []() noexcept {
    Shared<float4> s{16};          // 16 float4 elements
    s[thread_x()] = make_float4(1.0f);
    Var<float4> v = s[thread_x()];
    s.atomic(0).compare_exchange(0.f, 1.f);
    s.atomic(0).fetch_add(1.f);
};
```

## Constants

```cpp
Kernel1D k = []() noexcept {
    Constant floats = {1.0f, 2.0f};
    Constant ints = std::vector<int>{1, 2, 3, 4};
    Var<float> v = floats.read(0);
    Var<int> iv = ints[idx];
};

// Captured outside:
Constant floats = {1.0f, 2.0f};
Kernel1D k = [&floats]() noexcept { Var<float> v = floats[0]; };
```

## Type Casting

```cpp
Var<float> f = cast<float>(i);
Var<int> i = cast<int>(f);
Var<int> r = cast<int>(buf->read(a + b));
Var<float> m = i.cast<float>();  // method syntax
```

## Sugar Syntax

```cpp
#include <luisa/dsl/sugar.h>

// $ prefix = Var<T>
$int a; $float b; $float3 c; $uint2 d;
$ v = 10;          // $int
$ f = 1.0f;        // $float

// $constant, $shared, $array, $buffer
$constant floats = {1.0f, 2.0f};
$shared<float4> s{16};
$array<float, 5> arr;
Kernel1D k = &[$]($buffer<float> buf, $uint count) { /* ... */ };

// Control flow
$if (w.x < 5) { } $elif (w.x > 0) { } $else { };
$loop { $break; };
$switch (123) { $case (1) { }; $default { }; };
$for (x, n) { /* x is Var<uint>, 0..n-1 */ };
```

## Dispatch & Thread IDs

```cpp
// 1D
UInt idx = dispatch_id().x;    // or dispatch_x()
// 2D
UInt2 coord = dispatch_id().xy(); UInt2 size = dispatch_size().xy();
// 3D
UInt3 coord = dispatch_id().xyz();

// Thread within block
UInt tx = thread_id().x;       // or thread_x()
UInt bx = block_id().x;
UInt bs = block_size().x;
```

## Complete Example

```cpp
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
using namespace luisa::compute;

struct Particle { float3 position, velocity; float mass; };
LUISA_STRUCT(Particle, position, velocity, mass) {};

int main(int argc, char *argv[]) {
    Context ctx{argv[0]};
    Device device = ctx.create_device("cuda");
    Stream stream = device.create_stream();
    Buffer<Particle> particles = device.create_buffer<Particle>(1024);

    Callable update = []($Particle p, $float dt) noexcept {
        p.position = p.position + p.velocity * dt;
        return p;
    };

    Kernel1D k = [&update]($buffer<Particle> buf, $float dt) noexcept {
        $ idx = dispatch_x();
        $ p = buf.read(idx);
        p = update(p, dt);
        buf.write(idx, p);
    };

    auto shader = device.compile(k);
    stream << shader(particles, 0.016f).dispatch(1024) << synchronize();
}
```

## Summary

| Feature | Syntax |
|---|---|
| Kernel1D/2D/3D | `Kernel1D k = [](...) { ... };` |
| Callable | `Callable c = [](...) { ... };` / `Callable<Ret(Args...)>` |
| Struct | `LUISA_STRUCT(Name, m1, m2) {}` |
| Template Struct | `LUISA_TEMPLATE_STRUCT(TMPL_DEF, TMPL_USE, members) {}` |
| Variable | `Var<T> v` / `$T v` |
| Buffer Read/Write | `buf.read(idx)` / `buf.write(idx, val)` |
| Atomic | `buf.atomic(idx).fetch_add(val)` / `.compare_exchange(exp, new)` |
| Shared | `Shared<T> s{n}` |
| Constant | `Constant c = { ... }` |
| Cast | `cast<T>(val)` / `val.cast<T>()` |
| If | `if_(cond, [] {})` / `.elif_(cond, [] {})` / `.else_([] {})` |
| Switch | `switch_(val).case_(v, [] {})...default_([] {})` |
| Loop | `loop([] {})` / `$for (i, n) {}` |
| Dispatch ID | `dispatch_id().xy()` / `dispatch_x()` |
| Thread ID | `thread_id().x` / `thread_x()` |
| Compose | `compose(v1, v2)` → `.get<0>()`, `.get<1>()` |
