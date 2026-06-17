---
name: lc_dsl
description: DSL kernels, callables, structs, buffers, atomics, control flow, and dispatch.
---

# LuisaCompute DSL Usage Guide

Based on test cases in `src/tests/unit/dsl/test_dsl.cpp`, `test_dsl_sugar.cpp`, `test_var.cpp`, `test_callable.cpp` and `src/tests/unit/runtime/test_atomic.cpp`, `test_warp.cpp`, plus `src/tests/integration/runtime/test_rtx.cpp` and `test_indirect.cpp`.

## Headers

```cpp
#include <luisa/dsl/syntax.h>   // core DSL (includes func, buffers, textures, RTX, indirect dispatch, ...)
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

// Compile a raw lambda directly as a 2D kernel
auto shader2 = device.compile<2>(kernel_lambda);

kernel.function_builder()->set_name("my_kernel");        // debug name
// Or inline:
Kernel2D k = []() noexcept { set_name("my_kernel"); /* ... */ };
```

### Block Size
```cpp
Kernel2D k = []() noexcept { set_block_size(16u, 16u, 1u); /* ... */ };
// Equivalent shorthand:
set_block_size(make_uint2(16u, 16u));
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

// Explicit construction from an expression or C++ value
Float x = def(1.0f);
Float3 y = def<float3>(1.0f, 2.0f, 3.0f);

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
for (auto v : dynamic_range(count)) { /* v is Var<int>, 0..count-1 */ }
for (auto v : dynamic_range(begin, end, step)) { /* begin..end-1 with step */ }
loop(begin, end, step, [](auto i) { /* body */ });

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

## Warp/Wave Intrinsics

Warp (NVIDIA) / Wave (AMD) intrinsics enable cross-lane communication within a single warp.
Requires setting a warp size and uses lane indices for per-lane data exchange.

### Configuration

```cpp
Kernel1D k = []() noexcept {
    set_block_size(128u, 1u, 1u);
    set_warp_size(32u);            // 32 (NVIDIA) or 64 (AMD, some cases)

    UInt lane_count = warp_lane_count();  // total lanes in warp
    UInt lane_id    = warp_lane_id();     // this thread's lane index (0..31)
};
```

### Lane Identification

```cpp
// Check if current lane is the first active lane in the warp
Bool first = warp_is_first_active_lane();

// Get the index of the first active lane
UInt first_lane = warp_first_active_lane();
```

### Active Lane Vote & Ballot

```cpp
// Returns true if predicate is true for ALL active lanes
Bool all_true = warp_active_all(condition);

// Returns true if predicate is true for ANY active lane
Bool any_true = warp_active_any(condition);

// Returns a uint4 bitmask (up to 128 lanes, each bit = one lane)
UInt4 mask = warp_active_bit_mask(predicate);

// Count active lanes where predicate is true
UInt count = warp_active_count_bits(predicate);

// Exclusive prefix count of active lanes where predicate is true
UInt prefix_count = warp_prefix_count_bits(predicate);
```

### Active Lane Reductions

Operate on values from all active lanes in the warp. Accept `Float`, `Int`, `UInt`, and vectors.

```cpp
// Sum reduction
Float sum = warp_active_sum(value);         // scalar or vector

// Product reduction
Float prod = warp_active_product(value);

// Minimum / Maximum
Float min_val = warp_active_min(value);
Float max_val = warp_active_max(value);

// Bitwise reductions (integral types only)
UInt and_bits = warp_active_bit_and(value); // bitwise AND
UInt or_bits  = warp_active_bit_or(value);  // bitwise OR
UInt xor_bits = warp_active_bit_xor(value); // bitwise XOR

// Check if all active lanes have the same value
Bool equal = warp_active_all_equal(value);  // returns bool or Vector<bool,N>
```

### Prefix (Scan) Operations

Exclusive prefix scan across active lanes. Lane i gets the sum/product of lanes 0..i-1.

```cpp
// Exclusive prefix sum: lane i receives sum of lanes 0..i-1
Float prefix_sum = warp_prefix_sum(value);

// Exclusive prefix product: lane i receives product of lanes 0..i-1
Float prefix_prod = warp_prefix_product(value);
```

### Lane Data Exchange

```cpp
// Read value from a specific lane by index (broadcast)
// Supports scalar, vector, and matrix types; lane index must be integral
Float other_val = warp_read_lane(value, lane_index);

// Read value from the first active lane (convenient broadcast)
Float first_val = warp_read_first_active_lane(value);
```

### Block-Wide Barrier

```cpp
sync_block();  // synchronize all threads in a thread block
```

### Complete Warp MatMul Example

Based on `src/tests/unit/runtime/test_warp.cpp`:

```cpp
constexpr uint k_warp_size = 32;

auto mat_mul_kernel = [&](BufferVar<float> lhs, BufferVar<float> rhs,
                          BufferVar<float> result, UInt lhs_row_size) {
    set_block_size(128, 1, 1);
    set_warp_size(k_warp_size);

    UInt2 lhs_size = make_uint2(lhs_row_size, dispatch_size().y);
    UInt2 rhs_size = make_uint2(dispatch_size().x / k_warp_size, lhs_row_size);

    UInt lhs_y = dispatch_id().x / k_warp_size;
    UInt rhs_x = dispatch_id().y;
    UInt lane = warp_lane_id();

    UInt tile_count = (lhs_size.x + k_warp_size - 1) / k_warp_size;
    Float accum = 0.f;

    for (auto tile : dynamic_range(tile_count)) {
        UInt lhs_x = tile * k_warp_size + lane;
        Float v = 0.f;
        $if (lhs_x < lhs_size.x) {
            v = lhs.read(lhs_size.x * lhs_y + lhs_x);
            v *= rhs.read(rhs_size.x * rhs_x + lhs_x);
        };
        accum += warp_active_sum(v);  // sum across all 32 lanes
    }

    // Only lane 0 writes the result
    $if (lane == 0) {
        result.write(rhs_size.x * lhs_y + rhs_x, accum);
    };
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

// Bitwise reinterpretation (same size)
UInt bits = as<uint>(f);
UInt2 u2 = as<uint2>(make_float2(1.0f, 2.0f));
```

## Sugar Syntax

```cpp
#include <luisa/dsl/sugar.h>

// $ prefix = Var<T>
$int a; $float b; $float3 c; $uint2 d;
$ v = 10;          // $int
$ f = 1.0f;        // $float

// $constant, $shared, $array, $buffer, $image, $volume, $bindless, $accel, $atomic
$constant floats = {1.0f, 2.0f};
$shared<float4> s{16};
$array<float, 5> arr;
Kernel1D k = &[$]($buffer<float> buf, $uint count) { /* ... */ };

// Control flow
$if (w.x < 5) { } $elif (w.x > 0) { } $else { };
$loop { $break; };
$while (i > 0u) { i = i / b; };
$switch (123) { $case (1) { }; $default { }; };
$for (x, n) { /* x is Var<uint>, 0..n-1 */ };
$for (i, 0, n, 2) { /* i is Var<int>, step 2 */ };

// Return/break/continue/unreachable
$return(x + y);
$continue;
unreachable();            // or unreachable("reason")
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

// Which kernel in an indirect dispatch packet
UInt kid = kernel_id();
```

## Bindless Arrays

```cpp
Kernel1D k = [](Var<BindlessArray> heap, BufferVar<float4> out) noexcept {
    // Bindless buffer
    $float4 v = heap.buffer<float4>(0u).read(0u);
    // Bindless 2D texture
    $float4 t = heap.tex2d(1u).read(make_uint2(0u));
    out.write(0u, v + t);
};
```

## Ray-Tracing DSL

`syntax.h` pulls in `<luisa/dsl/rtx/*.h>`. Example:

```cpp
#include <luisa/dsl/sugar.h>

Kernel2D raytrace = [&](BufferFloat4 image, AccelVar accel, UInt frame) noexcept {
    UInt2 coord = dispatch_id().xy();
    Var<Ray> ray = make_ray(make_float3(0.0f), make_float3(0.0f, 0.0f, -1.0f));
    Var<TriangleHit> hit = accel.intersect(ray, {});
    $if (!hit->miss()) {
        Float3 color = triangle_interpolate(hit.bary,
                                            make_float3(1.0f, 0.0f, 0.0f),
                                            make_float3(0.0f, 1.0f, 0.0f),
                                            make_float3(0.0f, 0.0f, 1.0f));
        image.write(coord.y * dispatch_size_x() + coord.x, make_float4(color, 1.0f));
    };
};
```

## Indirect Dispatch

```cpp
#include <luisa/dsl/dispatch_indirect.h>
#include <luisa/runtime/dispatch_buffer.h>

Kernel1D clear = [](Var<IndirectDispatchBuffer> dispatch_buffer) noexcept {
    dispatch_buffer.set_dispatch_count(16u);
};
Kernel1D emplace = [](Var<IndirectDispatchBuffer> dispatch_buffer) noexcept {
    dispatch_buffer.set_kernel(dispatch_id().x,
                               make_uint3(64u, 1u, 1u),
                               make_uint3(dispatch_id().x, 1u, 1u),
                               dispatch_id().x);
};
Kernel1D work = [](BufferVar<uint> buf) noexcept {
    set_block_size(64u, 1u, 1u);
    buf.atomic(kernel_id()).fetch_add(dispatch_size().x);
};

IndirectDispatchBuffer idb = device.create_indirect_dispatch_buffer(16u);
auto clear_s = device.compile(clear);
auto emplace_s = device.compile(emplace);
auto work_s = device.compile(work);
stream << clear_s(idb).dispatch(1u)
       << emplace_s(idb).dispatch(16u)
       << work_s(buf).dispatch(idb)
       << synchronize();
```

## Hints & Device Debug

```cpp
assume(index >= 0 & index < size);    // optimizer hint (use bitwise & for scalar bools)
device_assert(x > 0.0f);             // device-side assertion
device_assert(x > 0.0f, "x must be positive");

// Clock
ULong t = device_clock();
```

## Coroutine Examples

Coroutine examples that expose scheduler selection should use `--scheduler <state_machine|wavefront|persistent>` after the explicit backend argument, with `state_machine` as the default unless the example has a documented reason to choose otherwise. Prefer the shared parser in `examples/common/coro_scheduler_options.h` over per-example parsing.

Keep unit tests different from examples: coroutine unit tests should require an explicit backend and exercise all schedulers internally for scheduler-agnostic behavior, while examples may let users specify a scheduler or rely on the default.

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

    Callable update = [](Var<Particle> p, $float dt) noexcept {
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
| Kernel1D/2D/3D | `Kernel1D k = [](...) { ... };` / `device.compile<N>(lambda)` |
| Callable | `Callable c = [](...) { ... };` / `Callable<Ret(Args...)>` |
| Struct | `LUISA_STRUCT(Name, m1, m2) {}` |
| Template Struct | `LUISA_TEMPLATE_STRUCT(TMPL_DEF, TMPL_USE, members) {}` |
| Variable | `Var<T> v` / `$T v` / `def<T>(...)` |
| Buffer Read/Write | `buf.read(idx)` / `buf.write(idx, val)` |
| Atomic | `buf.atomic(idx).fetch_add(val)` / `.compare_exchange(exp, new)` |
| Shared | `Shared<T> s{n}` |
| Constant | `Constant c = { ... }` |
| Cast | `cast<T>(val)` / `val.cast<T>()` / `as<T>(val)` |
| If | `if_(cond, [] {})` / `.elif_(cond, [] {})` / `.else_([] {})` / `$if ... $elif ... $else` |
| Switch | `switch_(val).case_(v, [] {})...default_([] {})` / `$switch ... $case ... $default` |
| Loop | `loop([] {})` / `$loop` / `$while` / `$for (i, n)` / `$for (i, begin, end, step)` |
| Dispatch ID | `dispatch_id().xy()` / `dispatch_x()` |
| Thread ID | `thread_id().x` / `thread_x()` |
| Bindless | `heap.buffer<T>(slot).read(idx)` / `heap.tex2d(slot).read(uv)` |
| RTX | `make_ray(...)`, `accel.intersect(ray, {})`, `TriangleHit` |
| Indirect | `Var<IndirectDispatchBuffer>` / `.set_dispatch_count` / `.set_kernel` |
| Compose | `compose(v1, v2)` → `.get<0>()`, `.get<1>()` |
| Warp Config | `set_warp_size(32)` / `warp_lane_id()` / `warp_lane_count()` |
| Warp Vote | `warp_active_all(pred)` / `warp_active_any(pred)` / `warp_active_bit_mask(pred)` |
| Warp Count | `warp_active_count_bits(pred)` / `warp_prefix_count_bits(pred)` |
| Warp Reduce | `warp_active_sum(v)` / `warp_active_min(v)` / `warp_active_max(v)` / `warp_active_product(v)` |
| Warp Bitwise | `warp_active_bit_and(v)` / `warp_active_bit_or(v)` / `warp_active_bit_xor(v)` |
| Warp Prefix | `warp_prefix_sum(v)` / `warp_prefix_product(v)` |
| Warp Broadcast | `warp_read_lane(v, lane)` / `warp_read_first_active_lane(v)` |
| Warp Equal | `warp_active_all_equal(v)` |
| Warp First Lane | `warp_is_first_active_lane()` / `warp_first_active_lane()` |
| Block Barrier | `sync_block()` |
| Hints | `assume(pred)` / `device_assert(pred, msg)` / `unreachable()` |
