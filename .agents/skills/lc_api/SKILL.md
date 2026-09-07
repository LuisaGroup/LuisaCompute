---
name: lc_api
description: LuisaCompute API guide covering the DSL (kernels, callables, structs, buffers, atomics, control flow, dispatch), the runtime API (Context, Device, Stream, buffers, images, ray tracing, rasterization), AST usage markers and AST-to-XIR lowering, the core library (traits, types, logging, math, fiber, allocators, STL containers), and vstd containers (HashMap, queues, pools, variant, smart pointers, utilities).
---

# LuisaCompute API Guide

## Overview

Combined agent-facing reference for the LuisaCompute DSL, AST, core library, runtime API, and VSTL utilities. Each section preserves the code examples, conventions, pitfalls, and source-file references of the five source skills (`lc_dsl`, `lc_ast`, `lc_runtime`, `lc_core`, `lc_vstl`).

## DSL: Kernels, Callables, Structs, Buffers, Atomics, Control Flow, and Dispatch

Based on test cases in `src/tests/unit/dsl/test_dsl.cpp`, `test_dsl_sugar.cpp`, `test_var.cpp`, `test_callable.cpp` and `src/tests/unit/runtime/test_atomic.cpp`, `test_warp.cpp`, plus `src/tests/integration/runtime/test_rtx.cpp` and `test_indirect.cpp`.

### Headers

```cpp
#include <luisa/dsl/syntax.h> // core DSL (includes func, buffers, textures, RTX, indirect dispatch, ...)
#include <luisa/dsl/sugar.h> // syntactic sugar macros
#include <luisa/dsl/struct.h> // struct registration
using namespace luisa;
using namespace luisa::compute;
```

Key concepts: **Kernel** (GPU entry, 1D/2D/3D), **Callable** (reusable function), **Var<T>** (DSL variable), **LUISA_STRUCT** (register C++ structs).

### Kernel Definitions

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
stream << shader(buf, count).dispatch(1024u); // 1D
stream << shader2d(img).dispatch(width, height); // 2D

// Compile a raw lambda directly as a 2D kernel
auto shader2 = device.compile<2>(kernel_lambda);

kernel.function_builder()->set_name("my_kernel"); // debug name
// Or inline:
Kernel2D k = []() noexcept { set_name("my_kernel"); /* ... */ };
```

### Block Size

```cpp
Kernel2D k = []() noexcept { set_block_size(16u, 16u, 1u); /* ... */ };
// Equivalent shorthand:
set_block_size(make_uint2(16u, 16u));
```

### Callable Functions

```cpp
Callable add = [](Var<int> a, Var<int> b) noexcept { a.set_name("a"); b.set_name("b"); return a + b; };
Callable<float(float, float)> mul = [](Var<float> a, Var<float> b) noexcept { return a * b; };
Callable<int(int, int)> add_t = []<typename T>(Var<T> a, Var<T> b) noexcept { return cast<int>(a + b); };
```

#### Captures (transitive)

```cpp
Buffer<float> buf = device.create_buffer<float>(1024);
Callable c1 = [&buf](UInt a) noexcept { return buf->read(a); };
Callable c2 = [&c1, &another_buffer](UInt b) noexcept { return c1(b) + another_buffer->read(b); };
// Kernel captures c2 → automatically captures buf + another_buffer
Kernel1D k = [&c2] { auto v = c2(dispatch_x()); };
```

#### Multiple Return Values

```cpp
Callable add_mul = [](Var<int> a, Var<int> b) noexcept { return compose(a + b, a * b); };
// Unpack: Var am = add_mul(3, 4); Var sum = am.get<0>(); Var prod = am.get<1>();
```

### Struct Definitions

```cpp
struct Point3D { float3 v; };
struct Material { float3 albedo; float roughness; float metallic; };
LUISA_STRUCT(Point3D, v) {};
LUISA_STRUCT(Material, albedo, roughness, metallic) {};
```

#### With Methods

```cpp
struct Onb { float3 tangent, binormal, normal; };
LUISA_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] Float3 to_world(Expr<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};
// Usage: Var<Onb> onb; Float3 world = onb->to_world(local_vec);
```

#### Arrays & Templates

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

#### Usage in Kernels

```cpp
Var<Point3D> p1; // default
Var<Point3D> p2{make_float3(1.0f)}; // init
Var<Point3D> p3{p2}; // copy
Var<float3> pos = p2.v; p2.v = make_float3(2,3,4);
```

### Variables

```cpp
Var<float> f; Var<int3> iv; Var<float4x4> m;
Var v = 10; // Var<int>
Var v2 = make_float3(1.0f); // Var<float3>

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

### Buffer Operations

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

### Control Flow

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

### Compile-Time vs Runtime Control Flow

DSL kernels are constructed by executing the host C++ lambda during `Kernel1D/2D/3D` creation (and again inside `device.compile()`). This means native C++ control flow on **host** values is evaluated at kernel construction time, while DSL control-flow constructs become real device instructions.

- **Native C++ `if` / `for` / `while` / `switch` on plain host variables** are resolved on the host. Only the taken path is recorded in the AST; no corresponding branch or loop appears in the generated GPU code.
- **DSL `$if` / `$else`, `$for` / `$while` / `$loop`, `$switch` / `$case` / `$default`** (and the non-sugar `if_`, `switch_`, `for (auto i : dynamic_range(...))`, `loop`) emit real device control flow. Their conditions must be DSL expressions such as `Var<bool>` or `Var<uint>`.

#### Examples

Native `if` on a host variable — the unselected branch is erased during AST construction:

```cpp
bool host_visible = true;
Kernel1D k = [&]() noexcept {
    Var<uint> x = 0u;
    if (host_visible) {
        x = 1u;
    } else {
        x = 2u; // never recorded; the kernel always writes 1
    }
};
```

To emit a real GPU branch, use the DSL form with a device expression:

```cpp
Kernel1D k = [&]() noexcept {
    Var<bool> visible = read_some_flag();  // DSL bool
    Var<uint> x = 0u;
    $if (visible) {
        x = 1u;
    } $else {
        x = 2u;
    };
};
```

The same distinction applies to loops. A native C++ `for` with a host-bound count duplicates the loop body into the AST once per iteration:

```cpp
// BAD for large N: the body is inlined N times, so compilation can become
// extremely slow or run out of memory.
uint host_n = 1024;
Kernel1D k = [&]() noexcept {
    Var<uint> x = 0u;
    for (uint i = 0; i < host_n; ++i) {
        x = x + 1u;
    }
};
```

Use `$for` or `dynamic_range` so the GPU executes the loop at runtime with a single AST node:

```cpp
// GOOD: one ForStmt is emitted; the loop runs on the device.
Kernel1D k = [&]() noexcept {
    Var<uint> x = 0u;
    Var<uint> n = 1024u;
    $for (i, n) {
        x = x + 1u;
    };
};
```

#### When to use which

| Host C++ construct | Evaluated | Emitted in GPU code? | Safe for large counts? |
|---|---|---|---|
| `if (host_bool)` | Kernel construction | No (only taken path) | N/A |
| `$if (Var<bool>)` / `if_(Expr<bool>)` | GPU runtime | Yes | Yes |
| `for (host i < N)` | Kernel construction | No (flattened N times) | **No** |
| `$for (i, N)` / `dynamic_range(N)` | GPU runtime | Yes | Yes |
| `switch (host_val)` | Kernel construction | No (only matching case) | N/A |
| `$switch (Var<T>)` / `switch_(Expr<T>)` | GPU runtime | Yes | Yes |

Reserve native C++ loops for small, compile-time-known unrolling (for example, a fixed 4×4 matrix operation). Use DSL loops whenever the bound comes from a runtime value or is large.

### Atomic Operations

```cpp
Kernel1D k = [](BufferUInt buf) noexcept {
    buf.atomic(3u).fetch_add(1u);
    buf.atomic(0u).fetch_sub(-1.f);
    buf.atomic(0u).fetch_max(100u);
    buf.atomic(0u).compare_exchange(expected, new_value);
};

// Vector component:  buf.atomic(0u).x.fetch_add(1.f);
// Matrix element: buf.atomic(0u)[1].x.fetch_add(1.f);  // [col][row]
// Nested array: buf.atomic(0u)[1][2][3].fetch_add(1.f);
// Struct member: auto a = buf.atomic(0u); a.v.x.fetch_max(1.f);
```

### Shared Memory

```cpp
Kernel1D k = []() noexcept {
    Shared<float4> s{16}; // 16 float4 elements
    s[thread_x()] = make_float4(1.0f);
    Var<float4> v = s[thread_x()];
    s.atomic(0).compare_exchange(0.f, 1.f);
    s.atomic(0).fetch_add(1.f);
};
```

### Warp/Wave Intrinsics

Warp (NVIDIA) / Wave (AMD) intrinsics enable cross-lane communication within a single warp.
Requires setting a warp size and uses lane indices for per-lane data exchange.

#### Configuration

```cpp
Kernel1D k = []() noexcept {
    set_block_size(128u, 1u, 1u);
    set_warp_size(32u); // 32 (NVIDIA) or 64 (AMD, some cases)

    UInt lane_count = warp_lane_count();  // total lanes in warp
    UInt lane_id = warp_lane_id(); // this thread's lane index (0..31)
};
```

#### Lane Identification

```cpp
// Check if current lane is the first active lane in the warp
Bool first = warp_is_first_active_lane();

// Get the index of the first active lane
UInt first_lane = warp_first_active_lane();
```

#### Active Lane Vote & Ballot

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

#### Active Lane Reductions

Operate on values from all active lanes in the warp. Accept `Float`, `Int`, `UInt`, and vectors.

```cpp
// Sum reduction
Float sum = warp_active_sum(value); // scalar or vector

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

#### Prefix (Scan) Operations

Exclusive prefix scan across active lanes. Lane i gets the sum/product of lanes 0..i-1.

```cpp
// Exclusive prefix sum: lane i receives sum of lanes 0..i-1
Float prefix_sum = warp_prefix_sum(value);

// Exclusive prefix product: lane i receives product of lanes 0..i-1
Float prefix_prod = warp_prefix_product(value);
```

#### Lane Data Exchange

```cpp
// Read value from a specific lane by index (broadcast)
// Supports scalar, vector, and matrix types; lane index must be integral
Float other_val = warp_read_lane(value, lane_index);

// Read value from the first active lane (convenient broadcast)
Float first_val = warp_read_first_active_lane(value);
```

#### Block-Wide Barrier

```cpp
sync_block();  // synchronize all threads in a thread block
```

#### Complete Warp MatMul Example

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

### Constants

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

### Type Casting

```cpp
Var<float> f = cast<float>(i);
Var<int> i = cast<int>(f);
Var<int> r = cast<int>(buf->read(a + b));
Var<float> m = i.cast<float>();  // method syntax

// Bitwise reinterpretation (same size)
UInt bits = as<uint>(f);
UInt2 u2 = as<uint2>(make_float2(1.0f, 2.0f));
```

### Sugar Syntax

```cpp
#include <luisa/dsl/sugar.h>

// $ prefix = Var<T>
$int a; $float b; $float3 c; $uint2 d;
$ v = 10; // $int
$ f = 1.0f; // $float

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
unreachable(); // or unreachable("reason")
```

### Dispatch & Thread IDs

```cpp
// 1D
UInt idx = dispatch_id().x; // or dispatch_x()
// 2D
UInt2 coord = dispatch_id().xy(); UInt2 size = dispatch_size().xy();
// 3D
UInt3 coord = dispatch_id().xyz();

// Thread within block
UInt tx = thread_id().x; // or thread_x()
UInt bx = block_id().x;
UInt bs = block_size().x;

// Which kernel in an indirect dispatch packet
UInt kid = kernel_id();
```

### Bindless Arrays (Kernel Side)

```cpp
Kernel1D k = [](Var<BindlessArray> heap, BufferVar<float4> out) noexcept {
    // Bindless buffer
    $float4 v = heap.buffer<float4>(0u).read(0u);
    // Bindless 2D texture
    $float4 t = heap.tex2d(1u).read(make_uint2(0u));
    out.write(0u, v + t);
};
```

### Ray-Tracing DSL

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

### Indirect Dispatch

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

### Hints & Device Debug

```cpp
assume(index >= 0 & index < size); // optimizer hint (use bitwise & for scalar bools)
device_assert(x > 0.0f); // device-side assertion
device_assert(x > 0.0f, "x must be positive");

// Clock
ULong t = device_clock();
```

### Coroutine Examples

Coroutine examples that expose scheduler selection should use `--scheduler <state_machine|wavefront|persistent>` after the explicit backend argument, with `state_machine` as the default unless the example has a documented reason to choose otherwise. Prefer the shared parser in `examples/common/coro_scheduler_options.h` over per-example parsing.

Coroutine frames reserve four scalar `uint` fields: frame indices 0, 1, and 2 store `coro_id.x/y/z`, and frame index 3 stores `target_token`. User frame fields start at `CoroFrameDesc::reserved_field_count` (currently 4). Do not reintroduce a skip flag; it was only needed by the old structured-CFG replay path, and XIR coroutine splitting now uses unstructured CFG continuations directly.

Rendering coroutine examples should keep the real fine-grained coroutine topology. Wavefront rebuilds or sorts work queues per suspend phase, so inner-loop suspends can dominate runtime even when the generated code is functionally correct; do not hide that by silently removing or coarsening suspends in the main example/test. If a coarser coroutine is useful for profiling, add it as a separate focused debug case. Log `coro.frame().total_size()`, `coro.frame().frame_type()->size()`, frame field count, subroutine count, and graph node count after compiling complex coroutines.

Keep unit tests different from examples: coroutine unit tests should require an explicit backend and exercise all schedulers internally for scheduler-agnostic behavior, while examples may let users specify a scheduler or rely on the default.

### Complete DSL Example

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

### Cooperative Vector Operations

Cooperative vectors are thread-local vectors of uniform size that participate in hardware-accelerated cooperative (cross-lane/warp) operations. They are backed by `CoopVector<T>`, `CoopVectorRef`, and `CoopMatrixRef` types defined in `<luisa/dsl/coop_vector.h>`. All free functions are in `<luisa/dsl/resource.h>`.

#### Headers

```cpp
#include <luisa/dsl/coop_vector.h> // CoopVector<T>, CoopVectorRef, CoopMatrixRef
#include <luisa/dsl/resource.h> // all cooperative_vector_*, cooperative_mat_*, bindless_cooperative_* functions
#include <luisa/dsl/expr.h> // Expr<CoopVector<T>> specialization
#include <luisa/dsl/sugar.h> // $ sugar macros (optional)
```

#### Backend Support

> **⚠️ Currently cooperative vector operations only support the Vulkan (`vk`) backend.**
> The DX backend requires Shader Model 6.8 with experimental features, which is not widely available.
> Check `src/tests/unit/ast/test_cooperative_vector.cpp` for the `create_test_device()` helper.

#### Type System

```cpp
// Create a cooperative vector type (element type + size)
auto cv_type = Type::cooperative_vector(Type::of<float>(), 16);  // coopvec<float,16>

// Create a cooperative vector reference type (used to describe buffer offsets)
auto cvr_type = Type::cooperative_vector_ref(CoopRefVecType::FLOAT32, 16);  // coopvec_ref<16,5>

// Create a cooperative matrix reference type
auto cmr_type = Type::cooperative_matrix_ref(CoopRefVecType::FLOAT32, 4, 8);  // coopmat_ref<4,8,5>
```

Available `CoopRefVecType` values: `FLOAT16`, `FLOAT32`, `INT8`, `UINT8`, `INT32`, `UINT32`.

#### DSL Object Construction

```cpp
// Cooperative vector of float with 8 elements
CoopVector<float> v{8};

// Cooperative vector reference (describes a byte-buffer region)
CoopVectorRef offset{CoopRefVecType::FLOAT32, 8};
offset.set_byte_offset(0u);  // set the byte offset into the buffer

// Cooperative matrix reference (for matrix multiply operations)
CoopMatrixRef mat_offset{CoopRefVecType::FLOAT32, 4, 8};
mat_offset.set_byte_offset(0u);
```

#### Element Access

Individual elements are accessed with `operator[]` (read/write):

```cpp
CoopVector<float> v{8};
for (auto i = 0u; i < 8u; ++i) {
    v[i] = static_cast<float>(i + 1);  // write
}
Var<float> elem = v[3];  // read
```

#### Load / Store (ByteBuffer)

Load a cooperative vector from a `ByteBuffer` into thread-local storage:

```cpp
ByteBufferVar buf{luisa::compute::detail::ArgumentCreation{}};
CoopVectorRef offset{CoopRefVecType::FLOAT32, 8};
offset.set_byte_offset(0u);
auto loaded = cooperative_vector_load<float>(buf, offset);
```

Store a cooperative vector to a `ByteBuffer`:

```cpp
CoopVector<float> input{8};
for (auto i = 0u; i < 8u; ++i) input[i] = static_cast<float>(i);
offset.set_byte_offset(0u);
cooperative_vector_store(buf, offset, Expr<CoopVector<float>>{input});
```

#### Accumulate

Atomically accumulate a cooperative vector into a `ByteBuffer` at a given offset:

```cpp
CoopVector<float> input{8};
for (auto i = 0u; i < 8u; ++i) input[i] = static_cast<float>(i + 1);
offset.set_byte_offset(0u);
cooperative_vector_accumulate(buf, offset, Expr<CoopVector<float>>{input});
```

#### Splat

Create a cooperative vector with all elements set to the same scalar value:

```cpp
auto result = cooperative_vector_splat<float>(42.0f, 8u);
```

#### Cast

Cast the element type of a cooperative vector:

```cpp
CoopVector<float> input{8};
// ... fill input ...
auto result = cooperative_vector_cast<int>(Expr<CoopVector<float>>{input});
```

#### Bindless Load / Store

Load from or store to a bindless (or typed bindless) buffer:

```cpp
BindlessVar bindless{luisa::compute::detail::ArgumentCreation{}};
CoopVectorRef offset{CoopRefVecType::FLOAT32, 8};
offset.set_byte_offset(0u);

// Bindless load
auto out0 = bindless_cooperative_vector_load<float>(bindless, 0u, offset);
auto out1 = typed_bindless_cooperative_vector_load<float>(bindless, 0u, offset);

// Bindless store
CoopVector<float> input{8};
// ... fill input ...
bindless_cooperative_vector_store(bindless, 0u, offset, Expr<CoopVector<float>>{input});
typed_bindless_cooperative_vector_store(bindless, 0u, offset, Expr<CoopVector<float>>{input});
```

#### Workgroup Load / Store

Load from or store to shared memory (workgroup-level cooperative vector transfer):

```cpp
Shared<float> shared_mem{8};

// Workgroup load: load from shared memory at index
auto result = cooperative_vector_workgroup_load(shared_mem, 0u);

// Workgroup store: store to shared memory at index
CoopVector<float> input{8};
// ... fill input ...
cooperative_vector_workgroup_store(shared_mem, 0u, Expr<CoopVector<float>>{input});
```

#### Matrix Multiply Operations

Compute `out = matrix * input_vector + bias` (cooperative matrix multiply with accumulator):

```cpp
ByteBufferVar matrix_buffer{luisa::compute::detail::ArgumentCreation{}};
ByteBufferVar bias_buffer{luisa::compute::detail::ArgumentCreation{}};
CoopMatrixRef matrix_offset{CoopRefVecType::FLOAT32, 4, 8};
CoopVectorRef bias_offset{CoopRefVecType::FLOAT32, 8};
CoopVector<float> input{4};

matrix_offset.set_byte_offset(0u);
bias_offset.set_byte_offset(0u);

auto out = cooperative_mat_mul_add<float, float>(
    matrix_buffer, matrix_offset,
    bias_buffer, bias_offset,
    Expr<CoopVector<float>>{input});
```

Compute `out = matrix * input_vector` (without bias):

```cpp
auto out = cooperative_mat_mul<float, float>(
    matrix_buffer, matrix_offset,
    Expr<CoopVector<float>>{input});
```

Bindless variants:

```cpp
BindlessVar bindless{luisa::compute::detail::ArgumentCreation{}};
CoopMatrixRef matrix_offset{CoopRefVecType::FLOAT32, 4, 8};
CoopVectorRef bias_offset{CoopRefVecType::FLOAT32, 8};
CoopVector<float> input{4};

// bindless mat_mul_add
auto out0 = bindless_cooperative_mat_mul_add<float, float>(
    bindless, 0u, matrix_offset, 0u, bias_offset,
    Expr<CoopVector<float>>{input});

// typed bindless mat_mul_add
auto out1 = typed_bindless_cooperative_mat_mul_add<float, float>(
    bindless, 0u, matrix_offset, 0u, bias_offset,
    Expr<CoopVector<float>>{input});

// bindless mat_mul (no bias)
auto out2 = bindless_cooperative_mat_mul<float, float>(
    bindless, 0u, matrix_offset,
    Expr<CoopVector<float>>{input});

// typed bindless mat_mul (no bias)
auto out3 = typed_bindless_cooperative_mat_mul<float, float>(
    bindless, 0u, matrix_offset,
    Expr<CoopVector<float>>{input});
```

#### Outer Product Accumulate

Accumulate the outer product of two cooperative vectors into a cooperative matrix:

```cpp
ByteBufferVar matrix_buffer{luisa::compute::detail::ArgumentCreation{}};
CoopMatrixRef matrix_offset{CoopRefVecType::FLOAT32, 4, 8};
CoopVector<float> input1{4};
CoopVector<float> input2{8};
// ... fill vectors ...
matrix_offset.set_byte_offset(0u);
cooperative_outer_product_accumulate(
    matrix_buffer, matrix_offset,
    Expr<CoopVector<float>>{input1},
    Expr<CoopVector<float>>{input2});
```

#### Element-wise Math Operations

These compute element-wise operations by iterating over each lane:

```cpp
CoopVector<float> a{4}, b{4}, c{4}, lo{4}, hi{4}, v{4};
// ... fill ...

auto r_min  = cooperative_vector_min(a, b); // element-wise min
auto r_max  = cooperative_vector_max(a, b); // element-wise max
auto r_clamp = cooperative_vector_clamp(v, lo, hi);  // element-wise clamp
auto r_exp  = cooperative_vector_exp(v); // element-wise exp
auto r_log  = cooperative_vector_log(v); // element-wise log
auto r_tanh = cooperative_vector_tanh(v); // element-wise tanh
auto r_atan = cooperative_vector_atan(v); // element-wise atan
auto r_fma  = cooperative_vector_fma(a, b, c); // element-wise fma(a,b,c) = a*b+c
```

#### Element-wise Bitwise Operations (Integer Element Types)

```cpp
CoopVector<uint> a{4}, b{4}, v{4};
// ... fill ...

auto r_and = cooperative_vector_bitwise_and(a, b); // element-wise &
auto r_or  = cooperative_vector_bitwise_or(a, b); // element-wise |
auto r_xor = cooperative_vector_bitwise_xor(a, b); // element-wise ^
auto r_not = cooperative_vector_bitwise_not(v); // element-wise ~
auto r_shl = cooperative_vector_shift_left(v, 1u); // element-wise <<
auto r_shr = cooperative_vector_shift_right(v, 4u);  // element-wise >>
```

#### Device Compilation Considerations

- Backends: DX (Shader Model 6.8 with experimental features) or Vulkan.
- For DX, enable experimental features via `DirectXDeviceConfigExt`:

```cpp
class DXExperimentalConfigExt final : public DirectXDeviceConfigExt {
public:
    [[nodiscard]] bool UseExperimental() const noexcept override { return true; }
};

auto dx_config = luisa::make_unique<DXExperimentalConfigExt>();
config.extension = std::move(dx_config);
Device device = context.create_device("dx", &config);
```

- Compile and dispatch like regular kernels:

```cpp
Kernel1D kernel = [&](ByteBufferVar buf) noexcept {
    CoopVectorRef offset{CoopRefVecType::FLOAT32, 8};
    CoopVector<float> input{8};
    // ... fill input ...
    offset.set_byte_offset(0u);
    cooperative_vector_accumulate(buf, offset, Expr<CoopVector<float>>{input});
};
auto shader = device.compile(kernel);
stream << shader(buf).dispatch(1u) << synchronize();
```

#### Complete Load/Store Round-Trip Example

```cpp
constexpr auto n = 8u;
ByteBuffer vector_buffer = device.create_byte_buffer(n * sizeof(float));

// Store kernel
Kernel1D store_kernel = [&](ByteBufferVar buf) noexcept {
    CoopVectorRef offset{CoopRefVecType::FLOAT32, n};
    CoopVector<float> input{n};
    for (auto i = 0u; i < n; ++i) input[i] = static_cast<float>(i + 1);
    offset.set_byte_offset(0u);
    cooperative_vector_accumulate(buf, offset, Expr<CoopVector<float>>{input});
};

// Load kernel
Kernel1D load_kernel = [&](ByteBufferVar buf, BufferVar<float> output) noexcept {
    CoopVectorRef offset{CoopRefVecType::FLOAT32, n};
    offset.set_byte_offset(0u);
    auto loaded = cooperative_vector_load<float>(buf, offset);
    for (auto i = 0u; i < n; ++i) {
        output.write(i, loaded[i]);
    }
};

auto store_shader = device.compile(store_kernel);
auto load_shader = device.compile(load_kernel);
```

#### DSL Source File References

| File | Contents |
|------|----------|
| `include/luisa/dsl/coop_vector.h` | `CoopVector<T>`, `CoopVectorRef`, `CoopMatrixRef` DSL type definitions |
| `include/luisa/dsl/resource.h` (lines 880–1322) | All free functions: `cooperative_vector_*`, `cooperative_mat_*`, `bindless_cooperative_*`, `cooperative_outer_product_*` |
| `include/luisa/dsl/expr.h` (lines 151–155) | `Expr<CoopVector<T>>` template specialization with subscript access |
| `src/tests/unit/ast/test_cooperative_vector.cpp` | AST construction, DSL sugar, and device execution tests for all cooperative operations |

### DSL Summary Table

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
| CoopVector Obj | `CoopVector<float> v{n}` / `CoopVectorRef{type, n}` / `CoopMatrixRef{type, n, m}` |
| CVec Load/Store | `cooperative_vector_load<T>(buf, offset)` / `cooperative_vector_store(buf, offset, val)` |
| CVec Accumulate | `cooperative_vector_accumulate(buf, offset, val)` |
| CVec Splat | `cooperative_vector_splat<T>(scalar, n)` |
| CVec Cast | `cooperative_vector_cast<T>(vec)` |
| CVec Bindless | `bindless_cooperative_vector_load<T>(arr, slot, offset)` / `typed_bindless_cooperative_vector_store(arr, slot, offset, val)` |
| CVec Workgroup | `cooperative_vector_workgroup_load(shared, idx)` / `cooperative_vector_workgroup_store(shared, idx, val)` |
| CVec MatMul | `cooperative_mat_mul_add<Out,In>(buf, mat_off, bias_buf, bias_off, vec)` / `cooperative_mat_mul<Out,In>(buf, mat_off, vec)` |
| CVec Bindless Mat | `bindless_cooperative_mat_mul_add<Out,In>(arr, mat_slot, mat_off, bias_slot, bias_off, vec)` |
| CVec Outer Product | `cooperative_outer_product_accumulate(buf, mat_off, v1, v2)` |
| CVec Element-wise | `cooperative_vector_min/max/clamp/exp/log/tanh/atan/fma(a, b...)` |
| CVec Bitwise | `cooperative_vector_bitwise_and/or/xor/not(v)` / `cooperative_vector_shift_left/right(v, bits)` |

## AST: Usage Markers, CallExpr Marking, and AST-to-XIR Lowering

Quick reference for how manual `FunctionBuilder` AST tracks variable read/write usage and how builtin `CallOp` calls propagate usage to their arguments.

### Usage Enum

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

### Two-Layer Marker Design

#### 1. Per-expression cache

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

#### 2. FunctionBuilder storage

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

### RefExpr Forwarding

`src/ast/expression.cpp`

```cpp
void RefExpr::_mark(Usage usage) const noexcept {
    if (auto fb = detail::FunctionBuilder::current(); fb == builder()) {
        fb->mark_variable_usage(_variable.uid(), usage);
    }
}
```

Only marks when the current builder owns the expression, preventing stale marking across function boundaries.

### Manual API Example

```cpp
auto &cur = *FunctionBuilder::current();
auto ref = cur.reference(Type::of<float4>());
cur.mark_variable_usage(ref->variable().uid(), Usage::READ_WRITE);
```

### Builtin CallOp Usage Marking

#### Builtin detection

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

#### CallExpr::_mark rules

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

#### Rule summary

- **Default builtin**: every argument marked `READ`.
- **Write-style builtins** (list above): argument 0 marked `WRITE`; remaining arguments marked `READ`.
- **`PACK(value, words, offset)`**: value and offset are `READ`; the destination `buffer<uint>` is `WRITE`. Do not put `PACK` in the ordinary argument-0 write group.
- **`UNPACK(words, offset)`**: follows the default rule, so both arguments are `READ`.
- Atomic ops mark their target reference (argument 0) as `WRITE`; `AtomicRefNode::operate()` builds the `CallExpr` with the target as `_arguments[0]` (`src/ast/atomic_ref_node.cpp`).
- External calls copy each `ExternalFunction::argument_usages()` entry to the matching argument.
- Custom callable reference, resource, and opaque-custom arguments propagate the callee variable usage. Ordinary value arguments are always `READ`.

### AST-to-XIR Call and ID Conventions

Use `src/xir/translators/ast2xir.cpp` as the source of truth for argument form.

- Cache external declarations by `ExternalFunction::hash()`, preserve their name and return type, and accept `void` returns.
- Lower a non-resource external argument with `READ` or `NONE` usage as an XIR value.
- Lower a non-resource external argument containing `WRITE`, and every opaque custom argument, as an XIR reference. Require the call operand to be an lvalue.
- Keep resource arguments as XIR resources; do not additionally wrap them in ordinary references.
- Represent opaque custom arguments to ordinary custom callables as references even when the AST surface presents the backend handle by value.
- Lower `TypeIDExpr` to `uint64(0)` until the source Metal/CUDA paths define a stable cross-backend type-ID ABI.
- Lower `StringIDExpr` to the 64-bit `luisa::hash_value` of the string contents.

### Metal4 AIR External Declarations

Metal4 AIR preserves these declarations as exact LLVM declarations. Every used symbol must be defined by `ShaderOption::native_include` as compatible textual LLVM IR or bitcode. CodeGen checks target/data layout, function ABI, address space, calling convention, ABI attributes, and reference alignment, then links needed definitions before O2 and LLVM-14 downgrade. Values use register ABI, references use generic pointers, and external calls receive no hidden Luisa state parameters. Missing or incompatible definitions fail shader creation; Metal4 has no MSL or legacy-IR fallback. Compute and raster AOT loaders consume their compiled archives without rerunning AST-to-XIR or preflight.

This policy belongs to the separate `metal4` backend. The original `metal` backend remains source-MSL codegen and must not acquire a dependency on the Metal4 LLVM/AIR pipeline.

### Preserve Raster-Stage Identity and Payloads

- An AST `Function::Tag::RASTER_STAGE` does not encode vertex versus fragment. Require the caller to set `AST2XIRConfig::raster_stage` and create a `RasterStageFunction` with that explicit role. Do not guess from argument or return types.
- Keep argument zero as the stage payload: `AppData` for vertex and the vertex return type for fragment. All later arguments form the shared host root ABI; do not reorder them with kernel-style binding sorting.
- Preserve reflected structure member attributes as part of the payload `Type` description. `LUISA_RASTER_VARYING_INTERPOLATION(...)` marks member zero as position and records one interpolation value for every remaining member; AST-to-XIR must carry that exact `Type *` into both paired stages so Metal4 AIR and the common HLSL raster path see the same semantics. Do not rebuild an unannotated structural type or store qualifiers only in a frontend side table.
- Keep `Function::arguments()` zipped with the full raster `bound_arguments()` array. Raster bindings may contain `monostate` entries in-place, so `unbound_arguments()` and a bound-prefix assumption are not valid for this path.
- Expose `raster_object_id()`, `raster_barycentrics()`, `raster_is_front_face()`, `raster_base_instance()`, and `raster_discard()` through the normal DSL. Object ID and base instance are `uint`, barycentrics are `float3`, front-facing is `bool`, and discard lowers to the XIR raster-discard terminator. Translate front-facing to `RASTER_FRONT_FACING`/`SPR_FrontFacing`; it is fragment-only even though the shared raster implementation ABI carries a placeholder value in vertex code. Translate base instance to `RASTER_BASE_INSTANCE`/`SPR_BaseInstance`; it is vertex-only and receives the nonzero draw-time value stored in `RasterMesh`.
- Lower `DDX` and `DDY` to raster quad derivative XIR operations. Reject these operations outside a fragment-stage backend configuration rather than treating them as compute thread-group operations.
- Preserve a void fragment return as a null stage type. It is valid for a depth-only pass when the stage calls one consistent `raster_set_z_depth*` operation; XIR inlining must move that operation into the fragment entry before Metal4 AIR preflight. A void fragment with no color or depth output remains invalid.

### Normalize Bindless Aliases and Ray-Query State

- Lower every supported `TYPED_BINDLESS_*` and `TYPED_UNIFORM_BINDLESS_*` query, read, or write alias to the corresponding ordinary XIR bindless `ResourceQueryOp`, `ResourceReadOp`, or `ResourceWriteOp`. Keep the original operands and result type; backends should not need duplicate typed or uniform opcode families.
- Require the first operand of a direct ray-query object operation to be the query lvalue. Pass that same lvalue to every XIR read or write emitted for one AST call.
- Lower direct `RAY_QUERY_PROCEED(query)` to a `RAY_QUERY_OBJECT_PROCEED(query)` write immediately followed by a `RAY_QUERY_OBJECT_IS_TERMINATED(query)` read, and return `UNARY_BIT_NOT` of the read. The AST operation means "a candidate is available," which is the logical inverse of termination.
- Treat the four top-level `RAY_TRACING_QUERY_{ALL,ANY}` constructors, including their motion-blur variants, as fresh mutable state in downstream XIR passes. They must not be commoned or hoisted even when their operands match.

### PACK/UNPACK Lowering Contract

Validate `PACK` as `(packable value, buffer<uint>, uint offset) -> void` and `UNPACK` as `(buffer<uint>, uint offset) -> packable value`. Reject resources and opaque custom types.

AST-to-XIR wraps the packed value in a one-member Luisa structure whose alignment is at least four bytes, bitwise-casts the complete wrapper to `array<uint, sizeof(wrapper) / 4>`, and emits consecutive buffer writes or reads. This wrapper makes scalar bool/byte/short values one full word and retains Luisa padding for values such as `float3`. A four-field `{bool, bool, bool, bool}` value occupies four bytes and must not become the one-byte LLVM vector `<4 x i1>`; a Luisa byte4 becomes `<4 x i8>` and also occupies four bytes. A `float3` wrapper occupies 16 bytes. Backends must initialize padding before the bitwise cast; the Metal4 AIR path uses zero so `PACK` never observes LLVM poison.

### AST Files of Record

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

### Common Modifications

- **Add a new write-style builtin op**: extend the switch in `src/ast/expression.cpp` `CallExpr::_mark()` so argument 0 is `WRITE`.
- **Add an op whose destination is not argument 0**: give it a dedicated case, as `PACK` does for argument 1.
- **Query usage after building**: call `Function::variable_usage(uid)` or `FunctionBuilder::variable_usage(uid)`.
- **Custom callable reference/resource/custom args**: explicitly mark the callee variable with its real usage so callers propagate it correctly.
- **Change external lowering**: update declaration form and call operand form together, then run `test_ast_external_lowering` and the `unit_xir` CTest label.
- **Change bindless aliases or direct ray queries**: preserve ordinary XIR op normalization and lvalue identity, then run `test_ast_typed_bindless_lowering` and `test_xir_pass_lower_ray_query_loop`.

### Broader AST Changes

Use `.agents/skills/ast/SKILL.md` for manual `FunctionBuilder` construction. When adding an expression, statement, or `CallOp`, inspect the current enum, builder, usage marking, hashing, traversal/visitor, validation, serialization, and every affected AST or XIR backend. Search the implementation rather than copying a backend inventory or enum boundary into this skill.

## Runtime API: Context, Device, Stream, Buffers, Images, Ray Tracing, Rasterization

Covers `luisa/runtime/` classes for GPU compute: context/device management, memory, execution, ray tracing, rasterization, presentation.

### Context

```cpp
#include <luisa/runtime/context.h>
luisa::compute::Context ctx{argv[0]};
// or: Context ctx{argv[0], data_dir};

for (auto &&backend : ctx.installed_backends()) {
    auto names = ctx.backend_device_names(backend);
}
Device device = ctx.create_default_device();
```

### Device

```cpp
#include <luisa/runtime/device.h>
Device device = ctx.create_device("cuda");  // or "dx", "metal", "metal4", "vk", "hip", "fallback"

DeviceConfig cfg{.device_index = 0, .inqueue_buffer_limit = false};
Device device = ctx.create_device("cuda", &cfg, true/*validation*/);

auto backend = device.backend_name();
auto warp = device.compute_warp_size();
```

### Resource Creation

```cpp
Buffer<float> buf = device.create_buffer<float>(1024);
Buffer<MyStruct> sbuf = device.create_buffer<MyStruct>(100);
Image<float> img = device.create_image<float>(PixelStorage::FLOAT4, w, h);
Image<float> mip = device.create_image<float>(PixelStorage::FLOAT4, w, h, mips);
Volume<float> vol = device.create_volume<float>(PixelStorage::FLOAT4, w, h, d);
ByteBuffer bb = device.create_byte_buffer(size_bytes);
BindlessArray heap = device.create_bindless_array(65536);
IndirectDispatchBuffer indirect = device.create_indirect_dispatch_buffer(capacity);
```

### Stream

```cpp
#include <luisa/runtime/stream.h>
Stream stream = device.create_stream();
Stream compute = device.create_stream(StreamTag::COMPUTE);
Stream graphics = device.create_stream(StreamTag::GRAPHICS);
stream.set_name("my stream");
```

### Events

```cpp
#include <luisa/runtime/event.h>
Event event = device.create_event();
TimelineEvent timeline = device.create_timeline_event();

stream << event.signal();
stream << event.wait();
stream << graphics_event.wait(frame_index);
stream << graphics_event.signal(frame_index);
timeline.synchronize(frame_index);
```

### Buffer

```cpp
#include <luisa/runtime/buffer.h>
Buffer<float> buf = device.create_buffer<float>(1024);

// Transfer
stream << buf.copy_from(host_data);
stream << buf.copy_to(host_data);

// Views
auto view = buf.view(offset, count);
auto elem_view = buf.view().as<float>();  // for atomic operations
buf.set_name("vertex data");
```

#### Buffer-to-Buffer Copy

Use `BufferView::copy_from(BufferView<T>)` or `BufferView::copy_to(BufferView<T>)` — **these work in both normal and SAFE builds**.

```cpp
Buffer<float> src = device.create_buffer<float>(1024);
Buffer<float> dst = device.create_buffer<float>(1024);
Buffer<float> readback = device.create_buffer<float>(1024);

// ✅ Correct (SAFE-mode compatible): go through .view()
stream << dst.view().copy_from(src); // BufferView::copy_from(BufferView)
stream << readback.view().copy_from(src); // BufferView::copy_from(BufferView)

// ❌ Wrong (fails in SAFE mode): Buffer::copy_from(BufferView<T>)
// is guarded by #ifndef LUISA_ENABLE_SAFE_MODE
// stream << dst.copy_from(src.view()); // compile error in SAFE
```

#### SAFE Build Mode (`LUISA_ENABLE_SAFE_MODE`)

Define `LUISA_ENABLE_SAFE_MODE` at build time to **disable unsafe raw-pointer overloads**, enabling runtime validation of buffer creation. This is controlled by the cmake option `ENABLE_SAFE_MODE` in the project.

**What is excluded in SAFE mode** (`#ifndef LUISA_ENABLE_SAFE_MODE` blocks in `include/luisa/runtime/buffer.h`):

| Class | Excluded overloads |
|---|---|
| `Buffer<T>` | `copy_to(void*)`
`copy_to(BufferView<T>)`
`copy_to(const ByteBufferView&)`
`copy_from(const void*)`
`copy_from(const void*, move_only_function)`
`copy_from(BufferView<T>)`
`copy_from(const ByteBufferView&)` |
| `BufferView<T>` | `copy_to(void*)`
`copy_from(const void*)` |

**What remains available** (works in both modes):

| API | Example |
|---|---|
| `Buffer::copy_to(luisa::span<U>)` / `Buffer::copy_from(luisa::span<U>)` | `buf.copy_from(luisa::span{host_vec})` |
| `BufferView::copy_to(luisa::span<U>)` / `BufferView::copy_from(luisa::span<U>)` | `buf.view().copy_to(luisa::span{host_vec})` |
| `BufferView::copy_to(BufferView<T>)` / `BufferView::copy_from(BufferView<T>)` | `dst.view().copy_from(src)` |
| `BufferView::copy_to(const ByteBufferView&)` / `BufferView::copy_from(const ByteBufferView&)` | `buf.view().copy_to(byte_view)` |

**To pass the build in SAFE mode**: Always go through `BufferView` or `luisa::span` overloads instead of the `Buffer<T>` convenience overloads that are guarded. For buffer-to-buffer copy, change `dst.copy_from(src.view())` → `dst.view().copy_from(src)`. For raw-pointer transfers, change `buf.copy_from(data_ptr)` → `buf.copy_from(luisa::span{ptr, count})`.

### Image & Volume

```cpp
#include <luisa/runtime/image.h>
Image<float> img = device.create_image<float>(PixelStorage::FLOAT4, w, h);
Image<float> img2 = device.create_image<float>(swapchain.backend_storage(), size);
// Mipmapped: device.create_image<float>(PixelStorage::FLOAT4, w, h, mips);
// Simultaneous access: device.create_image<float>(PixelStorage::FLOAT4, w, h, 1, true);

#include <luisa/runtime/volume.h>
Volume<float> vol = device.create_volume<float>(PixelStorage::FLOAT4, w, h, d);
```

#### Sparse Images and Volumes

Sparse image/volume mip counts follow the same convention as regular textures: zero requests the full chain and larger requests are clamped to the logical maximum. Tile map and unmap regions are validated against the selected mip's ceil-divided tile grid, not the base extent or a floor-divided grid. Counts must be nonzero and range arithmetic must not wrap. Sparse copy regions use the same validation, convert tiles to texel offsets, and clip the final partial tile to the selected mip extent; buffer-backed copies must provide enough bytes for that clipped texel region.

Sparse buffers use the same nonzero-count and checked-range rules over a ceil-divided byte tile grid. Every sparse map operation requires a valid heap created by the same `DeviceInterface` as the sparse resource.

#### Image in Kernels

```cpp
Kernel2D k = [&](ImageFloat img) {
    UInt2 coord = dispatch_id().xy();
    Float4 c = img.read(coord);
    img.write(coord, make_float4(1,0,0,1));
};
```

### BindlessArray (Host Side)

```cpp
#include <luisa/runtime/bindless_array.h>
BindlessArray heap = device.create_bindless_array(64);
heap.emplace_on_update(slot, buffer);
heap.emplace_on_update(slot, image, TextureSampler::linear_linear_mirror());
stream << heap.update() << synchronize();

// Kernel:
Kernel1D k = [&](Var<BindlessArray> heap) {
    auto v = heap.buffer<float>(slot).read(idx);
    auto c = heap.texture2d(slot).sample(uv);
};
```

### Swapchain

```cpp
#include <luisa/runtime/swapchain.h>
Swapchain swapchain = device.create_swapchain(stream, SwapchainOption{
    .display = window.native_display(),
    .window = window.native_handle(),
    .size = resolution,
    .wants_hdr = false,
    .wants_vsync = true,
    .back_buffer_count = 3});
stream << swapchain.present(image);
```

On iOS, UIKit owns the `UIView`/`CAMetalLayer`; rendering sources should still construct the ordinary `Window` and `Swapchain`. The app host installs a process-wide `Window::set_native_handle_provider(...)` before entering the example. The provider returns the native layer/display handles, and platform touch/keyboard/resize events are queued through the `post_native_*` functions and delivered by `Window::poll_events()` on the rendering thread. Clear the provider only after every provider-backed window has been destroyed. Do not teach each rendering example about UIKit or bypass `Window -> Swapchain`.

### Ray Tracing (Host Resources)

```cpp
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/rtx/curve.h>

Mesh mesh = device.create_mesh(vertex_buffer, triangle_buffer);
Accel accel = device.create_accel();
accel.emplace_back(mesh, transform);
accel.emplace_back(mesh, transform, visibility_mask);
stream << mesh.build() << accel.build();
stream << accel.update_instance_buffer();

Curve curve = device.create_curve(CurveBasis::CUBIC_BSPLINE, cp_buf, seg_buf);
```

#### Motion Instances and Metal4 Feature Queries

```cpp
#include <luisa/runtime/rtx/motion_instance.h>

AccelMotionOption motion_option{};
motion_option.mode = AccelMotionMode::MATRIX;
motion_option.keyframe_count = 2u;
auto moving = device.create_motion_instance(mesh, motion_option);
std::array keyframes{translation(-1.f, 0.f, 0.f),
                     translation(1.f, 0.f, 0.f)};
moving.set_keyframes(luisa::span{keyframes});

AccelOption accel_option{};
accel_option.allow_update = true;
auto accel = device.create_accel(accel_option);
accel.emplace_back(moving);
stream << mesh.build() << moving.build() << accel.build();

// After changing keyframes, rebuild the host motion resource and refit/rebuild
// the containing TLAS.
moving.set_keyframes(luisa::span{new_keyframes});
stream << moving.build() << accel.build();
```

For the `metal4` backend, use `device.query("metal_motion_blur")` before choosing matrix motion and `device.query("metal4_component_motion")` before choosing SRT/component motion. The latter is Apple9-only. The independent `metal4_address_driven_acceleration_structures` query reports whether AS build/refit uses the MTL4 encoder; a false value can still support matrix motion through the synchronized compatibility build path. These queries return the strings `"true"` or `"false"`.

#### Ray Tracing Kernel

```cpp
Kernel2D trace = [&](AccelVar accel, BufferFloat4 img) {
    Var<Ray> ray = make_ray(origin, direction);
    Var<TriangleHit> hit = accel.intersect(ray, {});
    $if (!hit->miss()) {
        Float3 c = triangle_interpolate(hit.bary, v0, v1, v2);
    };
};
```

### Rasterization

```cpp
#include <luisa/runtime/raster/depth_buffer.h>
#include <luisa/runtime/raster/raster_scene.h>
#include <luisa/runtime/raster/raster_shader.h>

DepthBuffer depth = device.create_depth_buffer(DepthFormat::D32, size);
auto raster_shader = device.compile(raster_kernel, mesh_format);
RasterScene scene = device.create_raster_scene(vertex_buffer, index_buffer);
```

`RasterMesh` carries both an instance count and an optional base instance. The base defaults to zero; pass it after `vertex_offset` when a vertex shader uses `raster_base_instance()`. The runtime forwards it to indexed and non-indexed Metal4, DX12, and Vulkan draws, and the ordinary instance ID starts at that base value.

```cpp
RasterMesh mesh{vertex_streams, index_buffer.view(),
                instance_count, object_id,
                vertex_offset, base_instance};
```

Depth-only raster draws use the existing zero-RTV form of `draw`. Return `void` from the fragment stage, call one of the `raster_set_z_depth*` builtins, and pass a non-null depth buffer without trailing color images. Metal4 AIR records a fragment color-output count of zero in both JIT and AOT shaders.

```cpp
std::move(shader(args...))
    .draw(std::move(meshes), mesh_format, viewport, state, &depth);
```

For fixed-reference stencil testing, create a stencil-bearing depth buffer and fill the complete public state. The same reference/mask semantics are forwarded by Metal4, DX12, and Vulkan.

```cpp
auto depth_stencil =
    device.create_depth_buffer(DepthFormat::D32S8A24, size);

StencilFaceOp face{
    .stencil_fail_op = StencilOp::Keep,
    .depth_fail_op = StencilOp::Keep,
    .pass_op = StencilOp::Replace,
    .comparison = Comparison::Equal};
RasterState state{};
state.stencil_state = StencilState{
    .enable_stencil = true,
    .front_face_op = face,
    .back_face_op = face,
    .read_mask = 0xffu,
    .write_mask = 0xffu,
    .reference = 1u};

stream << depth_stencil.clear(1.0f)
       << std::move(shader(args...))
              .draw(std::move(meshes), mesh_format,
                    viewport, state, &depth_stencil);
```

`DepthBuffer::clear()` clears the stencil plane to zero when the format has stencil. `D32S8A24` maps directly to depth32-float/stencil8. A requested `D24S8` remains the logical runtime format, but Metal4 transparently uses D32S8A24 physical storage with a warning on devices that do not support depth24-unorm/stencil8. Stencil requires such a depth-stencil attachment; shader-written stencil reference and conservative rasterization are not part of the current Metal4 AIR contract.

### CommandList

Batch commands for efficient submission:

```cpp
CommandList cmdlist = CommandList::create();
cmdlist << kernel.dispatch(w, h) << buffer.copy_to(host_data);
stream << cmdlist.commit() << synchronize();
```

> Prefer merging dispatch + transfers into one `CommandList` + single commit/synchronize over separate stream submissions.

### Complete Runtime Example

```cpp
#include <luisa/luisa-compute.h>
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context ctx{argv[0]};
    Device device = ctx.create_device("cuda");
    Stream stream = device.create_stream();
    Buffer<float> buf = device.create_buffer<float>(1024);

    Kernel1D k = [&](BufferVar<float> buf) {
        auto idx = dispatch_id().x;
        buf.write(idx, buf.read(idx) + 1.0f);
    };

    auto shader = device.compile(k);
    stream << shader(buf).dispatch(1024) << synchronize();
}
```

### Common Patterns

#### Multi-Stream Sync

```cpp
Stream compute = device.create_stream(StreamTag::COMPUTE);
Stream graphics = device.create_stream(StreamTag::GRAPHICS);
Event event = device.create_event();
compute << shader().dispatch(w, h) << event.signal();
graphics << event.wait() << swapchain.present(img);
```

#### Triple Buffering

```cpp
TimelineEvent timeline = device.create_timeline_event();
uint64_t frame = 0;
while (running) {
    if (frame >= 3) timeline.synchronize(frame - 2);
    stream << shader().dispatch(w, h) << timeline.signal(++frame);
}
```

#### Buffer Upload/Download

Always prefer `luisa::span<T>` overloads for SAFE-mode compatibility:

```cpp
luisa::vector<float> host_data(1024, 1.0f);

// ✅ span-based (SAFE-mode compatible)
stream << buf.copy_from(luisa::span{host_data}) << synchronize();
stream << buf.copy_to(luisa::span{host_data}) << synchronize();

// ❌ raw-pointer (fails in SAFE mode)
// stream << buf.copy_to(host_data.data()) << synchronize();
// stream << buf.copy_from(host_data.data()) << synchronize();
```

### Runtime Key Headers

| Header | Class |
|---|---|
| `luisa/runtime/context.h` | Context |
| `luisa/runtime/device.h` | Device |
| `luisa/runtime/stream.h` | Stream |
| `luisa/runtime/event.h` | Event, TimelineEvent |
| `luisa/runtime/buffer.h` | Buffer |
| `luisa/runtime/image.h` | Image |
| `luisa/runtime/volume.h` | Volume |
| `luisa/runtime/swapchain.h` | Swapchain |
| `luisa/runtime/bindless_array.h` | BindlessArray |
| `luisa/runtime/dispatch_buffer.h` | IndirectDispatchBuffer |
| `luisa/runtime/command_list.h` | CommandList |
| `luisa/runtime/rtx/accel.h` | Accel |
| `luisa/runtime/rtx/mesh.h` | Mesh |
| `luisa/runtime/rtx/curve.h` | Curve |
| `luisa/runtime/rtx/ray.h` | Ray, hit types |
| `luisa/runtime/raster/raster_shader.h` | RasterShader |
| `luisa/runtime/raster/raster_scene.h` | RasterScene |

## VSTL: Containers and Utilities

`vstd` = custom containers/utilities under `include/luisa/vstl/`. Many are aliases to `luisa::` STL replacements.

### Aliases

| vstd type | Actual |
|---|---|
| `vstd::vector<T>` | `luisa::vector<T>` |
| `vstd::fixed_vector<T,N>` | `luisa::fixed_vector<T,N>` |
| `vstd::span<T>` | `luisa::span<T>` |
| `vstd::string` | `std::basic_string<char,...,luisa::allocator<char>>` |
| `vstd::wstring` | `std::basic_string<wchar_t,...,luisa::allocator<wchar_t>>` |
| `vstd::string_view` | `luisa::string_view` |
| `vstd::function<T>` | `luisa::move_only_function<T>` |
| `vstd::shared_ptr<T>` | `luisa::shared_ptr<T>` |
| `vstd::spin_mutex` | `luisa::spin_mutex` |
| `vstd::unordered_map/set` | `luisa::unordered_map/set` |

### Vector Helpers

`#include <luisa/vstl/vector.h>`

```cpp
vstd::push_back_func(vec, 10, [&](size_t i) { return i * 2; });
vstd::push_back_func(vec, 5, [] { return Foo{}; });
vstd::push_back_all(vec, ptr, n);
vstd::push_back_all(vec, {1, 2, 3});
vstd::push_back_all(vec, some_span);
```

### HashMap

`#include <luisa/vstl/hash_map.h>`. Power-of-2 capacity, open addressing, per-bucket red-black trees.

```cpp
vstd::HashMap<Key, Value> map;           // default
vstd::HashMap<Key, Value> map(capacity); // pre-sized
vstd::HashMap<Key> set;                  // HashSet when V=void
```

Template: `HashMap<K, V=void, Hash=HashValue, Compare=compare<K>, allocType=VEngine>`

#### API

```cpp
auto [idx, ok] = map.try_emplace(key, args...);  // insert if absent
auto idx = map.force_emplace(key, args...);       // insert or overwrite
auto idx = map.emplace(key, args...);             // = try_emplace().first
auto idx = map.find(key);
if (idx) { auto& k = idx.key(); auto& v = idx.value(); }
map.remove(key);   // or map.remove(idx), map.remove(it)
map.clear(); map.reserve(n);
map.size(); map.empty(); map.capacity();
```

For `V=void` (sets): `idx.Get()`, `idx->`, `idx*`.

#### Iteration

```cpp
for (auto& kv : map) { }             // lvalue: Iterator → NodePair&
for (auto&& kv : std::move(map)) { } // move: MoveIterator → MoveNodePair&&
```

### ArenaHashMap

`#include <luisa/vstl/arena_hash_map.h>`. Arena-backed, **trivially destructible** K/V only.

```cpp
vstd::ArenaHashMap<ArenaType, Key, Value> map(capacity, std::move(arena));
// API: try_emplace, force_emplace, emplace, find, remove, clear, reserve
// No custom Index/remove(Index); key-based removal only.
```

### Object Pool

`#include <luisa/vstl/pool.h>`. Free-list pool using `vengine_malloc`.

```cpp
vstd::Pool<MyType> pool(initial_capacity, initialize=true);
// Pool<T, true>  — trivially destructible, lightweight
// Pool<T, false> — tracks live objects, supports iteration

T* obj = pool.create(args...);
T* obj = pool.create_lock(mtx, args...);  // thread-safe
pool.destroy(obj);
pool.destroy_lock(mtx, obj);
pool.destroy_all();

// Non-trivial only:
for (T* obj : pool.iterator()) { }
```

### Queues

`#include <luisa/vstl/lockfree_array_queue.h>`

#### LockFreeArrayQueue

Mostly lock-free circular queue, spin-mutex on resize.

```cpp
vstd::LockFreeArrayQueue<T> q(capacity);
q.enqueue(args...);              // blocking
q.try_push(args...);             // try-lock
auto opt = q.dequeue();          // optional<T>
bool ok = q.pop(&dst);           // pop into pre-constructed T*
auto opt = q.try_pop();          // non-blocking
q.reserve(newCapa); size_t len = q.length();
```

#### SingleThreadArrayQueue

SPSC, no locking.

```cpp
vstd::SingleThreadArrayQueue<T> q(capacity);
T* ptr = q.enqueue(args...);     // returns ptr to enqueued item
T* front = q.front();            // peek
auto opt = q.dequeue();
bool ok = q.pop(&dst);
q.pop_discard();
q.reserve(newCapa);
```

### StackAllocator

`#include <luisa/vstl/stack_allocator.h>`

```cpp
vstd::DefaultMallocVisitor visitor;
vstd::StackAllocator alloc(initCapacity, &visitor, expandRate=1.5);
auto chunk = alloc.allocate(size);        // {handle, offset}
auto chunk = alloc.allocate(size, align); // aligned
T* ptr = alloc.allocate_memory<T>();      // typed + zeroed
alloc.clear(); alloc.dispose();
```

### Smart Pointers

`#include <luisa/vstl/unique_ptr.h>`

```cpp
auto p = vstd::make_unique<T>(args...);
auto p = vstd::create_unique(raw_ptr);     // adopts raw, uses vengine_free
auto sp = vstd::make_shared<T>(args...);
auto sp = vstd::create_shared(raw_ptr);
// unique_ptr: if T derives from IDisposable, calls Dispose() on destroy
```

### Variant

`#include <luisa/vstl/meta_lib.h>`. Custom `vstd::variant<...>`.

```cpp
vstd::variant<int, float, std::string> v = 42;
size_t idx = v.index(); bool b = v.is_type_of<int>();
int& i = v.get<0>(); int* p = v.try_get<int>(); int& j = v.force_get<int>();

v.visit([&](auto& x) {});
v.multi_visit([&](int&){}, [&](float&){}, [&](std::string&){});
auto r = v.visit_or(fallback, [](auto& x) { return process(x); });

v.reset_as<int>(123); v.reset_as<2>(args...);  // reset by index
```

### Optional & StackObject

`#include <luisa/vstl/meta_lib.h>`

```cpp
vstd::StackObject<T, false> obj;  // manual lifetime
obj.create(args...); obj.destroy();
T& val = *obj; T* ptr = obj.ptr();

vstd::optional<T> opt(args...);   // = StackObject<T, true>, auto-destroy
if (opt.has_value()) { }
T val = opt.value_or(default_val);
```

### String Utilities

`#include <luisa/vstl/vstring.h>`, `<luisa/vstl/string_builder.h>`

```cpp
// vstd::string
vstd::string s = vstd::to_string(42);
s << value;  // operator<<

// StringBuilder (fixed_vector<char,32>)
vstd::StringBuilder sb;
sb.append("hello"); sb.append(view); sb.append('!'); sb << 42; sb += "suffix";
vstd::string_view v = sb.view();

// StringUtil
for (auto part : vstd::StringUtil::split(str, ',')) { }
vstd::StringUtil::to_lower(s); vstd::StringUtil::to_upper(s);
vstd::StringUtil::to_base64(binary_span, result);
vstd::StringUtil::from_base64(base64_str, byte_vec);
vstd::StringUtil::to_hex_string(binary_span, result, upper=true);
```

### Hash & Compare

`#include <luisa/vstl/hash.h>`, `<luisa/vstl/compare.h>`

```cpp
size_t h = vstd::hash<MyType>{}(value);
int32_t c = vstd::compare<MyType>{}(a, b);  // -1,0,1
// vstd::HashValue delegates to hash<T>; vstd::Hash::binary_hash uses xxHash64
// compare uses memcmp for non-arithmetic non-enum types
```

### Function Reference

`#include <luisa/vstl/functional.h>`

```cpp
vstd::FuncRef<void(int)> cb = vstd::make_func_ref(lambda);
cb(42);
vstd::FuncRef<int(double)> cb2 = &my_c_function;
```

### Ranges

`#include <luisa/vstl/ranges.h>`. One-shot ranges (debug: begin once).

```cpp
for (int64_t i : vstd::range(begin, end, step)) { }
for (int64_t i : vstd::range(end)) { }            // 0..end-1
for (T& x : vstd::ptr_range(ptr, count)) { }
for (auto& x : vstd::ite_range(container)) { }

// Chain
auto r = vstd::make_ite_range(container)
    | filter_range([](auto& x) { return x.active; })
    | transform_range([](auto& x) { return x.value; });

// Erased heap range: i_range()
```

### Others

```cpp
// Scope guard
auto guard = vstd::scope_exit([&] { cleanup(); });

// Macros
KILL_COPY_CONSTRUCT(ClassName)  KILL_MOVE_CONSTRUCT(ClassName)
VSTD_TRIVIAL_COMPARABLE(ClassName)  // memcmp-based == != > <

// Allocation (#include <luisa/vstl/memory.h>)
void* p = vengine_malloc(size); void* p = vengine_realloc(old, size); vengine_free(p);
T* obj = vengine_new<T>(args...); T* arr = vengine_new_array<T>(count, args...); vengine_delete(obj);

// Guid & MD5 (#include <luisa/vstl/v_guid.h>, <luisa/vstl/md5.h>)
vstd::Guid g(true);               // generate new
vstd::Guid g("12345678-...");
auto og = vstd::Guid::TryParseGuid(str);
vstd::MD5 md5(str);
vstd::string s = md5.to_string(upper=true);
```


## LuisaCompute Core Library (lc_core)

Based on test cases in `src/tests/unit/core/*.cpp`.

### Basic Traits

**Header**: `<luisa/core/basic_traits.h>`

#### Type Predicates
```cpp
luisa::always_false_v<T...>           // always false (for static_assert)
luisa::always_true_v<T...>            // always true
// Scalars
luisa::is_integral_v<T>               // is_boolean_v, is_floating_point_v, is_signed_v, is_unsigned_v
luisa::is_signed_integral_v<T>        // is_unsigned_integral_v, is_scalar_v
// Vectors
luisa::is_vector_v<T> / is_vector_v<T,N> / is_vector2_v / is_vector3_v / is_vector4_v
luisa::is_boolean_vector_v<T>         // is_floating_point_vector_v, is_integral_vector_v
luisa::is_signed_integral_vector_v<T> // is_unsigned_integral_vector_v
// Matrices
luisa::is_matrix_v<T> / is_matrix_v<T,N> / is_matrix2_v / is_matrix3_v / is_matrix4_v
// Combined
luisa::is_basic_v<T>                  // scalar||vector||matrix
luisa::is_boolean_or_vector_v<T>      // is_floating_point_or_vector_v, is_integral_or_vector_v
luisa::is_signed_integral_or_vector_v<T> // is_unsigned_integral_or_vector_v
luisa::is_vector_same_dimension_v<T1, T2, ...>
```

#### Type Transformations
```cpp
using Elem = luisa::vector_element_t<VecType>;
using Elem = luisa::matrix_element_t<MatType>;
auto val = luisa::to_underlying(Enum::Value);
constexpr size_t dim = luisa::vector_dimension_v<T>;  // 1 for scalars
constexpr size_t dim = luisa::matrix_dimension_v<T>;  // 1 for scalars
```

### Basic Types

**Header**: `<luisa/core/basic_types.h>`

#### Aliases
```cpp
// Scalars
luisa::byte(int8_t), ubyte(uint8_t), ushort(uint16_t), uint(uint32_t), ulong(uint64_t), slong(int64_t), half(16-bit)
// Vectors (2/3/4): bool, short, ushort, byte, ubyte, int, uint, slong, ulong, half, float, double
// Matrices (2x2/3x3/4x4): float, double, half
```

#### Construction
```cpp
float2 f(1.0f);                // broadcast
int2 i(1, 2);                  // component-wise
auto z = float2::zero();       // (0,0)
auto o = float2::one();        // (1,1)

// Matrix (default = identity)
float2x2 m2;                   // identity
float2x2 m2c(float2(1,2), float2(3,4)); // from cols
auto eye = float2x2::eye(2.0f); // 2*identity
auto fill = float2x2::fill(3.0f);
```

#### Element Access
```cpp
float3 v(1,2,3); v.x, v.y, v.z; v[0]; v[1] = 5.0f;
float3x3 m; m[0], m[1], m[2];  // column access
float e = m[col][row];
```

#### Operators
```cpp
a+b, a-b, a*b, a/b            // component-wise
a*2.0f, 3.0f*a                // scalar
-a, +a                         // unary
~i, i<<1                       // bitwise (integral)
a==b, a<b                      // comparison → bool vector
b1||b2, b1&&b2                 // bool logic
any(b), all(b), none(b)        // bool vector reduce

// Matrix
m*2.0f, 3.0f*m, m/2.0f        // scalar
m * v                          // matrix-vector
a * b                          // matrix-matrix
a + b, a - b                   // element-wise
```

#### Make Functions
```cpp
make_float2(1.0f);                        // broadcast
make_float2(1.0f, 2.0f);                 // components
make_float2(float3(1,2,3));              // from larger vec
make_float3(float2(1,2), 3.0f);          // vec + scalar
make_float3(1.0f, float2(2,3));          // scalar + vec
make_float4(float2(1,2), float2(3,4));   // vec + vec

make_float2x2(2.0f);                      // diagonal fill
make_float2x2(1,2,3,4);                  // row-major
make_float2x2(float2(1,2), float2(3,4)); // columns
make_float3x3(1,2,3, 4,5,6, 7,8,9);     // row-major
make_float4x4(1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16);
```

### Binary File Stream

**Header**: `<luisa/core/binary_file_stream.h>`. Use instead of `std::ifstream` for binary I/O.

```cpp
luisa::BinaryFileStream stream("file.bin");
if (stream.valid()) { /* or operator bool */ }
size_t len = stream.length(), pos = stream.pos();
stream.set_pos(128); stream.set_pos(0);
stream.read(luisa::span<std::byte>(buf.data(), buf.size()));
stream.close();
luisa::BinaryFileStream s2(std::move(stream));  // move semantics
```

### Binary IO (BinaryBlob)

**Header**: `<luisa/core/binary_io.h>`

```cpp
luisa::BinaryBlob blob{ptr, size, [](void* p) { ::operator delete(p); }};
luisa::BinaryBlob empty;
std::byte* d = blob.data(); size_t sz = blob.size(); bool e = blob.empty();
luisa::span<std::byte> sp = static_cast<luisa::span<std::byte>>(blob);
luisa::BinaryBlob b2(std::move(blob)); b3 = std::move(b2);
void* raw = blob.release();  // blob becomes empty; manual delete required
```

### Clock

**Header**: `<luisa/core/clock.h>`

```cpp
luisa::Clock clock;           // starts timing on construction
clock.tic();                  // reset
double ms = clock.toc();      // elapsed ms since tic (does NOT reset)
double t1 = clock.toc();      // cumulative
```

### Dynamic Module

**Header**: `<luisa/core/dynamic_module.h>`

```cpp
auto mod = luisa::DynamicModule::load("name");           // platform-specific
auto mod = luisa::DynamicModule::load("/path", "name");  // from dir
auto mod = luisa::DynamicModule::load_exact("/path/lib.so");
if (mod) { void* h = mod.handle(); }
void* addr = mod.address("fn");
auto* fn = mod.function<MyFunc>("fn");
void* raw = mod.release(); luisa::dynamic_module_destroy(raw);
mod.reset();

// Search paths
luisa::DynamicModule::add_search_path(dir);
luisa::DynamicModule::remove_search_path(dir);
```

### First Fit Allocator

**Header**: `<luisa/core/first_fit.h>`

```cpp
luisa::FirstFit alloc(1024, 8);          // size, alignment
luisa::FirstFit::Node* n = alloc.allocate(100);
if (n) { size_t off = n->offset(), sz = n->size(); }
n = alloc.allocate_best_fit(100);
alloc.free(n);
size_t sz = alloc.size(), align = alloc.alignment();
luisa::string fl = alloc.dump_free_list();
```

### Logging

**Header**: `<luisa/core/logging.h>`

```cpp
luisa::log_level_verbose();  // log_level_info(), log_level_warning(), log_level_error()
luisa::log_flush();

// Function-style
luisa::log_verbose("msg"); luisa::log_info("v: {}", 42); luisa::log_warning("warn");

// Macro-style (recommended)
LUISA_VERBOSE("msg"); LUISA_INFO("v: {}, {}", 1, 2); LUISA_WARNING("warn");
LUISA_VERBOSE_WITH_LOCATION("dbg: {}", val);
LUISA_INFO_WITH_LOCATION("proc: {}", name);
LUISA_WARNING_WITH_LOCATION("deprecated: {}", api);
```

Format: `{}` (default), `{:x}` (hex), `{:b}` (binary), `{:e}` (scientific), `{:.2f}` (fixed precision).

### Mathematics

**Header**: `<luisa/core/mathematics.h>`

#### Scalar
```cpp
luisa::next_pow2(100u);          // 128
luisa::fract(3.7f);              // 0.7 (-3.7f → 0.3)
luisa::radians(180.0f);          // pi
luisa::degrees(constants::pi);   // 180
luisa::sin(x), cos(x), sqrt(x), abs(x), min(a,b), max(a,b)
```

#### Vector (component-wise)
```cpp
luisa::sin(v2), cos(v2), sqrt(v2), abs(v2), floor(v2), ceil(v2), fract(v2)
luisa::min(a, b), max(a, b), pow(a, b), atan2(y, x), fmod(a, b)
luisa::min(2.0f, a)              // scalar-vector
luisa::isnan(v2), isinf(v2)

luisa::dot(a, b);                // scalar result
luisa::length(a);                // sqrt(dot(a,a))
luisa::distance(a, b);
luisa::normalize(a);             // a/length(a)
luisa::cross(c, d);              // 3D only
```

#### Matrix
```cpp
luisa::transpose(m);             // 2x2/3x3/4x4
luisa::inverse(m);
luisa::determinant(m);

// Transformations (float4x4)
luisa::translation(1,2,3);      // or translation(float3)
luisa::scaling(2,3,4);          // non-uniform; scaling(5.0f) = uniform
luisa::rotation(axis_float3, angle_rad);
```

#### Interpolation & Selection
```cpp
luisa::select(false_val, true_val, cond);          // scalar/vector
luisa::lerp(a, b, t);                               // a+(b-a)*t (scalar/vector)
luisa::clamp(v, lo, hi);                            // scalar/vector
luisa::sign(x);                                     // 1.0/-1.0 (scalar/vector, float/int)
luisa::fma(a, b, c);                                // a*b+c
```

#### Constants
```cpp
luisa::constants::pi, pi_over_2, pi_over_4, two_pi, inv_pi, e
```

### Pool Allocator

**Header**: `<luisa/core/pool.h>`

```cpp
luisa::Pool<MyClass> pool;                  // thread-safe
luisa::Pool<MyClass, false> pool_nt;        // non-thread-safe
MyClass* obj = pool.allocate();             // raw (no ctor)
pool.deallocate(obj);
MyClass* obj2 = pool.create();              // default-construct
MyClass* obj3 = pool.create(args...);       // construct with args
pool.destroy(obj2);
```

### Fiber

**Header**: `<luisa/core/fiber.h>`. Built on marl.

#### Scheduler
```cpp
luisa::fiber::scheduler sched;     // all cores
luisa::fiber::scheduler sched(4);  // fixed threads
// RAII: binds on construction, unbinds on destruction
```

#### Sync Primitives
```cpp
luisa::fiber::event evt(luisa::fiber::event::Mode::Manual, false);
evt.signal(); evt.clear(); evt.wait(); bool r = evt.test()/evt.is_signalled();

luisa::fiber::counter cnt(3);
cnt.add(2); cnt.done(); cnt.wait();

luisa::fiber::mutex mtx;
luisa::fiber::lock lck(mtx);
luisa::fiber::condition_variable cv;
```

#### Tasks
```cpp
luisa::fiber::schedule([]() noexcept { /* work */ });
auto evt = luisa::fiber::async([]() noexcept { return 42; }); evt.wait();
```

#### Parallel For
```cpp
// Blocking
luisa::fiber::parallel(100, [](uint32_t i) noexcept {});
luisa::fiber::parallel(100, [](uint32_t begin, uint32_t end) noexcept {});
// Async
auto cnt = luisa::fiber::async_parallel(100, [](uint32_t i) noexcept {}); cnt.wait();
// Iterator
luisa::fiber::parallel(data.begin(), data.end(), 64, [](auto l, auto r) {});
auto cnt = luisa::fiber::async_parallel(data.begin(), data.end(), 64, [](auto l, auto r) {}); cnt.wait();
// With external counter
luisa::fiber::counter cnt(0); luisa::fiber::async_parallel(cnt, 100, [](uint32_t i){});
// Control batch size
luisa::fiber::parallel(1000, []{}, /*internal_jobs=*/10);

uint32_t n = luisa::fiber::worker_thread_count();
```

#### Defer
```cpp
{ luisa_fiber_defer(printf("world\n")); printf("hello "); }
```

### STL Containers & Utilities

**Location**: `include/luisa/core/stl/`. Uses `std::` or EASTL based on `LUISA_USE_SYSTEM_STL`. All in `luisa` namespace.

#### Memory
**Header**: `<luisa/core/stl/memory.h>`
```cpp
luisa::allocator<T> alloc;
auto sz1 = 64_k;  // 65536 (also 16_M, 2_G)
luisa::unique_ptr<T> up = luisa::make_unique<T>(args...);
luisa::shared_ptr<T> sp = luisa::make_shared<T>(args...);
luisa::weak_ptr<T> wp = sp;
luisa::span<T> s(data, count);
T* p = luisa::allocate_with_allocator<T>(n); luisa::deallocate_with_allocator(p);
T* obj = luisa::new_with_allocator<T>(args...); luisa::delete_with_allocator(obj);
auto u = luisa::bit_cast<uint32_t>(3.14f);
```

#### Strings & Format
**Header**: `<luisa/core/stl/string.h>`, `<luisa/core/stl/format.h>`
```cpp
luisa::string s = "hello"; luisa::u8string u8s = u8"hello"; luisa::wstring ws = L"hello";
luisa::string_hash h; uint64_t hash = h("hello");
luisa::string s = luisa::format("Value: {}, {}", 42, 3.14);
auto s2 = luisa::to_string(float3(1,2,3));
auto hex = luisa::hash_to_string(0x1234ABCD);
```

#### Containers
```cpp
// <stl/vector.h>
luisa::vector<int> vec = {1,2,3}; vec.push_back(4);
luisa::fixed_vector<int, 64> fvec;
luisa::bitvector bits(100);
auto* raw = luisa::enlarge_by(vec, 10);  // push 10 uninit, return ptr to first
luisa::vector_resize(vec, 100);
size_t bytes = luisa::size_bytes(vec);

// <stl/unordered_map.h> — dense hash map (faster than std)
luisa::unordered_map<string, int> map; map.emplace("key", 42);
luisa::unordered_set<int> set;

// <stl/map.h> — ordered
luisa::map<string,int> omap; luisa::set<int> oset; luisa::multimap<string,int> mm; luisa::multiset<int> ms;

// <stl/fixed_map.h> — fixed-capacity
luisa::fixed_map<int,string,64> fmap; luisa::fixed_set<int,64> fset;
luisa::fixed_unordered_map<int,string,64> fumap; luisa::fixed_unordered_set<int,64> fus;
luisa::fixed_multimap<int,string,64> fmm; luisa::fixed_multiset<int,64> fms;

// <stl/vector_map.h> — sorted-vector-based (better cache locality)
luisa::vector_map<int,string> vm; luisa::vector_set<int> vs;
luisa::vector_multimap<int,string> vmm; luisa::vector_multiset<int> vms;

// <stl/deque.h>, <stl/queue.h>, <stl/stack.h>, <stl/priority_queue.h>
luisa::deque<int> dq; luisa::queue<int> q; luisa::stack<int> st; luisa::priority_queue<int> pq;

// <stl/list.h>
luisa::list<int> lst; luisa::forward_list<int> flst;
luisa::fixed_list<int,64> fl; luisa::fixed_forward_list<int,64> ffl;

// <stl/ring_buffer.h> (EASTL only)
luisa::ring_buffer<int> rb; luisa::fixed_ring_buffer<int,64> frb;
```

#### LRU Cache
**Header**: `<luisa/core/stl/lru_cache.h>`
```cpp
luisa::lru_cache<string, int> cache(100); cache.emplace("key", 42);
auto val = cache.at("key");  // luisa::optional<int>
cache.touch("key");
auto tc = luisa::LRUCache<string, int>::create(100);  // thread-safe
tc->set_delete_callback([](const int &v) {});
auto v = tc->fetch("key"); tc->update("key", 42);
```

#### Optional & Variant
**Header**: `<luisa/core/stl/optional.h>`, `<luisa/core/stl/variant.h>`
```cpp
luisa::optional<int> opt = 42; if (opt) { int v = *opt; }
auto o = luisa::make_optional(3.14); auto n = luisa::nullopt;

luisa::variant<int,float,string> v = 3.14f;
if (luisa::holds_alternative<float>(v)) {}
auto f = luisa::get<float>(v);
auto p = luisa::get_if<int>(&v);
luisa::visit([](auto&& x){}, v);
```

#### Functional
**Header**: `<luisa/core/stl/functional.h>`
```cpp
luisa::function<void(int)> fn = [](int){};
luisa::move_only_function<void(int)> mfn = [p=make_unique<int>()](int){};
luisa::less<> lt; luisa::equal_to<> eq; luisa::greater<> gt;
auto visitor = luisa::make_overloaded([](int i){ return "int"; }, [](float f){ return "float"; });
auto obj = luisa::lazy_construct([]{ return make_unique<Resource>(); });
auto guard = luisa::make_finally([]{ cleanup(); });  // EASTL only
```

#### Hashing
**Header**: `<luisa/core/stl/hash.h>`
```cpp
uint64_t h = luisa::hash_value(42);
uint64_t hc = luisa::hash_combine({h1, h2, h3});
luisa::Hash128 h128 = luisa::hash128(data, size, seed);
luisa::string s = h128.to_string();
```

#### Iterators & Algorithms
**Header**: `<luisa/core/stl/iterator.h>`, `<luisa/core/stl/algorithm.h>`
```cpp
for (auto i : luisa::range(10)) {}          // 0..9
for (auto i : luisa::range(2, 10)) {}       // 2..9
for (auto i : luisa::range(0, 10, 2)) {}    // 0,2,4,6,8

luisa::sort(vec.begin(), vec.end());         // pdqsort
luisa::sort(vec.begin(), vec.end(), luisa::greater<>{});
luisa::transform(a.begin(), a.end(), b.begin(), op);
bool found = luisa::binary_search(vec.begin(), vec.end(), val);

#include <luisa/core/stl/pdqsort.h>
pdqsort(vec.begin(), vec.end());
pdqsort_branchless(vec.begin(), vec.end());
```

#### Other
```cpp
// <stl/filesystem.h>
luisa::filesystem::path p = "/some/path"; luisa::string s = luisa::to_string(p);
// <stl/sstream.h>
luisa::stringstream ss; ss << "value=" << 42; luisa::string s = ss.str();
luisa::ostringstream oss; luisa::istringstream iss("42 3.14");
```

### Summary

| Component | Header | Key |
|---|---|---|
| Basic Traits | `<luisa/core/basic_traits.h>` | `is_vector_v`, `is_matrix_v`, `vector_element_t`, `to_underlying` |
| Basic Types | `<luisa/core/basic_types.h>` | `float2/3/4`, `float2x2/3x3/4x4`, `make_float2/3/4`, `make_float2x2/3x3/4x4` |
| Binary File Stream | `<luisa/core/binary_file_stream.h>` | `BinaryFileStream` |
| Binary IO | `<luisa/core/binary_io.h>` | `BinaryBlob` |
| Clock | `<luisa/core/clock.h>` | `Clock::tic()`, `Clock::toc()` |
| Dynamic Module | `<luisa/core/dynamic_module.h>` | `DynamicModule::load()`, `address()`, `function<>()` |
| First Fit | `<luisa/core/first_fit.h>` | `FirstFit::allocate()`, `allocate_best_fit()`, `free()` |
| Logging | `<luisa/core/logging.h>` | `LUISA_INFO()`, `LUISA_WARNING()`, `log_level_info()` |
| Mathematics | `<luisa/core/mathematics.h>` | `sin()`, `dot()`, `normalize()`, `transpose()`, `inverse()`, `lerp()`, `clamp()` |
| Pool | `<luisa/core/pool.h>` | `Pool<T>::allocate()`, `create()`, `destroy()` |
| Fiber | `<luisa/core/fiber.h>` | `scheduler`, `schedule()`, `async()`, `parallel()`, `async_parallel()`, `event`, `counter` |
| STL Memory | `<luisa/core/stl/memory.h>` | `allocator`, `unique_ptr`, `shared_ptr`, `span`, `make_unique`, `make_shared` |
| STL String | `<luisa/core/stl/string.h>` | `string`, `string_hash`, `u8string`, `wstring` |
| STL Format | `<luisa/core/stl/format.h>` | `format()`, `to_string()`, `hash_to_string()` |
| STL Vector | `<luisa/core/stl/vector.h>` | `vector`, `fixed_vector`, `bitvector`, `enlarge_by`, `vector_resize` |
| STL Map | `<luisa/core/stl/unordered_map.h>` | `unordered_map`, `unordered_set` |
| STL Algorithm | `<luisa/core/stl/algorithm.h>` | `sort`, `transform`, `binary_search`, `pdqsort` |
| STL Functional | `<luisa/core/stl/functional.h>` | `function`, `move_only_function`, `overloaded`, `lazy_construct` |
| STL Hash | `<luisa/core/stl/hash.h>` | `hash_value`, `hash_combine`, `Hash128` |
| STL LRU Cache | `<luisa/core/stl/lru_cache.h>` | `lru_cache`, `LRUCache` |
