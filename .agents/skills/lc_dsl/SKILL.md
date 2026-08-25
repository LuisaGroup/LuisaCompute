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

## Compile-Time vs Runtime Control Flow

DSL kernels are constructed by executing the host C++ lambda during `Kernel1D/2D/3D` creation (and again inside `device.compile()`). This means native C++ control flow on **host** values is evaluated at kernel construction time, while DSL control-flow constructs become real device instructions.

- **Native C++ `if` / `for` / `while` / `switch` on plain host variables** are resolved on the host. Only the taken path is recorded in the AST; no corresponding branch or loop appears in the generated GPU code.
- **DSL `$if` / `$else`, `$for` / `$while` / `$loop`, `$switch` / `$case` / `$default`** (and the non-sugar `if_`, `switch_`, `for (auto i : dynamic_range(...))`, `loop`) emit real device control flow. Their conditions must be DSL expressions such as `Var<bool>` or `Var<uint>`.

### Examples

Native `if` on a host variable — the unselected branch is erased during AST construction:

```cpp
bool host_visible = true;
Kernel1D k = [&]() noexcept {
    Var<uint> x = 0u;
    if (host_visible) {
        x = 1u;
    } else {
        x = 2u;   // never recorded; the kernel always writes 1
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

### When to use which

| Host C++ construct | Evaluated | Emitted in GPU code? | Safe for large counts? |
|---|---|---|---|
| `if (host_bool)` | Kernel construction | No (only taken path) | N/A |
| `$if (Var<bool>)` / `if_(Expr<bool>)` | GPU runtime | Yes | Yes |
| `for (host i < N)` | Kernel construction | No (flattened N times) | **No** |
| `$for (i, N)` / `dynamic_range(N)` | GPU runtime | Yes | Yes |
| `switch (host_val)` | Kernel construction | No (only matching case) | N/A |
| `$switch (Var<T>)` / `switch_(Expr<T>)` | GPU runtime | Yes | Yes |

Reserve native C++ loops for small, compile-time-known unrolling (for example, a fixed 4×4 matrix operation). Use DSL loops whenever the bound comes from a runtime value or is large.

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

Coroutine frames reserve four scalar `uint` fields: frame indices 0, 1, and 2 store `coro_id.x/y/z`, and frame index 3 stores `target_token`. User frame fields start at `CoroFrameDesc::reserved_field_count` (currently 4). Do not reintroduce a skip flag; it was only needed by the old structured-CFG replay path, and XIR coroutine splitting now uses unstructured CFG continuations directly.

Rendering coroutine examples should keep the real fine-grained coroutine topology. Wavefront rebuilds or sorts work queues per suspend phase, so inner-loop suspends can dominate runtime even when the generated code is functionally correct; do not hide that by silently removing or coarsening suspends in the main example/test. If a coarser coroutine is useful for profiling, add it as a separate focused debug case. Log `coro.frame().total_size()`, `coro.frame().frame_type()->size()`, frame field count, subroutine count, and graph node count after compiling complex coroutines.

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

## Cooperative Vector Operations

Cooperative vectors are thread-local vectors of uniform size that participate in hardware-accelerated cooperative (cross-lane/warp) operations. They are backed by `CoopVector<T>`, `CoopVectorRef`, and `CoopMatrixRef` types defined in `<luisa/dsl/coop_vector.h>`. All free functions are in `<luisa/dsl/resource.h>`.

### Headers

```cpp
#include <luisa/dsl/coop_vector.h>   // CoopVector<T>, CoopVectorRef, CoopMatrixRef
#include <luisa/dsl/resource.h>       // all cooperative_vector_*, cooperative_mat_*, bindless_cooperative_* functions
#include <luisa/dsl/expr.h>           // Expr<CoopVector<T>> specialization
#include <luisa/dsl/sugar.h>          // $ sugar macros (optional)
```

### Backend Support

> **⚠️ Currently cooperative vector operations only support the Vulkan (`vk`) backend.**
> The DX backend requires Shader Model 6.8 with experimental features, which is not widely available.
> Check `src/tests/unit/ast/test_cooperative_vector.cpp` for the `create_test_device()` helper.

### Type System

```cpp
// Create a cooperative vector type (element type + size)
auto cv_type = Type::cooperative_vector(Type::of<float>(), 16);  // coopvec<float,16>

// Create a cooperative vector reference type (used to describe buffer offsets)
auto cvr_type = Type::cooperative_vector_ref(CoopRefVecType::FLOAT32, 16);  // coopvec_ref<16,5>

// Create a cooperative matrix reference type
auto cmr_type = Type::cooperative_matrix_ref(CoopRefVecType::FLOAT32, 4, 8);  // coopmat_ref<4,8,5>
```

Available `CoopRefVecType` values: `FLOAT16`, `FLOAT32`, `INT8`, `UINT8`, `INT32`, `UINT32`.

### DSL Object Construction

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

### Element Access

Individual elements are accessed with `operator[]` (read/write):

```cpp
CoopVector<float> v{8};
for (auto i = 0u; i < 8u; ++i) {
    v[i] = static_cast<float>(i + 1);  // write
}
Var<float> elem = v[3];  // read
```

### Load / Store (ByteBuffer)

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

### Accumulate

Atomically accumulate a cooperative vector into a `ByteBuffer` at a given offset:

```cpp
CoopVector<float> input{8};
for (auto i = 0u; i < 8u; ++i) input[i] = static_cast<float>(i + 1);
offset.set_byte_offset(0u);
cooperative_vector_accumulate(buf, offset, Expr<CoopVector<float>>{input});
```

### Splat

Create a cooperative vector with all elements set to the same scalar value:

```cpp
auto result = cooperative_vector_splat<float>(42.0f, 8u);
```

### Cast

Cast the element type of a cooperative vector:

```cpp
CoopVector<float> input{8};
// ... fill input ...
auto result = cooperative_vector_cast<int>(Expr<CoopVector<float>>{input});
```

### Bindless Load / Store

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

### Workgroup Load / Store

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

### Matrix Multiply Operations

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

### Outer Product Accumulate

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

### Element-wise Math Operations

These compute element-wise operations by iterating over each lane:

```cpp
CoopVector<float> a{4}, b{4}, c{4}, lo{4}, hi{4}, v{4};
// ... fill ...

auto r_min  = cooperative_vector_min(a, b);    // element-wise min
auto r_max  = cooperative_vector_max(a, b);    // element-wise max
auto r_clamp = cooperative_vector_clamp(v, lo, hi);  // element-wise clamp
auto r_exp  = cooperative_vector_exp(v);       // element-wise exp
auto r_log  = cooperative_vector_log(v);       // element-wise log
auto r_tanh = cooperative_vector_tanh(v);      // element-wise tanh
auto r_atan = cooperative_vector_atan(v);      // element-wise atan
auto r_fma  = cooperative_vector_fma(a, b, c); // element-wise fma(a,b,c) = a*b+c
```

### Element-wise Bitwise Operations (Integer Element Types)

```cpp
CoopVector<uint> a{4}, b{4}, v{4};
// ... fill ...

auto r_and = cooperative_vector_bitwise_and(a, b);   // element-wise &
auto r_or  = cooperative_vector_bitwise_or(a, b);    // element-wise |
auto r_xor = cooperative_vector_bitwise_xor(a, b);   // element-wise ^
auto r_not = cooperative_vector_bitwise_not(v);      // element-wise ~
auto r_shl = cooperative_vector_shift_left(v, 1u);   // element-wise <<
auto r_shr = cooperative_vector_shift_right(v, 4u);  // element-wise >>
```

### Device Compilation Considerations

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

### Complete Load/Store Round-Trip Example

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

### DSL Source File References

| File | Contents |
|------|----------|
| `include/luisa/dsl/coop_vector.h` | `CoopVector<T>`, `CoopVectorRef`, `CoopMatrixRef` DSL type definitions |
| `include/luisa/dsl/resource.h` (lines 880–1322) | All free functions: `cooperative_vector_*`, `cooperative_mat_*`, `bindless_cooperative_*`, `cooperative_outer_product_*` |
| `include/luisa/dsl/expr.h` (lines 151–155) | `Expr<CoopVector<T>>` template specialization with subscript access |
| `src/tests/unit/ast/test_cooperative_vector.cpp` | AST construction, DSL sugar, and device execution tests for all cooperative operations |

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
