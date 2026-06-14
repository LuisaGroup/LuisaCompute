---
name: ast
description: Manual AST construction with FunctionBuilder for kernels and callables without DSL sugar.
---

# Manual AST Construction

Two approaches: **DSL** (`Kernel2D<>`, `Callable<>` lambdas) vs **Manual AST** (`FunctionBuilder`). Use manual AST for codegen, metaprogramming, programmatic kernel building.

**Header**: `#include <luisa/ast/function_builder.h>`

## FunctionBuilder

```cpp
using FuncBuilder = luisa::compute::detail::FunctionBuilder;

auto kernel   = FuncBuilder::define_kernel([&]() { auto &cur = *FuncBuilder::current(); ... });
auto callable = FuncBuilder::define_callable([&]() { ... });
auto raster   = FuncBuilder::define_raster_stage([&]() { ... });
```

All three return `luisa::shared_ptr<const FuncBuilder>`. `define_kernel` may duplicate the builder if outlined functions leak locals, so keep the returned pointer.

### Built-in Variables

```cpp
auto &cur = *FuncBuilder::current();
cur.dispatch_id();        // uint3
cur.dispatch_size();      // uint3
cur.thread_id();          // uint3
cur.block_id();           // uint3
cur.kernel_id();          // uint  (indirect kernels only)
cur.warp_lane_id();       // uint
cur.warp_lane_count();    // uint

// Rasterization stage only
cur.raster_object_id();   // uint
cur.raster_barycentrics();// uint
```

### Config

```cpp
cur.set_block_size(uint3(16, 16, 1));
cur.set_name("my_kernel");
cur.set_variable_name(var_uid, "my_var");
auto name = cur.get_variable_name(var_uid);
cur.mark_variable_usage(var_uid, Usage::READ_WRITE);
```

## Variables

```cpp
// Arguments
cur.argument(Type::of<float3>());            // input value
auto ref = cur.reference(Type::of<uint2>()); // inout parameter

// Memory
cur.local(Type::of<float>());                // local variable
auto arr = cur.shared(Type::array(Type::of<float>(), count)); // shared array

// Constants
float4 v{1,2,3,4};
auto cdata = ConstantData::create(Type::of<float4>(), &v, sizeof(v));
auto c = cur.constant(cdata);

// Resources
cur.buffer(Type::of<Buffer<float>>());       // buffer
cur.texture(Type::of<Image<float>>());       // 2D texture
cur.texture(Type::of<Image3D<float>>());     // 3D texture
cur.accel();                                 // acceleration structure
cur.bindless_array();                        // bindless array
```

### Bindings

Binding methods create a bound argument and return its `RefExpr*`. Use that expression directly in the function body.

```cpp
auto bound_buf = cur.buffer_binding(Type::of<Buffer<float>>(), handle, offset_bytes, size_bytes);
auto bound_tex = cur.texture_binding(Type::of<Image<float>>(), handle, level);
auto bound_ba  = cur.bindless_array_binding(handle);
auto bound_acc = cur.accel_binding(handle);

// Example: read from a bound buffer
auto idx = cur.literal(Type::of<uint>(), 0u);
auto value = cur.call(Type::of<float>(), CallOp::BUFFER_READ, {bound_buf, idx});
```

## Expressions

### Literals

```cpp
cur.literal(Type::of<float>(), 1.0f);
cur.literal(Type::of<int>(), 42);
cur.literal(Type::of<uint>(), 0u);
cur.literal(Type::of<bool>(), true);
cur.literal(Type::of<float2>(), float2(0.5f, 0.5f));
```

### Binary Operations

Comparison operators are named, not symbolic.

```cpp
cur.binary(Type::of<float>(), BinaryOp::ADD, a, b);     // +
cur.binary(Type::of<float>(), BinaryOp::SUB, a, b);     // -
cur.binary(Type::of<float>(), BinaryOp::MUL, a, b);     // *
cur.binary(Type::of<float>(), BinaryOp::DIV, a, b);     // /
cur.binary(Type::of<int>(),   BinaryOp::MOD, a, b);     // %
cur.binary(Type::of<uint>(),  BinaryOp::BIT_AND, a, b); // &
cur.binary(Type::of<uint>(),  BinaryOp::BIT_OR, a, b);  // |
cur.binary(Type::of<uint>(),  BinaryOp::BIT_XOR, a, b); // ^
cur.binary(Type::of<uint>(),  BinaryOp::SHL, a, b);     // <<
cur.binary(Type::of<uint>(),  BinaryOp::SHR, a, b);     // >>
cur.binary(Type::of<bool>(),  BinaryOp::AND, a, b);     // &&
cur.binary(Type::of<bool>(),  BinaryOp::OR, a, b);      // ||
cur.binary(Type::of<bool>(),  BinaryOp::EQUAL, a, b);          // ==
cur.binary(Type::of<bool>(),  BinaryOp::NOT_EQUAL, a, b);      // !=
cur.binary(Type::of<bool>(),  BinaryOp::LESS, a, b);           // <
cur.binary(Type::of<bool>(),  BinaryOp::LESS_EQUAL, a, b);     // <=
cur.binary(Type::of<bool>(),  BinaryOp::GREATER, a, b);        // >
cur.binary(Type::of<bool>(),  BinaryOp::GREATER_EQUAL, a, b);  // >=
```

### Unary Operations

```cpp
cur.unary(Type::of<float>(), UnaryOp::PLUS, value);     // +
cur.unary(Type::of<float>(), UnaryOp::MINUS, value);    // -
cur.unary(Type::of<bool>(),  UnaryOp::NOT, value);      // !
cur.unary(Type::of<uint>(),  UnaryOp::BIT_NOT, value);  // ~
```

### Swizzle

Component indices are packed in 4-bit nibbles, lowest bits first:

```cpp
// .xy from uint3: (0) | (1 << 4)
uint64_t swizzle_xy = (0ull) | (1ull << 4ull);
cur.swizzle(Type::of<uint2>(), coord_uint3, 2, swizzle_xy);

// .xyzw (all): 0x3210u
cur.swizzle(Type::of<float4>(), vec, 4, 0x3210u);

// .x: 0ull   .y: 1ull<<4   .z: 2ull<<8   .w: 3ull<<12
```

### Function Calls

```cpp
// Built-in
cur.call(Type::of<float4>(), CallOp::MAKE_FLOAT4, {r, g, b, a});
cur.call(CallOp::TEXTURE_WRITE, {texture, coord, color});
cur.call(Type::of<float4>(), CallOp::TEXTURE_READ, {texture, coord});

// Buffer
cur.call(Type::of<float>(), CallOp::BUFFER_READ, {buffer, index});
cur.call(CallOp::BUFFER_WRITE, {buffer, index, value});

// Atomic (use AtomicRefNode)
auto ref = luisa::compute::detail::AtomicRefNode::create(buffer)
                ->access(index);
auto old = ref->operate(CallOp::ATOMIC_EXCHANGE, {new_value});

// Custom callable
cur.call(Function(callable.get()), {arg1, arg2});
```

### Other Expressions

```cpp
cur.cast(Type::of<float>(), CastOp::STATIC, int_value);          // type cast
cur.cast(Type::of<float>(), CastOp::BITWISE, int_value);         // bitwise reinterpretation
cur.access(Type::of<float>(), buffer_expr, index_expr);          // array/buffer access
cur.member(Type::of<float>(), struct_expr, member_index);        // struct member
cur.make_vector(Type::of<float4>(), luisa::vector{x, y, z, w});  // vector construction
cur.string_id("my_string");                                      // string ID -> uint64
cur.type_id(Type::of<float3>());                                 // type ID -> uint64
```

## Statements

```cpp
cur.assign(lhs_expr, rhs_expr);

// Control flow
cur.break_();
cur.continue_();
cur.return_(value_expr);   // with value
cur.return_();             // void

// Use cur.with(scope, body) to populate nested scopes
auto if_stmt = cur.if_(cond);
cur.with(if_stmt->true_branch(), [&] { /* then */ });
cur.with(if_stmt->false_branch(), [&] { /* else */ });

auto loop_stmt = cur.loop_();
cur.with(loop_stmt->body(), [&] { /* loop body */ });

auto for_stmt = cur.for_(var, cond, step);
cur.with(for_stmt->body(), [&] { /* for body */ });

auto switch_stmt = cur.switch_(expr);
cur.with(switch_stmt->body(), [&] {
    auto case0 = cur.case_(cur.literal(Type::of<int>(), 0));
    cur.with(case0->body(), [&] { ...; cur.break_(); });
    auto case1 = cur.case_(cur.literal(Type::of<int>(), 1));
    cur.with(case1->body(), [&] { ...; cur.break_(); });
    auto def = cur.default_();
    cur.with(def->body(), [&] { ...; cur.break_(); });
});

auto ray_query_stmt = cur.ray_query_(query_expr);
cur.with(ray_query_stmt->on_triangle_candidate(), [&] { ... });
cur.with(ray_query_stmt->on_procedural_candidate(), [&] { ... });

auto ad_stmt = cur.autodiff_();
cur.with(ad_stmt->body(), [&] { ... });

// Print
cur.print_("value = {}", luisa::vector<const Expression *>{value_expr});

// Comment
cur.comment_("marker");
```

## Type System

### Getting Types

```cpp
// Scalars
Type::of<float>(); Type::of<int>(); Type::of<uint>(); Type::of<bool>();
Type::of<half>(); Type::of<double>();
Type::of<short>(); Type::of<ushort>();
Type::of<int8_t>(); Type::of<uint8_t>();
Type::of<slong>(); Type::of<ulong>();

// Vectors
Type::of<float2>(); Type::of<float3>(); Type::of<float4>();
Type::of<int2>(); Type::of<int3>(); Type::of<int4>();
Type::of<uint2>(); Type::of<uint3>(); Type::of<uint4>();
Type::of<half2>(); Type::of<double4>(); // etc.

// Matrices
Type::of<float2x2>(); Type::of<float3x3>(); Type::of<float4x4>();

// Resources
Type::of<Buffer<float>>();
Type::of<Image<float>>(); Type::of<Image3D<float>>();
Type::of<Accel>(); Type::of<BindlessArray>();
```

### Constructing Types

```cpp
Type::vector(Type::of<float>(), 2);                         // float2
Type::matrix(4);                                            // float4x4
Type::array(Type::of<float>(), 100);                        // float[100]
Type::structure({Type::of<float>(), Type::of<int>()});      // struct
Type::buffer(Type::of<float>());                            // buffer<float>
Type::texture(Type::of<float>(), 2);                        // 2D texture
Type::texture(Type::of<float>(), 3);                        // 3D texture
Type::custom("MyOpaqueType");
Type::from("vector<float,4>");                              // from string
```

## Operators

### BinaryOp

```
ADD, SUB, MUL, DIV, MOD        // arithmetic
BIT_AND, BIT_OR, BIT_XOR       // bitwise
SHL, SHR                       // shift
AND, OR                        // logical
LESS, GREATER, LESS_EQUAL, GREATER_EQUAL, EQUAL, NOT_EQUAL  // comparison
```

### UnaryOp

```
PLUS, MINUS, NOT, BIT_NOT
```

### CallOp

The full set is defined in `include/luisa/ast/op.h`. Common groups:

```
// Vector construction
MAKE_FLOAT2/3/4, MAKE_INT2/3/4, MAKE_UINT2/3/4, MAKE_BOOL2/3/4
MAKE_SHORT2/3/4, MAKE_USHORT2/3/4, MAKE_LONG2/3/4, MAKE_ULONG2/3/4
MAKE_HALF2/3/4, MAKE_DOUBLE2/3/4, MAKE_BYTE2/3/4, MAKE_UBYTE2/3/4
MAKE_FLOAT2X2/3X3/4X4

// Buffer/Texture
BUFFER_READ, BUFFER_WRITE, BUFFER_SIZE, BUFFER_ADDRESS
BUFFER_VOLATILE_READ, BUFFER_VOLATILE_WRITE
BYTE_BUFFER_READ, BYTE_BUFFER_WRITE, BYTE_BUFFER_SIZE
TEXTURE_READ, TEXTURE_WRITE, TEXTURE_SIZE
TEXTURE2D_SAMPLE, TEXTURE2D_SAMPLE_LEVEL, TEXTURE2D_SAMPLE_GRAD, ...

// Atomic
ATOMIC_EXCHANGE, ATOMIC_COMPARE_EXCHANGE, ATOMIC_FETCH_ADD, ATOMIC_FETCH_SUB
ATOMIC_FETCH_AND, ATOMIC_FETCH_OR, ATOMIC_FETCH_XOR, ATOMIC_FETCH_MIN, ATOMIC_FETCH_MAX

// Bindless
BINDLESS_TEXTURE2D_SAMPLE, BINDLESS_TEXTURE2D_READ, BINDLESS_TEXTURE2D_SIZE
BINDLESS_BUFFER_READ, BINDLESS_BUFFER_WRITE, BINDLESS_BUFFER_SIZE, ...
UNIFORM_BINDLESS_*, TYPED_BINDLESS_*, TYPED_UNIFORM_BINDLESS_*

// Ray tracing
RAY_TRACING_TRACE_CLOSEST, RAY_TRACING_TRACE_ANY
RAY_TRACING_QUERY_ALL, RAY_TRACING_QUERY_ANY
RAY_TRACING_SET_INSTANCE_TRANSFORM, RAY_TRACING_SET_INSTANCE_VISIBILITY, ...
RAY_QUERY_WORLD_SPACE_RAY, RAY_QUERY_TRIANGLE_CANDIDATE_HIT,
RAY_QUERY_COMMIT_TRIANGLE, RAY_QUERY_TERMINATE, RAY_QUERY_PROCEED, ...

// Math
ALL, ANY, SELECT, CLAMP, SATURATE, LERP, SMOOTHSTEP, STEP
ABS, MIN, MAX, CLZ, CTZ, POPCOUNT, REVERSE
ISINF, ISNAN
SIN, COS, TAN, ASIN, ACOS, ATAN, ATAN2, SINH, COSH, TANH, ASINH, ACOSH, ATANH
EXP, EXP2, EXP10, LOG, LOG2, LOG10, POW, SQRT, RSQRT
CEIL, FLOOR, FRACT, TRUNC, ROUND, FMA, COPYSIGN

// Vector/Matrix
DOT, CROSS, LENGTH, LENGTH_SQUARED, NORMALIZE, FACEFORWARD, REFLECT, REFRACT
OUTER_PRODUCT, MATRIX_COMPONENT_WISE_MULTIPLICATION
DETERMINANT, TRANSPOSE, INVERSE

// Warp/Wave
WARP_IS_FIRST_ACTIVE_LANE, WARP_FIRST_ACTIVE_LANE, WARP_ACTIVE_ALL_EQUAL
WARP_ACTIVE_BIT_AND/OR/XOR, WARP_ACTIVE_COUNT_BITS, WARP_ACTIVE_MAX/MIN
WARP_ACTIVE_PRODUCT/SUM, WARP_ACTIVE_ALL/ANY, WARP_ACTIVE_BIT_MASK
WARP_PREFIX_SUM, WARP_PREFIX_PRODUCT, WARP_PREFIX_COUNT_BITS
WARP_READ_LANE, WARP_READ_FIRST_ACTIVE_LANE

// Sync
SYNCHRONIZE_BLOCK

// Rasterization
RASTER_DISCARD, RASTER_SET_Z_DEPTH,
RASTER_SET_Z_DEPTH_GREATER_EQUAL, RASTER_SET_Z_DEPTH_LESS_EQUAL

// Derivatives
DDX, DDY

// Indirect dispatch
INDIRECT_SET_DISPATCH_KERNEL, INDIRECT_SET_DISPATCH_COUNT

// Debugging/optimization
ASSERT, ASSUME, UNREACHABLE, FLATTEN, BRANCH, FORCE_CASE

// Clock
CLOCK
```

## Usage Flags

```cpp
enum struct Usage : uint32_t {
    NONE = 0u, READ = 0x01u, WRITE = 0x02u, READ_WRITE = READ | WRITE
};
```

References must be marked explicitly:

```cpp
cur.mark_variable_usage(ref->variable().uid(), Usage::READ_WRITE);
```

`mark_variable_usage` ORs flags, so it is safe to call multiple times.

## Examples

### Simple Kernel

```cpp
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    cur.set_block_size(uint3(16, 16, 1));
    auto dispatch = cur.dispatch_id();
    auto img = cur.texture(Type::of<Image<float>>());
    auto color = cur.argument(Type::of<float4>());
    auto coord = cur.swizzle(Type::of<uint2>(), dispatch, 2, (0ull) | (1ull << 4ull));
    cur.call(CallOp::TEXTURE_WRITE, {img, coord, color});
});
```

### Callable with Reference

```cpp
auto callable = FuncBuilder::define_callable([&]() {
    auto &cur = *FuncBuilder::current();
    auto tex = cur.texture(Type::of<Image<float>>());
    auto coord_ref = cur.reference(Type::of<uint2>());
    cur.mark_variable_usage(coord_ref->variable().uid(), Usage::READ_WRITE);
    auto color = cur.argument(Type::of<float3>());
    auto alpha = cur.literal(Type::of<float>(), 1.0f);
    auto value = cur.make_vector(Type::of<float4>(),
                                  luisa::vector<const Expression *>{color, alpha});
    cur.call(CallOp::TEXTURE_WRITE, {tex, coord_ref, value});
});
```

### Kernel Calling Callable

```cpp
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    cur.set_block_size(uint3(16, 16, 1));
    auto img = cur.texture(Type::of<Image<float>>());
    auto color = cur.argument(Type::of<float3>());
    auto coord_uint3 = cur.dispatch_id();
    auto coord = cur.local(Type::of<uint2>());
    cur.assign(coord, cur.swizzle(Type::of<uint2>(), coord_uint3, 2, (0ull) | (1ull << 4ull)));
    cur.call(Function(callable.get()), {img, coord, color});
});
```

### Swizzle Operations

```cpp
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    auto input = cur.argument(Type::of<float4>());
    auto output = cur.reference(Type::of<float4>());
    cur.mark_variable_usage(output->variable().uid(), Usage::READ_WRITE);
    uint64_t swizzle_xyz = (0ull) | (1ull << 4ull) | (2ull << 8ull);
    auto xyz = cur.swizzle(Type::of<float3>(), input, 3, swizzle_xyz);
    auto w = cur.swizzle(Type::of<float>(), input, 1, 3ull); // .w
    cur.assign(output, cur.make_vector(Type::of<float4>(), luisa::vector{x, w}));
});
```

### Buffer Operations

```cpp
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    cur.set_block_size(uint3(256, 1, 1));
    auto input_buf = cur.buffer(Type::of<Buffer<float>>());
    auto output_buf = cur.buffer(Type::of<Buffer<float>>());
    auto idx = cur.swizzle(Type::of<uint>(), cur.thread_id(), 1, 0ull);
    auto value = cur.call(Type::of<float>(), CallOp::BUFFER_READ, {input_buf, idx});
    auto scaled = cur.binary(Type::of<float>(), BinaryOp::MUL, value, cur.literal(Type::of<float>(), 2.0f));
    auto result = cur.binary(Type::of<float>(), BinaryOp::ADD, scaled, cur.literal(Type::of<float>(), 1.0f));
    cur.call(CallOp::BUFFER_WRITE, {output_buf, idx, result});
});
```

### For Loop

```cpp
auto kernel = FuncBuilder::define_kernel([&]() {
    auto &cur = *FuncBuilder::current();
    auto buf = cur.buffer(Type::of<Buffer<float>>());
    auto i = cur.local(Type::of<uint>());
    cur.assign(i, cur.literal(Type::of<uint>(), 0u));
    auto ten = cur.literal(Type::of<uint>(), 10u);
    auto cond = cur.binary(Type::of<bool>(), BinaryOp::LESS, i, ten);
    auto step = cur.literal(Type::of<uint>(), 1u);
    auto for_stmt = cur.for_(i, cond, step);
    cur.with(for_stmt->body(), [&] {
        auto idx = i;
        auto v = cur.call(Type::of<float>(), CallOp::BUFFER_READ, {buf, idx});
        cur.call(CallOp::BUFFER_WRITE, {buf, idx,
            cur.binary(Type::of<float>(), BinaryOp::ADD, v,
                cur.literal(Type::of<float>(), 1.0f))});
    });
});
```

## Key Rules

1. Always use `Type::of<T>()` for explicit types.
2. Mark reference usage with `mark_variable_usage(uid, Usage::READ_WRITE)`.
3. Swizzle: component indices in nibbles, lowest bits first.
4. Use `FunctionBuilder::current()` within define callbacks.
5. Statements and expressions are owned by `FunctionBuilder`; do not delete them.
6. Set block size for compute kernels (typically `uint3(16, 16, 1)` for 2D).
7. Use `cur.with(scope, body)` to append statements into `if`/`loop`/`for`/`switch`/`ray_query`/`autodiff` bodies.
8. Atomic operations require `AtomicRefNode`, not raw buffer variables.
9. `print_` takes a format string and a `luisa::span`/`vector` of expressions, not an initializer list.
10. `BinaryOp` comparison names are `EQUAL`, `NOT_EQUAL`, `LESS`, `GREATER`, `LESS_EQUAL`, `GREATER_EQUAL`.
