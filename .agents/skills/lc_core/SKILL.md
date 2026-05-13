---
name: lc_core
---

# LuisaCompute Core Library (lc_core) Usage Guide

This skill documents the usage patterns for the LuisaCompute core library based on test cases in `src/tests/for_agent/*.cpp`.

## Table of Contents

1. [Basic Traits](#basic-traits)
2. [Basic Types](#basic-types)
3. [Binary File Stream](#binary-file-stream)
4. [Binary IO (BinaryBlob)](#binary-io-binaryblob)
5. [Clock](#clock)
6. [Dynamic Module](#dynamic-module)
7. [First Fit Allocator](#first-fit-allocator)
8. [Logging](#logging)
9. [Mathematics](#mathematics)
10. [Pool Allocator](#pool-allocator)
11. [Fiber](#fiber)

---

## Basic Traits

**Header**: `<luisa/core/basic_traits.h>`

Type traits for compile-time type checking and vector/matrix type analysis.

### Type Predicates

```cpp
// Boolean type predicates
luisa::always_false_v<T...>           // Always false (for static_assert)
luisa::always_true_v<T...>            // Always true

// Scalar type checks
luisa::is_integral_v<T>               // Check if integral type
luisa::is_boolean_v<T>                // Check if bool
luisa::is_floating_point_v<T>         // Check if float/double/half
luisa::is_signed_v<T>                 // Check if signed
luisa::is_unsigned_v<T>               // Check if unsigned
luisa::is_signed_integral_v<T>        // Check if signed integral
luisa::is_unsigned_integral_v<T>      // Check if unsigned integral
luisa::is_scalar_v<T>                 // Check if scalar (integral, float, or bool)

// Vector type checks
luisa::is_vector_v<T>                 // Check if any vector
luisa::is_vector_v<T, N>              // Check if vector with dimension N
luisa::is_vector2_v<T>                // Check if 2-component vector
luisa::is_vector3_v<T>                // Check if 3-component vector
luisa::is_vector4_v<T>                // Check if 4-component vector
luisa::is_boolean_vector_v<T>         // Check if vector<bool, N>
luisa::is_floating_point_vector_v<T>  // Check if vector of floats
luisa::is_integral_vector_v<T>        // Check if vector of integrals
luisa::is_signed_integral_vector_v<T> // Check if vector of signed integrals
luisa::is_unsigned_integral_vector_v<T>

// Matrix type checks
luisa::is_matrix_v<T>                 // Check if any matrix
luisa::is_matrix_v<T, N>              // Check if NxN matrix
luisa::is_matrix2_v<T>                // Check if 2x2 matrix
luisa::is_matrix3_v<T>                // Check if 3x3 matrix
luisa::is_matrix4_v<T>                // Check if 4x4 matrix

// Combined checks
luisa::is_basic_v<T>                  // Check if scalar, vector, or matrix
luisa::is_boolean_or_vector_v<T>
luisa::is_floating_point_or_vector_v<T>
luisa::is_integral_or_vector_v<T>
luisa::is_signed_integral_or_vector_v<T>
luisa::is_unsigned_integral_or_vector_v<T>
luisa::is_vector_same_dimension_v<T1, T2, ...>
```

### Type Transformations

```cpp
// Get element type of vector
using ElementType = luisa::vector_element_t<VectorType>;

// Get element type of matrix
using ElementType = luisa::matrix_element_t<MatrixType>;

// Convert enum to underlying type
auto value = luisa::to_underlying(Enum::Value);  // Returns underlying integer
```

### Dimension Queries

```cpp
constexpr size_t dim = luisa::vector_dimension_v<VectorType>;  // 1 for scalars
constexpr size_t dim = luisa::matrix_dimension_v<MatrixType>;  // 1 for scalars
```

---

## Basic Types

**Header**: `<luisa/core/basic_types.h>`

Vector and matrix types with GLSL/HLSL-like syntax.

### Type Aliases

```cpp
// Scalar types
using luisa::byte;       // int8_t
using luisa::ubyte;      // uint8_t
using luisa::ushort;     // uint16_t
using luisa::uint;       // uint32_t
using luisa::ulong;      // uint64_t
using luisa::slong;      // int64_t
using luisa::half;       // 16-bit float

// Vector types (2, 3, 4 components)
using luisa::bool2, bool3, bool4;
using luisa::short2, short3, short4;
using luisa::ushort2, ushort3, ushort4;
using luisa::byte2, byte3, byte4;
using luisa::ubyte2, ubyte3, ubyte4;
using luisa::int2, int3, int4;
using luisa::uint2, uint3, uint4;
using luisa::slong2, slong3, slong4;
using luisa::ulong2, ulong3, ulong4;
using luisa::half2, half3, half4;
using luisa::float2, float3, float4;
using luisa::double2, double3, double4;

// Matrix types (NxN)
using luisa::float2x2, float3x3, float4x4;
using luisa::double2x2, double3x3, double4x4;
using luisa::half2x2, half3x3, half4x4;
```

### Vector Construction

```cpp
// Scalar broadcast
float2 f2(1.0f);                    // (1.0, 1.0)
float3 f3(2.0f);                    // (2.0, 2.0, 2.0)
float4 f4(3.0f);                    // (3.0, 3.0, 3.0, 3.0)

// Component-wise
int2 i2(1, 2);                      // (1, 2)
int3 i3(1, 2, 3);                   // (1, 2, 3)
int4 i4(1, 2, 3, 4);                // (1, 2, 3, 4)

// Factory methods
auto z2 = float2::zero();           // (0.0, 0.0)
auto o2 = float2::one();            // (1.0, 1.0)
```

### Vector Element Access

```cpp
float3 v(1.0f, 2.0f, 3.0f);

// Named accessors
float x = v.x;
float y = v.y;
float z = v.z;

// Array accessor
float first = v[0];
v[1] = 5.0f;                        // Modify through index
```

### Vector Operators

```cpp
float2 a(1.0f, 2.0f);
float2 b(3.0f, 4.0f);

// Arithmetic
auto sum = a + b;                   // (4, 6)
auto diff = b - a;                  // (2, 2)
auto prod = a * b;                  // (3, 8) - component-wise
auto quot = b / a;                  // (3, 2) - component-wise

// Scalar operations
auto scaled = a * 2.0f;             // (2, 4)
auto scaled2 = 3.0f * a;            // (3, 6)

// Unary
auto neg = -a;                      // (-1, -2)
auto pos = +a;                      // (1, 2)

// Bitwise (integral vectors)
int2 i(0x0F, 0xF0);
int2 xi = ~i;                       // Bitwise NOT
int2 shifted = i << 1;              // Left shift

// Comparison (returns bool vector)
bool2 eq = (a == b);                // Component-wise equality
bool2 lt = (a < b);                 // Component-wise less than

// Boolean logic (bool vectors)
bool2 b1(true, false);
bool2 b2(true, true);
bool2 bor = b1 || b2;               // (true, true)
bool2 band = b1 && b2;              // (true, false)
```

### Vector Assignment Operators

```cpp
float2 a(1.0f, 2.0f);
a += float2(3.0f, 4.0f);            // (4, 6)
a -= float2(1.0f, 2.0f);            // (3, 4)
a *= float2(2.0f, 2.0f);            // (6, 8)
a /= float2(3.0f, 4.0f);            // (2, 2)
```

### Boolean Vector Functions

```cpp
bool2 b(true, false);

bool any_result = any(b);           // true if any component is true
bool all_result = all(b);           // true if all components are true
bool none_result = none(b);         // true if no component is true
```

### Matrix Construction

```cpp
// Default (identity matrix)
float2x2 m2;                        // [[1,0],[0,1]]
float3x3 m3;                        // Identity 3x3
float4x4 m4;                        // Identity 4x4

// From column vectors
float2x2 m2c(float2(1.0f, 2.0f), float2(3.0f, 4.0f));
// Results in: col0=(1,2), col1=(3,4)
// Matrix: [[1,3],[2,4]] (row-major view)

// Factory methods
auto eye2 = float2x2::eye(2.0f);    // 2 * identity
auto fill2 = float2x2::fill(3.0f);  // All elements = 3.0
```

### Matrix Element Access

```cpp
float3x3 m(...);

// Column access (returns vector)
float3 col0 = m[0];
float3 col1 = m[1];
float3 col2 = m[2];

// Element access
float elem = m[col][row];

// Modify column
m[1] = float3(10.0f, 11.0f, 12.0f);
```

### Matrix Operations

```cpp
float2x2 m(...);

// Scalar multiplication
auto scaled = m * 2.0f;
auto scaled2 = 3.0f * m;

// Scalar division
auto divided = m / 2.0f;

// Matrix-vector multiplication
float2 v(1.0f, 1.0f);
float2 r = m * v;                   // Linear transformation

// Matrix-matrix multiplication
float2x2 a(...), b(...);
float2x2 c = a * b;

// Addition/Subtraction
float2x2 sum = a + b;
float2x2 diff = a - b;
```

### Make Functions

```cpp
// Vector make functions
float2 f2a = make_float2(1.0f);                         // Broadcast
float2 f2b = make_float2(1.0f, 2.0f);                   // Components
float2 f2c = make_float2(float3(1.0f, 2.0f, 3.0f));     // From larger vector
float2 f2d = make_float2(float4(1.0f, 2.0f, 3.0f, 4.0f));

float3 f3a = make_float3(1.0f);                         // Broadcast
float3 f3b = make_float3(1.0f, 2.0f, 3.0f);             // Components
float3 f3c = make_float3(float2(1.0f, 2.0f), 3.0f);     // From float2 + scalar
float3 f3d = make_float3(1.0f, float2(2.0f, 3.0f));     // From scalar + float2
float3 f3e = make_float3(float4(1.0f, 2.0f, 3.0f, 4.0f));

float4 f4a = make_float4(1.0f);                         // Broadcast
float4 f4b = make_float4(1.0f, 2.0f, 3.0f, 4.0f);       // Components
float4 f4c = make_float4(float2(1.0f, 2.0f), 3.0f, 4.0f);
float4 f4d = make_float4(float3(1.0f, 2.0f, 3.0f), 4.0f);
float4 f4e = make_float4(1.0f, float3(2.0f, 3.0f, 4.0f));
float4 f4f = make_float4(float2(1.0f, 2.0f), float2(3.0f, 4.0f));

// Matrix make functions
float2x2 m2a = make_float2x2(2.0f);                     // Diagonal fill
float2x2 m2b = make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);   // Row-major elements
float2x2 m2c = make_float2x2(float2(1.0f, 2.0f), float2(3.0f, 4.0f)); // Columns

float3x3 m3a = make_float3x3(2.0f);                     // Diagonal fill
float3x3 m3b = make_float3x3(
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f,
    7.0f, 8.0f, 9.0f);                                  // Row-major elements

float4x4 m4a = make_float4x4(2.0f);                     // Diagonal fill
float4x4 m4b = make_float4x4(
    1.0f, 2.0f, 3.0f, 4.0f,
    5.0f, 6.0f, 7.0f, 8.0f,
    9.0f, 10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f, 16.0f);                        // Row-major elements
```

---

## Binary File Stream

**Header**: `<luisa/core/binary_file_stream.h>`

Read-only binary file streaming with seeking support.

```cpp
#include <luisa/core/binary_file_stream.h>

// Construction from file path
luisa::BinaryFileStream stream(luisa::string{"file.bin"});

// Check validity
if (stream.valid()) { /* ... */ }
if (stream) { /* operator bool */ }

// File info
size_t len = stream.length();               // File size in bytes
size_t pos = stream.pos();                  // Current position

// Seeking
stream.set_pos(128);                        // Seek to offset
stream.set_pos(0);                          // Seek to beginning

// Reading
luisa::vector<std::byte> buffer(256);
stream.read(luisa::span<std::byte>(buffer.data(), buffer.size()));

// Explicit close
stream.close();

// Move semantics
luisa::BinaryFileStream stream2(std::move(stream));
```

---

## Binary IO (BinaryBlob)

**Header**: `<luisa/core/binary_io.h>`

RAII wrapper for binary data with custom disposal.

```cpp
#include <luisa/core/binary_io.h>

// Construction with custom disposer
size_t size = 1024;
auto* ptr = static_cast<std::byte*>(::operator new(size));
luisa::BinaryBlob blob{
    ptr,
    size,
    [](void* p) { ::operator delete(p); }    // Custom disposer
};

// Default construction (empty)
luisa::BinaryBlob empty_blob;

// Data access
std::byte* data = blob.data();
const std::byte* cdata = blob.data();       // For const blob
size_t sz = blob.size();
bool is_empty = blob.empty();

// Span conversion
luisa::span<std::byte> mutable_span = static_cast<luisa::span<std::byte>>(blob);
luisa::span<const std::byte> const_span = static_cast<luisa::span<const std::byte>>(blob);

// Move semantics
luisa::BinaryBlob blob2(std::move(blob));   // Move construct
luisa::BinaryBlob blob3;
blob3 = std::move(blob2);                   // Move assign

// Release (returns raw pointer, blob becomes empty)
void* released = blob.release();
// Remember to manually delete released pointer!
```

---

## Clock

**Header**: `<luisa/core/clock.h>`

High-resolution timer for performance measurement.

```cpp
#include <luisa/core/clock.h>

// Construction (starts timing automatically)
luisa::Clock clock;

// Reset/start timing
clock.tic();

// Get elapsed time in milliseconds (does NOT reset)
double elapsed_ms = clock.toc();

// Multiple measurements
clock.tic();
// ... work ...
double t1 = clock.toc();
// ... more work ...
double t2 = clock.toc();                    // t2 > t1
```

---

## Dynamic Module

**Header**: `<luisa/core/dynamic_module.h>`

Cross-platform dynamic library loading.

```cpp
#include <luisa/core/dynamic_module.h>

// Load by name (platform-specific)
auto module = luisa::DynamicModule::load("library_name");
// Windows: "kernel32" -> kernel32.dll
// Linux: "c" -> libc.so

// Load from specific directory
auto module = luisa::DynamicModule::load("/path/to/libs", "library_name");

// Load by full path
auto module = luisa::DynamicModule::load_exact("/path/to/lib.so");

// Check if loaded
if (module) { /* valid */ }
if (!module) { /* failed */ }

// Get raw handle
void* handle = module.handle();

// Get function address
void* addr = module.address("function_name");

// Get typed function pointer
using MyFunc = int(int, int);
auto* func = module.function<MyFunc>("function_name");
if (func) {
    int result = func(1, 2);
}

// Release handle (manual cleanup)
void* raw = module.release();
if (raw) {
    luisa::dynamic_module_destroy(raw);
}

// Reset (unload)
module.reset();

// Move semantics
luisa::DynamicModule mod2(std::move(module));
```

### Search Path Management

```cpp
#include <luisa/core/platform.h>

// Add search path
auto exe_dir = std::filesystem::path(luisa::current_executable_path()).parent_path();
luisa::DynamicModule::add_search_path(exe_dir);

// Remove search path (decrements ref count)
luisa::DynamicModule::remove_search_path(exe_dir);
```

---

## First Fit Allocator

**Header**: `<luisa/core/first_fit.h>`

Simple memory allocator with first-fit and best-fit strategies.

```cpp
#include <luisa/core/first_fit.h>

// Construction
luisa::FirstFit allocator(1024, 8);         // 1024 bytes, 8-byte alignment
luisa::FirstFit allocator(4096, 64);        // 4096 bytes, 64-byte alignment

// Allocation
luisa::FirstFit::Node* node = allocator.allocate(100);
if (node) {
    size_t offset = node->offset();         // Byte offset in arena
    size_t size = node->size();             // Allocated size (may be larger than requested)
}

// Best-fit allocation (finds smallest fitting block)
luisa::FirstFit::Node* node = allocator.allocate_best_fit(100);

// Deallocation
allocator.free(node);

// Properties
size_t sz = allocator.size();               // Total arena size
size_t align = allocator.alignment();       // Alignment requirement

// Debug - dump free list
luisa::string free_list = allocator.dump_free_list();

// Move semantics
luisa::FirstFit allocator2(std::move(allocator));
```

---

## Logging

**Header**: `<luisa/core/logging.h>`

Formatted logging with multiple severity levels.

### Log Level Control

```cpp
#include <luisa/core/logging.h>

// Set log level
luisa::log_level_verbose();                 // Show all
luisa::log_level_info();                    // Info and above
luisa::log_level_warning();                 // Warnings and above
luisa::log_level_error();                   // Errors only

// Flush log buffer
luisa::log_flush();
```

### Function-Style Logging

```cpp
// Simple messages
luisa::log_verbose("Verbose message");
luisa::log_info("Info message");
luisa::log_warning("Warning message");

// Formatted messages
luisa::log_info("Value: {}, Result: {}", 42, 3.14);
luisa::log_warning("File: {}, Line: {}", filename, line);
```

### Macro-Style Logging (Recommended)

```cpp
// Basic macros
LUISA_VERBOSE("Verbose message");
LUISA_INFO("Info message");
LUISA_WARNING("Warning message");

// With format arguments
LUISA_INFO("Values: {}, {}", 1, 2);
LUISA_WARNING("Result: {}", 3.14159);

// With source location
LUISA_VERBOSE_WITH_LOCATION("Debug: {}", value);
LUISA_INFO_WITH_LOCATION("Processing: {}", name);
LUISA_WARNING_WITH_LOCATION("Deprecated: {}", api);
```

### Format Specifiers

```cpp
LUISA_INFO("Integer: {}", -123456);
LUISA_INFO("Unsigned: {}", 123456u);
LUISA_INFO("Float: {}", 3.14159f);
LUISA_INFO("Double: {}", 2.718281828);
LUISA_INFO("String: {}", "text");
LUISA_INFO("Boolean: {}", true);
LUISA_INFO("Pointer: {}", static_cast<void*>(ptr));
LUISA_INFO("Hex: {:x}", 255);               // ff
LUISA_INFO("Binary: {:b}", 170);            // 10101010
LUISA_INFO("Scientific: {:e}", 12345.6789);
LUISA_INFO("Fixed: {:.2f}", 3.14159);       // 3.14
```

---

## Mathematics

**Header**: `<luisa/core/mathematics.h>`

Math functions for scalars and vectors (SIMD-friendly).

### Power of 2

```cpp
// Next power of 2
uint32_t p2 = luisa::next_pow2(100u);       // 128
uint64_t p2_64 = luisa::next_pow2(1025ull); // 2048
```

### Scalar Math

```cpp
// Fractional part
float f = luisa::fract(3.7f);               // 0.7
float f_neg = luisa::fract(-3.7f);          // 0.3

// Angle conversion
float rad = luisa::radians(180.0f);         // pi
float deg = luisa::degrees(luisa::constants::pi); // 180

// Standard functions
float s = luisa::sin(angle);
float c = luisa::cos(angle);
float sq = luisa::sqrt(x);
float a = luisa::abs(x);
float mn = luisa::min(a, b);
float mx = luisa::max(a, b);
```

### Vector Math

```cpp
// Unary functions (component-wise)
float2 v2 = luisa::make_float2(0.0f, luisa::constants::pi / 2.0f);
float2 sin2 = luisa::sin(v2);               // (0, 1)
float2 cos2 = luisa::cos(v2);
float2 sqrt2 = luisa::sqrt(v2);
float2 abs2 = luisa::abs(v2);
float2 floor2 = luisa::floor(v2);
float2 ceil2 = luisa::ceil(v2);
float2 fract2 = luisa::fract(v2);
float2 rad2 = luisa::radians(v2);
float2 deg2 = luisa::degrees(v2);

// Binary functions (component-wise)
float2 a = luisa::make_float2(1.0f, 2.0f);
float2 b = luisa::make_float2(3.0f, 4.0f);
float2 min2 = luisa::min(a, b);             // (1, 2)
float2 max2 = luisa::max(a, b);             // (3, 4)
float2 pow2 = luisa::pow(a, b);             // (1^3, 2^4) = (1, 16)
float2 atan2_2 = luisa::atan2(y, x);
float2 fmod2 = luisa::fmod(a, b);

// Scalar-vector operations
float2 min_s2 = luisa::min(2.0f, a);        // min(2, each component)
float2 max_s2 = luisa::max(a, 2.0f);        // max(each component, 2)

// Special values check
bool2 isnan2 = luisa::isnan(v2);
bool2 isinf2 = luisa::isinf(v2);
```

### Vector Operations

```cpp
float2 a = luisa::make_float2(1.0f, 2.0f);
float2 b = luisa::make_float2(3.0f, 4.0f);
float3 c = luisa::make_float3(1.0f, 2.0f, 3.0f);
float3 d = luisa::make_float3(4.0f, 5.0f, 6.0f);

// Dot product
float dot2 = luisa::dot(a, b);              // 1*3 + 2*4 = 11
float dot3 = luisa::dot(c, d);              // 1*4 + 2*5 + 3*6 = 32

// Length (magnitude)
float len = luisa::length(a);               // sqrt(1+4) = sqrt(5)

// Distance
float dist = luisa::distance(a, b);

// Normalize
float2 n = luisa::normalize(a);             // a / length(a)

// Cross product (3D only)
float3 cr = luisa::cross(c, d);
```

### Matrix Operations

```cpp
float2x2 m2 = luisa::make_float2x2(1.0f, 2.0f, 3.0f, 4.0f);
float3x3 m3 = luisa::make_float3x3(...);
float4x4 m4 = luisa::make_float4x4(...);

// Transpose
float2x2 t2 = luisa::transpose(m2);
float3x3 t3 = luisa::transpose(m3);
float4x4 t4 = luisa::transpose(m4);

// Inverse
float2x2 inv2 = luisa::inverse(m2);
float3x3 inv3 = luisa::inverse(m3);
float4x4 inv4 = luisa::inverse(m4);

// Determinant
float det2 = luisa::determinant(m2);
float det3 = luisa::determinant(m3);
float det4 = luisa::determinant(m4);
```

### Transformation Matrices

```cpp
// Translation
float4x4 trans = luisa::translation(1.0f, 2.0f, 3.0f);
float4x4 trans_vec = luisa::translation(luisa::make_float3(1.0f, 2.0f, 3.0f));

// Scaling
float4x4 scale = luisa::scaling(2.0f, 3.0f, 4.0f);     // Non-uniform
float4x4 scale_u = luisa::scaling(5.0f);                // Uniform

// Rotation (axis + angle)
float4x4 rot = luisa::rotation(
    luisa::make_float3(0.0f, 0.0f, 1.0f),               // Axis
    luisa::constants::pi / 2.0f                         // Angle (radians)
);
```

### Selection and Interpolation

```cpp
// Select (ternary operator)
float result = luisa::select(0.0f, 1.0f, true);         // 1.0 (false -> 0.0, true -> 1.0)
float2 sel2 = luisa::select(a, b, luisa::make_bool2(true, false));

// Linear interpolation
float lerp = luisa::lerp(0.0f, 10.0f, 0.5f);            // 5.0
float2 lerp2 = luisa::lerp(a, b, 0.5f);                 // Component-wise

// Clamp
float clamped = luisa::clamp(5.0f, 0.0f, 10.0f);        // 5.0
float clamped2 = luisa::clamp(-5.0f, 0.0f, 10.0f);      // 0.0
float clamped3 = luisa::clamp(15.0f, 0.0f, 10.0f);      // 10.0
float3 clamp3 = luisa::clamp(v3, 0.0f, 10.0f);

// Sign
float s = luisa::sign(5.0f);                            // 1.0
float s2 = luisa::sign(-3.0f);                          // -1.0
float s3 = luisa::sign(0.0f);                           // 1.0
float3 sign3 = luisa::sign(v3);
int si = luisa::sign(-5);                               // -1

// Fused multiply-add: a * b + c
float fma = luisa::fma(2.0f, 3.0f, 4.0f);               // 2*3 + 4 = 10
float3 fma3 = luisa::fma(a3, b3, c3);                   // Component-wise
float3 fma_s3 = luisa::fma(2.0f, a3, c3);               // Scalar * vector + vector
```

### Constants

```cpp
luisa::constants::pi;           // 3.14159265...
luisa::constants::pi_over_2;    // pi / 2
luisa::constants::pi_over_4;    // pi / 4
luisa::constants::two_pi;       // 2 * pi
luisa::constants::inv_pi;       // 1 / pi
luisa::constants::e;            // 2.71828182... (Euler's number)
```

---

## Pool Allocator

**Header**: `<luisa/core/pool.h>`

Fast object pool allocator with thread-safe and non-thread-safe variants.

```cpp
#include <luisa/core/pool.h>

// Thread-safe pool (default)
luisa::Pool<MyClass> pool;

// Non-thread-safe pool (for single-threaded use)
luisa::Pool<MyClass, false> pool_nt;

// Raw allocation (constructor not called)
MyClass* obj = pool.allocate();
pool.deallocate(obj);

// Constructed allocation
MyClass* obj2 = pool.create();              // Default construct
MyClass* obj3 = pool.create(args...);       // Construct with args
pool.destroy(obj2);                         // Destruct and return to pool
pool.destroy(obj3);

// Move semantics
luisa::Pool<MyClass> pool2(std::move(pool));
```

---

## Fiber

**Header**: `<luisa/core/fiber.h>`

Fiber-based task scheduler and parallel primitives built on top of marl.

### Scheduler

```cpp
#include <luisa/core/fiber.h>

// Default scheduler (uses all cores)
luisa::fiber::scheduler sched;

// Scheduler with fixed thread count
luisa::fiber::scheduler sched(4);

// RAII: binds on construction, unbinds on destruction
```

### Defer

```cpp
// Defer execution until end of scope (like Go's defer but scope-based)
{
    luisa_fiber_defer(printf("world\n"));
    printf("hello ");
}
```

### Synchronization Primitives

```cpp
// Event (Manual or Auto reset)
luisa::fiber::event evt(luisa::fiber::event::Mode::Manual, false);
evt.signal();        // Set signalled
evt.clear();         // Reset to unsignalled
evt.wait();          // Block until signalled
bool ready = evt.test();
bool ready2 = evt.is_signalled();

// Counter (WaitGroup)
luisa::fiber::counter cnt(3);
cnt.add(2);          // Increment count
cnt.done();          // Decrement count
cnt.wait();          // Block until count reaches 0

// Mutex / Lock / ConditionVariable
luisa::fiber::mutex mtx;
luisa::fiber::lock lck(mtx);
luisa::fiber::condition_variable cv;
```

### Task Submission

```cpp
// Schedule fire-and-forget task
luisa::fiber::schedule([]() noexcept {
    // do work
});

// Async task with completion event/future
auto evt = luisa::fiber::async([]() noexcept {
    return 42;
});
evt.wait();

// Async task returning void
auto evt_void = luisa::fiber::async([]() noexcept {
    // do work
});
evt_void.wait();
```

### Parallel For

```cpp
// Blocking parallel for (waits for completion)
luisa::fiber::parallel(100, [](uint32_t i) noexcept {
    // process item i
});

// Blocking parallel for with range callback
luisa::fiber::parallel(100, [](uint32_t begin, uint32_t end) noexcept {
    // process range [begin, end)
});

// Async parallel for (returns counter)
auto cnt = luisa::fiber::async_parallel(100, [](uint32_t i) noexcept {
    // process item i
});
cnt.wait();

// Async parallel for with external counter
luisa::fiber::counter cnt(0);
luisa::fiber::async_parallel(cnt, 100, [](uint32_t i) noexcept {
    // process item i
});
cnt.wait();

// Control batch size (items per fiber task)
luisa::fiber::parallel(1000, [](uint32_t i) noexcept {}, /*internal_jobs=*/10);
```

### Iterator Parallel For

```cpp
std::vector<int> data(1000);

// Blocking iterator parallel for
luisa::fiber::parallel(data.begin(), data.end(), 64,
    [](auto l, auto r) {
        // process range [l, r)
    });

// Async iterator parallel for
auto cnt = luisa::fiber::async_parallel(data.begin(), data.end(), 64,
    [](auto l, auto r) {
        // process range [l, r)
    });
cnt.wait();

// Async with external counter
luisa::fiber::counter cnt(0);
luisa::fiber::async_parallel(cnt, data.begin(), data.end(), 64,
    [](auto l, auto r) {
        // process range [l, r)
    });
cnt.wait();
```

### Query Worker Threads

```cpp
uint32_t n = luisa::fiber::worker_thread_count();
```

---

## Summary Table

| Component | Header | Key Classes/Functions |
|-----------|--------|----------------------|
| Basic Traits | `<luisa/core/basic_traits.h>` | `is_vector_v`, `is_matrix_v`, `vector_element_t`, `to_underlying` |
| Basic Types | `<luisa/core/basic_types.h>` | `float2/3/4`, `float2x2/3x3/4x4`, `make_float2/3/4`, `make_float2x2/3x3/4x4` |
| Binary Buffer | `<luisa/core/binary_buffer.h>` | `BinaryBuffer`, `BinaryBufferReader` |
| Binary File Stream | `<luisa/core/binary_file_stream.h>` | `BinaryFileStream` |
| Binary IO | `<luisa/core/binary_io.h>` | `BinaryBlob` |
| Clock | `<luisa/core/clock.h>` | `Clock::tic()`, `Clock::toc()` |
| Dynamic Module | `<luisa/core/dynamic_module.h>` | `DynamicModule::load()`, `address()`, `function<>()` |
| First Fit | `<luisa/core/first_fit.h>` | `FirstFit::allocate()`, `allocate_best_fit()`, `free()` |
| Logging | `<luisa/core/logging.h>` | `LUISA_INFO()`, `LUISA_WARNING()`, `log_level_info()` |
| Mathematics | `<luisa/core/mathematics.h>` | `sin()`, `dot()`, `normalize()`, `transpose()`, `inverse()`, `lerp()`, `clamp()` |
| Pool | `<luisa/core/pool.h>` | `Pool<T>::allocate()`, `create()`, `destroy()` |
| Fiber | `<luisa/core/fiber.h>` | `scheduler`, `schedule()`, `async()`, `parallel()`, `async_parallel()`, `event`, `counter` |
