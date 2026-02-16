# LuisaCompute Python DSL v2

A modern, high-performance multistage programming system for GPU/CPU compute shaders, featuring native Python integration, comprehensive type hinting, and an optimized JIT compilation pipeline.

## 🚀 Overview

The LuisaCompute Python DSL v2 allows you to write high-performance compute kernels using idiomatic Python syntax. It leverages advanced AST transformation to bridge the gap between Python's flexibility and the rigorous performance requirements of modern GPU programming.

### Key Features

*   **Template Specialization**: C++-style generics with `@callable['T']` for type-safe generic programming
*   **Unified AST Rewriter**: Dynamically transforms Python code into high-efficiency IR-building logic
*   **Multistage Programming**: Seamlessly mix host-side Python logic with device-side DSL operations
*   **Automatic Constant Folding**: Evaluates complex math and logical expressions on constants at compile time
*   **Structured, AST-like IR**: Generates clean, high-level IR with structured `if`, `while`, and `switch` nodes
*   **Complete Type Hinting**: Leverages Python type annotations for static analysis and robust IR generation
*   **LLVM-Style Debugging**: Includes a pretty-printer that generates human-readable, LLVM-inspired IR
*   **Polymorphic Dispatch**: Support for defining and specializing `@callable` functions within kernels

---

## 🛠 Quick Start

### Basic Kernel

```python
from luisa import Float, kernel, Buffer, dispatch_id

@kernel
def gradient_kernel(result: Buffer[Float], start: Float, end: Float):
    idx = dispatch_id().x
    t = Float(idx) / 1024.0
    result[idx] = start + (end - start) * t

# Compile to IR (no device required for IR generation)
ir = gradient_kernel.ir
```

### Template Functions

```python
from luisa import callable, kernel, Float, Int, Buffer

# Generic identity function
@callable['T']
def identity(x: T) -> T:
    return x

# Generic buffer fill
@kernel['T']
def fill_buffer(buf: Buffer[T], value: T, count: Int):
    for i in range(count):
        buf[i] = value

# Usage with explicit specialization
int_fill = fill_buffer[Int]
float_fill = fill_buffer[Float]

# Or implicit specialization (types inferred from arguments)
@kernel
def test_kernel():
    result = identity(Float(3.14))  # T inferred as Float
```

### LLVM-Style IR Output
```llvm
kernel void gradient_kernel(buffer<f32> arg0, f32 arg1, f32 arg2) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  f32 v2 = cast(v1);
  f32 v3 = div(v2, 1024.0);
  f32 v4 = add(arg1, mul(sub(arg2, arg1), v3));
  buffer_write(arg0, v1, v4);
}
```

---

## 💎 Template Programming Guide

### Basic Templates

Define generic functions with template parameters:

```python
from luisa import callable, kernel, Float, Int, Bool, Buffer

# Single template parameter
@callable['T']
def scale(x: T, factor: Float) -> T:
    return x * T(factor)

# Multiple template parameters
@callable['T', 'U']
def cast_and_add(a: T, b: U) -> U:
    return U(a) + b

# Generic buffer operations
@callable['T']
def buffer_sum(buf: Buffer[T], n: Int) -> T:
    result = T(0)
    for i in range(n):
        result = result + buf[i]
    return result
```

### Specialization

Templates can be specialized in three ways:

#### 1. Explicit Specialization
```python
int_scale = scale[Int]      # Fix T=Int
float_scale = scale[Float]  # Fix T=Float

# Multiple params
int_float_add = cast_and_add[Int, Float]
```

#### 2. Implicit Specialization (Type Inference)
```python
@kernel
def test():
    # T is inferred from argument type
    a = scale(Int(10), 2.0)      # Uses scale[Int]
    b = scale(Float(3.0), 2.0)   # Uses scale[Float]
```

#### 3. Partial Specialization
```python
@callable['T', 'U']
def combine(a: T, b: U) -> T:
    return a + T(b)

# Fix T, leave U polymorphic
partial = combine[Int]  # Returns TemplatedFunction

# Later, specialize U
full = partial[Float]   # Now returns StagedFunction
```

### Nested Templates

Inner functions can capture outer template parameters:

```python
@callable['T']
def outer_transform(x: T):
    # Inner callable captures T from outer scope
    @callable
    def inner_scale(factor: Float) -> T:
        return x * T(factor)
    
    return inner_scale(2.0)

@kernel
def test():
    result = outer_transform[Float](Float(5.0))  # Returns 10.0
```

### Kernel Templates

Kernels support templates for buffer element types:

```python
@kernel['T']
def process_buffer(buf: Buffer[T], n: Int):
    for i in range(n):
        val = buf[i]
        buf[i] = val * T(2.0) + T(1.0)

# Specialize for different types
int_kernel = process_buffer[Int]
float_kernel = process_buffer[Float]
```

---

## ⚡ Advanced Features

### Automatic Constant Folding
The DSL automatically identifies and optimizes compile-time constants. Complex math expressions on literals or `Const` variables are evaluated on the host, producing a single constant in the final IR.

```python
from luisa import kernel, Buffer, sin, pi

@kernel
def optimized_kernel(buf: Buffer[Float]):
    # sin(pi / 2.0) is evaluated during JIT compilation
    val = sin(pi / 2.0) 
    buf[0] = val  # Emits: buffer_write(buf, 0, 1.0)
```

### Static Loop Unrolling
Use `static_range()` for compile-time loop unrolling:

```python
from luisa import kernel, static_range, Buffer, Float

@kernel
def unrolled_kernel(buf: Buffer[Float]):
    # This loop is unrolled at compile time
    for i in static_range(4):
        buf[i] = Float(i) * 2.0
```

### Reference Arguments
Mutable arguments are supported via `Ref[T]`, allowing functions to modify values in-place.

```python
from luisa import callable, Ref, Int

@callable
def increment(x: Ref[Int]):
    x = x + 1  # Transparently handled as load + op + store

@callable
def swap(a: Ref[Int], b: Ref[Int]):
    temp = a
    a = b
    b = temp
```

### Native `match` Support
Python's native `match` statement is directly translated to structured `SWITCH` operations in the IR.

```python
@callable
def classify(tag: Int) -> Int:
    match tag:
        case 0: return 10
        case 1: return 20
        case 2: return 30
        case _: return -1
```

### Vector and Matrix Types

```python
from luisa import Float3, Float4x4, Float2, Int3

@callable
def vector_ops(a: Float3, b: Float3) -> Float3:
    return a + b * 2.0

@callable
def matrix_transform(m: Float4x4, v: Float3) -> Float3:
    # Matrix-vector multiplication
    return m @ v
```

---

## 🔍 Debugging & Profiling

### AST Dumps
Set `LUISA_DUMP_REWRITTEN_AST=1` to see how the rewriter transforms your Python code into IR-building logic.

```bash
LUISA_DUMP_REWRITTEN_AST=1 python my_kernel.py
```

### IR Inspection
```python
from luisa import pprint
from luisa.lang.inspect import format_ir_summary

@kernel
def my_kernel(buf: Buffer[Float]):
    buf[0] = 1.0

# Get the IR
ir = my_kernel.ir

# Pretty-print
print(pprint(ir))

# Get statistics
print(format_ir_summary(ir))
```

### Template Debugging
```python
from luisa.lang.jit import StagedFunction, TemplatedFunction

# Check if specialized
spec = my_template[Int]
print(f"Is staged: {isinstance(spec, StagedFunction)}")
print(f"Arg types: {spec.arg_types}")
print(f"Template values: {spec.specialization_values}")
```

---

## 🧪 Testing

The DSL includes an extensive test suite verifying IR correctness, constant folding, template specialization, and control flow.

```bash
cd python
pytest                    # Run all tests
pytest -v               # Verbose output
pytest tests/test_jit/test_specialization.py -v  # Specific test file
```

### Test Coverage
- **Template Specialization**: 30+ tests for explicit, implicit, partial, and nested templates
- **Constant Folding**: Compile-time expression evaluation
- **Control Flow**: If/else, loops, match/switch
- **Type System**: Scalars, vectors, matrices, buffers, textures
- **IR Correctness**: Structured IR generation and validation

---

## 📚 Architecture Overview

The DSL uses a **multistage programming** approach:

1. **Parse**: Python AST is extracted from decorated functions
2. **Rewrite**: AST is transformed to IR builder calls
3. **Specialize** (for templates): Template params injected as local variables
4. **Execute**: Builder function runs to generate structured IR
5. **Cache**: IR is cached per argument type combination

For detailed design documentation, see [DESIGN.md](DESIGN.md).

---

## 📝 Example: Ray Tracing Kernel

```python
from luisa import *
from luisa.builtins import rtx

@callable
def hit_sphere(center: Float3, radius: Float, ray: Ray) -> Float:
    oc = ray.origin - center
    a = dot(ray.direction, ray.direction)
    b = 2.0 * dot(oc, ray.direction)
    c = dot(oc, oc) - radius * radius
    discriminant = b * b - 4.0 * a * c
    
    if discriminant < 0.0:
        return -1.0
    else:
        return (-b - sqrt(discriminant)) / (2.0 * a)

@kernel
def ray_trace_kernel(pixels: Buffer[Float3], width: Int, height: Int):
    x = dispatch_id().x
    y = dispatch_id().y
    
    u = Float(x) / Float(width)
    v = Float(y) / Float(height)
    
    ray = rtx.make_ray(
        origin=Float3(0.0, 0.0, 0.0),
        direction=normalize(Float3(u - 0.5, v - 0.5, -1.0))
    )
    
    t = hit_sphere(Float3(0.0, 0.0, -1.0), 0.5, ray)
    
    if t > 0.0:
        pixels[y * width + x] = Float3(1.0, 0.0, 0.0)  # Red
    else:
        pixels[y * width + x] = Float3(0.0, 0.0, 0.0)  # Black
```

---

## 📜 License

This project is licensed under the same terms as the main LuisaCompute project.
