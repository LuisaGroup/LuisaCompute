# LuisaCompute Python DSL v2

A modern, high-performance multistage programming system for GPU/CPU compute shaders, featuring native Python integration, comprehensive type hinting, and an optimized JIT compilation pipeline.

## 🚀 Overview

The LuisaCompute Python DSL v2 allows you to write high-performance compute kernels using idiomatic Python syntax. It leverages advanced AST transformation to bridge the gap between Python's flexibility and the rigorous performance requirements of modern GPU programming.

### Key Features

*   **Unified AST Rewriter**: Dynamically transforms Python code into high-efficiency IR-building logic.
*   **Multistage Programming**: Seamlessly mix host-side Python logic with device-side DSL operations.
*   **Automatic Constant Folding**: Automatically evaluates complex math and logical expressions on constants at compile time.
*   **Structured, AST-like IR**: Generates clean, high-level IR with structured `if`, `while`, and `switch` nodes.
*   **Complete Type Hinting**: Leverages Python type annotations for static analysis and robust IR generation.
*   **LLVM-Style Debugging**: Includes a pretty-printer that generates human-readable, LLVM-inspired IR for easy debugging.
*   **Polymorphic Dispatch**: Support for defining and specializing `@callable` functions within kernels for complex shading and compute logic.

---

## 🛠 Quick Start

Writing a gradient kernel with constant folding and structured IR:

```python
from luisa import Float, kernel, callable, Buffer, dispatch_id, pprint

@callable
def lerp(a: Float, b: Float, t: Float) -> Float:
    return a + (b - a) * t

@kernel
def gradient_kernel(result: Buffer[Float], start: Float, end: Float):
    # dispatch_id() provides the execution index
    idx = dispatch_id().x
    
    # Constant folding: This division is partially optimized if 1024.0 is static
    t = Float(idx) / 1024.0
    
    # Call the staged function
    result[idx] = lerp(start, end, t)

# Compile to IR (no device required for IR generation)
ir = gradient_kernel(None, 0.0, 1.0)
print(pprint(ir))
```

### LLVM-Style IR Output
```llvm
kernel void gradient_kernel(buffer<f32> arg0, f32 arg1, f32 arg2) {
  <3 x u32> v0 = dispatch_id();
  u32 v1 = swizzle(v0, 'x');
  f32 v2 = cast(v1);
  f32 v3 = div(v2, 1024.0);
  f32 v4 = call(@lerp, arg1, arg2, v3);
  buffer_write(arg0, v1, v4);
}
```

---

## 💎 Advanced Features

### ⚡ Automatic Constant Folding
The DSL automatically identifies and optimizes compile-time constants. Complex math expressions on literals or `Const` variables are evaluated on the host, producing a single constant in the final IR.

```python
from luisa import kernel, Buffer, sin, pi

@kernel
def optimized_kernel(buf: Buffer[Float]):
    # sin(pi / 2.0) is evaluated during JIT compilation
    val = sin(pi / 2.0) 
    buf[0] = val # Emits: buffer_write(buf, 0, 1.0)
```

### 🔗 Reference Arguments
Mutable arguments are supported via `Ref[T]`, allowing functions to modify values in-place.

```python
from luisa import callable, Ref, Int

@callable
def increment(x: Ref[Int]):
    x = x + 1 # Transparently handled as load + op + store
```

### 🔀 Native `match` Support
Python's native `match` statement is directly translated to structured `SWITCH` operations in the IR, providing a clean way to implement complex branching logic.

```python
@callable
def classify(tag: Int) -> Int:
    match tag:
        case 0: return 10
        case 1: return 20
        case _: return -1
```

---

## 🔍 Debugging & Profiling

*   **AST Dumps**: Set `LUISA_DUMP_REWRITTEN_AST=1` to see how the rewriter transforms your Python code into IR-building logic.
*   **IR Summary**: Use `luisa.lang.inspect.format_ir_summary(ir)` to get statistics on instruction counts and block structure.

---

## 🧪 Testing

The DSL includes an extensive test suite verifying IR correctness, constant folding, and control flow.

```bash
cd python
pytest
```

## 📜 License

This project is licensed under the same terms as the main LuisaCompute project.
