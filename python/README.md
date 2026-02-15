# LuisaCompute Python DSL v2

A modern, multistage programming system for GPU/CPU compute shaders with complete type hinting support and structured IR.

## Overview

This is a new implementation of the LuisaCompute Python DSL that features:

1.  **Unified AST Rewriter**: Dynamically transforms Python AST into IR-building code.
2.  **Multistage Programming**: Seamlessly mix DSL and host logic to control code generation at runtime.
3.  **Structured, AST-like IR**: Uses high-level `IF`, `LOOP`, and `SWITCH` operations instead of flat basic blocks with jumps.
4.  **Complete Type Hinting**: Full support for Python type annotations with static type checking.
5.  **LLVM-Style Pretty Printing**: Human-readable IR output for debugging with LLVM-style type names (e.g., `i32`, `f32`, `<4 x f32>`, `buffer<f32>`).
6.  **Native `match` Support**: Translates Python's native `match` statement directly to structured `SWITCH` in the IR.
7.  **Mixed DSL + Python**: Support for nested functions and using standard Python helpers within DSL code.

## Architecture

### Multistage Compilation Pipeline

The DSL utilizes a sophisticated transformation process:

1.  **Parse (Decoration Time)**: The Python source is parsed into an AST.
2.  **Rewrite (Decoration Time)**: The AST is rewritten into a "Builder Function" that, when executed, will generate the equivalent Luisa IR.
3.  **Execute (Call Time)**: When called with specific types, the Builder Function executes. Host-side logic (like `for i in range(n)`) is expanded, and DSL operations are recorded into a structured IR tree.
4.  **CodeGen**: The resulting structured IR can be serialized or pretty-printed.

## Features

### Reference Arguments
Support for mutable reference arguments using `Ref[T]`:

```python
@callable
def increment(x: Ref[int32]):
    x = x + 1
```

In the pretty-printed IR, this appears as `ref<i32>`.

### Nested Polymorphic Callables
Define and call specialized functions within kernels:

```python
@kernel
def nested_kernel(tags: Buffer[int32]):
    @callable
    def add_one(x: float32) -> float32:
        return x + 1.0
    
    # ... use add_one in a dispatch switch ...
```

## Quick Start

```python
from luisa import float32, kernel, callable, Buffer, dispatch_id

@callable
def lerp(a: float32, b: float32, t: float32) -> float32:
    return a + (b - a) * t

@kernel
def gradient_kernel(result: Buffer[float32], start: float32, end: float32):
    idx = dispatch_id().x
    result[idx] = lerp(start, end, float32(idx) / 1024.0)

# Build IR
ir = gradient_kernel(None, 0.0, 1.0)
from luisa import pprint
print(pprint(ir))
```

Example Output:
```llvm
kernel void gradient_kernel(buffer<f32> arg0, f32 arg1, f32 arg2) {
  entry:
    <3 x i32> t0 = dispatch_id();
    i32 t1 = swizzle(t0, 'x');
    f32 t2 = cast(t1);
    f32 t3 = div(t2, 1024.0);
    f32 t4 = call('lerp', arg1, arg2, t3);
    void t5 = buffer_write(arg0, t1, t4);
}
```

## Debugging

- **`LUISA_DUMP_REWRITTEN_AST=1`**: Set this environment variable to see the AST transformation performed by the rewriter.

## Running Tests

```bash
cd python
pytest
```

## Implementation Status

- ✅ Unified AST Rewriter (`rewriter.py`)
- ✅ Multistage Runtime Support (`multistage.py`)
- ✅ Structured IR Nodes (`IF`, `LOOP`, `SWITCH`)
- ✅ LLVM-style Pretty Printer
- ✅ Python `match` translation
- ✅ Polymorphic dispatch support
- ✅ Reference arguments (`Ref[T]`)
- ✅ Extensive test suite (140+ tests)

## License

Same as LuisaCompute project.
