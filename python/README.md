# LuisaCompute Python DSL v2

A modern, multistage programming system for GPU/CPU compute shaders with complete type hinting support and structured IR.

## Overview

This is a new implementation of the LuisaCompute Python DSL that features:

1.  **Unified AST Rewriter**: Dynamically transforms Python AST into IR-building code.
2.  **Multistage Programming**: Seamlessly mix DSL and host logic to control code generation at runtime.
3.  **Structured, AST-like IR**: Uses high-level `IF`, `LOOP`, and `SWITCH` operations instead of flat basic blocks with jumps.
4.  **Complete Type Hinting**: Full support for Python type annotations with static type checking.
5.  **C-Like Pretty Printing**: Human-readable IR output for debugging (e.g., `if (cond) { ... } else { ... }`).
6.  **Native `match` Support**: Translates Python's native `match` statement directly to structured `SWITCH` in the IR.

## Architecture

### Multistage Compilation Pipeline

The DSL utilizes a sophisticated transformation process:

1.  **Parse (Decoration Time)**: The Python source is parsed into an AST.
2.  **Rewrite (Decoration Time)**: The AST is rewritten into a "Builder Function" that, when executed, will generate the equivalent Luisa IR.
3.  **Execute (Call Time)**: When called with specific types, the Builder Function executes. Host-side logic (like `for i in static_range(n)`) is expanded, and DSL operations are recorded into a structured IR tree.
4.  **CodeGen**: The resulting structured IR can be serialized or pretty-printed.

## Control Flow

### Structured Dynamic Flow
Device-side control flow generated in the final shader:

```python
@callable
def abs_val(x: float32) -> float32:
    if x >= 0.0:
        return x
    else:
        return -x
```

### Structured Static Flow (Meta-programming)
Host-side logic that controls what code is generated:

```python
@kernel
def polymorphic_kernel(buf: Buffer[float32], mode: int32):
    # mode is a host-side constant here if passed via capture or logic
    if StaticIf(mode == 0):
        buf[0] = 1.0
    else:
        buf[0] = 2.0
```

### Static Loops and Ranges
Fully unroll loops at generation time:

```python
@callable
def sum_elements(buf: Buffer[float32]):
    total = float32(0.0)
    for i in static_range(4): # Loop expanded at generation time
        total += buf[i]
    return total
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

## Running Tests

```bash
cd python
pytest
```

## Implementation Status

- ✅ Unified AST Rewriter (`rewriter.py`)
- ✅ Multistage Runtime Support (`multistage.py`)
- ✅ Structured IR Nodes (`IF`, `LOOP`, `SWITCH`)
- ✅ C-style Pretty Printer
- ✅ Python `match` translation
- ✅ Polymorphic dispatch support
- ✅ Extensive test suite (135+ tests)

## License

Same as LuisaCompute project.
