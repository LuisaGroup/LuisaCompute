# LuisaCompute Python DSL v2 Design Document

## 🏗 Architecture Overview

The LuisaCompute Python DSL v2 is built on a **Multistage Programming** architecture. Instead of interpreting Python at runtime, we utilize a tiered compilation pipeline that transforms high-level Python syntax into an optimized, structured Intermediate Representation (IR).

### The Compilation Pipeline

1.  **Transformation (Decoration Time)**
    *   Triggered by `@kernel` or `@callable`.
    *   The function's AST is passed through `ASTRewriter`.
    *   Every DSL-relevant operation (arithmetic, control flow, built-in calls) is replaced with a call to a runtime router (`__luisa_rt`).
    *   The result is a **Builder Function** that, when executed, will generate the equivalent Luisa IR.

2.  **Generation (JIT/Call Time)**
    *   Triggered when the staged function is called with specific argument types.
    *   The Builder Function is executed. 
    *   **Host-side logic** (standard Python `if`, `for`, list comprehensions) is expanded normally by Python.
    *   **DSL operations** call into the `Builder` to record instructions into a **Structured IR Tree**.
    *   The resulting IR is cached for subsequent calls with the same argument types.

3.  **Lowering (Backend)**
    *   The Structured IR is lowered to the LuisaCompute C++ AST or XIR.
    *   Structured nodes like `IF`, `LOOP`, and `SWITCH` are preserved, allowing the backend to perform higher-level optimizations before final machine code generation.

---

## ⚡ Constant Folding & Routing

A core innovation in v2 is the **Host/Device Routing** system, facilitated by the `@router` decorator.

### The `@router` Mechanism
The router intelligently decides whether an operation should be evaluated immediately on the host (Constant Folding) or emitted as a device instruction.

*   **Rule**: If all arguments to a routed function are compile-time constants (Python primitives or `ConstantValue`), the host-side implementation is executed.
*   **Optimization**: This allows complex expressions like `sin(1.0) + cos(2.0)` to be folded into a single numeric literal in the final IR, reducing GPU execution overhead.

### Constant Value Arithmetic
The `ConstantValue` class overrides standard Python operators. This enables expressions involving both Python literals and DSL constants to be folded seamlessly:
```python
a = sin(1.0) # Evaluated on host -> ConstantValue
b = a * 2.0  # Also evaluated on host -> ConstantValue
```

---

## 🌳 Structured IR

Unlike traditional compiler IRs that rely on flat basic blocks and explicit jumps (`GOTO`), this DSL produces a **Structured IR**.

### Core Nodes
*   **`IF`**: Encapsulates a condition and two distinct sub-blocks (`true` and `false`).
*   **`LOOP`**: Encapsulates a body block. Control flow is managed via explicit `BREAK` and `CONTINUE` instructions.
*   **`SWITCH`**: Maps a value to multiple case blocks, including an optional `DEFAULT`.

### Type System
Types are represented using an LLVM-inspired naming convention for clarity during debugging and serialization:
*   **Scalars**: `i1` (bool), `i32` (int), `f32` (float).
*   **Vectors**: `<N x T>` (e.g., `<4 x f32>`).
*   **Matrices**: `[N x <N x T>]` (column-major arrays of vectors).
*   **Resources**: `buffer<T>`, `texture2d<T>`, `accel`.

---

## 🔗 Variable Management

### DSL Variables vs. Host Constants
The `ASTRewriter` distinguishes between standard Python variables and DSL variables.
*   Variables assigned from **dynamic device values** (e.g., `buf.read()`) are transformed into DSL variables using `alloca`, `load`, and `store`.
*   Variables assigned from **constants** (and never reassigned with dynamic values) remain plain Python variables, maximizing the potential for host-side meta-programming.

### Reference Arguments (`Ref[T]`)
References are implemented as a property of function arguments. When a variable is identified as a `Ref`, the rewriter automatically injects the necessary pointer-logic (`load`/`store`) to ensure that in-place modifications within a `@callable` are reflected in the caller's scope.

---

## 🛠 Key Components

*   `jit.py`: Manages the staging process and JIT caching.
*   `rewriter.py`: Implements the `ast.NodeTransformer` for stage 1 transformation.
*   `builder.py`: A fluent API for manual IR construction and global context management.
*   `ir.py`: Defines the data structures for the IR tree and instructions.
*   `router.py`: Handles the logic for constant folding and host/device dispatch.
*   `printer.py`: Generates the LLVM-style human-readable IR representation.
