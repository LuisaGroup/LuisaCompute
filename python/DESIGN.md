# LuisaCompute Python DSL v2 Design Document

## Architecture Overview

The LuisaCompute Python DSL v2 is built on a **Multistage Programming** architecture. Instead of interpreting the Python AST at call time, we rewrite the AST into a specialized **IR-building function**.

### The Three Stages

1.  **Stage 1: Transformation (Static)**
    *   Triggered by `@kernel` or `@callable`.
    *   The Python function's AST is passed through `ASTRewriter`.
    *   Every operation (arithmetic, control flow, calls) is replaced with a call to a runtime helper (e.g., `l_binop`, `l_if`).
    *   **Distinction**: The rewriter distinguishes between DSL code and plain Python code. Nested functions decorated with `@kernel` or `@callable` are preserved as plain Python to be staged independently.
    *   The result is a "Builder Function" that accepts a `builder` object and original arguments.

2.  **Stage 2: Generation (JIT)**
    *   Triggered when the staged function is called.
    *   The Builder Function is executed.
    *   Host-side logic (standard Python loops, `StaticIf`) is executed normally by Python.
    *   DSL operations call the `builder` to record instructions into a **Structured IR Tree**.
    *   **Nested Functions**: Staged functions can be defined and specialized within other functions, enabling powerful polymorphic dispatch patterns.

3.  **Stage 3: Lowering (Backend)**
    *   The Structured IR is lower to Luisa AST or XIR.
    *   Structured nodes like `IF` and `LOOP` are preserved until the final backend lowering.

## Structured IR

Unlike traditional compiler IRs that use flat basic blocks and `GOTO`s, this DSL uses a **Structured IR** with LLVM-style type naming.

### Core Structured Nodes

*   **`IF`**: Contains a condition, a `true_block`, and a `false_block`.
*   **`LOOP`**: Contains a `body_block`. Jumps within the block are handled by `BREAK` and `CONTINUE`.
*   **`SWITCH`**: Contains a value, a list of `(values, block)` cases, and an optional `default_block`.

### Type System (LLVM-Style)

Types are pretty-printed in LLVM style for clarity:
- Scalars: `i1`, `i8`, `i32`, `f32`, `f64`.
- Vectors: `<4 x f32>`.
- Matrices: `[4 x <4 x f32>]`.
- Buffers: `buffer<f32>`.
- References: `ref<f32>`.

## Reference Arguments

References (`Ref[T]`) are implemented as a property of the function argument rather than a separate IR node. This simplifies the IR while preserving the semantics of mutable arguments. The `ASTRewriter` automatically generates `builder.load()` and `builder.store()` calls for variables identified as references.

## Control Flow Dispatch

The DSL automatically handles the "Dispatch Problem" between host and device logic:

*   **Dynamic Dispatch**: If a condition is an IR `Value` (result of a device operation), `l_if` generates an `IF` instruction in the IR.
*   **Static Dispatch**: If a condition is a Python primitive (e.g., `bool`), `l_if` executes the branch on the host, controlling which code is generated.

## Key APIs

### `StagedFunction.compile(builder, *args)`
Ensures the function is compiled for the given argument types and returns the `IRFunction`.

### `IRBuilder.call(func, *args)`
A unified API to emit a function call. It accepts either an `IRFunction` or a `StagedFunction`. If a `StagedFunction` is passed, it automatically calls `.compile()` to generate the IR for the specific arguments before emitting the call.

## Key Components

*   `builder.py`: The `IRBuilder` class and global builder context management (`get_current_builder()`).
*   `rewriter.py`: The `ast.NodeTransformer` that performs the Stage 1 transformation.
*   `multistage.py`: The runtime environment providing `l_xxx` helpers and the `StagedFunction` wrapper.
*   `control_flow.py`: Builder-side objects for structured instructions.
*   `ir.py`: Data structures for the structured IR tree.
*   `pretty_printer.py`: LLVM-style human-readable output for the structured IR.
