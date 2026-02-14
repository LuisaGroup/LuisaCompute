# LuisaCompute Python DSL v2 Design Document

## Architecture Overview

The LuisaCompute Python DSL v2 is built on a **Multistage Programming** architecture. Instead of interpreting the Python AST at call time, we rewrite the AST into a specialized **IR-building function**.

### The Three Stages

1.  **Stage 1: Transformation (Static)**
    *   Triggered by `@kernel` or `@callable`.
    *   The Python function's AST is passed through `ASTRewriter`.
    *   Every operation (arithmetic, control flow, calls) is replaced with a call to a runtime helper (e.g., `l_binop`, `l_if`).
    *   The result is a "Builder Function" that accepts a `builder` object and original arguments.

2.  **Stage 2: Generation (JIT)**
    *   Triggered when the staged function is called.
    *   The Builder Function is executed.
    *   Host-side logic (standard Python loops, `StaticIf`) is executed normally by Python.
    *   DSL operations call the `builder` to record instructions into a **Structured IR Tree**.

3.  **Stage 3: Lowering (Backend)**
    *   The Structured IR is lower to Luisa AST or XIR.
    *   Structured nodes like `IF` and `LOOP` are preserved until the final backend lowering.

## Structured IR

Unlike traditional compiler IRs that use flat basic blocks and `GOTO`s, this DSL uses a **Structured IR**.

### Core Structured Nodes

*   **`IF`**: Contains a condition, a `true_block`, and a `false_block`.
*   **`LOOP`**: Contains a `body_block`. Jumps within the block are handled by `BREAK` and `CONTINUE`.
*   **`SWITCH`**: Contains a value, a list of `(values, block)` cases, and an optional `default_block`.

This design preserves the high-level structure of the original code, making optimizations and pretty-printing much cleaner.

## Control Flow Dispatch

The DSL automatically handles the "Dispatch Problem" between host and device logic:

*   **Dynamic Dispatch**: If a condition is an IR `Value` (result of a device operation), `l_if` generates an `IF` instruction in the IR.
*   **Static Dispatch**: If a condition is a Python primitive (e.g., `bool`), `l_if` executes the branch on the host, controlling which code is generated.

## Key Components

*   `rewriter.py`: The `ast.NodeTransformer` that performs the Stage 1 transformation.
*   `multistage.py`: The runtime environment providing `l_xxx` helpers and the `StagedFunction` wrapper.
*   `control_flow.py`: Builder-side objects for structured instructions.
*   `ir.py`: Data structures for the structured IR tree.
*   `pretty_printer.py`: C-like human-readable output for the structured IR.

## Comparison with Polymorphic Dispatch

The system is designed to enable patterns like `polymorphic.h`:

```python
class MyDispatch:
    def __init__(self):
        self.impls = []
    def call(self, builder, tag, *args):
        sw = builder.switch(tag)
        for i, impl in enumerate(self.impls):
            with sw.case_scope(i):
                impl.builder_func(builder, *args)
```

In this example, the `for` loop is a **host-side loop** that expands into multiple **device-side cases** inside a single IR `SWITCH` instruction.
