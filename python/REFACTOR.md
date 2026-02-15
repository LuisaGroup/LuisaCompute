# LuisaCompute Python DSL v2 Refactor Plan (Final Version)

This document outlines the final plan for refactoring the LuisaCompute Python DSL v2 package. It is designed to be flat, pythonic, and structurally optimal by cleanly separating language definitions from transformation machinery.

## 1. Package Structure

### `luisa/` (Root)
- `__init__.py`: Public API (exports `kernel`, `callable`, `types`, etc. from `lang`).
- `printer.py`: (Merged) LLVM-style IR pretty printer.
- `serialize.py`: IR-to-JSON serialization.
- `version.py`: Version information.

### `luisa/lang/` (Language Definition)
Contains everything the user interacts with when writing DSL code.
- `__init__.py`: Aggregates and exports all public language features.
- `jit.py`: The `@kernel` and `@callable` decorators and JIT compilation logic.
- `types.py`: Consolidated type system (Scalar, Vector, Matrix, Struct, Buffer, Texture, etc.).
- `ops.py`: Runtime operator support for rewritten AST code (`binop`, `unaryop`, `compare`).
- `control_flow.py`: High-level control flow helpers (`StaticIf`, `StaticWhile`).
- `router.py`: Dispatch and routing logic.
- `builtins/`: (Subpackage) Well-organized implementation of DSL built-in functions.
    - `math.py`, `atomic.py`, `warp.py`, `rtx.py`, `resource.py`, `core.py`.

### `luisa/transform/` (Transformation Machinery)
The internal engine that converts Python AST into Luisa IR.
- `op.py`: (New) The `Op` Enum definition to break circular dependencies between types and IR.
- `ir.py`: Data structures for the structured IR.
- `builder.py`: The IR construction API.
- `rewriter.py`: The `ast.NodeTransformer` for DSL rewriting.
- `inspect.py`: Python introspection (source extraction, closure analysis).

---

## 2. Structural Optimizations

- **Dependency Management**: Moving the `Op` enum to `transform/op.py` allows both `lang/types.py` and `transform/ir.py` to import it without circularity.
- **Flattest Possible Hierarchy**: We avoid sub-nesting wherever possible, using modules instead of subpackages unless there is a clear collection (like `builtins`).
- **Separation of Concerns**: `lang/` is the "Front-end" (what users see), while `transform/` is the "Back-end" (how it works).
- **Consolidated Types**: Merging `type.py` and `dsl_types.py` into `lang/types.py` simplifies the type system's internal structure.

---

## 3. Implementation Steps

1.  **Preparation**: Initialize the `luisa/lang/` and `luisa/transform/` directories.
2.  **Op Enum**: Extract `Op` from `ir.py` to `luisa/transform/op.py`.
3.  **Types**: Consolidate and migrate all types to `luisa/lang/types.py`.
4.  **Transformation Pipeline**: 
    - Move `ir.py` (minus `Op`), `builder.py`, and `compiler.py` (renamed to `rewriter.py`) to `luisa/transform/`.
    - Move introspection logic to `luisa/transform/inspect.py`.
5.  **Language Runtime**:
    - Move `staged.py` to `luisa/lang/jit.py`.
    - Move `ops.py`, `control_flow.py`, `router.py`, and `builtins/` to `luisa/lang/`.
6.  **Utilities**: Move and merge `printer.py` and `serialize.py` to the root.
7.  **Finalization**: 
    - Update all imports across the package.
    - Reorganize the `tests/` directory to match the new structure.
    - Update `pyproject.toml`, `README.md`, and `DESIGN.md`.
    - Execute the full test suite.
