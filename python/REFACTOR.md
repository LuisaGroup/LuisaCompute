# LuisaCompute Python DSL v2 Refactor Plan

This document outlines the plan for refactoring the LuisaCompute Python DSL v2 package to be more simple, flat, easy to understand, and pythonic.

## 1. Package Structure Reorganization

The current structure is somewhat deeply nested under `lang/` and has overlapping responsibilities. We will move to a more orthogonal structure.

### `luisa/` (Root)
- `__init__.py`: Clean exports of all user-facing DSL features.
- `version.py`: Version information.
- `py.typed`: PEP 561 marker.

### `luisa/core/` (DSL Engine)
The "engine" responsible for turning Python code into Luisa IR.
- `builder.py`: IR construction logic.
- `ir.py`: IR node definitions and data structures.
- `ast_rewriter.py`: (Renamed from `compiler.py`) AST transformation logic.
- `jit.py`: (Renamed from `staged.py`) `StagedFunction`, compilation, and caching.
- `inspect.py`: Python introspection utilities (source extraction, closure analysis).
- `router.py`: Dispatch and routing logic.

### `luisa/types/` (Type System)
A dedicated subpackage for the Luisa type system.
- `basic.py`: Scalar, Vector, Matrix types and their aliases.
- `aggregate.py`: Array, Struct, and the `@struct` decorator.
- `resource.py`: Buffer, Texture, BindlessArray, Accel, RayQuery.
- `utils.py`: Type inference, conversion, and promotion utilities.

### `luisa/dsl/` (User-Facing DSL)
High-level constructs and runtime support for writing DSL code.
- `jit.py`: Re-export `kernel` and `callable` from `core/jit.py`.
- `ops.py`: Operator overloading and runtime support for AST-rewritten code.
- `control_flow.py`: `StaticIf`, `StaticWhile`, and IR control flow helpers.
- `builtin/`: Subpackage for built-in functions.
    - `math.py`, `atomic.py`, `warp.py`, `rtx.py`, `resource.py`, `core.py`.

### `luisa/codegen/` (Output & Serialization)
- `printer.py`: Merged and refined IR-to-text (LLVM-style) printer.
- `json.py`: IR-to-JSON serialization.

---

## 2. Detailed Module Changes

### `type.py` Split
The current `type.py` is too large. It will be split:
- Move `Scalar`, `Vector`, `Matrix` to `types/basic.py`.
- Move `Array`, `Struct`, `Ref` to `types/aggregate.py`.
- Move `Buffer`, `Texture2D`, etc., to `types/resource.py`.
- Move `value_to_type`, `annotation_to_type`, etc., to `types/utils.py`.

### `compiler.py` and `staged.py`
- Rename `compiler.py` to `ast_rewriter.py` to better reflect its purpose.
- Rename `staged.py` to `jit.py`.
- Move `parse_function` and closure analysis from `compiler.py` to `core/inspect.py`.

### `printer.py`
- Merge `lang/printer.py` and `codegen/pretty_printer.py` into a single, robust `codegen/printer.py`.

---

## 3. Orthogonal Tests and Examples

### Tests (`python/tests/`)
Reorganize tests to match the new structure:
- `test_types/`: `test_basic.py`, `test_aggregate.py`, `test_resources.py`.
- `test_core/`: `test_rewriter.py`, `test_jit.py`, `test_ir.py`.
- `test_dsl/`: `test_ops.py`, `test_control_flow.py`, `test_builtins.py`.
- `test_integration/`: End-to-end tests.

### Examples (`python/examples/`)
- Ensure examples are grouped by complexity: `basic/`, `advanced/`, `features/`.
- Add docstrings to all examples.

---

## 4. Documentation Updates

- Update `python/README.md` with the new architecture diagram and package structure.
- Update `python/DESIGN.md` to reflect the finalized 2.0 architecture.
- Ensure all modules have high-quality docstrings.

---

## 5. Tooling and Standards (`pyproject.toml`)

- Update `pyproject.toml` to use modern `setuptools` or `hatchling` (if desired).
- Configure `ruff` for linting and formatting (replacing `black` and `isort`).
- Strict `mypy` configuration for the entire package.
- Minimum Python version set to 3.10 (for `match` statement support).

---

## 6. Implementation Strategy

1.  **Preparation**: Create the new directory structure.
2.  **Types Migration**: Move and split type definitions first (they are the most depended upon).
3.  **Core Migration**: Move IR and Builder logic.
4.  **Rewriter & JIT**: Move the compilation pipeline.
5.  **DSL & Builtins**: Move the high-level API.
6.  **Codegen**: Clean up serialization.
7.  **Verification**: Update all imports and run the test suite.
8.  **Refinement**: Polish docstrings, update `pyproject.toml`, and docs.
