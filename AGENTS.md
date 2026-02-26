# Agent Instructions for LuisaCompute

Welcome to LuisaCompute, a high-performance cross-platform computing framework for graphics and beyond.
When operating in this repository, strictly adhere to the following rules, conventions, and workflows.

## 1. Build, Lint, and Test Commands

### Building
The project uses CMake (recommended) or XMake and requires C++20 (Clang-15, GCC-11, or MSVC-17).
- **Bootstrap script (recommended):** `python bootstrap.py cmake -f cuda -b` (change `cuda` to `dx`, `metal`, `all`, etc.)
- **CMake build:**
  ```bash
  cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
  cmake --build build
  ```
- **XMake build:** `xmake f -m release -c && xmake`

### Linting & Formatting
- **Format:** The codebase uses `.clang-format` (LLVM-based). Always format C++ code before committing.
  - Command: `find src include -name "*.cpp" -o -name "*.h" -o -name "*.hpp" | xargs clang-format -i`
- **Lint:** The project uses `.clang-tidy` for static analysis. Fix any reported warnings.

### Testing
- Tests are located in `src/tests/` and use the **doctest** framework (header-only).
- **Running a single test:** Tests compile to standalone executables in the build binary directory (usually `build/bin/` or `bin/`). You MUST pass the backend as an argument.
  ```bash
  # Example: Run path tracing test on CUDA
  ./build/bin/test_path_tracing_clangcxx cuda

  # Other backends: dx, metal, cpu
  ./build/bin/test_matrix_multiply metal
  ```

## 2. Code Style & Conventions

### General C++ Practices
- **C++20 Heavily Used:** Leverage modern C++20 features.
- **Resource Management:** Always use RAII for resource management. Avoid raw new/delete.
- **Constexpr & Templates:** Prefer templates and constexpr where appropriate for compile-time evaluation.

### Formatting Rules
- **Indentation:** 4 spaces (NEVER tabs).
- **Line Length:** No strict column limit.
- **Braces:** Attach style (e.g., `if (x) {` on the same line).
- **Pointers/References:** Right-aligned (e.g., `Type *ptr`, `Type &ref`).
- **Namespaces:** Compact, with NO indentation inside the namespace blocks.
- **Comments:** No specific format, but keep them sparse. Document the *why*, not the *what*. DO NOT leave conversational comments.

### Domain-Specific Language (DSL) Macros
When writing DSL kernel code inside C++, you must use the provided custom macros for control flow:
- `$if`, `$elif`, `$else`, `$for`, `$while`, `$loop`, `$switch`, `$case`, `$default`.
- Structs use `LUISA_STRUCT`, `LUISA_BINDING_GROUP`, `LUISA_BINDING_GROUP_TEMPLATE`.
- Typical variable wrappers: `Var<T>` (e.g., `Var<float3>`). Note that type aliases exist, like `Float3` for `Var<float3>`.
- Type conversions: use `make_*` for construction, `cast<T>` for static casting, and `as<T>` for bitwise casting.

### Architecture & Workflows
- **Context & Device:** A typical workflow begins by creating a `Context` and loading a backend `Device` (CUDA, Metal, DX, CPU).
- **Resources:** Use `Stream`, `Buffer<T>`, and `Image<T>` to manage commands and data.
- **Kernels:** Kernels are authored and then compiled into `Shader` objects.
- **Backends:** If making changes to the DSL or runtime, consider cross-backend compatibility. Backend specific logic lives in `src/backends/<backend-name>/`.
- **Generated Code:** Do NOT manually edit auto-generated files (e.g., `src/xir/op.cpp`). Modify the underlying python scripts (e.g., `src/xir/update_op_name_map.py`) instead.

## 3. Python Bindings
- Located in `src/py/` and `src/tests/python/`.
- No dedicated reference type qualifier in Python.
- Structures and arrays are passed as references to `@luisa.func`, but built-in types (scalars, vectors, matrices) are passed by value by default.

## 4. Agent Operational Rules
- **No Assumption of Tools:** Read configuration files (`CMakeLists.txt`, `xmake.lua`) if you need to determine library links or build definitions.
- **Path Resolution:** Always construct absolute paths by combining the repository root with the relative path.
- **Refactoring:** When modifying core structures, ensure `AST`/`IR` compatibility (note: IR is actively replacing AST).
- **Dependencies:** Stored in `src/ext/` as Git submodules. Do not add dependencies unless explicitly instructed.
- **High Performance:** This is a performance-critical graphics framework. Avoid unnecessary allocations, deep copies, and synchronization stalls.