---
name: cpp-style
description: C++ naming, formatting, static analysis, and RTTI rules for LuisaCompute.
---

## Naming

- **Classes / structs / enums**: `CamelCase` (`Device`, `ResourceCreationInfo`, `Resource::Tag`)
- **Functions & public vars**: `snake_case` (`get_value`, `copy_from`)
- **Private/protected members & functions**: `_snake_case` (`_size`, `_move_from()`)
- **Constants**: `kCamelCase` (`kTensorShaderModel`); macros are `UPPER_SNAKE_CASE` (`LUISA_STRUCT`)
- **Template params**: `CamelCase` (`T`, `U`, `Key`)
- **Namespaces**: `luisa`, `luisa::compute`, `vstd`, etc. Keep compact.
- Exception: the CUDA `lcub` device wrappers (`include/luisa/backends/ext/cuda/lcub/`) intentionally keep CUB's `PascalCase` entry points (`SortPairs`, `DeviceReduce`).

## Syntax Check

Use the project C++ syntax checker:

```bash
python scripts/check_cpp_syntax.py <file>.cpp
```

Flags: `--project-root DIR` (default `.`), `--clangd PATH` (default `clangd`; falls back to `clangd.path` in `.vscode/settings.json` when present), `--compile-commands-dir DIR` (overrides auto-detection), `--clang-tidy` (off by default), `--diagnostic-timeout SEC` (default 30), `-v`/`--verbose`.

It drives `clangd` over LSP with `--compile-commands-dir=<dir>`, the database being auto-detected in the order `.vscode`, `build`, project root, then any `build*` directory, and only accepted if it actually contains the requested file — otherwise (e.g. a brand-new header) clangd falls back to the project root with no per-TU flags rather than skipping the file. `-Werror`/`-Werror=`/`/WX` are demoted before the check. `clangd` still reads the root `.clangd` (`CompileFlags.Add`/`Remove`) for the project's warning suppressions. Exit status: 0 clean, 1 errors, 2 tool failure. `scripts/check_all_cpp_syntax.py` runs the same check over every file of a database (default `--compile-commands .vscode/compile_commands.json`).

## Formatting

Format with the project-root `.clang-format` (this skill ships no copy of it). CI does **not** run `clang-format`.

Base: **LLVM style**. Key overrides:

- **Indent**: 4 spaces, no tabs (`IndentWidth: 4`, `UseTab: Never`, `TabWidth: 4`). Continuation indent 4. Case labels indented. Preprocessor body indent 2 (`PPIndentWidth: 2`) but directives stay at column 0 (`IndentPPDirectives: None`).
- **Braces**: K&R (attach). No break before braces. Indent braces off.
- **Line width**: unlimited (`ColumnLimit: 0`).
- **Pointers/refs**: right-aligned (`int *p`, `int &r`; `PointerAlignment: Right`, `ReferenceAlignment: Right`).
- **Access modifiers**: indent offset `-4` (flush with `class`). Empty lines before/after left as-is.
- **Short constructs**: allow single-line for short blocks, functions, ifs, loops, lambdas, enums, case labels.
- **Constructor init**: not forced one-per-line; no break before comma.
- **Templates / concepts**: break declarations only when multiline; indent requires clause.
- **Spaces**: before `=`, ctor-initializer `:`, inheritance `:`, range-for `:`. No space after C-style casts, `!`, `template` keyword, before braced lists. No space in empty parens or before trailing comments.
- **Alignment**: after open brackets & operands; don't align consecutive assignments.
- **Includes/using**: never auto-sort.
- **Namespaces**: compact when short (`CompactNamespaces: true`, `ShortNamespaceLines: 0` = no line limit); no indentation inside (`NamespaceIndentation: None`).
- **Strings/comments**: break string literals; don't reflow comments.
- **Macros** (`SpaceBeforeParens: Custom`): the `IfMacros` list gets a space before `(` (`AfterIfMacros: true`), the `ForEachMacros` list does not (`AfterForeachMacros: false`); function-like macros never get one. Lists:
  - `ForEachMacros`: `LUISA_STRUCT`, `LUISA_BINDING_GROUP`, `LUISA_BINDING_GROUP_TEMPLATE`
  - `IfMacros`: `$if`, `$elif`, `$else`, `$for`, `$while`, `$loop`, `$switch`, `$case`, `$default`
  - `StatementMacros`: `LUISA_MAP`

## Static Analysis

Run the project-root `.clang-tidy` (this skill ships no copy of it). It is a single `Checks:` string: everything off with `-*`, then **142 individually named checks** — there are no category globs and no `WarningsAsErrors`/`HeaderFilterRegex`/per-check options. Prefix coverage:

- **bugprone-** (50 checks)
- **cert-** (9)
- **cppcoreguidelines-** (4: `interfaces-global-init`, `narrowing-conversions`, `pro-type-member-init`, `slicing`)
- **google-** (3: `default-arguments`, `explicit-constructor`, `runtime-operator`)
- **hicpp-** (2: `exception-baseclass`, `multiway-paths-covered`)
- **misc-** (6)
- **modernize-** (27)
- **mpi-** (2), **openmp-** (1)
- **performance-** (14)
- **portability-** (1: `simd-intrinsics`)
- **readability-** (23)

See the project-root `.clang-tidy` for the exact check list — enabling a whole category (e.g. `readability-*`) is not what the project config does.

## No RTTI

RTTI is off for project code by default: `lc_rtti` (`xmake.lua:71`, default `false`) drives `/GR-` for `cl`/`clang_cl` and `-fno-rtti` (`-fno-rtti-data` for clang, `-fno-rtti` for gcc) in `scripts/xmake_func.lua:435-455`. RTTI is force-enabled when `lc_rtti` is on **or** the Python bindings are built (`_lc_enable_py`, `scripts/xmake_func.lua:438`) because pybind11 needs it; the CMake build adds no global RTTI flag, only `-fno-rtti`/`/GR-` on LLVM-linked targets when the LLVM package has RTTI off (`src/backends/common/CMakeLists.txt:197-200`). Do **not** use:

- `dynamic_cast` — use `static_cast` when the type is known (e.g. `static_cast<CUDADeviceConfigExt *>(_device_config_ext.get())`, `src/backends/cuda/cuda_device.cpp:390`)
- `typeid`
- `std::type_info`

Prefer virtual dispatch or explicit type tags: `Resource::Tag` (`include/luisa/runtime/rhi/resource.h:227`), the `luisa::compute::Type` registry (`Type::of<T>()`, `include/luisa/ast/type.h:416`, backed by `detail::TypeDesc<T>` in `include/luisa/ast/type_registry.h`), and `luisa::to_string`/magic_enum (`include/luisa/core/magic_enum.h`) for enum names. Vendored third-party code under `src/ext` is exempt; project glue inside `src/ext` is not.

## No C++ Exception Raising in Project Code

The entire repository's own C++ code must not use `throw`, rethrow, or an
exception-raising helper/macro as a substitute. This includes libraries,
backends, bindings, examples, tests, benchmarks, and generated project code.
Third-party libraries and archived third-party compiler output are exempt;
project-owned glue under a dependency directory is not automatically exempt.

Build reality: the rule is about **raising**, not about `try`/`catch` — the scanner
forbids only the tokens `throw`, `rethrow_exception`, `throw_with_nested`,
`__cxa_throw`, `_CxxThrowException`, `throws`. xmake targets compile with
`exceptions = "no-cxx"` plus `_HAS_EXCEPTIONS=0` on Windows unless a target sets
`enable_exception = true` (`scripts/xmake_func.lua:345-356`); the only targets that do are
`src/py` (`src/py/xmake.lua:23`) and `examples/extension/clangcxx_compiler`
(`examples/extension/clangcxx_compiler/xmake.lua:4`). The CMake build disables nothing and
adds `/EHsc` for clang-cl (`src/CMakeLists.txt:34`), so `catch` at integration boundaries
(`src/backends/remote/remote_server.cpp`, `src/py/export_dlpack.cpp`) still compiles.

- Use `LUISA_ASSERT` for fatal preconditions and `LUISA_ERROR` for unconditional
  failures, with an explicit `<luisa/core/logging.h>` include. Use a literal
  format string for dynamic messages, such as `LUISA_ERROR("{}", message)`.
- Preserve recoverable API contracts with explicit error/status results and
  propagate failures before consuming invalid values. Do not turn compilation
  rejection, protocol errors, or Python validation errors into process aborts
  merely to remove an exception. Python bindings can use the Python C API error
  indicator and a null return from a C API entry point.
- Preserve cleanup with RAII. Catching a third-party exception at an integration
  boundary is allowed; do not remove that boundary or enable exceptions on an
  otherwise exception-free target to accommodate project-owned raising code.
- Test expected fatal failures in a separate process; do not use `expect(throws(...))`
  for a function that now reports through Luisa fatal checks. See
  [the test skill](../test/SKILL.md#fatal-checks-without-exceptions).

Run `python scripts/check_cpp_no_throw.py` (no arguments; it scans every tracked
`.c/.cpp/.h/...` file via `git ls-files`) to check the rule locally. Its third-party
exemptions are the explicit `VENDORED_DIRECTORIES`/`VENDORED_FILES` lists in the script.
CI runs it as the `no-project-exceptions` job of `.github/workflows/check-cpp-style.yml`
— that job is the **only** C++ style check in CI. Python `raise` and other host-language
error handling remain governed by their own API contracts.

## Integer Types

Prefer fixed-width integer types:

- Use: `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `int16_t`, `uint16_t`, `int8_t`, `uint8_t`
- `size_t` is acceptable for sizes/indices per STL convention.
- Prefer `std::byte` for raw byte data.
- Avoid `unsigned int`, `long long`, `unsigned long`, `short`, and `char` for arithmetic. The project defines its own short aliases in `include/luisa/core/basic_traits.h:47-56` — `luisa::uint = uint32_t`, `luisa::ushort = uint16_t`, `luisa::byte = int8_t`, `luisa::ubyte = uint8_t`, `luisa::slong = long long`, `luisa::ulong = unsigned long long` — and `luisa::uint` is the dominant spelling in public headers, so keep using them where the surrounding API already does; do not invent new aliases.
- GPU-side vector types (`int3`, `uint3`, `float2`/`float3`/`float4`, …) are `luisa::Vector<T, N>` aliases produced by `LUISA_MAKE_VECTOR_TYPES` in `include/luisa/core/basic_types.h:119-140`.

## Verification

After editing C++ files:

```bash
# Syntax + clang-tidy diagnostics through the project checker (--clang-tidy is opt-in)
python scripts/check_cpp_syntax.py src/core/logging.cpp --clang-tidy
# Check formatting (dry run; replace --dry-run with -i to apply) — not enforced by CI
clang-format --dry-run --Werror src/core/logging.cpp
# Run clang-tidy on a specific file (build/compile_commands.json)
clang-tidy -p build src/core/logging.cpp
# Exception-raising scan (no arguments)
python scripts/check_cpp_no_throw.py
```

When changing build-affecting files, configure and build a relevant target:

```bash
xmake f -m debug -c
xmake build <target>
```

## Resources

- `.clang-format` — project-root formatter config (referenced above; not copied into this skill).
- `.clang-tidy` — project-root static-analysis config (referenced above; not copied into this skill).
- `.clangd` — project-root configuration for clangd diagnostics (`CompileFlags.Add`/`Remove`, suppressions); not copied into this skill.
