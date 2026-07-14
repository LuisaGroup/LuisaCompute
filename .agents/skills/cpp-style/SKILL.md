---
name: cpp-style
description: C++ naming, formatting, static analysis, and RTTI rules for LuisaCompute.
---

## Naming

- **Classes / structs / enums**: `CamelCase` (`MyClass`, `RenderPipeline`)
- **Functions & public vars**: `snake_case` (`get_value`, `process_data`)
- **Private/protected members & functions**: `_snake_case` (`_private_var`, `_internal_helper()`)
- **Constants**: `kCamelCase` or `UPPER_SNAKE_CASE` for macros
- **Template params**: `CamelCase`
- **Namespaces**: `luisa`, `luisa::compute`, `vstd`, etc. Keep compact.

## Syntax Check

Use the project C++ syntax checker:

```bash
python scripts/check_cpp_syntax.py <file>.cpp
```

It runs `clangd` with the project's `compile_commands.json` and `.clang-tidy`. Skip files not in `compile_commands.json`.

## Formatting

Format with the project `.clang-format` (bundled in this skill; the project root copy is authoritative).

Base: **LLVM style**. Key overrides:

- **Indent**: 4 spaces, no tabs. Continuation indent 4. Case labels indented. Preprocessor indent 2.
- **Braces**: K&R (attach). No break before braces. Indent braces off.
- **Line width**: unlimited (`ColumnLimit: 0`).
- **Pointers/refs**: right-aligned (`int *p`, `int &r`).
- **Access modifiers**: indent offset `-4` (flush with `class`). Empty lines before/after left as-is.
- **Short constructs**: allow single-line for short blocks, functions, ifs, loops, lambdas, enums, case labels.
- **Constructor init**: not forced one-per-line; no break before comma.
- **Templates / concepts**: break declarations only when multiline; indent requires clause.
- **Spaces**: before `=`, ctor-initializer `:`, inheritance `:`, range-for `:`. No space after C-style casts, `!`, `template` keyword, before braced lists. No space in empty parens or before trailing comments.
- **Alignment**: after open brackets & operands; don't align consecutive assignments.
- **Includes/using**: never auto-sort.
- **Namespaces**: compact single-line when short; no indentation inside (`ShortNamespaceLines: 0`).
- **Strings/comments**: break string literals; don't reflow comments.
- **Macros**: control-flow-like (`$if`, `$elif`, `$else`, `$for`, `$while`, `$loop`, `$switch`, `$case`, `$default`) get space before `(`. Function-like macros don't. Special lists:
  - `ForEachMacros`: `LUISA_STRUCT`, `LUISA_BINDING_GROUP`, `LUISA_BINDING_GROUP_TEMPLATE`
  - `IfMacros`: `$if`, `$elif`, `$else`, `$for`, `$while`, `$loop`, `$switch`, `$case`, `$default`
  - `StatementMacros`: `LUISA_MAP`

## Static Analysis

Run `.clang-tidy` (bundled in this skill; the project root copy is authoritative).

All checks disabled (`-*`), then enabled by category:

- **bugprone-***
- **cert-***
- **cppcoreguidelines-***
- **google-*** (default-arguments, explicit-constructor, runtime-operator)
- **hicpp-***
- **misc-***
- **modernize-***
- **mpi-***, **openmp-***
- **performance-***
- **portability-***
- **readability-***

See the bundled `.clang-tidy` for the exact check list.

## No RTTI

RTTI is disabled for project code. Do **not** use:

- `dynamic_cast` — use `static_cast` when type is known
- `typeid`
- `std::type_info`

Prefer virtual dispatch or explicit type tags for type-safe downcasting. Third-party code under `src/ext` is exempt.

## Integer Types

Prefer fixed-width integer types:

- Use: `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `int16_t`, `uint16_t`, `int8_t`, `uint8_t`
- `size_t` is acceptable for sizes/indices per STL convention.
- Prefer `std::byte` for raw byte data.
- Avoid `unsigned int`, `long long`, `unsigned long`, `short`, and `char` for arithmetic. Some platform/system headers may define aliases such as `uint`; avoid introducing new uses in project code.

## Verification

After editing C++ files:

```bash
# Check syntax / tidy diagnostics
python scripts/check_cpp_syntax.py src/foo.cpp

# Check formatting (dry run; replace --dry-run with -i to apply)
clang-format --dry-run --Werror src/foo.cpp

# Run clang-tidy on a specific file
clang-tidy -p build src/foo.cpp
```

When changing build-affecting files, configure and build a relevant target:

```bash
xmake f -m debug -c
xmake build <target>
```

## Resources

- `.clang-format` — bundled copy of the project formatter config.
- `.clang-tidy` — bundled copy of the project static-analysis config.
- `.clangd` — project root configuration for clangd diagnostics (not bundled; see project root).
