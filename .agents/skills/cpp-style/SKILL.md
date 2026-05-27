---
name: cpp-style
description: C++ naming, formatting, static analysis, and RTTI rules for LuisaCompute.
---

## Naming

- **Classes**: `CamelCase` (`MyClass`, `RenderPipeline`)
- **Functions & public vars**: `snake_case` (`get_value`, `process_data`)
- **Private/protected members & functions**: `_snake_case` (`_private_var`, `_internal_helper()`)
- **Constants**: `kCamelCase` or `UPPER_SNAKE_CASE` for macros
- **Template params**: `CamelCase`

## Syntax Check

Use `tool:Cpplint`. Skip files not in `compile_commands.json`.

## Formatting (`.clang-format`)

Base: **LLVM style**. Key overrides:
- **Indent**: 4 spaces, no tabs. Continuation indent 4. Case labels indented.
- **Braces**: K&R (attach). No break before braces for any construct. Indent braces off.
- **Line width**: unlimited (`ColumnLimit: 0`).
- **Pointers/refs**: right-aligned (`int *p`, `int &r`).
- **Access modifiers**: indent offset `-4` (flush with `class`).
- **Short constructs**: allow single-line for short blocks, functions, ifs, loops, lambdas, enums, case labels.
- **Constructor init**: not forced one-per-line; no break before comma.
- **Templates**: break only when multiline.
- **Spaces**: before `=`, ctor-initializer `:`, inheritance `:`, range-for `:`. No space after C-style casts, `!`, `template` keyword, before braced lists. No space in empty parens or before trailing comments.
- **Alignment**: after open brackets & operands; don't align consecutive assignments.
- **Includes/using**: never auto-sort.
- **Namespaces**: compact single-line when short; no indentation inside.
- **Macros**: control-flow-like (`$if`, `$elif`, `$else`, `$for`, `$while`, `$loop`, `$switch`, `$case`, `$default`) get space before `(`; function-like macros don't. Special lists: `LUISA_STRUCT`, `LUISA_BINDING_GROUP`, `LUISA_MAP`.

## Static Analysis (`.clang-tidy`)

All checks disabled (`-*`), then enabled by category:

- **Bugprone**: argument-comment mismatch, assert side effects, dangling handles, dynamic static init, forwarding-ref overloads, incorrect erasure/rounding/division, lambda misuse, macro side effects, misplaced arithmetic in allocation, move-after-use, missing commas/semicolons, no-escape paths, non-null-terminated results, parent virtual calls, reserved identifiers, suspicious memset/memcmp/realloc/enum/include/string-compare, swapped args, terminating continue, thrown temporaries, too-small loop vars, unhandled self-assignment, unused RAII/return values, use-after-move, virtual near-miss
- **CERT**: const-correctness for bitwise ops (dcl21-cpp), std modification (dcl58-cpp), getenv/signal safety (err52-cpp), destructor noexcept (err60-cpp), floating-point comparisons (flp30-c), random seeding (msc50/51-cpp), safe conversions (err34-c, str34-c)
- **C++ Core Guidelines**: avoid global init in interfaces, prevent narrowing, init all members, prevent slicing
- **Google**: ban default args where inappropriate, require explicit single-arg ctors, forbid operator overload abuse
- **HICPP**: throw by value/catch by reference, multiway path coverage
- **Misc**: misplaced const, paired new/delete overloads, non-copyable objects, conventional assignment ops, unique_ptr reset/release misuse
- **Modernize**: bind→lambda, nested namespace concatenation, deprecated headers/ios aliases, range-for, make_shared/make_unique, pass-by-value sink args, raw string literals, redundant void args, auto_ptr/random_shuffle replacement, braced-init-list return, shrink_to_fit, unary static_assert, auto, bool literals, emplace, =default/=delete, [[nodiscard]], noexcept, nullptr, override, transparent functors, uncaught_exceptions
- **MPI/OpenMP**: buffer deref safety, type mismatch, default(none)
- **Performance**: faster string find, avoid implicit copies in range-for, avoid implicit loop conversions, prefer efficient algorithms, pre-size vectors, move correctly, noexcept move ctors, trivially destructible types, avoid type promotion in math, remove unnecessary copies/value params
- **Portability**: avoid SIMD intrinsics without wrappers
- **Readability**: avoid const in param decls, avoid const return for values, empty() over size()==0, static/const member functions where possible, remove delete nullptr, remove deleted defaults, consistent param names, fix misleading indentation, remove redundant control flow/declarations/fn-ptr deref/smartptr get()/string c_str()/string init, simplify subscripts, avoid static access via instances, avoid static defs in anonymous namespaces, starts_with/contains over compare, remove unique_ptr manual delete/release, use any_of/all_of

## No RTTI

RTTI is disabled. Do **not** use:
- `dynamic_cast` — use `static_cast` when type is known
- `typeid`
- `std::type_info`

Prefer virtual dispatch or explicit type tags for type-safe downcasting.

## Integer Types

Never use platform-specific type aliases. Always use fixed-width types:
- Forbidden: `unsigned int`, `long long`, `unsigned long`, `short`, `char` (for arithmetic)
- Required: `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `int16_t`, `uint16_t`, `int8_t`, `uint8_t`
- `size_t` is acceptable for sizes/indices per STL convention.
- prefer `std::byte`