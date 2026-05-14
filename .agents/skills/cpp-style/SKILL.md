---
name: cpp-style
---

## Naming

- **Class names**: Use `CamelCase` (e.g., `MyClass`, `RenderPipeline`)
- **Functions and public variables**: Use `snake_case` (e.g., `get_value`, `process_data`)
- **Private/protected member variables and functions**: Use `_snake_case` (prefix with underscore)
  - Example: `_private_var`, `_internal_helper()`
- **Constants**: Use `kCamelCase` or `UPPER_SNAKE_CASE` for macros
- **Template parameters**: Use `CamelCase`

## Syntax Check

- Use `tool:Cpplint` to check file syntax. Give up if file not in `compile_commands.json`.

## Formatting (`.clang-format`)

- **Base**: LLVM style
- **Indent**: 4 spaces, no tabs; continuation indent 4; case labels indented
- **Line width**: unlimited (`ColumnLimit: 0`)
- **Braces**: attach (K&R style); no break before braces for classes, functions, namespaces, control statements, enums, unions, lambdas, `catch`, or `else`; indent braces false
- **Short constructs**: allow single-line for short blocks, functions, ifs, loops, lambdas, enums, and case labels
- **Pointers / references**: right-aligned (`int *p`, `int &r`)
- **Access modifiers**: indent offset `-4` (flush with `class`); keep existing empty lines around them
- **Constructor initializers**: not forced to one-per-line; do not break before comma
- **Templates**: break only when multiline
- **Spaces**: before assignment operators, ctor-initializer colons, inheritance colons, and range-based-for colons; no space after C-style casts, `!`, `template` keyword, or before C++11 braced lists; no space in empty parentheses; no space before trailing comments
- **Alignment**: align after open brackets and operands; do not align consecutive assignments
- **Includes / using**: never auto-sort
- **Namespaces**: compact on a single line when short; no indentation inside namespace blocks
- **Macros**: custom spacing treats control-statement-like macros (`$if`, `$for`, etc.) with a space before `(`, but function-like macro names have no space; special macro lists include `LUISA_STRUCT`, `LUISA_BINDING_GROUP`, `$if`/`$elif`/`$else`/`$for`/`$while`/`$loop`/`$switch`/`$case`/`$default`, and `LUISA_MAP`

## Static Analysis (`.clang-tidy`)

All checks disabled by default (`-*`), then explicitly enabled by category:

- **Bugprone**: catches argument-comment mismatches, assert side effects, dangling handles, dynamic static initializers, forwarding-reference overloads, incorrect erasure/rounding/division, lambda misuse, macro side effects, misplaced arithmetic in allocation, move-after-use, missing commas/semicolons, no-escape paths, non-null-terminated results, parent virtual calls, reserved identifiers, suspicious `memset`/`memcmp`/`realloc`/enum/include/string-compare, swapped arguments, terminating `continue`, thrown temporaries, too-small loop variables, unhandled self-assignment, unused RAII/return values, use-after-move, virtual near-miss
- **CERT**: enforce `const`-correctness for bitwise operators (`dcl21-cpp`), modification of `std` (`dcl58-cpp`), `getenv`/signal safety (`err52-cpp`), destructor noexcept (`err60-cpp`), floating-point comparisons (`flp30-c`), proper random seeding (`msc50/51-cpp`), safe conversions (`err34-c`, `str34-c`)
- **C++ Core Guidelines**: avoid global init in interfaces, prevent narrowing conversions, initialize all members, prevent slicing
- **Google**: ban default arguments where inappropriate, require `explicit` single-argument constructors, forbid `operator` overload abuse
- **HICPP**: throw by value/catch by reference, ensure multiway paths are fully covered
- **Misc**: misplaced `const`, paired new/delete overloads, non-copyable objects, conventional assignment operators, `unique_ptr` reset/release misuse
- **Modernize**: replace `bind` with lambdas, concatenate nested namespaces, replace deprecated headers/ios aliases, range-based for loops, `make_shared`/`make_unique`, pass-by-value for sink arguments, raw string literals, remove redundant `void` args, replace `auto_ptr`/random_shuffle/`DISALLOW_COPY_AND_ASSIGN`, return braced-init-lists, `shrink_to_fit`, unary `static_assert`, `auto`, bool literals, `emplace`, `= default`/`= delete`, `[[nodiscard]]`, `noexcept`, `nullptr`, `override`, transparent functors, `uncaught_exceptions`
- **MPI / OpenMP**: buffer dereference safety, type mismatch, `default(none)`
- **Performance**: faster string find (literal vs. single char), avoid implicit copies in range-for, avoid implicit conversions in loops, prefer efficient algorithms, pre-size vectors, move correctly, mark move constructors `noexcept`, use trivially destructible types, avoid type promotion in math functions, remove unnecessary copies/value params
- **Portability**: avoid SIMD intrinsics without wrappers
- **Readability**: avoid `const` in parameter declarations, avoid `const` return types for values, prefer `empty()` over `size() == 0`, make member functions `static`/`const` where possible, remove `delete nullptr`, remove deleted defaults, keep parameter names consistent across declarations, fix misleading indentation, remove redundant control flow/declarations/function-pointer dereferences/smartptr `get()`/string `c_str()`/string init, simplify subscript expressions, avoid static access through instances, avoid static definitions in anonymous namespaces, prefer `starts_with`/`contains` over `compare`, remove `unique_ptr` manual delete/release, use `any_of`/`all_of`

## No RTTI Usage

RTTI (Run-Time Type Information) is disabled. Do **not** use:

- `dynamic_cast` — use `static_cast` instead when the type is known
- `typeid`
- `std::type_info`

If type-safe downcasting is required, prefer virtual dispatch or maintain explicit type tags rather than relying on RTTI.
