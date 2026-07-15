from kimix import *
# make sure you are using a low-hallucination model, or low-temperature settings to run this script
# review by human after run. Remember, trust only the human. May the humanity be with you.
# kimix agent script: https://github.com/Sikao-Engine/Kimi-CLI-X
clear_default_context()
prompt("""# Skill Update Prompt: ast

- **Skill name:** ast
- **Original skill path:** `D:/compute/.agents/skills/ast/SKILL.md`

## Guideline

make minimal changes

## Goal

Review, update, and verify the `ast` skill so it remains accurate, complete, and aligned with current LuisaCompute project practices.

## Background

The `ast` skill documents manual AST construction using `luisa::compute::detail::FunctionBuilder` for kernels, callables, and raster stages without DSL sugar. It covers built-in variables, variables/resources/bindings, expressions (literals, binary/unary ops, swizzle, calls), statements, the type system, operators, usage flags, and worked examples.

## Instructions

1. **Read and analyze**
   - Open `D:/compute/.agents/skills/ast/SKILL.md`.
   - Read any linked or bundled resources referenced from the skill directory (e.g., code samples, headers, tests).

2. **Review for correctness and completeness**
   - Verify API names, signatures, and behaviors against the current codebase, especially `src/ast/function_builder.h` and related AST headers.
   - Check that all `BinaryOp`, `UnaryOp`, and `CallOp` entries are still valid.
   - Confirm swizzle encoding rules and `mark_variable_usage` guidance are correct.
   - Look for outdated types, removed features, or new AST constructs that should be documented.

3. **Identify gaps and broken links**
   - Flag any broken internal links, missing examples, or unclear sections.
   - Note missing topics such as error handling, debug tips, performance guidance, or recently added statement/expression kinds.
   - Compare against the `lc_dsl` skill and project style guides (`D:/compute/.agents/skills/cpp-style/SKILL.md`) for consistency.

4. **Update the skill and bundled resources**
   - Edit `D:/compute/.agents/skills/ast/SKILL.md` to fix inaccuracies, add missing coverage, and improve clarity.
   - Add or update code examples so they compile against current headers.
   - Update any bundled snippets or test files in the same skill directory if needed.

5. **Run domain-relevant verification**
   - For any Python helper scripts touched, run `python scripts/py_lint.py <file>`.
   - For any C++ code changes or new examples, build and test them:
     - `xmake f -m debug -c`
     - `xmake build <target>`
     - `xmake run <target> <args>`
   - Run `python scripts/check_cpp_syntax.py <file>` on modified C++ examples if they are part of the project tree.
   - Confirm no build regressions and that examples execute correctly.

6. **Summarize changes**
   - Produce a concise summary listing what was reviewed, what was changed, and the verification results.
   - Note any remaining risks or follow-up items.

## Constraints

- Do not change unrelated project files.
- Keep changes minimal and focused on the `ast` skill.
- Preserve the skill's front-matter (`name`, `description`) unless the project conventions have changed.

## Output

- Updated `D:/compute/.agents/skills/ast/SKILL.md` (and any bundled resources).
- A short change summary reported back to the parent agent.
""")

# backend_architecture update prompt
clear_default_context()
prompt("""# Skill Update Prompt: `backend_architecture`

- **Skill Name:** backend_architecture
- **Original Skill Path:** `D:/compute/.agents/skills/backend_architecture/SKILL.md`
- **Scope:** Review, update, and verify the backend plugin architecture skill.

## Guideline

make minimal changes

## Objective

You are a coding agent tasked with reviewing and updating the `backend_architecture` skill documentation for the LuisaCompute project. The skill covers backend plugin architecture, `DeviceInterface`, dynamic loading, command encoding, and related CMake/codegen patterns.

Your job is to ensure the skill remains correct, complete, and aligned with current project practices.

## Instructions

1. **Read the Original Skill**
   - Open `D:/compute/.agents/skills/backend_architecture/SKILL.md`.
   - Read it in full, including any code snippets, tables, and linked resources.

2. **Review for Quality**
   - Check for outdated API names, method signatures, file paths, or CMake functions.
   - Verify that the described `DeviceInterface` methods match `include/luisa/runtime/rhi/device_interface.h`.
   - Verify dynamic loading details match `src/runtime/context.cpp`.
   - Check that backend registration patterns and exported C functions match current usage in `src/backends/<name>/`.
   - Identify missing resources such as Buffer, Texture, BindlessArray, Stream, Event, Shader, Mesh, Curve, Procedural Primitive, Motion Instance, Accel, Swapchain, or newer additions.
   - Look for broken links, stale examples, or incomplete codegen/CMake sections.
   - Assess whether examples are sufficient for a new backend author.

3. **Update Skill and Bundled Resources**
   - Edit only the skill file at `D:/compute/.agents/skills/backend_architecture/SKILL.md`.
   - If the skill references bundled examples, headers, or scripts in its directory, update those as needed.
   - Keep changes minimal and focused on correctness, completeness, and clarity.
   - Preserve the existing structure unless a clear reorganization improves usability.

4. **Run Domain-Relevant Verification**
   - For any modified Python scripts, run `python scripts/py_lint.py <file>`.
   - For any modified C++ examples or code snippets, run the C++ syntax checker: `python scripts/check_cpp_syntax.py <file>`.
   - If the skill changes affect build configuration, run `xmake f -m debug -c` and `xmake build <relevant_target>` to confirm no build regressions.
   - Run backend-related tests if applicable (e.g., `xmake run <test_target>`).
   - Confirm all checks pass before finishing.

5. **Summarize Changes**
   - Provide a concise summary of what was updated, added, removed, or fixed.
   - List any verification steps performed and their results.
   - Note any remaining gaps or follow-up work.

## Constraints

- Do not modify project source files unless they are bundled resources owned by this skill.
- Do not introduce changes that break existing builds or tests.
- Keep the prompt scope focused on the `backend_architecture` skill.

## Output

- Updated `D:/compute/.agents/skills/backend_architecture/SKILL.md` (and any bundled resources).
- A short summary report describing the changes and verification results.
""")

# cmake update prompt
clear_default_context()
prompt("""---
skill: cmake
original_path: D:/compute/.agents/skills/cmake/SKILL.md
---

# Update Prompt: `cmake` Skill

You are a coding agent. Your task is to review and, if necessary, update the CMake skill documentation for the LuisaCompute project.

## Guideline

make minimal changes

## Original Skill

- **Path**: `D:/compute/.agents/skills/cmake/SKILL.md`
- **Topic**: CMake build options, custom functions, and backend patterns for LuisaCompute.

## Instructions

1. **Read the current skill** at `D:/compute/.agents/skills/cmake/SKILL.md`.
2. **Review for correctness, completeness, and alignment** with current project practices. Pay special attention to:
   - CMake version requirements and compiler requirements.
   - Quick-start commands for Linux, macOS, and Windows.
   - The `scripts/agent_windows_cmake.py` helper and its flags/options.
   - Build options (`LUISA_COMPUTE_ENABLE_*`) and their defaults.
   - Target naming conventions.
   - Module hierarchy.
   - Custom CMake functions: `luisa_compute_add_backend`, `luisa_compute_install`, `luisa_compute_add_executable`, `luisa_compute_add_test`, `luisa_compute_add_example`, `luisa_example_pair_link`.
   - Backend plugin build rules and output/RPATH behavior.
   - Rust integration via `src/rust/CMakeLists.txt`.
   - Third-party extension pattern under `src/ext/<lib>/`.
3. **Identify outdated information, missing examples, broken links, or gaps.** Compare against the actual CMake files in the project (e.g., `CMakeLists.txt`, `src/*/CMakeLists.txt`, `scripts/agent_windows_cmake.py`) where needed.
4. **Update `SKILL.md` and any bundled resources** if needed. Keep changes minimal and focused on documentation accuracy.
5. **Run domain-relevant verification** after any edits:
   - Check Python script syntax for referenced helpers: `python scripts/py_lint.py scripts/agent_windows_cmake.py`.
   - Validate the documented configure/build flow by running `scripts/agent_windows_cmake.py` (or equivalent platform-specific commands) to confirm no regressions.
   - Optionally run CMake configure with the documented CI minimal flags to ensure options still exist.
6. **Summarize changes** at the end of your response, listing what was modified and the verification steps performed.

## Acceptance Criteria

- `D:/compute/.agents/skills/cmake/SKILL.md` is accurate and up to date.
- No unrelated files are modified.
- Verification steps are run and pass; any failures are reported with mitigation.
- A concise change summary is provided.
""")

# cpp-style update prompt
clear_default_context()
prompt("""# Skill Update Prompt: `cpp-style`

- **Skill name:** cpp-style
- **Original skill path:** `D:/compute/.agents/skills/cpp-style/SKILL.md`
- **Output/save path:** `D:/compute/.kimix/cpp-style_update_prompt.md`

## Guideline

make minimal changes

## Goal

Review and refresh the `cpp-style` skill so it stays correct, complete, and aligned with the current LuisaCompute project's C++ style and tooling practices.

## Instructions

1. **Read the original skill**
   - Open `D:/compute/.agents/skills/cpp-style/SKILL.md`.
   - Read any bundled resources in the same directory (e.g., `.clang-format`, `.clang-tidy`, example snippets). Currently only `SKILL.md` exists; note if that changes.

2. **Review for correctness, completeness, and alignment**
   - Verify that the naming conventions, `.clang-format` rules, `.clang-tidy` checks, RTTI policy, and integer-type rules match current project usage.
   - Check for outdated instructions, broken or stale links, and examples that no longer compile or reflect project conventions.
   - Look for gaps: missing modern C++ guidance, missing project-specific macros, missing formatting edge cases, or missing verification steps.

3. **Identify issues**
   - List concrete problems: contradictions, obsolete checks, missing examples, missing `.clang-format`/`.clang-tidy` references, etc.

4. **Update the skill and bundled resources**
   - Edit `D:/compute/.agents/skills/cpp-style/SKILL.md` only as needed to fix issues and close gaps.
   - Add or update examples, resource links, and verification commands.
   - If `.clang-format` or `.clang-tidy` snippets belong in the skill directory, create or refresh them.

5. **Run domain-relevant verification**
   - Confirm that `.clang-format` and `.clang-tidy` files in the project root still align with the documented rules.
   - Validate any example C++ snippets by running:
     - `python scripts/check_cpp_syntax.py <example>.cpp` where applicable.
     - `xmake f -m debug -c` and `xmake build <target>` for a relevant target if the changes could affect build flags or style enforcement.
   - If a lint/format tool is referenced (e.g., `cpplint`, `clang-format`, `clang-tidy`), run it on a representative source file and confirm no unexpected regressions.

6. **Summarize changes**
   - After updating, write a brief change summary at the bottom of this prompt file or in a new note in `D:/compute/.kimix/cpp-style_update_summary.md`.
   - Include what was reviewed, what was changed, and the results of verification.

## Constraints

- Do not modify files outside the skill directory except for project-level `.clang-format` / `.clang-tidy` files if they are bundled with the skill.
- Do not break existing project style conventions.
- Keep the skill concise; add examples only where they clarify non-obvious rules.

## Definition of Done

- `D:/compute/.agents/skills/cpp-style/SKILL.md` and any bundled resources are up to date.
- All verification steps above have been run and passed.
- A change summary is recorded.
""")

# debug update prompt
clear_default_context()
prompt("""# Skill Update Prompt: debug

- **Skill name:** debug
- **Original skill path:** `D:/compute/.agents/skills/debug/SKILL.md`
- **Purpose:** Debug crashes and test failures via stack-traces, host/device logging, and DSL buffer inspection.

---

## Guideline

make minimal changes

## Objective

Review, update, and verify the `debug` skill for the LuisaCompute project. Make the skill accurate, complete, and aligned with current project practices, without changing unrelated files.

## Instructions

1. **Read the current skill.** Start from `D:/compute/.agents/skills/debug/SKILL.md` and inspect any bundled resources, code snippets, or linked files in the same directory.

2. **Review for quality.** Check the skill for:
   - **Correctness:** Are logging APIs (`LUISA_INFO`, `LUISA_VERBOSE`, `LUISA_VERBOSE_WITH_LOCATION`, `device_log`, `log_level_verbose`, etc.) and paths (`bin/debug/spirv_output.spvasm`, `bin/debug/hlsl_output.hlsl`) still valid?
   - **Completeness:** Does it cover common debug scenarios (CPU, CUDA, DirectX, Metal, LLVM/SPIR-V backends)? Are troubleshooting steps for hangs, wrong results, crashes, and backend compile errors sufficient?
   - **Alignment:** Does the advice match current LuisaCompute DSL/runtime conventions, build system (`xmake.lua`), and directory layout?
   - **Examples:** Are the C++ snippets syntactically correct and illustrative? Could additional examples help (e.g., `Printer` usage, capture of device callbacks, backend-specific env vars)?
   - **Broken links / stale references:** Verify any file names, environment variables, or external references.
   - **Gaps:** Note missing topics such as multi-device debugging, async stream errors, validation layers, shader dumping for SPIR-V/HLSL/DXIL, or Windows-specific crash diagnostics.

3. **Update the skill.** Edit `D:/compute/.agents/skills/debug/SKILL.md` (and bundled resources if needed) to fix outdated information, add missing examples, and improve clarity. Keep changes minimal and focused. Do not rewrite the skill from scratch unless necessary.

4. **Verify changes.** Run domain-relevant checks to confirm no regressions:
   - Validate any new or modified Python scripts with `python scripts/py_lint.py <file>`.
   - Validate any new or modified C++ examples with the clangd LSP checker (`python scripts/check_cpp_syntax.py <file>`) where applicable.
   - If you add buildable examples or touch build files, run `xmake f -m debug -c` and `xmake build <target>`.
   - If you add runnable examples, run `xmake run <target> <args>`.
   - Confirm the updated skill file parses cleanly as Markdown and retains its YAML front matter.

5. **Summarize changes.** After updating, produce a concise summary that lists:
   - What was reviewed.
   - Issues or gaps found.
   - Specific edits made.
   - Verification steps run and their results.

## Constraints

- Do **not** modify files outside the skill directory except where build/test verification requires it.
- Do **not** change unrelated skills or project code.
- Preserve the existing structure unless a reorganization materially improves clarity.

## Output

- Updated `D:/compute/.agents/skills/debug/SKILL.md` (and bundled resources, if any).
- A summary of changes and verification results.
""")

# glslang update prompt
clear_default_context()
prompt("""# Skill Update Prompt: glslang

**Skill name:** `glslang`  
**Original skill path:** `D:/compute/.agents/skills/glslang/SKILL.md`  
**Goal:** Review and, if necessary, update the `glslang` skill so it remains accurate, complete, and aligned with current project practices.

## Guideline

make minimal changes

## Context

The `glslang` skill documents the glslang SPIR-V Builder API used in this project. It is located in `src/ext/glslang/SPIRV` and is consumed by code that programmatically constructs SPIR-V modules via `spv::Builder`. The skill covers the builder lifecycle, types, constants, variables, functions, control flow, arithmetic/memory instructions, access chains, texture operations, decorations, barriers, debug info, serialization, IR classes, and GlslangToSpv traversal patterns.

## Task

1. **Review the original skill** at `D:/compute/.agents/skills/glslang/SKILL.md`.
2. **Assess correctness and completeness:**
   - Verify that API names, signatures, constants, and examples still match the current glslang source in `src/ext/glslang/SPIRV`.
   - Check for outdated instructions, deprecated functions, or missing newer capabilities (e.g., cooperative matrix/vector/tensor APIs, non-semantic debug info, untyped pointers).
   - Identify missing examples or common usage patterns, especially for texture calls, access chains, control flow, and debug info.
   - Check for broken links, stale file paths, or references to moved headers.
3. **Align with project practices:**
   - Compare the examples and recommendations with current project C++ style, naming conventions, and backend usage.
   - Ensure sample snippets are syntactically plausible and follow the conventions described in `D:/compute/.agents/skills/cpp-style/SKILL.md` and other relevant project skills.
4. **Update the skill:**
   - Edit `D:/compute/.agents/skills/glslang/SKILL.md` directly to fix issues, add missing examples, clarify ambiguous sections, and remove outdated content.
   - If additional bundled resources would help (e.g., small example files), place them under `D:/compute/.agents/skills/glslang/`.
   - Keep changes minimal and focused; preserve the existing structure unless a clear improvement justifies reorganization.
5. **Run verification:**
   - For any modified or added Python helper scripts, run `python scripts/py_lint.py <file>`.
   - For any modified or added C++ example files, run the C++ syntax checker (`python scripts/check_cpp_syntax.py <file>`) if applicable.
   - If the skill touches a buildable area, build the relevant xmake target with `xmake f -m debug -c` and `xmake build <target>`, then run any related tests with `xmake run <target>`.
   - Confirm that the project builds cleanly and no regressions are introduced.
6. **Summarize changes:**
   - Provide a concise summary listing what was updated, added, fixed, or removed.
   - Note any issues that could not be resolved and why.

## Constraints

- Do not modify files outside the skill directory and directly related verification/build files.
- Do not introduce unrelated refactors.
- Preserve the YAML front matter (`name` and `description`) unless the description itself is outdated.

## Output

- Updated `D:/compute/.agents/skills/glslang/SKILL.md` (and any new bundled resources, if needed).
- A short summary of changes, verification steps run, and their results.
""")

# hlsl update prompt
clear_default_context()
prompt("""# Skill Update Prompt: `hlsl`

- **Skill name:** hlsl
- **Original skill path:** `D:/compute/.agents/skills/hlsl/SKILL.md`
- **Scope:** HLSL code generation, `vstd::StringBuilder` patterns, builtin headers, DXIL embedding, and codegen debug utilities in LuisaCompute.

## Guideline

make minimal changes

## Objective

Review and update the `hlsl` skill so it remains correct, complete, and aligned with current project practices. Make minimal, targeted changes to the skill itself and to any bundled resources it references.

## Instructions

1. **Read the original skill** at `D:/compute/.agents/skills/hlsl/SKILL.md`.
2. **Review for correctness:**
   - Verify all code snippets compile or are consistent with the current API (e.g., `vstd::StringBuilder`, `luisa::format`, `vstd::to_string`, `lc_hlsl::get_hlsl_builtin`).
   - Confirm the include paths, macro names (`LC_HLSL_DECL_VARNAME`, `LC_HLSL_INSERT_VARNAME`), and builtin keys listed in the tables still exist in `src/backends/common/hlsl/builtin/`.
3. **Review for completeness and alignment:**
   - Check whether the StringBuilder, code-generation patterns, builtin headers, DXIL files, and debug sections cover how the DX and VK HLSL backends currently work.
   - Identify outdated information, missing examples, broken links, or gaps (e.g., new builtin keys, changed output-file naming, additional codegen utilities).
4. **Update the skill and bundled resources if needed:**
   - Edit `D:/compute/.agents/skills/hlsl/SKILL.md` only when you find verified issues.
   - Update any linked header/code examples inside the skill directory if they are bundled with the skill.
   - Preserve the existing structure and tone; make minimal changes.
5. **Run domain-relevant verification:**
   - For any modified Python helper scripts, run `python scripts/py_lint.py <file>`.
   - For any modified C++ snippets/files, run the project’s C++ lint/syntax checks (e.g., `python scripts/check_cpp_syntax.py <file>`) and, if applicable, `xmake build <target>` for the relevant HLSL backend target.
   - Confirm no regressions: the skill builds, referenced files exist, and documented commands still behave as described.
6. **Summarize changes:**
   - List what was reviewed, what was updated, and the verification results.
   - Note any items that could not be verified or require follow-up.

## Constraints

- Do NOT modify files outside the skill scope unless required for verification.
- Do NOT change unrelated project code.
- Keep the prompt actionable and concise.

## Output

Provide a brief summary of the review, the exact path to the updated `SKILL.md`, and the outcome of verification steps.
""")

# ir_pipeline update prompt
clear_default_context()
prompt("""# Skill Update Prompt: ir_pipeline

**Skill Path:** `D:/compute/.agents/skills/ir_pipeline/SKILL.md`
**Skill Description:** Legacy IR and XIR compiler pipeline, AST lowering, SSA IR, and optimization passes.

## Guideline

make minimal changes

## Task

Review and update the `ir_pipeline` skill so it remains accurate, complete, and aligned with current project practices in LuisaCompute.

## Instructions

1. **Read the current skill** at `D:/compute/.agents/skills/ir_pipeline/SKILL.md`. Do not modify it yet.
2. **Review for correctness and completeness:**
   - Verify the comparison between legacy IR and XIR still matches the current codebase.
   - Confirm pipeline flow, file paths (`src/ir/`, `src/xir/`, `src/rust/luisa_compute_ir/`, `include/luisa/xir/`, etc.), and class names are still accurate.
   - Check expression/statement mapping tables against `src/xir/translators/ast2xir.cpp`.
   - Validate the XIR value hierarchy and instruction-set descriptions against the current headers.
   - Confirm the optimization pass list under `src/xir/passes/` is complete and filenames/purposes are correct.
   - Look for outdated information, broken links, missing examples, or gaps (e.g., new passes, new instructions, metadata types, serialization changes).
3. **Update the skill:**
   - Edit `D:/compute/.agents/skills/ir_pipeline/SKILL.md` directly.
   - Make minimal, focused changes: fix errors, add missing passes/instructions, refresh examples, clarify explanations.
   - If bundled resources exist in `D:/compute/.agents/skills/ir_pipeline/`, update them consistently.
   - Preserve the existing structure and style unless it hinders clarity.
4. **Run domain-relevant verification:**
   - For any Python helper scripts changed, run `python scripts/py_lint.py <file>`.
   - For C++ sources referenced or touched, run `python scripts/check_cpp_syntax.py <file>` where appropriate.
   - Build and run relevant tests: use `xmake f -m debug -c`, `xmake build <target>`, and `xmake run <target>` for C++ targets.
   - Confirm no regressions (clean build, tests pass, lint clean).
5. **Summarize changes:**
   - At the end of your response, provide a concise summary of what was updated, verified, and any remaining concerns.

## Constraints

- Do not change files outside the skill directory unless required for verification.
- Do not execute the update work in this prompt; only perform it when acting as the coding agent.
- Keep the prompt actionable and specific.
""")

# lc_dsl update prompt
clear_default_context()
prompt("""# Skill Update Prompt: lc_dsl

- Skill name: lc_dsl
- Original skill path: `D:/compute/.agents/skills/lc_dsl/SKILL.md`
- Description: DSL kernels, callables, structs, buffers, atomics, control flow, and dispatch.

## Guideline

make minimal changes

## Objective

Review and update the `lc_dsl` skill so it remains correct, complete, and aligned with the current LuisaCompute project. Make minimal, focused changes to `SKILL.md` and any bundled resources it references. Do not modify unrelated files.

## Tasks

1. **Read and analyze** `D:/compute/.agents/skills/lc_dsl/SKILL.md`. Understand its purpose, audience, rules, examples, and linked resources.

2. **Review for correctness and completeness**
   - Verify all DSL APIs, headers, types, and macros match the current codebase (e.g., `<luisa/dsl/syntax.h>`, `<luisa/dsl/sugar.h>`, `<luisa/dsl/struct.h>`, `Kernel1D/2D/3D`, `Callable`, `LUISA_STRUCT`, `LUISA_TEMPLATE_STRUCT`, `Var<T>`, `BufferVar`, `Shared`, `Constant`, warp intrinsics, `sync_block`).
   - Check that code examples compile conceptually and reflect current DSL syntax.
   - Confirm the summary table matches the sections above it.

3. **Identify issues**
   - Outdated API names, signatures, or semantics.
   - Missing common patterns (e.g., ray tracing DSL, bindless resources, indirect dispatch, `Graph`, additional sugar macros).
   - Broken or stale links to test files or documentation.
   - Gaps in examples (e.g., no dispatch with explicit block size, missing atomic on struct member usage, missing `$for` with step).
   - Inconsistencies in naming, formatting, or style.

4. **Update SKILL.md**
   - Edit only `D:/compute/.agents/skills/lc_dsl/SKILL.md`.
   - Preserve the front-matter (`name`, `description`).
   - Add, remove, or rewrite sections/examples to fix issues while keeping the guide concise.
   - If examples become large, prefer short, self-contained snippets over full files.
   - Update the summary table if new features are added or old ones removed.

5. **Verify changes**
   - Run project-relevant checks on any modified code snippets or bundled scripts, such as:
     - `python scripts/py_lint.py <file>` for Python helpers (if any).
     - `python scripts/check_cpp_syntax.py <file>` for new or extracted C++ examples.
     - Build and run relevant tests (`xmake build <target>` / `xmake run <target>`) if you modify or add runnable examples. Prefer existing DSL test targets like `test_dsl`, `test_dsl_sugar`, `test_atomic`, or `test_warp`.
   - Confirm no regressions: existing tests and lint checks still pass.

6. **Summarize**
   - Produce a concise summary of what was reviewed, what changed, and what verification was run.
   - List any remaining gaps or follow-ups.

## Deliverables

- Updated `D:/compute/.agents/skills/lc_dsl/SKILL.md`.
- Summary of changes and verification results.

## Constraints

- Do not change files outside `D:/compute/.agents/skills/lc_dsl/`.
- Do not perform the update work unless instructed by the parent agent; this prompt file is the deliverable for now.
""")

# llvm-spirv update prompt
clear_default_context()
prompt("""# Skill Update Prompt: `llvm-spirv`

- **Skill name**: llvm-spirv
- **Original skill path**: `D:/compute/.agents/skills/llvm-spirv/SKILL.md`
- **Backend location**: `src/backends/common/spirv_llvm/`

## Guideline

make minimal changes

## Goal

Review and, if necessary, update the `llvm-spirv` skill so it remains correct, complete, and aligned with current project practices. Do not change unrelated files.

## Scope

1. **Read the current skill** at `D:/compute/.agents/skills/llvm-spirv/SKILL.md`.
2. **Review the backend source** in `src/backends/common/spirv_llvm/`:
   - `llvm_codegen_result.h`
   - `llvm_codegen_stack_data.h` / `.cpp`
   - `llvm_codegen_utility.h` / `.cpp`
   - `llvm_state_visitor.h` / `.cpp`
   - `spirv_llvm.cpp`
   - `xmake.lua`
3. **Compare the skill's claims** (file roles, pipeline, type mapping, variable naming, builtin codegen, SPIR-V emission, build config, pitfalls, LLVM API cheatsheet, Luisa AST reference) against the actual code.

## Tasks

1. **Correctness & completeness**
   - Verify that all line-number references still point to the described code.
   - Confirm the described pipeline (`Function` AST → `LLVMStateVisitor` → `llvm::Module` → legalization → SPIR-V binary) is still accurate.
   - Check that type mapping, variable naming, function codegen, and `CallOp` handling match the current implementation.
   - Ensure the LLVM API cheatsheet and Luisa AST reference reflect the APIs and enums actually used.

2. **Identify gaps / outdated content**
   - Missing recently added `CallOp` values, types, or variable tags.
   - Outdated file lists, struct members, or function signatures.
   - Broken or stale links; missing links to related skills (e.g., `glslang`, `hlsl`, `spv-opt`, `backend_architecture`).
   - Missing examples for common tasks (e.g., adding a new builtin, debugging a codegen crash, reading `llvm_ir_debug.ll`).
   - New pitfalls or build dependencies not documented.

3. **Update the skill**
   - Edit `D:/compute/.agents/skills/llvm-spirv/SKILL.md` directly.
   - Keep changes minimal and focused on accuracy.
   - Preserve the existing structure (front-matter, sections, tables, code snippets) unless reorganizing improves clarity.
   - If bundled resources (diagrams, helper scripts) exist under the skill directory, update them too.

4. **Verification**
   - Run `python scripts/check_cpp_syntax.py` on every modified `src/backends/common/spirv_llvm/*.cpp` and `*.h` file.
   - Configure and build the target: `xmake f -m debug -c && xmake build lc-spirv-llvm` (or the relevant xmake target name).
   - If any Python scripts were touched, run `python scripts/py_lint.py <file>`.
   - Confirm no build regressions and that the skill's line references and code snippets match the source after updates.

5. **Summarize changes**
   - Produce a short changelog listing what was updated, added, removed, or verified.
   - Note any issues that could not be resolved.

## Non-goals

- Do not modify backend code unless it is strictly necessary to fix a broken skill example or link; if you do, document it in the changelog.
- Do not change unrelated skills or project files.

## Output

- Updated `D:/compute/.agents/skills/llvm-spirv/SKILL.md` (and any bundled resources).
- A brief summary of changes and verification results.
""")

# lsp update prompt
clear_default_context()
prompt("""# Skill Update Prompt: `lsp`

- **Skill name:** `lsp`
- **Original skill path:** `D:/compute/.agents/skills/lsp/SKILL.md`
- **Bundled resources to review:**
  - `D:/compute/scripts/cpp_lsp_server.py`
  - `D:/compute/scripts/cpp_lsp_client.py`

## Guideline

make minimal changes

## Goal

Review, update, and verify the `lsp` skill and its bundled scripts so that the documentation stays correct, complete, and aligned with current project practices in LuisaCompute.

## Instructions

1. **Read the current skill file** at `D:/compute/.agents/skills/lsp/SKILL.md` and the scripts it references (`scripts/cpp_lsp_server.py`, `scripts/cpp_lsp_client.py`).
2. **Review for correctness and completeness:**
   - Verify that the described architecture, workflow, commands, flags, and defaults match the actual script implementations.
   - Identify outdated information, missing options, broken examples, or gaps (e.g., missing error-handling notes, missing authentication/CLI options, stale Python package requirements).
   - Check that the documented symbol actions match those supported by `cpp_lsp_client.py`.
   - Confirm the `compile_commands.json` generation examples (`xmake project -k compile_commands` / CMake) are still valid for the project.
3. **Update the skill and bundled resources if needed:**
   - Edit `D:/compute/.agents/skills/lsp/SKILL.md` only when necessary to fix errors or add missing guidance.
   - Fix bugs or inconsistencies in `scripts/cpp_lsp_server.py` and/or `scripts/cpp_lsp_client.py` if discovered.
   - Keep changes minimal and focused on the skill's scope.
4. **Run verification:**
   - Check Python syntax for both scripts:  
     ```bash
     python scripts/py_lint.py scripts/cpp_lsp_server.py
     python scripts/py_lint.py scripts/cpp_lsp_client.py
     ```
   - If possible, start the server on a free port and exercise `check`, `hover`, and `documentSymbol` commands against a small C++ source file. Confirm the client exit codes match the documented behavior.
   - Ensure no regressions in the project build or other skills.
5. **Summarize changes:** produce a concise summary of what was reviewed, what was updated, and the verification results.

## Constraints

- Do not modify files outside the scope of the `lsp` skill unless a bug in a dependent script directly affects the skill's accuracy.
- Do not delete the original skill file.
- Keep the prompt-driven nature of this work clear in your final report.

## Output

- Confirm the saved file path.
- Provide a brief summary of what the prompt covers and any changes made.
""")

# project_structure update prompt
clear_default_context()
prompt("""# Skill Update Prompt: project_structure

- **Skill name:** project_structure
- **Skill path:** `D:/compute/.agents/skills/project_structure/SKILL.md`
- **Output/report path:** `D:/compute/.kimix/project_structure_update_prompt.md`

## Guideline

make minimal changes

## Task

Review and update the `project_structure` skill for the LuisaCompute repository. The skill documents the project layout, module architecture, compiler pipeline, design patterns, and maintenance rules.

## Instructions

1. **Read the skill**
   - Open `D:/compute/.agents/skills/project_structure/SKILL.md`.
   - Understand its purpose, rules, examples, and any linked resources.

2. **Review for correctness and completeness**
   - Verify the top-level directory map and module descriptions match the current repository state in `D:/compute`.
   - Check that listed backends, build systems, compiler pipeline, key headers, design patterns, naming conventions, and maintenance rules are accurate.
   - Identify outdated information, missing modules or subdirectories, broken links, stale examples, or gaps.

3. **Update the skill and bundled resources**
   - If you find issues, update `D:/compute/.agents/skills/project_structure/SKILL.md` directly.
   - Keep changes minimal and focused.
   - Preserve the existing structure and style unless it is itself outdated.
   - If the skill references other bundled files, update those as well.

4. **Run domain-relevant verification**
   - For any Python helper scripts mentioned or changed, run `python scripts/py_lint.py <file>`.
   - If C++ references or headers are changed, use `python scripts/check_cpp_syntax.py <file>`.
   - If build-system examples are updated, run `xmake f -m debug -c` and `xmake build <target>` (or equivalent CMake checks) to confirm no regressions.
   - Confirm that all verification steps pass before finishing.

5. **Summarize changes**
   - Write a short summary of what was reviewed, what was changed, and what verification was run.
   - Save or append the summary to `D:/compute/.kimix/project_structure_update_prompt.md` (or report it if no changes were needed).

## Constraints

- Do NOT modify files outside of `D:/compute/.agents/skills/project_structure/` and this prompt/report file.
- Do NOT execute the actual project update work unless it is required to verify the skill's accuracy.
- Keep the prompt concise but specific enough to be actionable.
""")

# rust_workspace update prompt
clear_default_context()
prompt("""# Skill Update Prompt: `rust_workspace`

- **Skill name:** `rust_workspace`
- **Original skill path:** `D:/compute/.agents/skills/rust_workspace/SKILL.md`
- **Scope:** Review and, if necessary, update the skill documentation and any bundled resources so it stays correct, complete, and aligned with current project practices.
- **Constraint:** Do not modify files outside the skill directory except to run verification. Do not change the original `SKILL.md` unless updates are required; if changes are made, preserve the original structure and intent as much as possible.

## Guideline

make minimal changes

## Task

1. **Read the current skill** at `D:/compute/.agents/skills/rust_workspace/SKILL.md` and inspect any linked files or resources under `D:/compute/.agents/skills/rust_workspace/`.

2. **Review for quality and currency:**
   - Correctness of crate names, paths, and dependency graph under `src/rust/`.
   - Accuracy of IR data structures, transforms, backend responsibilities, and FFI conventions.
   - Completeness: missing crates, recent compiler transforms, backend features, or build/CI details.
   - Examples: are code snippets up-to-date and do they compile against the current code base?
   - Links: any broken or stale references to files, headers, or external documentation.
   - Gaps: common workflows (e.g., adding a transform, exposing a new FFI function, running Rust tests, debugging cbindgen output) that should be covered.

3. **Update the skill:**
   - Edit `D:/compute/.agents/skills/rust_workspace/SKILL.md` directly with fixes and additions.
   - Add or update bundled resources in the skill directory if examples, diagrams, or scripts are needed.
   - Keep the same general format (front matter, sections, tables, code blocks) unless a reorganization clearly improves clarity.
   - Make minimal, purposeful changes; do not rewrite for style alone.

4. **Run domain-relevant verification:**
   - `cd D:/compute/src/rust && cargo check` (with relevant features such as `cpu`, `remote` if applicable).
   - `cargo test` for the workspace, or for the crates the skill describes.
   - `cargo fmt --check` and `cargo clippy` (resolve or document any new warnings tied to the skill's content).
   - Verify `cbindgen` headers are still generated correctly if the skill mentions FFI header generation.
   - If the skill references C++ integration, build the relevant CMake target(s) and confirm no regressions.

5. **Confirm no regressions:** ensure builds, tests, and lints pass after any changes. If a check fails and cannot be fixed as part of this update, document the failure and the reason.

6. **Summarize changes:** produce a short changelog covering what was updated, added, removed, or verified, and note any remaining issues.

## Output

- Updated `D:/compute/.agents/skills/rust_workspace/SKILL.md` (if changes were needed).
- Any supporting resources under `D:/compute/.agents/skills/rust_workspace/`.
- A summary of changes and verification results.
""")

# skill-creator update prompt
clear_default_context()
prompt("""# Update Prompt: skill-creator

- **Skill name:** `skill-creator`
- **Original skill path:** `D:/kimi-test/kimi-cli/src/kimi_cli/skills/skill-creator/SKILL.md`
- **Skill bundle path:** `D:/kimi-test/kimi-cli/src/kimi_cli/skills/skill-creator/`

## Guideline

make minimal changes

## Goal

Review and, if necessary, update the `skill-creator` skill so it remains correct, complete, and aligned with current project conventions for creating and maintaining Kimi skills.

## Instructions

1. **Read the current skill.** Open `D:/kimi-test/kimi-cli/src/kimi_cli/skills/skill-creator/SKILL.md` and inspect the full bundle directory for any linked resources (scripts, references, assets).

2. **Evaluate for quality and freshness.** Check for:
   - Outdated advice, broken external/internal links, or stale examples.
   - Missing guidance on current project practices (e.g., discovery paths, packaging, naming, frontmatter).
   - Gaps in the creation workflow, testing, or validation steps.
   - Inconsistencies between the skill body and any bundled resources.
   - Opportunities to make instructions more concise or actionable (per the skill's own "Concise is Key" principle).

3. **Update the skill and bundled resources.**
   - Edit `SKILL.md` to fix issues and add missing content. Keep the file lean; move large reference material into `references/` if needed.
   - If you add or update bundled resources, ensure they are deterministic, tested, and referenced clearly from `SKILL.md`.
   - Do **not** change the skill name unless it is inconsistent with the folder name or project conventions.

4. **Run domain-relevant verification.**
   - Validate YAML frontmatter (name, description present and well-formed).
   - If any scripts were added or modified, run syntax checks and execute representative tests.
   - Confirm the skill directory structure matches the documented anatomy and contains no forbidden auxiliary files (README.md, CHANGELOG.md, etc.).
   - Run any project-level tests or linting that apply to skill content.

5. **Confirm no regressions.** Verify that the updated skill still packages correctly and that no other project files were altered.

6. **Summarize changes.** Report:
   - What was reviewed.
   - Issues found.
   - Specific edits made (with file paths).
   - Verification steps run and their outcomes.

## Constraints

- Do **not** modify files outside the `skill-creator` skill bundle.
- Do **not** delete the original skill path metadata.
- Keep the prompt's own guidance concise and actionable.
""")

# spv-opt update prompt
clear_default_context()
prompt("""---
Skill: spv-opt
Original Path: D:/compute/.agents/skills/spv-opt/SKILL.md
---

# Update Prompt: spv-opt

You are updating the `spv-opt` skill, which guides developers in writing optimizer passes for SPIRV-Tools under `src/ext/SPIRV-Tools`. The skill covers `Pass`/`MemPass` skeletons, IR manipulation APIs (`Module`, `IRContext`, `Instruction`, `BasicBlock`, `Function`), `InstructionBuilder`, def-use and CFG helpers, testing with `PassTest`, pass registration, analysis invalidation, and the optimizer C++ API.

## Guideline

make minimal changes

## Scope

Review, correct, and improve the skill documented at:

- **D:/compute/.agents/skills/spv-opt/SKILL.md**

Do not modify unrelated files. If bundled resources exist in `D:/compute/.agents/skills/spv-opt/`, review and update those too.

## Tasks

1. **Review for correctness**
   - Verify all C++ snippets compile against the current SPIRV-Tools headers (e.g., `source/opt/pass.h`, `source/opt/ir_builder.h`, `source/opt/pass_fixture.h`, `include/spirv-tools/optimizer.hpp`).
   - Confirm class names, method signatures, analysis bit names, and fixture helper signatures match the current codebase.
   - Ensure the `name()` / CLI flag convention and `Process()` return semantics are accurate.

2. **Check completeness**
   - Cover the full pass lifecycle: header, implementation, registration, factory declaration, CLI flag dispatch, testing, and analysis preservation.
   - Add or expand examples for common tasks if missing (e.g., iterating instructions safely, creating new instructions, replacing uses, killing instructions, handling phi nodes, preserving analyses, writing match tests with `CHECK:`).

3. **Identify outdated information**
   - Look for deprecated APIs, moved headers, renamed methods, or stale analysis bits.
   - Check that paths like `source/opt/`, `test/opt/`, and `include/spirv-tools/optimizer.hpp` still apply.
   - Verify built-in recipes and environment constants are current.

4. **Fix broken links or references**
   - Ensure all file paths, header names, and fixture helper signatures are accurate.
   - If examples reference real files, confirm they exist.

5. **Run domain-relevant verification**
   - Use `python scripts/check_cpp_syntax.py <file>` for any new or updated C++ example files.
   - If you add or change Python scripts, run `python scripts/py_lint.py <py_file>`.
   - Build and run SPIRV-Tools optimizer tests that exercise the patterns described (e.g., a minimal pass test) to confirm no regressions.
   - Record the commands run and their results.

6. **Summarize changes**
   - Provide a concise summary listing what was updated, added, or removed, and why.

## Constraints

- Do not modify files outside `D:/compute/.agents/skills/spv-opt/`.
- Do not break existing project tests.
- Keep the skill concise and actionable.

## Deliverables

- Updated `D:/compute/.agents/skills/spv-opt/SKILL.md`.
- Any new bundled resources in `D:/compute/.agents/skills/spv-opt/`.
- A short summary of changes and verification results.
""")

# test update prompt
clear_default_context()
prompt("""# Skill Update Prompt: `test`

- **Skill name:** test
- **Original skill path:** `D:/compute/.agents/skills/test/SKILL.md`
- **Goal:** Review, update, and verify the `test` skill so it remains correct, complete, and aligned with current LuisaCompute project practices.

---

## Guideline

make minimal changes

## Background

The `test` skill documents how LuisaCompute tests are authored, organized, built, and run. It covers:

- Test directory layout under `src/tests/` (`unit/core/`, `unit/ext/`, `unit/ast/`, `unit/dsl/`, `unit/runtime/`, `unit/xir/`, `integration/runtime/`, `integration/ir/`, `common/`, `python/`, `cxx_shaders/`).
- Adding tests via CMake (`luisa_compute_add_test`) and xmake (`test_proj`).
- Mirroring `examples/` as test executables with `MIRROR_AS_TEST`.
- C++ test templates for no-device unit tests and device-needed tests.
- Style conventions: include order, using declarations, naming, assertions, `LUISA_STRUCT`, file headers, main shapes.
- Device helpers from `common/test_device.h`.
- Running tests under CMake and xmake.
- Reference image comparison rules and "what not to do" guidance.

## Your Task

1. **Read and review** `D:/compute/.agents/skills/test/SKILL.md`.
2. **Check correctness and alignment** with the current project:
   - Directory layout still matches `src/tests/`.
   - CMake/xmake helpers and signatures are still accurate.
   - Templates compile with current headers and Boost.UT API.
   - Naming conventions and include paths still match build system setup.
   - Backend names, CLI flags, and reference comparison behavior are current.
3. **Identify gaps or outdated information**:
   - Missing test categories or new test directories.
   - Missing helper functions or changed `test_device.h` APIs.
   - New assertion patterns, DSL features, or XIR conventions not reflected.
   - Broken or stale links (e.g., Boost.UT URL, internal paths).
   - Missing guidance for Python tests, Rust IR tests, or GUI-dependent tests.
   - Inconsistent examples or contradictory style advice.
4. **Update the skill**:
   - Edit `D:/compute/.agents/skills/test/SKILL.md` directly.
   - Add, remove, or rewrite sections to fix issues and close gaps.
   - Keep the existing structure unless a reorganization materially improves clarity.
   - If any bundled resources referenced by the skill need updates, update those too.
   - Make minimal, targeted changes; do not rewrite for style alone.
5. **Verify your changes**:
   - Run `python scripts/py_lint.py <any_changed_py_file>` if Python files were modified.
   - For C++ snippets or templates, build a representative test target with xmake or CMake to confirm the patterns compile:
     - `xmake f -m debug -c`
     - `xmake build <target>`
     - `xmake run <target> <args>`
   - Use `python scripts/check_cpp_syntax.py <file>` where applicable.
   - Confirm CTest labels and xmake test group behavior still match the documented rules.
   - Ensure no existing tests or build files are broken by skill-only changes (if you only edited SKILL.md, confirm the markdown renders correctly and links resolve).
6. **Summarize changes**:
   - List what was outdated, missing, or wrong.
   - List what you updated, added, or removed.
   - Report verification steps run and their results.

## Constraints

- Do **not** modify project source code, build files, or reference images unless the skill itself explicitly requires a bundled resource update.
- Do **not** delete the original skill or move it.
- Keep the prompt file at `D:/compute/.kimix/test_update_prompt.md` untouched.
- Preserve the YAML front matter (`name`, `description`) unless the description is no longer accurate.

## Output

When finished, report:

1. The absolute path of the updated skill file.
2. A concise summary of changes made.
3. The verification commands run and whether they passed.
4. Any remaining concerns or follow-up work you could not resolve.
""")

# xir_passes update prompt
clear_default_context()
prompt("""# Update Prompt: `xir_passes`

- **Skill name:** xir_passes
- **Original skill path:** `D:/compute/.agents/skills/xir_passes/SKILL.md`
- **Scope:** Authoring XIR transformation passes under `src/xir/passes/` and `include/luisa/xir/passes/`.

## Guideline

make minimal changes

## Task

Review and update the `xir_passes` skill so it remains correct, complete, and aligned with current project practices. Do not modify any other files.

## Instructions

1. **Read the current skill.** Open `D:/compute/.agents/skills/xir_passes/SKILL.md` and any resources it links to (e.g., `src/xir/passes/CFG_NORMALIZATION_PLAN.md`).

2. **Review for correctness and completeness.** Check the following:
   - Code examples compile against the current XIR API (`Function::function_list`, `BasicBlock::instructions`, `XIRBuilder` methods, terminator APIs, `isa<T>` + `static_cast` patterns, etc.).
   - API names, member names, and header paths match the current codebase.
   - The pass registration layout (`src/xir/CMakeLists.txt`, `src/tests/CMakeLists.txt`) is still accurate.
   - The Pipeline B pass status table reflects the current implementation.
   - Pitfalls still apply; remove any that are no longer true or add new ones you encounter.
   - Build/test commands use the project's current convention (`cmake-build-release` or whatever is now standard).
   - Linked resources exist and are reachable.
   - Missing examples (e.g., for newer passes, tests, or common mutation patterns) are identified.

3. **Identify gaps and outdated information.** Look for:
   - Broken or stale links.
   - Passes mentioned in the LLVM equivalents table that no longer exist or have been renamed.
   - Outdated hook rules or project conventions.
   - New passes under `src/xir/passes/` that are not documented.

4. **Update the skill.** Make minimal, targeted changes to `D:/compute/.agents/skills/xir_passes/SKILL.md` (and only bundled resources if explicitly needed). Preserve the existing structure and tone. Add working examples where gaps were found.

5. **Run domain-relevant verification.** After any edits, run at least:
   - C++ syntax / LSP checks on any new or modified code examples: `python scripts/check_cpp_syntax.py <relevant_file>`.
   - The XIR library build: `cmake --build cmake-build-release --target luisa-compute-xir -j` (or the current equivalent target).
   - Any affected XIR pass unit tests: `cmake --build cmake-build-release --target test_xir_pass_<name> -j` and execute the resulting binary, or use `ctest --test-dir cmake-build-release -R xir_pass --output-on-failure`.
   - Confirm no regressions in existing tests.

6. **Summarize changes.** In your final response, list what was updated, added, removed, or verified, and note any issues that remain unresolved or require follow-up.

## Constraints

- Do NOT modify files outside the skill directory (`D:/compute/.agents/skills/xir_passes/`).
- Do NOT execute broad unrelated changes.
- Keep the prompt's original intent: a practical guide for XIR pass authors.

## Output

Confirm the saved file path and provide a brief summary of the review/update results.
""")

# xmake update prompt
clear_default_context()
prompt("""# Skill Update Prompt: xmake

- **Skill name:** xmake
- **Original skill path:** `D:/compute/.agents/skills/xmake/SKILL.md`
- **Goal:** Review and update the XMake skill for correctness, completeness, and alignment with current LuisaCompute project practices.

## Guideline

make minimal changes

## Instructions

1. **Read the current skill** at `D:/compute/.agents/skills/xmake/SKILL.md` and any resources it links to (e.g., test scripts under `scripts/test/xmake/`, `xmake.lua` in the repository root).

2. **Review for quality and accuracy:**
   - Verify all version requirements (e.g., XMake 3.0.6+), platform commands, and option defaults are still correct.
   - Check that listed project options match the actual options defined in `xmake.lua`.
   - Confirm build examples still work and reflect recommended workflows.
   - Validate test script names, paths, and backend lists.
   - Look for broken links, outdated commands, missing options, or obsolete advice.
   - Identify gaps in examples (e.g., macOS, cross-compilation, backend-specific configuration, install/packaging).

3. **Update the skill** (`D:/compute/.agents/skills/xmake/SKILL.md`) and any bundled resources if needed:
   - Fix incorrect facts, commands, or defaults.
   - Add missing examples or sections that improve usability.
   - Remove obsolete content.
   - Keep formatting consistent with the existing file style.

4. **Run domain-relevant verification:**
   - Validate any changed Lua snippets or `xmake.lua` examples by running `xmake` configuration/build commands where applicable.
   - Run Python syntax checks on any updated test helper scripts using `python scripts/py_lint.py <file>`.
   - If the skill references C++ build rules, run `xmake f -m debug -c` and `xmake build <target>` for a representative target to confirm no regressions.
   - For modified C++ files referenced by the skill, run `python scripts/check_cpp_syntax.py <file>`.

5. **Confirm no regressions:** Ensure the skill still loads cleanly, examples are syntactically valid, and verification commands succeed.

6. **Summarize changes:** Provide a concise changelog covering what was reviewed, what was updated, what verification was run, and the outcome.

## Constraints

- Do not modify files outside the skill directory unless required for verification or to fix a referenced bundled resource.
- Do not delete the original skill file.
- Keep the prompt scope focused on XMake build system documentation and supporting resources.
""")

# yyjson update prompt
clear_default_context()
prompt("""---
skill: yyjson
path: D:/compute/.agents/skills/yyjson/SKILL.md
---

# yyjson Skill Update Prompt

You are asked to review and, if necessary, update the yyjson skill file at `D:/compute/.agents/skills/yyjson/SKILL.md`.

## Guideline

make minimal changes

## Background

This skill documents the project's use of ibireme's yyjson library for high-performance C JSON parsing, creation, and modification. It covers both the immutable (`yyjson_doc`/`yyjson_val`) and mutable (`yyjson_mut_doc`/`yyjson_mut_val`) APIs, reading/writing, iteration, JSON Pointer/Patch, memory allocators, compile-time flags, null safety, and a common read-modify-write pattern.

## Task

1. **Review for correctness**
   - Read `D:/compute/.agents/skills/yyjson/SKILL.md` in full.
   - Verify that every API name, signature snippet, flag name, macro, and code example is accurate with respect to the version of yyjson used by the project.
   - Check for typos, mismatched parentheses, incorrect return types, or stale function names.

2. **Check completeness and alignment**
   - Compare the skill against current project practices (search the codebase for `yyjson_` usage, `yyjson.h`, and any bundled third-party copies).
   - Identify missing common patterns (e.g., error handling, allocator lifetimes, NDJSON, number formatting, large file handling) that project code actually relies on.
   - Note any project-specific conventions (naming, allocator choice, error logging) that should be reflected in the skill.

3. **Identify outdated information, broken links, or gaps**
   - If the skill references external documentation, ensure links are still valid and point to the correct version.
   - Look for contradictions with the installed yyjson headers or source in the project.
   - Flag sections that are unclear or lack context for a coding agent.

4. **Update SKILL.md and bundled resources if needed**
   - Make minimal, focused edits to `D:/compute/.agents/skills/yyjson/SKILL.md`.
   - Add, remove, or revise examples to match real project usage.
   - If the skill directory contains additional resources (headers, scripts, tests), update those too only as needed.

5. **Run domain-relevant verification**
   - If you add or change C code examples, compile them or use available project tooling to confirm they build.
   - Run any relevant tests that exercise yyjson in the project.
   - Run lint or syntax checks where applicable (e.g., `python scripts/py_lint.py` for Python helpers, or the C++ lint tooling for any C/C++ code touched).
   - Confirm no regressions: existing tests and builds should still pass.

6. **Summarize changes**
   - Produce a short summary of what was reviewed, what was changed, and why.
   - List any issues found that were *not* fixed, with a brief reason.

## Constraints

- Do not change files outside the `D:/compute/.agents/skills/yyjson/` directory unless a project test/verification step requires it.
- Keep the skill concise and actionable for a coding agent; avoid unnecessary verbosity.
- Preserve the existing front matter (`name`, `description`) unless the purpose of the skill itself has changed.

## Output

When finished, report:
- The full path of the updated skill: `D:/compute/.agents/skills/yyjson/SKILL.md`
- A brief summary of what the prompt covers and which areas, if any, were updated.
""")

