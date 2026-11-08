---
name: xmake
description: XMake 3.1.2 CLI, Lua APIs, and LuisaCompute build configuration; use for configuring, building, testing, debugging, or writing targets, rules, toolchains, packages, and scripts.
---

# XMake Build System

> Documentation baseline: the English reference in `D:\xmake-docs\docs`, synchronized through XMake 3.1.2. For API details, read the file named in the appended [XMake API Reference](#xmake-api-reference-xmake-312); use `D:\xmake` source or `xmake <command> -h` only to resolve version/source drift.

Primary build system. Requires XMake 3.0.6+ (`set_xmakever("3.0.6")` in root `xmake.lua`). Optional: CUDA Toolkit, LLVM 22 (`--lc_llvm_path=`, used by `lc_enable_clangcxx`/`lc_tvm_llvm`), and Embree (`--lc_embree_path=`). The DX/VK shader-compiler SDKs are downloaded automatically into `SDKs/` by the `lc_backend_sdk` target (`scripts/find_sdk.lua`; override the directory with `--lc_sdk_dir=`, an empty string disables installation).

## Quick Start

```bash
xmake f -m debug -c -y
# explicit plat/arch (the form AGENTS.md prescribes):
xmake f -p windows -a x64 -m debug -c
xmake build              # build all default targets
xmake build <target>     # one target, e.g. lc-runtime, test_dsl, example_path_tracing
xmake run <target> <args>
# Update compile_commands.json:
xmake project -k compile_commands --lsp=clangd .vscode
```

## Configuration

| Platform | Command |
|---|---|
| Linux GCC | `xmake f -p linux -a x86_64 --toolchain=gcc -m release -c` |
| Linux Clang | `xmake f -p linux -a x86_64 --toolchain=clang -m release -c` |
| Windows MSVC | `xmake f -p windows -a x64 --toolchain=msvc -m release -c` |
| Windows Clang-CL | `xmake f -p windows -a x64 --toolchain=clang-cl -m release -c` |
| Windows LLVM | `xmake f -p windows -a x64 --toolchain=llvm -m release -c` |
| macOS Clang | `xmake f -p macosx -a arm64 --toolchain=clang -m release -c` |

### Flags
`-c` cleans cached user configs and the detection cache; `-m <mode>` selects `release`/`debug`/`releasedbg` (the only modes this project declares); `-p <plat>` selects `linux`/`windows`/`macosx`/`android` here (XMake itself supports more platforms); `-a <arch>` accepts `x86_64`/`x64`/`arm64` in `_lc_check_env`; `--check` drops only the detection cache and re-checks while keeping user configs; `-y` auto-accepts prompts. Use `-o <dir>`/`--builddir=<dir>` for XMake's build directory (`--buildir` is deprecated), independently of this project's `lc_bin_dir`.

`debug` does **not** enable sanitizers here: xmake's `mode.debug` rule only sets `symbols = "debug"` and `optimize = "none"` (`-G` for CUDA), and this project adds `/GS /Gd` plus the debug CRT (`MDd`) in `lc_basic_settings`. To build with ASan, set the policy explicitly: `xmake f -m debug --policies=build.sanitizer.address -c`.

### Project Options

Every option is declared as `option("lc_*")` in root `xmake.lua` — including the internal `lc_scripts_path` and `lc_ext_path` (defaults below are the `{default = ...}` values there); `scripts/xmake_func.lua` adds the underscore-prefixed `_lc_check_env`, `_lc_bin_dir` and `_lc_enable_py` options. There are no other valid option names — pass them as `--lc_name=value`.

| Option | Default | Meaning |
|---|---|---|
| `lc_dx_backend`, `lc_vk_backend`, `lc_cuda_backend`, `lc_metal_backend` | true | GPU backends → targets `lc-backend-dx`, `lc-backend-vk`, `lc-backend-cuda`, `lc-backend-metal` |
| `lc_fallback_backend` | false | CPU fallback backend `lc-backend-fallback`; needs `lc_llvm_path` + `lc_embree_path` |
| `lc_enable_xir` | true | XIR support; also pulls in `lc-coro` (`src/xmake.lua`) and gates `lc_vk_backend_use_xir_spirv` |
| `lc_vk_backend_use_xir_spirv` | false | native XIR→SPIR-V codegen for Vulkan (the CMake side, `LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV`, defaults ON) |
| `lc_vk_backend_use_ast_llvm_spirv` | false | experimental AST→LLVM→SPIR-V route; requires `lc_llvm_path`, mutually exclusive with the row above |
| `lc_vk_backend_enable_dxc_compatibility` | true | keep the legacy runtime DXC route for unsupported Vulkan shader features |
| `lc_vk_min_api_version` | `"auto"` | Vulkan API floor for device negotiation (`auto`, `1.2`, `1.3`) |
| `lc_vk_bindless_heap_capacity` | 262144 | requested update-after-bind heap capacity (0 disables bindless) |
| `lc_cuda_ext_lcub` | false | CUDA CUB extension (`lc-compute-cuda-ext-lcub`), off by default due to compile time |
| `lc_dx_cuda_interop`, `lc_vk_cuda_interop` | false | auto-enabled when both sides of the interop are enabled |
| `lc_enable_dsl`, `lc_enable_gui`, `lc_enable_imgui`, `lc_enable_osl`, `lc_enable_py`, `lc_enable_tests` | true | module switches; `lc_enable_clangcxx` defaults to false |
| `lc_enable_simd` | true | `avx`+`avx2` (x64) / `neon` vectorexts applied by `lc_basic_settings` |
| `lc_enable_unity_build`, `lc_enable_pch` | true | compile-speed rules (`c/c++.unity_build`, `lc_set_pcxxheader`) |
| `lc_check_unity_build` | false | adds the phony target `lc-check-unity-build` as a dependency of `lc-core`, so every build runs `scripts/check_unity_build_conflicts.py` and fails when two files of one unity batch declare the same file-local symbol |
| `lc_enable_mimalloc` | true | switches bundled EASTL (and SPIRV-Tools) onto the mimalloc override (`src/ext/xmake.lua`) |
| `lc_enable_custom_malloc` | false | custom allocator instead of the system allocator |
| `lc_external_marl` | false | external marl instead of the bundled `src/ext/marl` |
| `lc_use_lto` | false | declared in `xmake.lua`, but `lc_basic_settings` reads the **`lto`** key (`_get_or("lto")` → `lc_lto`), so `--lc_use_lto=true` alone changes nothing; set LTO per target with `_config_project({lto = true})`, which applies the `build.optimization.lto` policy |
| `lc_rtti`, `lc_safe_mode` | false | RTTI (off: `-fno-rtti` / `/GR-`), runtime safe mode |
| `lc_cxx_standard`, `lc_c_standard` | `'cxx20'`, `'clatest'` | language standards applied by `lc_basic_settings` |
| `lc_bin_dir` | `"bin"` | output directory root; the mode is appended (`bin/debug`) |
| `lc_win_runtime`, `lc_optimize` | false | override the Windows CRT (`MD`/`MT`/…) and the `optimize` setting |
| `lc_toolchain` | false | toolchain applied to every target (`target:set("toolchains", ...)`) |
| `lc_sdk_dir` | false | SDK download directory; an empty string disables SDK installation |
| `lc_llvm_path`, `lc_embree_path`, `lc_ndk` | false | external LLVM / Embree / Android NDK locations |
| `lc_py_include`, `lc_py_linkdir`, `lc_py_libs` | false | manual Python binding locations (needed for `lc_enable_py` to take effect) |
| `lc_use_system_stl` | false | defines `LUISA_USE_SYSTEM_STL` (system STL instead of bundled EASTL) |
| `lc_spdlog_use_xrepo`, `lc_reproc_use_xrepo`, `lc_lmdb_use_xrepo`, `lc_imgui_use_xrepo`, `lc_glfw_use_xrepo`, `lc_yyjson_use_xrepo` | false | use xmake-repo packages instead of the bundled `src/ext/*` submodules |
| `lc_disable_win_message_box` | true | defines `LUISA_DISABLE_WIN_MESSAGE_BOX` (Windows, debug mode) |
| `lc_tile_tirx_bridge`, `lc_tvm_stack`, `lc_tvm_llvm` | false | TileIR→TVM TIRx bridge, bundled TVM stack, TVM LLVM codegen |

`scripts/options.lua` holds the per-machine defaults: it assigns `lc_options = { toolchain = "...", lc_<name> = <value>, ... }` and root `xmake.lua` applies each entry with `set_config(k, v)` before the targets are loaded, so command-line `--lc_*=` values win over the file. Generate a starter file with `xmake lua scripts/write_options.lua`.

### Internal options and target names

`_lc_check_env` (`scripts/xmake_func.lua`) validates `-a`/`-m` and gates the `includes("src")`/`examples`/`tutorials` at the bottom of root `xmake.lua`; `_lc_bin_dir` collects the mode-suffixed output directory. Library/target names to build against are `lc-core`, `lc-runtime`, `lc-dsl`, `lc-vstl`, `lc-tile`, `lc-gui`, `lc-coro`, `lc-osl`, `lc-clangcxx`, `lcapi`, the `lc-backend-*` targets, `lc-backends-dummy` (phony, depends on the enabled backends) and `lc-validation-layer`.

## Sanitizer Modes

XMake supports sanitizer builds through sanitizer policies. Policies propagate the sanitizer configuration to dependent packages and avoid the deprecation warnings produced by legacy sanitizer mode rules such as `mode.asan`, `mode.tsan`, and `mode.ubsan`.

### ASan in debug mode

`-m debug` on its own produces an unsanitized build (xmake's `mode.debug` rule only sets `symbols`/`optimize`), and the legacy `-m asan` mode is rejected here because `_lc_check_env` accepts only `debug`/`release`/`releasedbg` (`scripts/xmake_func.lua`). Combine debug with the address-sanitizer policy instead:

```bash
xmake f -m debug --policies=build.sanitizer.address -c -y
xmake build
xmake run <target>
```

### Enable via policy manually

To enable a sanitizer for a different mode, use the corresponding policy. In `xmake.lua`:

```lua
set_policy("build.sanitizer.address", true)
```

Or from the command line:

```bash
xmake f --policies=build.sanitizer.address -c -y
xmake build
xmake run <target>
```

Available policies:

| Policy | Sanitizer |
|---|---|
| `build.sanitizer.address` | AddressSanitizer |
| `build.sanitizer.thread` | ThreadSanitizer |
| `build.sanitizer.memory` | MemorySanitizer |
| `build.sanitizer.leak` | LeakSanitizer |
| `build.sanitizer.undefined` | UndefinedBehaviorSanitizer |

Multiple sanitizers can be combined, e.g.:

```bash
xmake f --policies=build.sanitizer.address,build.sanitizer.undefined -c -y
```

A bare policy name enables it, `name:n` disables it, and `name:value` sets a value; separate entries with commas. List supported policies with `xmake show -l policies` or `xmake l core.project.policy.policies`.

## Commands

| Command | Description |
|---|---|
| `xmake build [target ...]` | Build default targets or one/more named targets (multiple targets require XMake 3.1.0+) |
| `xmake -r [target]` / `xmake build -r [target]` | Rebuild |
| `xmake -a` / `xmake build -a` | Build all targets, not only default ones (`-a` means `--arch` under `xmake f`) |
| `xmake build -g <pattern>` / `--files=<pattern>` | Build a target group / selected source files |
| `xmake run <target> [args]` | Run a target; `-w <dir>` sets its working directory and `-d` invokes the debugger |
| `xmake clean [target ...]` | Clean targets |
| `xmake show -l targets` | List all targets; also lists `rules`, `policies`, `toolchains`, `buildmodes`, `platforms`, `architectures`, `apis`, `envs`, `themes`, or `packages` |
| `xmake show -t <target> --format=json` | Inspect a target |
| `xmake show --info=depgraph --target=<target> --format=<plain,json,dot>` | Inspect dependencies (3.0.9+) |
| `xmake test [target/test or pattern] [-g group] [--output-on-failure]` | Run tests declared with `add_tests`; LuisaCompute's plain `test_*` binaries still use `xmake run` |
| `xmake check [api, clang.tidy, syntax, or cuda.devlink]` | Run built-in checkers; `--list` lists them |
| `xmake watch` | Watch sources and rebuild/run; see `xmake watch -h` |
| `xmake lua <script.lua>` / `xmake lua -c '<code>'` | Run a script or snippet in the XMake sandbox |
| `xmake install -o <dir>` | Install to `<dir>`; `--headers=n`, `--bindir=`, `--libdir=`, and `--includedir=` control content/layout |
| `xmake project -k compile_commands --lsp=clangd .vscode` | Generate `compile_commands.json` (`--lsp` also accepts `cpptools`/`ccls`) |
| `xmake f --menu` / `--export=<file>` / `--import=<file>` | Interactive configuration / export or import configuration |
| `xmake addon --list` | List installed addons (3.1.1+; replaces the deprecated plugin manager) |
| `xmake -y` | Auto-accept prompts and skip interaction |

## Common Issues

- `-v`/`--verbose` (verbose user output) and `-D`/`--diagnosis` (developer diagnostics) are both valid xmake flags and can be combined (`xmake -vD`); `xmake --verbose` is simply the long form of `-v`
- Boolean options: `--lc_option=true`/`=false`; value options: `--lc_option=<value>` (e.g. `--lc_cxx_standard=cxx23`, `--lc_bin_dir=out`)
- Use `-c` to clean cache when reconfiguring with different options
- Use `-y` to auto-accept all prompts and skip interaction — essential in automated scripts and CI pipelines
- `lc_fallback_backend` requires both `lc_llvm_path` and `lc_embree_path`
- `lc_dx_backend` is silently disabled on non-Windows platforms
- `lc_metal_backend` is silently disabled on non-macOS platforms
- `lc_cuda_backend` is silently disabled outside Windows/Linux
- Unity (jumbo) build collision (`redefinition of ...`, `use of undeclared identifier`, ambiguous name): XMake merges translation units in each `batchsize` batch (LuisaCompute's `_config_project{batch_size = N}` maps to that rule option). Prefer built-in controls where possible: rule `{uniqueid = "MY_UNITY_ID"}` isolates identifiers, per-file `{unity_group = "name"}` groups files, and `{unity_ignored = true}` excludes a file. The project checker `python scripts/check_unity_build_conflicts.py` (or `xmake build lc-check-unity-build` with `--lc_check_unity_build=true`) finds collisions; `scripts/unity_isolate_locals.py` can move helpers into file-unique namespaces when rule controls are insufficient.
- PCH error like `has been modified since the precompiled header` / `redefinition of ...`: use `xmake build -r <target>` to rebuild; if stale compiler-cache output is involved, reconfigure with `xmake f --ccache=n -c`. XMake 3.1.2 also fixes clang-cl PCH dependency tracking and disables compiler caching while creating MSVC PCH artifacts. PCH is emitted by `lc_set_pcxxheader()` only while `lc_enable_pch` is on.
- The root project sets `check.auto_ignore_flags=false`, so unsupported flags are errors rather than being silently discarded. Use a valid tool filter or fix the flag; do not add `{force=true}` unless bypassing detection is intentional.
- `build.ccache` controls XMake's compiler cache (`xmake f --ccache=n` disables it). In release/releasedbg, `build.release.strip` defaults on and may strip symbols unless `set_strip()` was explicit; disable it with `set_policy("build.release.strip", false)` when needed. With clang-cl, LTO selects `lld-link`; `clang-cl[llvm]` explicitly selects the full LLVM toolset.
- `lc_vk_backend_use_xir_spirv` and `lc_vk_backend_use_ast_llvm_spirv` are mutually exclusive: enabling both raises at configure time (root `xmake.lua`). `lc_vk_backend_use_ast_llvm_spirv` also requires `lc_llvm_path` and force-disables the XIR→SPIR-V option (`scripts/xmake_func.lua`).
- `lc_enable_tests=false` makes the `test_*`, `benchmark_*`, `example_*` and `tutorial_*` targets disappear — `src/tests`, `examples` and `tutorials` are only `includes()`-ed when it is enabled (`src/xmake.lua`, root `xmake.lua`). They are created by the local helpers `test_proj(...)`, `example_proj(...)` and `tutorial_proj(...)`.
- Dependent options are force-applied at configure time (`scripts/xmake_func.lua`): `lc_enable_tests` forces `lc_enable_dsl` on; `lc_fallback_backend` or `lc_llvm_path` force `lc_enable_xir` on; Python bindings only activate when `lc_enable_py` *and* a non-empty `lc_py_include` are set; `lc_dx_cuda_interop`/`lc_vk_cuda_interop` are auto-enabled when both sides are present.
- Windows builds need the Windows SDK ≥ 10.0.22000.0; the phony target `lc-check-winsdk` (`scripts/xmake_func.lua`) raises otherwise.
- `-p android` requires the NDK: pass `--lc_ndk=<dir>` (auto-detected with `detect.sdks.find_ndk` when unset).
- Binaries land in `<lc_bin_dir>/<mode>` — default `bin/debug`, `bin/release` or `bin/releasedbg` (`lc_bin_dir` defaults to `bin`).

# Xmake Target Writing Tutorial

## Overview

This tutorial covers how to write xmake targets using the standard xmake API, with examples drawn from real projects like LuisaCompute. The recommended style uses `on_load` callbacks for dynamic configuration, with static declarations outside.

---

## 1. Basic Target Structure

```lua
target("<name>", {kind = "static"})   -- Optional: pass kind inline
-- or
target("<name>")
set_kind("static")  -- documented: "static", "shared", "binary", "object", "phony", "headeronly"

-- Static settings (outside on_load)
set_basename("my-lib")        -- Override output filename
add_deps("dep1", "dep2")      -- Target dependencies
add_rules("my-rule")          -- Custom build rules (MUST be outside on_load)
add_files("src/*.cpp")        -- Source files (simple globs outside)
add_headerfiles("include/**.h") -- Header files

on_load(function(target)
    -- Dynamic settings (preferred for conditional config)
    target:add("includedirs", "include", {public = true})
    target:add("defines", "MY_DEFINE", {public = true})
    target:add("deps", "another-dep")   -- Same as add_deps() outside
    target:set("kind", "shared")
    target:add("links", "pthread")
    target:add("syslinks", "dl")
    target:add("packages", "spdlog")    -- For xrepo packages
end)

after_build(function(target)
    -- Post-build steps (e.g., copy DLLs)
end)
target_end()
```

### Key rules:

- **Declare `add_rules()` in target script scope, outside `on_load`** — rules are resolved before `on_load`; `target:add("rules", ...)` there is silently ignored.
- **`add_deps()` outside = `target:add("deps", ...)` inside** — they are equivalent.
- **Simple globs** (`add_files`, `add_headerfiles`) can go outside; conditional additions go inside `on_load`.
- **Visibility** — pass `{public = true}`, `{interface = true}`, or `{private = true}` (default) to control inheritance.

---

## 2. Config Fields and Script Fields

Every xmake target has two kinds of declarations:

- **Config fields** — static target properties (what to build, how to build it).
- **Script fields** — lifecycle callbacks (when to run custom Lua code).

### Config Fields

Config fields are the key/value pairs that describe a target. They are set with `set_*` / `add_*` outside `on_load`, or equivalently with `target:set()` / `target:add()` inside a script field.

Common config field categories:

| Category | Fields |
|---|---|
| Identity | `kind`, `basename`, `filename`, `prefixname`, `suffixname`, `extension`, `group` |
| Output | `targetdir`, `objectdir`, `dependir`, `rundir`, `installdir`, `prefixdir`, `configdir`, `autogendir` |
| Sources | `files`, `headerfiles`, `extrafiles`, `remove_files`, `remove_headerfiles`, `configfiles`, `installfiles`, `filegroups`, `tests` |
| Includes | `includedirs`, `sysincludedirs` |
| Defines | `defines`, `undefines`, `configvar` |
| Links | `links`, `syslinks`, `linkdirs`, `rpathdirs`, `linkorders`, `linkgroups`, `frameworks`, `frameworkdirs` |
| Compilation | `languages`, `optimize`, `warnings`, `symbols`, `runtimes`, `exceptions`, `fpmodels`, `encodings`, `strip`, `vectorexts`, `forceincludes`, `pcheader`, `pcxxheader`, `pmheader`, `pmxxheader` |
| Dependencies | `deps`, `packages`, `options`, `rules` |
| Misc | `values.*`, `runenv` (replace one variable), `runenvs` (append values), `runargs`, `enabled`, `default`, `toolchains`, `toolset`, `plat`, `arch`, `policy` |

Rules for config fields:

1. **Most config fields can be set either outside or inside `on_load`** using the equivalent `target:set("field", value)` / `target:add("field", value)` form.
2. **Declare rules outside `on_load`** — they are loaded before the callback; adding them there has no effect.
3. **Static config goes outside** for readability; **dynamic/conditional config goes inside `on_load`**.
4. **Use `{public = true}` / `{interface = true}` / `{private = true}`** with `target:add()` to control inheritance of includedirs, defines, links, etc.
5. **`target:get("field")` reads a config field** inside a script field; `has_config("opt")` reads project-level options.

### Script Fields

Script fields are the lifecycle hooks where you write imperative Lua code. They receive the `target` object (and sometimes other arguments) and run at specific build phases.

Common script fields:

| Script field | Runs when | Typical use |
|---|---|---|
| `on_load(function(target) ... end)` | Target is loaded (early) | Dynamic config, conditional deps/files |
| `after_load(function(target) ... end)` | After target load/dependency resolution | Inspect deps; register generated source/object files in custom rules |
| `on_config(function(target) ... end)` | After `xmake config`, before build | Validate toolchain/options |
| `before_build(function(target) ... end)` | Before compilation starts | Pre-build checks/code generation |
| `on_build(function(target) ... end)` | Build phase | Override entire build |
| `after_build(function(target) ... end)` | After build finishes | Copy outputs, print reports |
| `before_link(function(target) ... end)` | Before linking | Inject link args |
| `after_link(function(target) ... end)` | After linking | Sign/post-process binary |
| `on_install(function(target) ... end)` | Install phase | Custom install logic |
| `on_run(function(target) ... end)` | `xmake run` | Override run behavior |

Rules for writing code in script fields:

1. **Always operate on the `target` argument** for target-local config: `target:add("field", value)`, `target:set("field", value)`, `target:get("field")`.
2. **Project-scope helpers are still available**: `is_plat()`, `is_arch()`, `is_mode()`, `has_config()`, `get_config()`, `os.*`, `io.*`, `path.*`, etc.
3. **You can `import()` extension modules** at the top of the script field callback (or at file scope).
4. **`on_load` is for configuration** — it should set/add target config fields. It runs very early, so dependencies may not be fully resolved yet.
5. **`before_build` / `after_build` are for actions** — they run around compilation and are the right place to generate files, copy DLLs, run validators, or emit summaries.
6. **Returning `false` from some hooks (e.g. `on_test`) signals failure**; most hooks ignore return values.

### Example: Writing Code Inside Script Fields

```lua
target("my-scripted-target")
set_kind("binary")
add_files("src/*.cpp")
add_includedirs("include")

-- Static config field outside
set_basename("myapp")
set_warnings("all")

on_load(function(target)
    -- Dynamic config field inside script field
    target:add("defines", "VERSION=\"1.0.0\"", {public = true})

    if target:is_plat("windows") then
        target:add("syslinks", "Advapi32", "Ole32")
    elseif target:is_plat("linux") then
        target:add("syslinks", "pthread", "dl")
    end

    if is_mode("debug") then
        target:set("symbols", "debug")
        target:set("optimize", "none")
    end

    -- Record data for later script fields
    target:data_set("build_start", os.mclock())
end)

before_build(function(target)
    -- Script field code: validate before compiling
    local main = path.join(target:scriptdir(), "src/main.cpp")
    if not os.isfile(main) then
        raise("missing entry point: " .. main)
    end

    -- Generate a version header
    local out = path.join(target:autogendir(), "version.h")
    os.mkdir(path.directory(out))
    io.writefile(out, string.format("#define BUILD_TIME %d\n", os.time()))
    target:add("includedirs", path.directory(out))
end)

after_build(function(target)
    -- Script field code: post-build action
    local exe = target:targetfile()
    if os.isfile(exe) then
        local dest = path.join("$(builddir)", "publish")
        os.mkdir(dest)
        os.cp(exe, dest)
        print("published:", exe)
    end

    local start = target:data("build_start")
    if start then
        print("elapsed:", os.mclock() - start, "ms")
    end
end)

target_end()
```

---

## 3. API Equivalence: Inside `on_load`

Use `target:add()` and `target:set()` inside `on_load` to dynamically configure targets:

### `target:add()` — cumulative (equivalent to `add_*`)

| Inside `on_load(target)` |
|---|
| `target:add("deps", "foo")` |
| `target:add("files", "*.cpp")` |
| `target:add("headerfiles", "*.h")` |
| `target:add("includedirs", "inc")` |
| `target:add("sysincludedirs", "inc")` |
| `target:add("defines", "FOO")` |
| `target:add("undefines", "BAR")` |
| `target:add("links", "foo")` |
| `target:add("syslinks", "dl")` |
| `target:add("linkorders", ...)` |
| `target:add("linkgroups", {group = true})` |
| `target:add("linkdirs", "lib")` |
| `target:add("rpathdirs", "lib")` |
| `target:add("frameworks", "Foundation")` |
| `target:add("frameworkdirs", "dir")` |
| `target:add("embeddirs", "dir")` |
| `target:add("packages", "spdlog")` |
| `target:add("options", "myopt")` |
| `target:add("vectorexts", "avx2")` |
| `target:add("languages", "cxx20")` |
| `target:add("imports", "module")` |
| `target:add("runenvs", "PATH", "/usr/bin")` |
| `target:add("forceincludes", "inc.h")` |
| `target:add("configfiles", "config.h.in")` |
| `target:add("installfiles", "data/*")` |
| `target:add("extrafiles", "readme.md")` |
| `target:add("filegroups", "src", files)` |

### `target:set()` — singular (equivalent to `set_*`)

| Inside `on_load(target)` |
|---|
| `target:set("kind", "static")` |
| `target:set("basename", "foo")` |
| `target:set("filename", "foo.dll")` |
| `target:set("prefixname", "lib")` |
| `target:set("suffixname", "-d")` |
| `target:set("extension", ".dll")` |
| `target:set("targetdir", "lib")` |
| `target:set("objectdir", "obj")` |
| `target:set("dependir", "deps")` |
| `target:set("rundir", "bin")` |
| `target:set("runargs", "--verbose")` |
| `target:set("installdir", "/usr")` |
| `target:set("prefixdir", "subdir")` |
| `target:set("configdir", "out")` |
| `target:set("group", "mygroup")` |
| `target:set("languages", "cxx20")` |
| `target:set("optimize", "fastest")` |
| `target:set("warnings", "all")` |
| `target:set("symbols", "debug")` |
| `target:set("exceptions", "cxx")` |
| `target:set("runtimes", "MD")` |
| `target:set("fpmodels", "fast")` |
| `target:set("encodings", "utf-8")` |
| `target:set("strip", "all")` |
| `target:set("enabled", true)` |
| `target:set("default", false)` |
| `target:set("toolchains", "clang")` |
| `target:set("toolset", "cc", "/usr/bin/gcc")` |
| `target:set("plat", "linux")` |
| `target:set("arch", "x64")` |
| `target:set("policy", "build.optimization.lto", true)` |
| `target:set("options", "opt1")` |
| `target:set("values.mykey", "val")` |
| `target:set("configvar", "VAR", "value")` |
| `target:set("runenv", "PATH", "/usr/bin")` |
| `target:set("pcheader", "header.h")` |
| `target:set("pcxxheader", "header.hpp")` |
| `target:set("pmheader", "header.m")` |
| `target:set("pmxxheader", "header.mm")` |

> **Note:** For the `target:add("name", ...)` / `target:set("name", ...)` pattern, any key name works through xmake's generic values mechanism. Only explicitly defined APIs (like `files`, `deps`, `kind`) have special handling.

---

## 4. Compilation Flags (by Language)

These APIs pass compiler-specific flags:

| API | Description |
|---|---|
| `add_cflags(...)` | C compilation flags |
| `add_cxflags(...)` | C/C++ compilation flags |
| `add_cxxflags(...)` | C++ compilation flags |
| `add_mflags(...)` | ObjC compilation flags |
| `add_mxflags(...)` | ObjC/ObjC++ compilation flags |
| `add_mxxflags(...)` | ObjC++ compilation flags |
| `add_scflags(...)` | Swift compilation flags |
| `add_asflags(...)` | Assembly compilation flags |
| `add_gcflags(...)` | Go compilation flags |
| `add_dcflags(...)` | D language compilation flags |
| `add_fcflags(...)` | Fortran compilation flags |
| `add_zcflags(...)` | Zig compilation flags |
| `add_cuflags(...)` | CUDA compilation flags |
| `add_culdflags(...)` | CUDA device link flags |
| `add_cugencodes(...)` | CUDA gencode settings (e.g., `"sm_90"`, `"compute_90,sm_90"`, `"native"`) |
| `add_rcflags(...)` | Windows resource-compiler flags |
| `add_ascnpuarchs(...)` | Ascend NPU architectures (3.0.9+) |

### Linker Flags

| API | Description |
|---|---|
| `add_ldflags(...)` | Static library/exe link flags |
| `add_arflags(...)` | Archive (static library) flags |
| `add_shflags(...)` | Dynamic library link flags |

Example with per-tool flags:

```lua
on_load(function(target)
    target:add("cxflags", "-fPIC", {tools = {"clang", "gcc"}, public = true})
    target:add("cxflags", "/Zc:preprocessor", {tools = "cl"})
    target:add("ldflags", "-Wl,-rpath,.", {force = true, expand = false})
end)
```

The `tool::flag` prefix is shorthand for tool filtering, e.g. `add_cxxflags("clang::-stdlib=libc++")`. Use `{expand = false}` for a flag array element containing spaces (for example `{"-L/my lib"}`); `{force = true}` bypasses automatic flag detection.

---

## 5. Precompiled Headers (PCH)

```lua
target("my-target")
set_pcheader("precompiled.h")     -- C PCH
set_pcxxheader("precompiled.hpp") -- C++ PCH
```

Enable conditionally with:

```lua
if has_config("lc_enable_pch") then
    set_pcxxheader("mypch.hpp")
end
```

This project wraps exactly that check in `lc_set_pcxxheader(...)` (root `xmake.lua`), which only forwards to `set_pcxxheader(...)` while `lc_enable_pch` is on — use it instead of `set_pcxxheader` in `src/**/xmake.lua`.

---

## 6. Conditional Configuration with Conditions

```lua
on_load(function(target)
    -- Platform checks
    if target:is_plat("windows") then
        target:add("defines", "NOMINMAX", "PLATFORM_WINDOWS")
        target:add("syslinks", "Advapi32", "Ole32")
    elseif target:is_plat("linux") then
        target:add("syslinks", "dl", "uuid", "pthread")
        target:add("cxflags", "-fPIC")
    elseif target:is_plat("macosx") then
        target:add("frameworks", "CoreFoundation", "Metal")
    end

    -- Architecture checks
    if target:is_arch("x64", "x86_64") then
        target:add("vectorexts", "avx2")
    elseif target:is_arch("arm64", "aarch64") then
        target:add("defines", "PLATFORM_ARM")
    end

    -- Build mode
    if is_mode("debug") then
        target:set("symbols", "debug")
        target:set("optimize", "none")
        target:set("runtimes", "MDd")
    elseif is_mode("release") then
        target:set("optimize", "aggressive")
        target:set("symbols", "hidden")
        target:set("runtimes", "MD")
    end

    -- Config option checks
    if has_config("my_feature") then
        target:add("defines", "MY_FEATURE_ENABLED")
        target:add("deps", "my-feature-dep")
    end

    if has_package("spdlog") then
        target:add("packages", "spdlog")
    end

    -- Target kind checks
    if target:get("kind") == "static" then
        target:add("defines", "MYLIB_STATIC", {public = true})
    elseif target:get("kind") == "shared" then
        target:add("defines", "MYLIB_EXPORT", {public = true})
    end
end)
```

### Standalone Condition Functions (usable in any scope)

```lua
if is_plat("windows") then ... end    -- Current target platform
if is_arch("x64") then ... end        -- Current target architecture
if is_mode("debug") then ... end      -- Current build mode
if is_os("windows") then ... end      -- Target operating system
if is_host("windows") then ... end    -- Host OS running xmake
if is_subhost("msys") then ... end    -- Subsystem (e.g., "msys", "cygwin")
if os.is_subarch(...) then ... end    -- Host subsystem architecture; docs list bare `is_subarch`, but installed 3.1.1 exposes only `os.is_subarch`
if is_cross() then ... end            -- Cross-compilation check
if is_kind("static") then ... end     -- Target kind check
if is_config("var", "value") then ... end  -- Config option value check
if has_config("feature") then ... end     -- Config option exists/enabled?
if has_package("pkg") then ... end        -- Package exists/enabled?
```

---

## 7. Lifecycle Hooks

```lua
target("my-target")

-- Loading/configuration order: on_load -> after_load -> on_config
on_load(function(target) end)
after_load(function(target) end) -- Dependencies are available here
on_config(function(target) end)

-- Prepare phase (before the build phase)
on_prepare(function(target, opt) end)
on_prepare_file(function(target, sourcefile, opt) end)
on_prepare_files(function(target, sourcebatch, opt) end)
-- Jobgraph overload:
-- on_prepare_files(function(target, jobgraph, sourcebatch, opt) end, {jobgraph = true})

-- Build/link overrides
on_build(function(target) end)
on_build_file(function(target, sourcefile, opt) end)
on_build_files(function(target, sourcebatch, opt) end)
on_link(function(target) end)

-- Clean/package/install/uninstall/run overrides
on_clean(function(target) end)
on_package(function(target) end)
on_install(function(target) end)
on_uninstall(function(target) end)
on_run(function(target) end)

-- Custom test; opt contains the add_tests payload
before_test(function(target, opt) end)
on_test(function(target, opt)
    return true -- or false, errors
end)
after_test(function(target, opt) end)

-- Non-overriding before_/after_ families:
-- prepare, prepare_file, prepare_files, build, build_file, build_files,
-- link, clean, package, install, uninstall, run, test
```

The main build sequence is `on_load -> after_load -> on_config -> before_build -> on_build -> after_build`. Use `after_load`, not `on_load`, when generated files or object lists depend on resolved dependencies.

### Common Use of `after_build` — Copy DLLs

```lua
after_build(function(target)
    if is_plat("windows") then
        os.cp("path/to/mylib.dll", target:targetdir())
    elseif is_plat("linux") then
        os.cp("path/to/libmylib.so", target:targetdir())
    end
end)
```

---

## 8. Visibility and Inheritance

Many `target:add()` / `target:set()` calls accept a visibility table to control propagation:

```lua
-- Public: propagated to dependent targets + current target
target:add("includedirs", "include", {public = true})
target:add("defines", "PUBLIC_DEF", {public = true})
target:add("links", "mylib", {public = true})

-- Interface: only propagated to dependents (not current target)
target:add("includedirs", "include", {interface = true})

-- Private: only for current target (default)
target:add("defines", "PRIVATE_DEF", {private = true})
```

Dependency inheritance can be controlled per-target:

```lua
add_deps("foo", {inherit = false})     -- No inheritance from this dep
add_deps("bar", {inherit = true})      -- Default: inherit
add_deps("baz", {links = false})       -- Don't inherit links from this dep
```

---

## 9. Tests

```lua
target("my-test")
set_kind("binary")
add_files("test_*.cpp")

add_tests("test_foo", {
    runargs = {"--arg1", "--arg2"},
    rundir = "$(projectdir)",
    runenvs = {PATH = "/usr/bin"},
    timeout = 30,
    group = "unit",
    trim_output = true,
    plain = false,
    pass_outputs = {"PASSED"},
    fail_outputs = {"FAILED"},
    should_fail = false,
    build_should_pass = true,
})

-- Or with custom test script
on_test(function(target, opt)
    -- opt contains name, runargs, runenvs, timeout, and other test settings
    -- Perform custom checks here. Do not use `if not os.execv(...)`:
    -- os.execv returns numeric 0 on success and raises on failure.
    return true -- or false, errors
end)
```

Run with `xmake test [target/test|target/*|*/name]`, `-g <group-pattern>`, `-jN`, or `--output-on-failure` (3.1.2+). Other options include `pass_output_files`, `fail_output_files`, `realtime_output`, per-test `files`/`defines`/`remove_files`/`languages`/`packages`, and `build_should_fail`. Policies `test.stop_on_first_failure` and `test.return_zero_on_failure` control failure behavior. LuisaCompute currently defines plain `test_*` executable targets without `add_tests`, so run those with `xmake run <target>`.

---

## 10. Common Target Patterns

### 10.1 Shared Library

```lua
target("mylib")
set_kind("shared")
set_basename("mylib")
add_deps("core")
add_headerfiles("include/**.h")

on_load(function(target)
    target:add("defines", "MYLIB_EXPORT_DLL")
    target:add("includedirs", "include", {public = true})
    target:add("files", "src/*.cpp")

    if target:is_plat("windows") then
        target:add("defines", "NOMINMAX")
        target:add("syslinks", "Advapi32")
    elseif target:is_plat("macosx") then
        target:add("frameworks", "Foundation")
    end

    if has_config("enable_extra") then
        target:add("defines", "EXTRA_FEATURE")
        target:add("files", "src/extra/*.cpp")
    end
end)

if has_config("lc_enable_pch") then
    set_pcxxheader("src/mylib_pch.h")
end
target_end()
```

### 10.2 Static Library

```lua
target("mystatic")
set_kind("static")
set_basename("mystatic")
add_deps("core")
add_headerfiles("include/**.h")
add_files("src/*.cpp")
add_defines("MYSTATIC_STATIC_LIB", {public = true})
target_end()
```

### 10.3 Executable (Binary)

```lua
target("my-tool")
set_kind("binary")
add_deps("lc-runtime", "lc-dsl")
add_files("main.cpp")
add_includedirs("include")

on_load(function(target)
 if has_config("lc_enable_gui") then
 target:add("deps", "lc-gui")
        target:add("defines", "ENABLE_GUI")
    end
end)
target_end()
```

### 10.4 Phony Target (Meta / Validation)

```lua
target("my-validator")
set_kind("phony")  -- No build output
add_deps("runtime")

on_config(function(target)
    if target:is_plat("windows") then
        local toolchain = target:toolchain("msvc")
        -- Validate SDK version, toolchain, etc.
    end
end)
target_end()
```

### 10.5 Header-only Target

```lua
target("my-headers")
set_kind("headeronly")
add_headerfiles("include/**.h")
add_includedirs("include", {public = true})
target_end()
```

### 10.6 Test Target (using a helper function)

```lua
local function test_proj(name, source, extra)
 target(name)
 set_kind("binary")
 add_deps("lc-runtime", "lc-dsl", "lc-vstl")
 add_files(source)
 add_includedirs("common")
 if extra then extra() end
 target_end()
end

test_proj("test_foo", "tests/test_foo.cpp")
test_proj("test_bar", "tests/test_bar.cpp", function()
 add_defines("EXTRA")
 add_deps("extra-dep")
end)
```

The real helper is `local function test_proj(name, source, gui_dep, callable, kind, cxx_standard)` in `src/tests/xmake.lua`; it always adds `lc-runtime`, `lc-dsl`, `lc-vstl`, `stb-image` and `lc-backends-dummy` (with `{inherit = false, links = false}`), calls `_config_project({...})` for the mode/standard settings, and skips the target when `gui_dep` is set but `lc_enable_gui` is off. Test target names are `test_*` and benchmark names `benchmark_*` (most created through that helper; a few, e.g. `test_reproc_win32_contract`, are plain `target(...)` blocks), so `xmake build test_dsl` / `xmake run benchmark_tile_native` work directly.

### 10.7 Object Target (Intermediate objects only)

```lua
target("my-objects")
set_kind("object")  -- Compiles sources but does not link
add_files("src/*.cpp")
target_end()
```

---

## 11. Custom Rules

`add_rules()` must be **outside** `on_load`:

```lua
target("my-target")
add_rules("c.unity_build", {batchsize = 8})    -- Unity build
add_rules("c++.unity_build", {batchsize = 8})
add_rules("utils.bin2obj", {extensions = {".cu", ".h"}})  -- Binary embedding
add_rules("lc_llvm") -- LLVM integration
target_end()
```

The rules this project defines (use these names, they are not xmake built-ins):

| Rule | Defined in | What it does |
|---|---|---|
| `lc_basic_settings` | `scripts/xmake_func.lua` | applied to every target through `_config_project`: toolchain, `project_kind`, `c_standard`/`cxx_standard`, exceptions, `optimize`/`warnings`/`runtimes` per mode, `-mfma`, `/Zc:preprocessor`, SIMD `vectorexts`, and the `lto`/`rtti` settings (each read by `_get_or("<key>")`: rule extraconf first, then the `lc_<key>` config) |
| `lc-rename-ext` | `scripts/xmake_func.lua` | prefixes the output basename with `luisa-ext-` |
| `lc_install_sdk` | `scripts/xmake_func.lua` (+ `scripts/find_sdk.lua`) | downloads/extracts SDK zips (`dx_sdk`, `vk_sdk`) before build; takes `{sdk_dir = ..., libnames = ...}` |
| `lc_llvm` | `scripts/xmake_func.lua` | links against the `lc_llvm_path` LLVM install and copies its DLLs on Windows |
| `lc_run_target` | `scripts/xmake_func.lua` | `on_run` that execs the built binary with the target directory as cwd |
| `lc-backend-deps` | `src/backends/xmake.lua` | pulls in the enabled `lc-backend-*` targets plus `lc_backend_sdk` and `lc-validation-layer` |

`_config_project({project_kind = ..., batch_size = ..., ...})` (`scripts/xmake_func.lua`) is the wrapper used by nearly every `xmake.lua` here: it adds `c.unity_build`/`c++.unity_build` when `batch_size > 1` (unless `lc_enable_unity_build` is off) and then `add_rules(_config_rules, config)` with `_config_rules = {"lc_basic_settings"}`.

### Rules with Custom Values

```lua
target("my-target")
add_rules("my-rule")
set_values("mykey", "value1", "value2")
add_values("mykey", "value3")
target_end()
```

---

## 12. The `on_load` / `on_config` Target Object

Inside lifecycle hooks, the `target` object provides these methods:

| Method | Description |
|---|---|
| `target:name()` | Get target name |
| `target:fullname()` | Get full name (with namespace) |
| `target:targetdir()` | Get output directory |
| `target:targetfile()` | Get target file path |
| `target:scriptdir()` | Get directory of the xmake.lua file |
| `target:arch()` | Get target architecture |
| `target:plat()` | Get target platform |
| `target:is_plat("windows")` | Check platform |
| `target:is_arch("x64")` | Check architecture |
| `target:is_arch64()` | Is 64-bit architecture? |
| `is_mode("debug")` | Build mode — a **global** function (`D:/xmake/core/project/project.lua`), not a target method; there is no `target:is_mode()` |
| `target:is_cross()` | Is cross-compilation? |
| `target:has_tool("cxx", "clang")` | Check if using specific tool |
| `target:get("kind")` | Get any target property |
| `target:get_from("links", "*")` | Get values from all sources (self, deps, options, packages) |
| `target:add("key", "value", {public=true})` | Add configuration |
| `target:set("key", "value")` | Override configuration |
| `target:deps()` | Get all dependent targets (after_load only) |
| `target:dep("name")` | Get a specific dependency (after_load only) |
| `target:orderdeps({inherit=true})` | Get ordered deps |
| `target:toolchain("msvc")` | Get toolchain instance |
| `target:compiler("cxx")` | Get compiler instance |
| `target:linker()` | Get linker instance |
| `target:sourcebatches()` | Get source file batches |
| `target:objectdir()` | Get object directory |
| `target:dependir()` | Get dependency directory |
| `target:autogendir()` | Get auto-generated files directory |
| `target:data("key")` | Get user private data |
| `target:data_set("key", value)` | Set user private data |
| `target:values("name")` | Get custom values |
| `target:values_set("name", ...)` | Set custom values |
| `target:rule("name")` | Get a rule instance |
| `target:rule_enable("name", bool)` | Enable/disable a rule |
| `target:extraconf("name", "item", "key")` | Get extra configuration |
| `target:extraconf_from("name", "source")` | Get extra config from source |
| `target:pkgs()` | Get all packages |
| `target:pkg("name")` | Get a package instance |
| `is_kind("static")` | Target kind check — a **global** function, not a target method (on a target use `target:kind()` or `target:is_binary()`/`is_shared()`/`is_static()`/`is_object()`/`is_headeronly()`/`is_moduleonly()`/`is_phony()`) |
| `target:kind()` | Get target kind |
| `target:basename()` | Get output base name |
| `target:filename()` | Get output filename |
| `target:version()` | Get target version |
| `target:clone()` | Clone the target (after_load only) |
| `target:is_phony()` | Is phony target? |
| `target:is_binary()` | Is binary target? |
| `target:is_shared()` | Is shared library? |
| `target:is_static()` | Is static library? |
| `target:is_library()` | Is any library type? |
| `target:is_enabled()` | Is target enabled? |
| `target:is_default()` | Is default build target? |
| `target:is_rebuilt()` | Was target rebuilt? |

---

## 13. Dependencies: Options & Packages

```lua
target("my-target")
-- Option dependencies
add_options("my_option")
set_options("my_option")

-- Package dependencies (requires add_requires in root scope)
add_requires("spdlog", "fmt")
target("my-target")
add_packages("spdlog", "fmt")

-- With component selection
add_packages("sfml", {components = {"graphics", "window"}})

-- Internal target dependencies with fine-grained control
add_deps("lib-a", "lib-b", {inherit = true})   -- Full inheritance
add_deps("lib-c", {inherit = false})             -- No inheritance
add_deps("lib-d", {links = false})               -- Don't inherit links
```

---

## 14. Run Environment

```lua
target("my-target")
set_runenv("PATH", "/custom/path")       -- Override environment variable
add_runenvs("PATH", "/extra/path")       -- Append to environment variable
```

---

## 15. File Management

```lua
target("my-target")
add_files("src/*.cpp")
add_files("src/*.cpp", {sourcekind = "cxx"})  -- With per-file options
add_files("src/*.m", {sourcekind = "mxx"})    -- ObjC++ files
remove_files("src/old.cpp")                   -- Remove previously added files
add_headerfiles("include/**.h")
remove_headerfiles("include/deprecated.h")
add_installfiles("config/*.ini")
add_configfiles("config.h.in")    -- Template config files with @var@ substitution
add_extrafiles("README.md")       -- Extra files for IDE listing
add_forceincludes("precompiled.h") -- Force-include header
```

---

## 16. Complete Example

```lua
-- Root xmake.lua
set_xmakever("3.0.6")
add_rules("mode.release", "mode.debug", "mode.releasedbg") -- as in this project's xmake.lua
add_requires("spdlog")

-- Library target
target("mylib")
set_kind("shared")
set_basename("mylib")
add_deps("core")
add_headerfiles("include/**.h")
add_rules("c++.unity_build", {batchsize = 8})

if has_config("lc_enable_pch") then
    set_pcxxheader("src/mylib_pch.h")
end

on_load(function(target)
    -- Source files
    target:add("files", "src/*.cpp")
    if has_config("enable_extra") then
        target:add("files", "src/extra/*.cpp")
        target:add("defines", "ENABLE_EXTRA")
    end

    -- Public include dirs and defines
    target:add("includedirs", "include", {public = true})

    -- Platform config
    if target:is_plat("windows") then
        target:add("defines", "NOMINMAX", "PLATFORM_WIN", "MYLIB_EXPORT_DLL")
        target:add("syslinks", "Advapi32", "Ole32")
        target:add("cxflags", "/Zc:preprocessor", {tools = "cl"})
    elseif target:is_plat("macosx") then
        target:add("frameworks", "Foundation", "Metal")
        target:add("cxflags", "-fobjc-arc")
    elseif target:is_plat("linux") then
        target:add("syslinks", "dl", "pthread", "uuid")
        target:add("cxflags", "-fPIC")
    end

    -- Arch config
    if target:is_arch("x64", "x86_64") then
        target:add("vectorexts", "avx2")
    end

    -- Package dependencies
    if has_config("lc_spdlog_use_xrepo") then
        target:add("packages", "spdlog")
    else
        target:add("deps", "spdlog-bundled")
    end
end)

after_build(function(target)
    if is_plat("windows") then
        os.cp("$(builddir)/mydep.dll", target:targetdir())
    end
end)
target_end()

-- Test executable
target("test-mylib")
set_kind("binary")
add_deps("mylib")
add_files("tests/*.cpp")

add_tests("test_basic", {
    runargs = {"--verbose"},
    group = "unit",
})

on_load(function(target)
    target:add("includedirs", "tests")
end)
target_end()
```

---

## Summary

1. **Use `on_load` for conditional logic** — platform checks, feature flags, dynamic file lists.
2. **Declare `add_rules()` outside `on_load`** — rules added in the callback are silently ignored.
3. **Simple globs outside, conditional additions inside** — keep `add_files`/`add_headerfiles` outside for simple cases.
4. **Prefer `target:set()` / `target:add()` inside `on_load`** for most configuration — it's equivalent to outside calls.
5. **Visibility** — `{public = true}` propagates to dependents, `{interface = true}` propagates only to dependents, `{private = true}` (default) is local-only.
6. **`add_deps()` outside = `target:add("deps", ...)` inside** — choose whichever fits your style.
7. **Target configuration APIs have description and instance forms** — for example, use `set_kind(...)` in target scope or `target:set("kind", ...)` in a callback; conditions such as `is_mode()` remain global helpers.

---

# Lua Scripting in xmake

> Primary reference: `D:\xmake-docs\docs\api\scripts`; implementation source for drift checks: `D:\xmake\core\sandbox\modules`, `D:\xmake\modules`, `D:\xmake\core\base`.

XMake scripts (in `on_load`, `on_build`, `after_install`, etc.) run in a **sandboxed Lua environment**. This section covers common modules; the appended API reference indexes the remaining built-in, instance, and extension APIs.

---

## 1. Built-in Sandbox Modules

### 1.1 `print` / `printf` — Output

```lua
print("hello", "world")       -- Print with newline
printf("hello %s", "world")   -- Print without newline
vprint("verbose msg")         -- Only printed with -v/--verbose
dprint("diagnosis msg")       -- Only printed with --diagnosis
```

### 1.2 `cprint` / `cprintf` — Colored Output

```lua
cprint("${bright}hello${reset}")           -- Bright text
cprint("${red}error${reset}")               -- Red text
cprint("${color.dump.string}hello")         -- Dump color
cprint("${dim}%s${reset}", "world")         -- Dim text
```

Available color tags: `${red}`, `${green}`, `${blue}`, `${yellow}`, `${magenta}`, `${cyan}`, `${bright}`, `${dim}`, `${reset}`, `${underline}`, etc.

### 1.3 `utils` — Utilities

```lua
utils.dump(obj)                    -- Dump object for debugging
utils.assert(value, "msg", ...)    -- Assert with error message
utils.error("err %s", arg)         -- Error message
utils.warning("warn %s", arg)      -- Warning message
utils.trycall(func)                -- Call function safely (returns ok, ...)
```

### 1.4 Path Operations

```lua
path.join("a", "b", "c")            -- Join with OS separators; does not collapse ".."
path.join("a", "..", "b")           -- "a/../b" on POSIX
path.absolute("rel/path")           -- Full absolute path
path.relative("/abs/path", "/base") -- Relative path from base
path.basename("foo/bar.cpp")        -- "bar" (without suffix)
path.filename("foo/bar.cpp")        -- "bar.cpp" (with suffix)
path.extension("foo/bar.cpp")       -- ".cpp"
path.directory("foo/bar.cpp")       -- "foo"
path.normalize("a/./b/../c")        -- Normalize and use OS path separators
```

### 1.5 `string` — String Operations

All standard Lua string functions are available. Extended functions:

```lua
string.vformat("$(var) hello", ...) -- Format with built-in variables
string.format("hello %s", "world")  -- Standard Lua format
-- Standard: sub, gsub, find, match, gmatch, upper, lower, rep, reverse, char, byte, len
-- XMake extensions: startswith, endswith, split, trim/ltrim/rtrim, lastof,
-- replace, serialize, deserialize, ipattern, levenshtein
```

**Built-in variables** (resolved in strings via `$()` or vformat):

| Variable | Description |
|---|---|
| `$(os)` / `$(host)` | Target OS / host OS |
| `$(tmpdir)` / `$(curdir)` | Temporary / current directory |
| `$(scriptdir)` / `$(projectdir)` | Current `xmake.lua` / project root directory |
| `$(builddir)` / `$(configdir)` | Build-output / configuration directory (`builddir`, not deprecated `buildir`) |
| `$(globaldir)` / `$(programdir)` | Global XMake / installation directory |
| `$(shell)` / `$(env)` / `$(reg)` | Use `$(shell <cmd>)`, `$(env <name>)`, or `$(reg <path;name>)` for shell output / environment / Windows registry values |

`$(plat)`, `$(arch)`, `$(mode)`, and custom `xmake f --name=value` settings are also available as `$(name)`.

Example:
```lua
path.join("$(projectdir)", "build")  -- Resolves to /path/to/project/build
print("$(scriptdir)")                -- Prints script directory
```

### 1.6 `table` — Table Operations

```lua
table.join(t1, t2)                    -- Merge tables (new table)
table.join2(t1, t2)                   -- Merge into t1 (in-place)
table.clone(t)                        -- Deep clone
table.wrap(v)                         -- Wrap single value as table {v}
table.unwrap({v})                     -- Unwrap table to single value
table.contains(t, value)               -- Check if value exists
table.unique(t)                        -- Remove duplicates
table.reverse(t)                       -- Reverse array
table.slice(t, first, last)            -- Slice array
table.is_array(t)                      -- Is array-like?
table.is_dictionary(t)                 -- Is dict-like?
table.keys(t)                          -- Get keys array
table.values(t)                        -- Get values array
table.pack(...)                        -- Pack arguments (like {...} but with .n)
table.map(t, mapper)                   -- Map values
table.imap(t, mapper)                  -- In-place map
table.find(t, value)                   -- Find index of value
table.find_if(t, pred)                 -- Find if predicate matches
table.remove_if(t, pred)               -- Remove if predicate matches
table.empty(t)                         -- Is empty?
table.orderkeys(t, callback)           -- Ordered keys
table.orderpairs(t, callback)          -- Ordered pairs iterator
table.inherit(...)                     -- Prototype-based inheritance
```

### 1.7 `os` — Operating System

#### File/Directory Operations

```lua
os.cp("src/file", "dst/file")          -- Copy file/dir
os.mv("src/file", "dst/file")          -- Move file/dir
os.rm("file_or_dir")                    -- Remove file/dir
os.ln("target", "symlink")             -- Create symlink
os.mkdir("dir")                        -- Create directory
os.rmdir("dir")                        -- Remove directory
os.cd("dir")                           -- Change directory (returns old cwd)
os.touch("file")                       -- Touch file
os.isfile("path")                      -- Is file?
os.isdir("path")                       -- Is directory?
os.islink("path")                      -- Is symlink?
os.isexec("path")                      -- Is executable?
os.exists("path")                      -- Exists?
os.readlink("symlink")                 -- Read symlink target
os.filesize("file")                    -- File size
os.mtime("file")                       -- Modification time
```

#### File Matching (Globbing)

```lua
os.files("src/*.cpp")                  -- Match .cpp files
os.dirs("src/*")                       -- Match directories
os.filedirs("src/*")                   -- Match files and dirs
os.match("src/*.c", "f")               -- Files; "d" = dirs, "a" = all (source-supported, absent from current os.md)
```

#### Running Commands

```lua
-- Run command, raise on failure
os.run("gcc -c %s -o %s", "file.c", "file.o")
os.runv("gcc", {"-c", "file.c", "-o", "file.o"})

-- Run command, return output
local out, err = os.iorun("echo hello")
local out, err = os.iorunv("python", {"--version"})

-- Run command, capture exit code
local exitok, errors = os.exec("ls")
local exitok, errors = os.execv("python", {"script.py"})

-- Verbose variants (print command if -v enabled)
os.vrun("gcc %s", "file.c")
os.vrunv("gcc", {"-c", "file.c"})
os.vexec("echo hello")
os.vexecv("echo", {"hello"})

-- Try variants (no raise on failure)
os.trycp("src", "dst")
os.trymv("src", "dst")
os.tryrm("file")
```

#### Environment Variables

```lua
os.getenv("PATH")                      -- Get env var
os.setenv("MY_VAR", "value")           -- Set env var (override)
os.addenv("PATH", "/new/path")         -- Append to env var
os.getenvs()                           -- Get all env vars
os.setenvs({PATH = "/usr/bin"})        -- Set multiple env vars
os.addenvs({PATH = "/new/path"})       -- Append multiple env vars
os.joinenvs({PATH = "/a:/b"})          -- Join env values
```

#### Directory/System Info

```lua
os.curdir()                            -- Current directory
os.scriptdir()                         -- Directory of current xmake.lua
os.projectdir()                        -- Project root directory
os.tmpdir()                            -- System temp directory
os.tmpfile("key")                      -- Generate temp file path
os.host()                              -- Host OS name
os.arch()                              -- Host architecture
os.subhost()                           -- Subsystem host
os.subarch()                           -- Subsystem arch
os.is_host("windows")                  -- Check host OS
os.is_arch("x64")                      -- Check host arch
os.is_subhost("msys")                  -- Check subsystem
os.isroot()                            -- Is running as root?
os.fscase()                            -- Is filesystem case-sensitive?
os.mclock()                            -- Monotonic elapsed-time clock (ms)
os.sleep(1000)                         -- Sleep ms (coroutine-safe)
os.nuldev()                            -- Null device path
os.xmakever()                          -- xmake version (semver)
os.args({"-a", "-b"})                  -- Format args array to string
os.getpid()                            -- Current process ID
os.cpuinfo(key)                        -- Get one CPU information field
os.meminfo(key)                        -- Get one memory information field
```

### 1.8 `io` — File I/O

```lua
-- Read/write entire files
local data = io.readfile("path")       -- Read all text
io.writefile("path", "content")        -- Write text
local obj = io.load("data.lua")        -- Load Lua-table serialization (not JSON)
io.save("data.lua", obj)               -- Save Lua-table serialization
-- JSON: import("core.base.json"), then json.loadfile/json.savefile

-- Open file handle
local f = io.open("file.txt", "r")     -- "r" read, "w" write, "a" append
f:read("*a")                           -- Read all
f:read("*l")                           -- Read line
f:read(n)                              -- Read n bytes
f:write("data")                        -- Write
f:print("format %s", "arg")            -- Write formatted with newline
f:printf("format %s", "arg")           -- Write formatted without newline
f:close()                              -- Close
f:flush()                              -- Flush
f:seek("set", 0)                       -- Seek
f:size()                               -- File size
f:load()                               -- Load serialized object from file
f:save(obj)                            -- Save serialized object to file

-- Text replacement
io.gsub("file", "pattern", "replace")  -- Global replace in file
io.replace("file", "pattern", "rep")   -- Replace occurrences
io.insert("file", lineidx, "text")     -- Insert text at line

-- Cat/Tail
io.cat("file")                         -- Print the whole file
io.tail("file", 10)                    -- Print last 10 lines

-- stdin/stdout/stderr
io.stdin:read("*l")                    -- Read from stdin
io.stdout:write("hello")               -- Write to stdout
io.stderr:write("error")               -- Write to stderr
io.write("hello")                      -- Shortcut for stdout:write
io.print("file", "hello %s", "world") -- Write formatted text plus newline to a file
io.flush()                             -- Flush stdout
```

### 1.9 `hash` — Hashing

```lua
hash.md5("data")                       -- MD5 hash
hash.md5("filepath")                   -- MD5 of file
hash.sha1("data")                      -- SHA1
hash.sha256("data")                    -- SHA256
hash.xxhash32("data")                  -- xxHash32
hash.xxhash64("data")                  -- xxHash64
hash.xxhash128("data")                 -- xxHash128
hash.uuid([name])                      -- Deterministic UUID in v4 format (optionally name-derived)
hash.uuid4()                           -- Generate UUID v4
hash.strhash32("str")                  -- String hash32
hash.strhash64("str")                  -- String hash64
hash.strhash128("str")                 -- String hash128
hash.rand32()                          -- Random 32-bit
hash.rand64()                          -- Random 64-bit
hash.rand128()                         -- Random 128-bit
```

### 1.10 `xmake` — xmake Runtime Info

```lua
xmake.arch()                           -- xmake architecture
xmake.version()                        -- xmake version string
xmake.branch()                         -- xmake git branch
xmake.programdir()                     -- xmake installation dir
xmake.programfile()                    -- xmake executable path
xmake.luajit()                         -- Is running on LuaJIT?
xmake.is_embed()                       -- Is embedded xmake?
```

### 1.11 `math` — Standard Lua math
### 1.12 `coroutine` — Standard Lua coroutine

---

## 2. Variable Formatting (`vformat`)

xmake strings can contain built-in variables resolved with `$(var)` syntax:

```lua
print("$(projectdir)/build")           -- Resolves to /path/to/project/build
print("$(scriptdir)/src")              -- Resolves to /path/to/xmake.lua/src
print("$(builddir)/$(mode)")            -- Resolves to build/debug etc.
```

All `os.*` functions, `io.*`, `path.*` and print functions automatically resolve `$(var)` in their string arguments.

---

## 3. Error Handling: `try` / `catch` / `finally`

```lua
local ok = try {
    function()
        -- Risky operation
        local data = io.readfile("may_not_exist.txt")
        if not data then
            raise("file not found")
        end
        return data
    end,
    catch {
        function(errors)
            -- Handle error
            print("caught:", errors)
            -- errors is the error message string,
            -- or a table {errors=..., stderr=..., stdout=...}
        end
    },
    finally {
        function(ok, result_or_errors)
            -- Always runs (like Lua's __gc)
        end
    }
}
-- If try succeeds, returns the try function's return values
-- If catch is provided, errors are caught and execution continues
```

Short form (no catch):
```lua
local ok = try { function() return io.readfile("file") end }
```

### `raise` — Throw an error

```lua
raise("something went wrong")          -- String error
raise({errors = "msg", stderr = "..."}) -- Table error (for command failures)
```

### `assert` — from utils

```lua
utils.assert(io.readfile("f"), "cannot read file")
-- Raises if first argument is falsy
```

---

## 4. Module Import System (`import`)

xmake provides a module system for importing extension modules from the `modules/` directory.

```lua
import("core.project.depend")
import("lib.detect.find_tool")
import("detect.sdks.find_cuda")
import("core.base.option")

-- Then use the module
local tool = find_tool("gcc")
local cuda = find_cuda()
local opt = option.get("verbose")
```

### Common extension modules

| Module | Description |
|---|---|
| `lib.detect.find_tool` | Find a system tool/executable |
| `lib.detect.find_file` | Find a file in search paths |
| `lib.detect.find_library` | Find a library (name + paths) |
| `lib.detect.find_package` | Find a package (pkg-config, builtin detectors) |
| `lib.detect.find_program` | Find a program in PATH |
| `detect.sdks.find_cuda` | Find CUDA SDK |
| `detect.sdks.find_ndk` | Find Android NDK |
| `detect.packages.find_openssl` | Find OpenSSL |
| `detect.packages.find_zlib` | Find zlib |
| `core.project.config` | Access project configuration |
| `core.project.depend` | Dependency/file change tracking |
| `core.project.option` | Access option definitions |
| `core.base.option` | Access command-line options |
| `core.base.global` | Access global configuration |
| `core.base.task` | Run xmake tasks programmatically |
| `core.ui.*` | Terminal UI components |
| `core.language.language` | Language extension registration |
| `utils.archive.*` | Archive (.tar, .zip) extraction |
| `net.*` | Network/HTTP utilities |
| `devel.git.*` | Git operations |
| `async.runjobs` | Parallel job execution |
| `async.jobgraph` | Job dependency graph |

Example:
```lua
import("lib.detect.find_tool")

on_load(function(target)
    local gcc = find_tool("gcc")
    if gcc then
        print("found gcc at", gcc.program)
    end
end)
```

### `find_package` — shortcut

```lua
-- Find a single package
local pkg = find_package("openssl", {required = false})
if pkg then
    target:add("links", pkg.links)
    target:add("linkdirs", pkg.linkdirs)
    target:add("includedirs", pkg.includedirs)
end

-- Find multiple packages
local packages = find_packages("openssl", "zlib", "curl")
```

---

## 5. Compiler/Detect Libraries

### `lib.detect.find_tool`

```lua
import("lib.detect.find_tool")

local tool = find_tool("clang", {version = true})
-- Returns: {program = "/usr/bin/clang", version = "15.0.0"}
```

### `lib.detect.find_package`

```lua
import("lib.detect.find_package")

local pkg = find_package("openssl", {
    paths = {"/usr/local/opt/openssl"},
    required = false,  -- Don't error if not found
})
-- Returns: {links = {"ssl", "crypto"}, linkdirs = {...}, includedirs = {...}}
```

### `lib.detect.find_file`

```lua
import("lib.detect.find_file")

local header = find_file("python.h", {"/usr/include", "/usr/local/include"})
```

---

## 6. Build Batch Commands

Inside `on_buildcmd_file` or rule scripts, you can use batch commands:

```lua
on_buildcmd_file(function(target, batchcmds, sourcefile, opt)
    batchcmds:show("compiling %s", sourcefile)
    batchcmds:vrunv("gcc", {"-c", sourcefile, "-o", opt.objectfile})
    batchcmds:cp("src.txt", "dst.txt")
end)
```

---

## 7. Private Target Data

Store and retrieve arbitrary data on a target:

```lua
on_load(function(target)
    target:data_set("mykey", {some = "data"})
end)

after_build(function(target)
    local data = target:data("mykey")
    print(data.some)  -- "data"
end)
```

---

## 8. Getting Configuration

```lua
get_config("lc_enable_dsl")           -- Get config option value
get_config("my_option")               -- Get any config value
has_config("lc_enable_dsl")           -- Boolean check
has_package("spdlog")                 -- Check if package is available
```

---

## 9. Complete Scripting Example

```lua
import("lib.detect.find_tool")
import("core.base.option")

target("my-scripted-target")
set_kind("binary")
add_files("src/*.cpp")

on_load(function(target)
    -- Print build info
    local verbose = option.get("verbose")
    if verbose then
        print("Building for:", target:plat(), target:arch())
    end

    -- Find a tool
    local clang = find_tool("clang")
    if clang then
        print("using clang:", clang.program)
    end

    -- Dynamic files based on platform
    if target:is_plat("windows") then
        target:add("files", "src/*.win.cpp")
        target:add("syslinks", "Advapi32")
    end

    -- Store data for later use
    target:data_set("build_time", os.time())
end)

before_build(function(target)
    -- Pre-build validation
    if not os.exists("src/main.cpp") then
        raise("main.cpp not found")
    end
end)

after_build(function(target)
    -- Post-build: copy output
    local target_file = target:targetfile()
    if os.isfile(target_file) then
        os.cp(target_file, path.join("$(projectdir)", "dist"))
        print("copied to dist/")
    end

    -- Report build time
    local start = target:data("build_time")
    local elapsed = os.mclock() - start
    print("build completed in", elapsed, "ms")
end)

target_end()
```

---

## Summary

| Module | Key APIs |
|---|---|
| `os` | `cp`, `mv`, `rm`, `mkdir`, `run`, `exec`, `iorun`, `files`, `dirs`, `isfile`, `exists`, `getenv`, `setenv`, `host`, `arch`, `sleep`, `cd`, `scriptdir`, `projectdir` |
| `io` | `readfile`, `writefile`, `load`, `save`, `open`, `gsub`, `replace`, `cat`, `tail`, `stdin`, `stdout`, `stderr` |
| `path` | `join`, `absolute`, `relative`, `basename`, `filename`, `extension`, `directory`, `normalize` |
| `table` | `join`, `join2`, `clone`, `wrap`, `unwrap`, `contains`, `unique`, `keys`, `values`, `map`, `find`, `empty` |
| `string` | all Lua standard + `vformat`, `format` |
| `utils` | `dump`, `assert`, `error`, `warning`, `trycall` |
| `hash` | `md5`, `sha1`, `sha256`, `uuid`, `uuid4`, `rand32`, `rand64`, `xxhash32/64/128` |
| `xmake` | `version`, `arch`, `branch`, `programdir`, `luajit` |
| `print` | `print`, `printf`, `cprint`, `cprintf`, `vprint`, `dprint` |
| try/catch | `try`, `catch`, `finally`, `raise` |
| `import` | Load extension modules from `modules/` |
| `find_package` | Find system packages (pkg-config, builtin detectors) |
| `find_packages` | Find multiple packages at once |
| `get_config` / `has_config` | Access configuration options |


---

# XMake API Reference (XMake 3.1.2)

This appendix synchronizes the skill with the English documentation in `D:\xmake-docs\docs`. It is an API index, not a replacement for parameter-level documentation. Paths below are relative to that directory. Prefer the documented API for the project's minimum XMake version; badges such as **3.1.2+** require a newer XMake than LuisaCompute's minimum.

## Reference Map

| Area | Documentation | Use it for |
|---|---|---|
| Targets and lifecycle | `api\description\project-target.md` | Every target `set_*`/`add_*` API, file options, hooks, tests |
| Global project APIs | `api\description\global-interfaces.md` | Includes, project/version, configuration, requirements, namespaces, addons |
| Options | `api\description\configuration-option.md` | Option declaration, checks, menu metadata, detected headers/types/functions |
| Rules | `api\description\custom-rule.md` | Rule hooks and `batchcmds` |
| Toolchains | `api\description\custom-toolchain.md` | Custom toolchain declaration, discovery, toolsets |
| Conditions and checks | `api\description\conditions.md`, `helper-interfaces.md` | Platform queries, compile checks, config-header checks |
| Policies and rules | `api\description\builtin-policies.md`, `builtin-rules.md` | Build behavior and reusable rules |
| Packages | `api\description\package-dependencies.md` | Inline package definitions and package components |
| Tasks | `api\description\plugin-and-task.md` | Project command definitions and menus |
| Distribution packages | `api\description\xpack-interfaces.md`, `xpack-component-interfaces.md` | Installers and archives with `xmake pack` |
| Script instances | `api\scripts\target-instance.md`, `option-instance.md`, `package-instance.md` | Methods available on callback objects |
| Built-in Lua modules | `api\scripts\builtin-modules` | Sandbox `os`, `io`, `path`, `table`, `string`, and related modules |
| Extension modules | `api\scripts\extension-modules` | Detection, networking, concurrency, serialization, caches, UI, and tools |
| Native Lua modules | `api\scripts\native-modules.md` | Building/importing native module.shared/module.binary modules |
| CLI workflows | `guide\basic-commands`, `guide\extensions\builtin-plugins.md` | Current command options, inspection, checking, watching, testing |

## Description-Scope APIs

### Global and Project Configuration

| API | Purpose / important forms |
|---|---|
| `includes(path, ..., {rootdir = ...})` | Include subproject files/directories; parent definitions flow into included scopes, not back upward. Builtins include `@builtin/check`, `@builtin/qt`, and `@builtin/xpack`. |
| `set_project(name)`, `set_xmakever(version)` | Project identity and minimum XMake version. |
| `set_version(version, {build = "%Y%m%d%H%M", soname = true})` | Project version; `soname` may also be an explicit version string. |
| `add_moduledirs(dirs)`, `add_plugindirs(dirs)` | Add Lua module and project task/plugin search directories. |
| `get_config(name)`, `set_config(name, value)` | Read or set configuration; command-line values override defaults. |
| `set_defaultplat`, `set_defaultarchs`, `set_defaultmode` | Default platform, architectures, and mode. |
| `set_allowedplats`, `set_allowedarchs`, `set_allowedmodes` | Restrict accepted values. |
| `namespace(name)` ... `namespace_end()` | Isolate targets/options/rules/tasks/toolchains/packages; `namespace(name, function() ... end)` is the LSP-friendly alternative. Cross-namespace references use `ns::name`. |
| `add_requires`, `add_requireconfs`, `add_repositories` | Declare packages, override transitive requirements, and register repositories. |
| `add_addons("name 1.0.x", ...)` | Declare versioned addons and produce `xmake-addons.lock` (3.1.1+). |

Addon references use `@addon/<name>/<payload>` for files and `@addon.<name>.<module>` for imports; an addon refers to its own modules as `@self.<module>`.

### Option Declaration

Use `option("name")` ... `option_end()` or the inline form `option("name", {default = false, description = "...", showmenu = true})`. Attach options to targets with `add_options("name")` and query them with `get_config`, `has_config`, or `is_config`.

| Family | APIs |
|---|---|
| Dependencies/checks | `add_deps`, `before_check`, `on_check`, `after_check`; `on_check` performs detection when no fixed default short-circuits it |
| Metadata | `set_default`, `set_values`, `set_showmenu`, `set_category`, `set_description` |
| Link detection | `add_links`, `add_linkdirs`, `add_rpathdirs` |
| Source detection | `add_cincludes`, `add_cxxincludes`, `add_ctypes`, `add_cxxtypes`, `add_csnippets`, `add_cxxsnippets`, `add_cfuncs`, `add_cxxfuncs` |

### Target APIs and Options Not Covered Above

| API | Important options / semantics |
|---|---|
| `add_files(files, ..., opts)` | Per-file `defines`, `languages`, `includedirs`, `rules`, `sourcekind`, compiler/linker flags, `force = {cxflags = ...}`, `unity_group`, `unity_ignored`, and `values`. Per-file rules take precedence over target rules; use `override = true` where documented. |
| `add_headerfiles(..., opts)` | `prefixdir`, `rootdir`, `filename`; `install = false` keeps the header out of installation while retaining it in IDE/project file lists. |
| `add_installfiles(..., opts)` | Supports `prefixdir`, `rootdir`, and `filename`. |
| `add_configfiles(..., opts)` | `variables`, `pattern`, `onlycopy`, and output path controls; pair with `set_configdir`, `set_configvar(name, value, {quote = false, escape = true})`. |
| `add_linkgroups(..., {group = true})` | Linker groups; also supports documented `whole` and `static` modes. `add_linkorders` controls link sequence. |
| `set_toolchains(...)`, `add_toolchains(...)` | Supports `name[option]@package`, e.g. `clang@llvm-10`, `mingw[clang]@llvm-mingw`, `msvc[vs=2025]`, `ndk[gcc]`, and `clang-cl[llvm]`. |
| `set_toolset(tool, program)` | Override a specific tool; `set_toolset("cc", "/path/to/clang")`. |
| `set_enabled(false)` vs `set_default(false)` | Disabled targets are unavailable; non-default targets are merely skipped by a default build and can still be named explicitly or tested. |
| `set_group`, `add_filegroups(name, {rootdir = ..., files = ..., mode = "tree"})` | Target grouping for CLI/IDEs; filegroup mode is `plain` or `tree`. |
| `set_values`, `add_values` | Rule/target custom values, read with `target:values(name[, sourcefile])`. |
| `set_runtimes`, `set_languages`, `set_encodings`, `set_exceptions` | Runtime/standard/encoding/exception configuration; encodings include `utf-8` and `source:utf-8`. |
| `add_vectorexts`, `add_embeddirs`, `add_forceincludes`, `add_extrafiles` | SIMD extensions, C23 `#embed` directories, forced includes, and IDE-only files. |
| `add_tests` | See the test section and `guide\basic-commands\run-tests.md` for the complete option set. |

Complete target declaration-name checklist (use the target reference for signatures):

| Family | APIs |
|---|---|
| Definition/identity | `target`, `target_end`, `set_kind`, `set_enabled`, `set_default`, `set_basename`, `set_filename`, `set_prefixname`, `set_suffixname`, `set_extension`, `set_group` |
| Output/install/run paths | `set_targetdir`, `set_objectdir`, `set_dependir`, `set_installdir`, `set_prefixdir`, `set_rundir`, `set_runargs`, `add_runenvs`, `set_runenv` |
| Compilation | `set_symbols`, `set_warnings`, `set_optimize`, `set_languages`, `add_languages`, `set_fpmodels`, `set_runtimes`, `set_exceptions`, `set_encodings`, `set_strip`, `add_vectorexts` |
| PCH/imports | `set_pcheader`, `set_pcxxheader`, `set_pmheader`, `set_pmxxheader`, `add_imports`, `add_forceincludes` |
| Files | `add_files`, `remove_files`, `add_headerfiles`, `remove_headerfiles`, `add_installfiles`, `add_configfiles`, `add_extrafiles`, `add_filegroups`, `add_embeddirs` |
| Includes/definitions | `add_includedirs`, `add_sysincludedirs`, `add_defines`, `add_undefines`, `set_configvar` |
| Linking | `add_deps`, `add_links`, `add_syslinks`, `add_linkorders`, `add_linkgroups`, `add_linkdirs`, `add_rpathdirs`, `add_frameworks`, `add_frameworkdirs`, `add_ldflags`, `add_arflags`, `add_shflags` |
| Language flags | `add_cflags`, `add_cxflags`, `add_cxxflags`, `add_mflags`, `add_mxflags`, `add_mxxflags`, `add_scflags`, `add_asflags`, `add_gcflags`, `add_dcflags`, `add_rcflags`, `add_fcflags`, `add_zcflags`, `add_cuflags`, `add_culdflags`, `add_cugencodes`, `add_ascnpuarchs` |
| Dependencies/toolchain | `add_rules`, `add_options`, `add_packages`, `set_options`, `set_toolchains`, `set_toolset`, `set_plat`, `set_arch`, `set_policy` |
| Custom/generated config | `set_values`, `add_values`, `set_configdir`, `add_configfiles`, `add_tests` |
| Load/prepare hooks | `on_load`, `after_load`, `on_config`, `on_prepare`, `on_prepare_file`, `on_prepare_files`, plus `before_prepare`, `after_prepare`, `before_prepare_file`, `after_prepare_file`, `before_prepare_files`, `after_prepare_files` |
| Build/link hooks | `on_build`, `on_build_file`, `on_build_files`, `on_link`, plus `before_build`, `after_build`, `before_build_file`, `after_build_file`, `before_build_files`, `after_build_files`, `before_link`, `after_link` |
| Other action hooks | `on_clean`, `on_package`, `on_install`, `on_uninstall`, `on_run`, `on_test`, with corresponding `before_clean`/`after_clean`, `before_package`/`after_package`, `before_install`/`after_install`, `before_uninstall`/`after_uninstall`, `before_run`/`after_run`, and `before_test`/`after_test` |

`set_configvar` templates provide `${VERSION}`, `${VERSION_MAJOR}`, `${VERSION_MINOR}`, `${VERSION_ALTER}`, `${VERSION_BUILD}`, `${VERSION_PRERELEASE}`, `${PLAT}`, `${ARCH}`, `${MODE}`, `${DEBUG}`, and `${OS}` where applicable. In 3.1.2+, prerelease/build values also come from semantic-version metadata. Git templates include `${GIT_COMMIT}`, `${GIT_COMMIT_LONG}`, `${GIT_COMMIT_DATE}`, `${GIT_BRANCH}`, `${GIT_TAG}`, `${GIT_TAG_LONG}`, and `${GIT_CUSTOM}`.

### Custom Rules and Batch Commands

```lua
rule("generate")
    set_extensions(".in")
    add_imports("core.project.depend")
    on_buildcmd_file(function(target, batchcmds, sourcefile, opt)
        local output = target:autogenfile(sourcefile, {filename = path.basename(sourcefile) .. ".cpp"})
        batchcmds:show_progress(opt.progress, "generating %s", output)
        batchcmds:vrunv("generator", {sourcefile, output})
        batchcmds:add_depfiles(sourcefile)
    end)
rule_end()

target("app")
    add_files("src/*.in", {rules = "generate"})
```

- Rule scope: `rule`, `rule_end`, `set_extensions`, `add_deps`, `add_imports`, and `on_load`/`on_config` plus build/link/clean/package/install/uninstall hooks.
- File forms: `on_build_file(target, sourcefile, opt)`, `on_build_files(target, sourcebatch, opt)`, `on_buildcmd_file(target, batchcmds, sourcefile, opt)`, and `on_buildcmd_files(target, batchcmds, sourcebatch, opt)`. Use `on_buildcmd_*` when exported IDE/Ninja/Make generators must reproduce the rule.
- Batch commands: `show`, `show_progress`, `runv`, `vrunv`, `execv`, `vexecv`, `lua`/`vlua`, `mkdir`, `rmdir`, `cp`, `mv`, `rm`, `ln`, `cd`, `compile`/`compilev`, `link`, `add_depfiles`, `add_depvalues`, `set_depmtime`, `set_depcache`, and `call` (3.1.0+). `batchcmds:call(function_or_script, argv, opt)` is not exportable to all project generators.
- Register generated source/object files in `after_load`; use `target:autogenfile`, `target:objectfile`, and the mutable `target:objectfiles()` list when generated objects participate in linking.
- Rule dependencies accept ordering configuration such as `add_deps("other_rule", {order = true})`.
- Rule before/after hooks include `before_config`/`after_config`, `before_link`/`after_link`, `before_build`/`after_build`, `before_clean`/`after_clean`, `before_package`/`after_package`, `before_install`/`after_install`, `before_uninstall`/`after_uninstall`, and the file/batch forms `before_build_file`/`after_build_file`, `before_build_files`/`after_build_files`, `before_buildcmd_file`/`after_buildcmd_file`, and `before_buildcmd_files`/`after_buildcmd_files`.

### Custom Toolchains

Use `toolchain("name")` ... `toolchain_end()` with `set_kind("cross"|"standalone")`, `set_toolset`, `set_sdkdir`, `set_bindir`, `on_check(function(toolchain) ... end)`, and `on_load(function(toolchain) ... end)`. In `on_load`, use `toolchain:add("cxflags"|"ldflags"|"shflags"|"asflags"|"mxflags"|"includedirs"|"linkdirs", ...)`. Standalone toolchains are mutually exclusive with other toolchains. List discovered toolchains with `xmake show -l toolchains`.

### Detection Helpers and Conditions

After `includes("@builtin/check")`, use:

- `check_links`, `check_cfuncs`, `check_cxxfuncs`, `check_ctypes`, `check_cxxtypes`, `check_cincludes`, `check_cxxincludes`, `check_csnippets`, `check_cxxsnippets`.
- `check_features`, `check_macros`, `check_sizeof` (cross-safe), `check_alignof` (3.1.0+), and `check_bigendian`.
- The corresponding `configvar_check_*` family writes results into generated configuration headers.
- Conditions: `is_os`, `is_arch`, `is_plat`, `is_host`, `is_subhost`, `is_cross`, `is_mode`, `is_kind`, `is_config`, `has_config`, and `has_package`. Most accept Lua-pattern alternatives.

## Built-in Policies

Set globally with `set_policy(name, value)`, on a target with the same target-scope API, or with `--policies=name,other:n,value_policy:value`. This list follows `api\description\builtin-policies.md`; read that file before relying on a policy default.

| Family | Policies |
|---|---|
| Flag checking | `check.auto_ignore_flags`, `check.auto_map_flags`, `check.target_package_licenses` |
| Core build | `build.across_targets_in_parallel`, `build.fence`, `build.merge_archive`, `build.release.strip`, `build.ccache`, `build.ccache.global_storage`, `build.warning`, `build.optimization.lto`, `build.c++.dynamic_debugging`, `build.cuda.devlink`, `build.always_update_configfiles`, `build.intermediate_directory`, `build.rpath` |
| Build execution/output | `build.progress_style`, `build.linker.output`, `build.jobgraph`, `build.progress_show_target`, `build.distcc.remote_only` |
| Sanitizers | `build.sanitizer.address`, `.thread`, `.memory`, `.leak`, `.undefined` (use each full `build.sanitizer.*` name) |
| C++ modules | `build.c++.modules`, `build.c++.modules.std`, `build.c++.modules.culling`, `build.c++.modules.reuse`, `build.c++.modules.reuse.nocheck`, `build.c++.modules.reuse.strict`, `build.c++.modules.fallbackscanner`, `build.c++.modules.clang.fallbackscanner`, `build.c++.modules.msvc.fallbackscanner`, `build.c++.modules.gcc.fallbackscanner`, `build.c++.modules.gcc.cxx11abi`, `build.c++.modules.non_cascading_changes`, `build.c++.modules.hide_dependencies`, `build.c++.modules.two_phases`, `build.c++.modules.clang.precompile_reduced_bmi` (3.1.2+) |
| MSVC/runtime | `build.c++.msvc.runtime` (`MT` or `MD`) |
| Preprocessor | `preprocessor.linemarkers`, `preprocessor.gcc.directives_only` |
| Install/run | `install.rpath`, `install.strip_packagelibs`, `run.autobuild`, `run.windows_error_dialog` |
| Package requirements | `package.requires_lock`, `package.precompiled`, `package.fetch_only`, `package.install_only`, `package.install_always`, `package.install_locally`, `package.host.install_locally` (3.1.1+), `package.keep_source` |
| Package compatibility/config | `package.librarydeps.strict_compatibility`, `package.strict_compatibility`, `package.resolve_depconflict`, `package.sync_requires_to_deps`, `package.xmake.pass_depconfs`, `package.include_external_headers`, `package.inherit_external_configs`, `package.download.http_headers` |
| Package builds | `package.cmake_generator.ninja`, `package.msbuild.multi_tool_task`, `package.merge_staticlibs`, `package.build.ccache` |
| Tests | `test.stop_on_first_failure`, `test.return_zero_on_failure` |
| Platform/network/diagnostics | `platform.longpaths`, `windows.manifest.uac`, `windows.manifest.uac.ui`, `network.mode`, `diagnosis.check_build_deps` |
| Compatibility/generators | `compatibility.version` (`2.0` or `3.0`), `generator.compile_commands`, `generator.vsxmake.root_sln` |

High-impact defaults include `build.ccache`, `build.across_targets_in_parallel`, `build.release.strip`, `build.jobgraph`, `build.progress_show_target`, `generator.compile_commands`, and `run.windows_error_dialog` being enabled in current XMake. Confirm the installed version with `xmake show -l policies`.

## Built-in Rules

The complete definitions and options are in `api\description\builtin-rules.md`.

| Family | Rule names |
|---|---|
| Modes | `mode.debug`, `mode.release`, `mode.releasedbg`, `mode.minsizerel`, `mode.check`, `mode.profile`, `mode.coverage`, `mode.valgrind`; legacy `mode.asan`, `mode.tsan`, `mode.lsan`, and `mode.ubsan` are deprecated in favor of policies |
| Unity/modules | `c.unity_build`, `c++.unity_build` with `batchsize`, `uniqueid`; per-file `unity_group` and `unity_ignored` |
| Symbols | `utils.symbols.export_all` (`export_classes`, `export_filter`), `utils.symbols.export_list` |
| Embedding/replacement | `utils.bin2c`, `utils.bin2obj` (including `transform` in 3.1.0+), `utils.replace` (3.0.9+) |
| Shaders | `utils.glsl2spv`, `utils.hlsl2spv`; both support SPV output and `bin2c` integration |
| Install metadata | `utils.install.cmake_importfiles`, `utils.install.pkgconfig_importfiles` |
| Generator integration | `plugin.vsxmake.autoupdate`, `plugin.compile_commands.autoupdate` |
| Language bindings | `python.library`, `nodejs.module` |
| Qt | `qt.static`, `qt.shared`, `qt.console`, `qt.quickapp`, `qt.quickapp_static`, `qt.widgetapp`, `qt.widgetapp_static` |
| Xcode | `xcode.bundle`, `xcode.framework`, `xcode.application` |
| Windows/WDK | `win.sdk.application`, `wdk.env.kmdf`, `wdk.env.umdf`, `wdk.env.wdm`, `wdk.driver`, `wdk.binary`, `wdk.static`, `wdk.shared`, `wdk.tracewpp`, `wdk.sdk.dotnet` |
| Other platforms | `android.native_app`, `utils.ipsc` |

## Package APIs

### Requirements and Package Definitions

- `add_requires("zlib 1.2.x", "fmt ~10.0", {optional = true, system = false, verify = true, debug = false, private = true, alias = "...", configs = {...}})` supports version ranges, branches/commits, and third-party prefixes such as `vcpkg::`, `conan::`, `brew::`, and `pkgconfig::`.
- `add_requireconfs("libpng.*", {system = false, override = true, configs = {...}})` configures matching transitive dependencies; `add_repositories("name url [branch]")` registers custom repositories.
- Enable `package.requires_lock` for `xmake-requires.lock`; use `xmake require --upgrade` deliberately. `package.install_locally` is target-side in current XMake; `package.host.install_locally` controls host packages (3.1.1+).
- Inline definition: `package("name")` ... with metadata `set_homepage`, `set_description`, `set_kind("library"|"binary"|"toolchain")`; sources `set_urls`, `add_urls`, `add_versions`, `add_versionfiles`, `set_sourcedir`, `add_patches`, `add_extsources`.
- Consumer flags: `add_links`, `add_syslinks`, `add_linkorders`, `add_linkgroups`, `add_frameworks`, `add_linkdirs`, `add_includedirs`, `add_bindirs`, `add_defines`.
- Structure: `add_configs`, `add_schemes`, `add_deps`, `add_components`, `set_base`; hooks `on_load`, `on_fetch`, `on_check`, `on_download`, `on_install`, `on_test`, and `on_component`.
- For pkg-config packages in 3.1.2+, `configs.configdirs` supplies package-local search directories without changing global `PKG_CONFIG_PATH`.

### Package Build Tools

Import `package.tools.<tool>` and call the documented functions:

| Module | Functions |
|---|---|
| `package.tools.cmake` | `configure`, `build`, `install` |
| `package.tools.autoconf` | `configure`, `make`, `build`, `install` |
| `package.tools.meson` | `generate`, `build`, `install` |
| `package.tools.make` | `make`, `build`, `install` |
| `package.tools.ninja` | `build`, `install` |
| `package.tools.msbuild` | `build` |
| `package.tools.xmake` | `install` |

### Package Instance

Inside package hooks, the `package` object provides:

- Generic/configuration: `name`, `get`, `set`, `add`, `license`, `description`, `alias`, `urls`, `sourcehash`, `kind`, `version`, `version_str`, `versions`, `config(name)`, `config_set(name, value)`, `configs`, `buildhash`, `patches`.
- Platform: `plat`, `arch`, `targetos`, `targetarch`, `is_plat`, `is_arch`, `is_targetos`, `is_targetarch`, `is_cross`, `is_supported`.
- Classification: `is_binary`, `is_toolchain`, `is_library`, `is_toplevel`, `is_thirdparty`, `is_debug`, `debug`.
- Dependencies: `dep`, `deps`.
- Paths/environment: `cachedir`, `installdir(...)`, `scriptdir`, `envs`, `getenv(name)`, `setenv(name, ...)`, `addenv(name, ...)`.
- Schemes: `scheme`, `current_scheme`.
- Checks: `has_cfuncs`, `has_cxxfuncs`, `has_ctypes`, `has_cxxtypes`, `has_cincludes`, `has_cxxincludes`, `check_csnippets`, `check_cxxsnippets`, `check_fcsnippets`.

See `api\scripts\package-instance.md` for hook-specific availability and exact options.

## Script Instance APIs

### Target Instance

Documented methods in `api\scripts\target-instance.md` include:

- Identity/config: `target:name()`, `kind()`, `get`, `set`, `add`, `is_plat`, `is_arch`, `is_arch64`.
- Outputs: `targetfile`, `targetdir`, `basename`, `filename`, `artifactfile(kind)` (for example a Windows DLL import library), `symbolfile`, `installdir`, `autogendir`.
- Sources/objects: `objectfile(sourcefile)`, mutable `objectfiles()`, documented `headerfiles()`, `sourcebatches`; source also accepts `headerfiles(outputdir, opt)`.
- Detection: `has_cxxfuncs`, `has_ctypes`, `has_cxxtypes`, `has_cflags`, `has_cxxflags`, `has_cincludes`, `has_cxxincludes`, `check_csnippets`, `check_cxxsnippets`, `check_sizeof`, `has_features`.

Commonly used source-supported methods not all listed on that page include `fullname`, `sourcefiles`, `dependfile`/`dependfiles`, `autogenfile`, `deps`/`dep`/`orderdeps`, `toolchain`, `compiler`, `linker`, `tool`/`toolconfig`, `data`/`data_set`, `values`, `fileconfig`/`fileconfig_set`/`fileconfig_add`, `policy`, `runtimes`/`has_runtime`, `pkg`/`pkgs`, `rule`, `rule_enable`, `extraconf`, `clone`, and the `is_binary`/`is_shared`/`is_static`/`is_library`/`is_headeronly`/`is_phony`/`is_enabled`/`is_default`/`is_rebuilt` predicates. Confirm source-supported methods against the installed XMake before using them in public scripts.

### Option Instance

`option:name`, `fullname`, `namespace`, `description`, `value`, `enabled`, `enable(enabled[, opt])`, `set_value`, `clear`, `get`, `set`, `add`, `remove`, `deps`, `dep`, `orderdeps`, and `extraconf(name, item, key)` are documented in `api\scripts\option-instance.md`.

### Native Modules

Use `add_moduledirs("modules")`, `add_rules("module.shared")` or `add_rules("module.binary")`, then `import("foo", {always_build = true})`. Native dynamic modules include `<xmi.h>` and export `luaopen(foo, lua_State *lua)`; binary modules communicate through arguments/stdout. The correct rule names are singular `module.shared` and `module.binary` (the native-modules prose has a known `modules.shared` typo).

## Tasks, Addons, and XPacks

### Tasks and Addons

- Define a project command with `task("name")`, `set_menu({usage = ..., options = ...})`, `set_category`, `on_run(function(...) ... end)`, and `task_end()`. Menu option kinds are `k` (boolean), `kv` (key/value), `v` (single value), and `vs` (multiple values).
- Run tasks programmatically with `import("core.base.task")` or `import("core.project.task")` as appropriate, then `task.run(name, options, ...)`.
- Addons (3.1.1+) use an `addon("name")` manifest with metadata/source/dependency APIs and payload directories `plugins`, `rules`, `toolchains`, `templates`, `modules`, and `includes`.
- Manage them with `xmake addon --install|--remove|--search|--list|--upgrade`; use `-y` for automation. `xmake plugin` is deprecated in favor of `xmake addon`.

### XPack

Start with `includes("@builtin/xpack")` and `xpack("name")`; build with `xmake pack [-f format] [-o outputdir]`.

- Metadata: `set_version`, `set_homepage`, `set_title`, `set_description`, `set_author`, `set_maintainer`, `set_copyright`, `set_license`, `set_licensefile`, `set_company`, `set_iconfile`, `set_displayicon` (3.1.2+).
- Input/output: `set_inputkind("binary"|"source")`, `set_formats`, `set_basename`, `set_extension`, `set_bindir`, `set_libdir`, `set_includedir`, `set_prefixdir`, `set_specfile`, `set_specvar`.
- Contents: `add_targets`, `add_components`, `add_sourcefiles`, `add_installfiles`, `add_buildrequires`.
- Hooks: `on_load`, `before_package`, `on_package`, `after_package`, `before_buildcmd`, `on_buildcmd`, `after_buildcmd`, `before_installcmd`, `on_installcmd`, `after_installcmd`, `before_uninstallcmd`, `on_uninstallcmd`, and `after_uninstallcmd`.
- Components support `set_title`, `set_description`, `set_default`, `on_load`, `add_sourcefiles`, `add_installfiles`, and the same `before_`/`on_`/`after_` installcmd/uninstallcmd hooks.
- Documented formats include `nsis`, `inno` (3.1.2+), `zip`, `targz`, `srczip`, `srctargz`, `runself`, `rpm`, `srpm`, and `deb`; `wix`, `dmg`, `appimage`, `tarxz`, and `srctarxz` also exist in the packaging plugins but are omitted from the current format table. `set_nsis_displayicon` is deprecated; use `set_displayicon`.

## Built-in Lua Module Index

The full signatures live under `api\scripts\builtin-modules`. This index includes APIs missing from the earlier quick reference.

| Module | APIs |
|---|---|
| `os` files/processes | `access`, `cp`, `mv`, `rm`, `trycp`, `trymv`, `tryrm`, `cd`, `rmdir`, `mkdir`, `touch`, `isdir`, `isfile`, `exists`, `islink`, `dirs`, `files`, `filedirs`, `exit`, `isexec`, `run`, `runv`, `vrun`, `vrunv`, `exec`, `execv`, `iorun`, `iorunv` |
| `os` paths/system | `tmpdir`, `tmpfile`, `curdir`, `filesize`, `scriptdir`, `programdir`, `programfile`, `projectdir`, `projectfile`, `workingdir`, `arch`, `host`, `subhost`, `subarch`, `is_host`, `is_arch`, `is_subhost`, `is_subarch`, `ln`, `readlink`, `features`, `term`, `shell`, `cpuinfo(key)`, `meminfo(key)`, `default_njob`, `mclock`, `getpid`, `uid`, `gid`, `nuldev`, `fscase`, `isroot` |
| `os` environment/misc | `getenv`, `setenv`, `addenv`, `getenvs`, `setenvs`, `addenvs`, `joinenvs`, `setenvp`, `addenvp`, `argv`, `args`, `pbpaste`, `pbcopy`, `raise`, `raiselevel`, `atexit`; `os.files(pattern, {async = true})` is available in 3.0.5+ |
| `io` | `open`, `load`, `save` (Lua serialization), `readfile`, `writefile`, `gsub`, `replace`, `insert`, `tail`, `cat`, `print(filename, ...)`, `printf(filename, ...)`, `lines`, `stdfile`, `openlock`, `read`, `write`, `flush`, `readable`, `isatty`, `close`, `convert` |
| `path` | `new`, `normalize`, `join`, `translate`, `basename` (no suffix), `filename` (with suffix), `extension`, `directory`, `relative`, `absolute`, `is_absolute`, `split`, `sep`, `islastsep`, `splitenv`, `joinenv`, `envsep`, `pattern`, `unix`, `cygwin`, `instance_of` |
| `table` | `join`, `join2`, `shallow_join`, `shallow_join2`, `unique`, `reverse_unique`, `slice`, `contains`, `orderkeys`, `orderpairs`, `keys`, `values`, `clone`, `wrap`, `unwrap`, `reverse`, `append`, `swap`, `empty`, `is_array`, `is_dictionary`, `find`, `find_if`, `find_first`, `find_first_if`, `remove_if`, `to_array`, `inherit`, `wrap_lock`, `wrap_unlock`, `pack`, `unpack` |
| `string` extensions | `startswith`, `endswith`, `split`, `trim`, `ltrim`, `rtrim`, `lastof`, `replace`, `serialize`, `deserialize`, `ipattern`, `levenshtein`, plus `vformat` and standard Lua string functions |
| `hash` | `md5`, `sha1`, `sha256`, `uuid([name])`, `xxhash32`, `xxhash64`, `xxhash128`, `strhash32`, `strhash64`, `strhash128`, `rand32`, `rand64`, `rand128` |
| `utf8` | `len`, `char`, `codepoint`, `codes`, `offset`, `sub`, `reverse`, `lastof`, `find`, `width`, `byte` |
| OS-specific | `winos.version`, `processes`, `set_error_mode`, `registry_keys`, `registry_values`, `registry_query`, `file_signature` (3.0.8+); `linuxos.name`, `version`, `kernelver`; `macos.version` |
| Signals/coroutines | `signal.register`, `ignore`, `reset`; standard `coroutine.create`, `resume`, `yield`, `status`, `wrap`, `running` |
| Language helpers | `import`, `inherit`, `try`/`catch`/`finally`, `raise`, `pairs`, `ipairs`, `print`, `printf`, `cprint`, `cprintf`, `format`, `vformat` |

`import` supports documented options including `rootdir`, `alias`, `inherit`, `try`, and `anonymous`; native modules additionally support `always_build`.

## Extension Module Index

Import the exact module name shown below. See its same-named file below `api\scripts\extension-modules` for parameters and return values.

| Import / family | Main APIs |
|---|---|
| `lib.detect.*` | `find_file`, `find_path`, `find_library`, `find_program`, `find_programver`, `find_package`, `find_tool`, `find_toolname`, `find_cudadevices`, `features`, `has_features`, `has_flags`, C/C++ function/include/type checks, `check_cxsnippets` |
| `net.http` | `http.download(url, output, opt)`; 3.1.2+ supports `opt.downloader` values `aria2`, `curl`, `wget`, `powershell`, and `pwsh` |
| `net.ping` | `ping(hosts, {force = true})`; returns host-to-latency results and uses cached probes |
| `async.runjobs`, `async.jobgraph` | Parallel jobs; job graphs with `new`, `add`, `add_orders`, `group` |
| `core.project.config` | `get`, `load`, `arch`, `plat`, `mode`, `builddir`, `directory`, `dump` |
| `core.project.project` | `target`, `targets`, `option`, `options`, `name`, `version` |
| `core.tool.compiler` | `compile`, `compcmd`, `compargv`, `compflags`, `has_flags`, `features`, `has_features` |
| `core.tool.linker` | `link`, `linkcmd`, `linkargv`, `linkflags`, `has_flags` |
| `core.base.json`, `core.base.xml` | `decode`, `encode`, `loadfile`, `savefile`; XML also has `find`, `scan`, `text_of`, `text`, `empty`, `comment`, `cdata`, `doctype` |
| `core.base.semver` | `new`, `try_parse`, `match`, `is_valid`, `is_valid_range`, `compare`, `satisfies`, `select`, and version accessors |
| `core.base.bytes` | Byte buffers, slicing/copy/move, signed/unsigned and endian-specific accessors, `dump`, `readonly` |
| `core.base.graph` | Directed/undirected graphs, vertices/edges, topological and partial topological sorting, cycles, clone/reverse/dump |
| `core.base.hashset`, `heap`, `list`, `queue`, `bloom_filter` | Corresponding containers and priority queues; see method indexes in their files |
| `core.base.process`, `pipe`, `socket` | Processes, pipes, TCP/UDP/Unix sockets, waiting and lifecycle operations |
| `core.base.thread`, `scheduler` | Threads, mutex/event/semaphore/queue/sharedata; coroutine scheduling, groups, locks, semaphores |
| `core.base.tty` | Terminal erasing/cursor save-restore, capabilities, `cursor_move_*`, `cursor_hide`, `cursor_show` |
| `core.base.cpu`, `libc`, `bit`, `global`, `option`, `task`, `privilege` | CPU information, native memory/pointers, bit operations, global/CLI config, task execution, privilege storage |
| `core.cache.*` | `localcache`, `memcache`, `globalcache`, `detectcache`, `global_detectcache`, each with `set`/`get`/`set2`/`get2`/`clear` |
| `core.compress.lz4` | `compress`, `decompress`, block forms, file and stream forms |
| `core.base.binutils` | `bin2c`, `bin2obj`, `readsyms`, `deplibs`, `rpath_list`, `extractlib` |
| `core.language.language` | Language/sourcekind extension introspection and loading |
| `devel.git` | Clone/init/pull/push/checkout/reset/clean/apply/branch, commits/refs/tags/branches, URL checks, and submodule operations |
| `cli.iconv`, `cli.amalgamate` | Encoding conversion (`iconv.convert`, 3.0.7+) and source amalgamation |
| `lib.lua.package` | `package.loadlib` |
| `privilege.sudo` | `has`, `run`, `runv`, `exec`, `execv`, `iorun`, `iorunv` |
| `utils.archive` | `archive`, `extract` |
| `utils.binary` | Dependency/symbol/archive helpers and `rpath.list`, `insert`, `change`, `remove`, `clean` |
| `utils.platform` | `gnu2mslib` |
| `core.ui.*` | Terminal UI: `application`, `view`, `window`, `event`, `action`, dialogs, buttons, menus, labels, text editing, scrollbars, and status bars; consult `core\ui` pages for their constructors/methods |

For downloader selection (3.1.2+), precedence is call option, `--downloader`, global configuration, then `XMAKE_DOWNLOADER`. Set a persistent default with `xmake g --downloader=curl`; an explicitly selected unavailable downloader fails rather than silently falling back.

## Additional CLI Reference

| Task | Commands / options |
|---|---|
| Configure | `xmake f -p plat -a arch -m mode --toolchain=name --builddir=dir`; `--menu`, `--export=file`, `--import=file`, `--runtimes=`, `--ccache=y/n`, `--ccachedir=dir`, `--trybuild=`, `--policies=` |
| External working directory | `xmake f -P <projectdir>` binds the current working directory to a project; `-F <file>`/`--file=` selects a project file |
| Build selection | Multiple targets (3.1.0+), `-g group-pattern`, `--files=pattern`, `--shallow`, `--dry-run`, `--linkonly`, `-jN`, `--linkjobs=N` |
| Run/debug | `xmake run -w dir -d target`; configure a debugger with `xmake f --debugger=gdb` |
| Inspect | `xmake show -l ...`, `xmake show -t target`, `xmake show --info=depgraph`; `--format=json` replaces deprecated `--json` |
| Tests | `xmake test target/test`, `target/*`, `*/name`, `-g`, `-r`, `--output-on-failure` (3.1.2+) |
| Check/watch | `xmake check [api, clang.tidy, syntax, or cuda.devlink]`; `xmake watch -c 'xmake -rv'` or `xmake watch -- <command>` |
| Packages | `xmake require --info`, `--depgraph`, `--search`, `--list`, `--clean`, `--fetch`, `--upgrade`, `--export`, `--import`; use `xrepo` for global package operations |
| Package repositories | `xmake repo -a name url [branch]`, `-r`, `-u`, `-c`, `-l`; add `-g` for global repositories |
| Project generation | `xmake project -k <compile_commands,compile_flags,cmake,ninja,make,vsxmake,...>`, with `-t`, `-m`, `-a`, `--lsp=`, and an output directory |
| Install/package | `xmake install -o dir`; `xmake package` creates an XMake package; `xmake pack` creates an XPack installer/archive |
| Global/network | `xmake g --cachedir=`, `--pkg_searchdirs=`, `--proxy=`, `--downloader=`, `--insecure-ssl=y` (`--ccachedir` belongs to `xmake f`) |
| Cache/remote services | `xmake service --ccache`, `--distcc`, or `--remote` with `--start`, `--connect`, `--disconnect`, `--stop`; see `guide\extras` before configuring CI |
| Scripts/macros | `xmake lua [script]`, `xmake lua -c code`, `--stdin`, bare REPL; `xmake macro` records/runs command macros |
| Create/update | `xmake create -l c++ -t console`, `xmake update <latest,dev,~3.1.2>` |

## Recent Documentation Changes to Remember

| Version | Relevant additions/fixes |
|---|---|
| 3.0.9 | Dependency graph inspection, `utils.replace`, `add_ascnpuarchs`, `clang-cl[llvm]`/lld-link, package-library merging in `build.merge_archive`, multi-toolchain improvements |
| 3.1.0 | `check_alignof`, `batchcmds:call`, bin2c/bin2obj `transform`, JSON output for `xmake show`, multiple targets for build/clean/install/package actions |
| 3.1.1 | Addons and `add_addons`, `package.host.install_locally`, addon lockfile/reference syntax |
| 3.1.2 | `build.c++.modules.clang.precompile_reduced_bmi`, MSVC/clang-cl PCH correctness fixes, `xmake test --output-on-failure`, downloader selection, pkg-config `configs.configdirs`, config-version prerelease/build variables, XPack Inno Setup and `set_displayicon` |

## Deprecations and Documentation Drift

- Use `$(builddir)` and `--builddir`, not deprecated `buildir` spellings.
- Use `remove_files`/`remove_headerfiles`, not legacy `del_files`; use `set_toolchains`, not legacy singular `set_toolchain`.
- Use `--format=json`, not deprecated `--json`; use sanitizer policies instead of legacy sanitizer modes; use `build.warning` instead of deprecated global `-w` warning builds where applicable.
- XMake addons supersede the old plugin manager. LuisaCompute's `batch_size` helper maps to XMake rule option `batchsize`; they are not interchangeable names.
- The docs may lead an installed binary. The current docs describe bare `is_subarch`, while XMake 3.1.1 exposes `os.is_subarch`; prefer the `os` form for compatibility. `moduleonly`, some target methods, and dependency field filters such as `add_deps(..., {links = false})` exist in source but are not fully documented.
- Official documentation has a few known prototype/example typos (for example `modules.shared` instead of `module.shared`). Resolve discrepancies against the installed source and `xmake <command> -h`, and record the tested XMake version when doing so.
