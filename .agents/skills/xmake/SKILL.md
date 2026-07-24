---
name: xmake
description: XMake build configuration, options, commands, and patterns for LuisaCompute.
---

# XMake Build System

Primary build system. Requires XMake 3.0.6+. Optional: CUDA Toolkit, Vulkan SDK, LLVM 20, Rust.

## Quick Start

```bash
xmake f -m debug -c -y
xmake build
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
`-c` clean cache, `-m <mode>` (release/debug/releasedbg/check/profile/coverage), `-p <plat>` (linux/windows/macosx), `-a <arch>` (x86_64/x64/arm64), `--check` check before building, `-y` auto-accept all prompts and skip interaction (useful in scripts/CI).

In this project `debug` mode automatically enables AddressSanitizer (ASan). To enable ASan for other modes, use `--policies=build.sanitizer.address`.

## Sanitizer Modes

XMake supports sanitizer builds through sanitizer policies. Policies propagate the sanitizer configuration to dependent packages and avoid the deprecation warnings produced by the legacy `mode.asan`/`mode.tsan`/`mode.lsan`/`mode.ubsan` rules.

### ASan in debug mode

Configure and build with debug mode as usual:

```bash
xmake f -m debug -c -y
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

## Commands

| Command | Description |
|---|---|
| `xmake clean` | Clean |
| `xmake -r` | Rebuild |
| `xmake build <target>` | Build target |
| `xmake run <target>` | Run target |
| `xmake run <target> <args>` | Run target with arguments |
| `xmake -l` | List targets |
| `xmake install -o <dir>` | Install binaries to `<dir>` |
| `xmake -y` | Auto-accept all prompts (downloads, overwrites, etc.), skip interaction |
| `xmake project -k compile_commands --lsp=clangd .vscode` | Generate `compile_commands.json` |

## Common Issues

- `-v`, `-D`, `--diagnosis` invalid; use `--verbose`
- Boolean options: `--lc_option=true`/`=false`
- Use `-c` to clean cache when reconfiguring with different options
- Use `-y` to auto-accept all prompts and skip interaction — essential in automated scripts and CI pipelines
- `lc_fallback_backend` requires both `lc_llvm_path` and `lc_embree_path`
- `lc_dx_backend` is silently disabled on non-Windows platforms
- `lc_metal_backend` is silently disabled on non-macOS platforms
- `lc_cuda_backend` is silently disabled outside Windows/Linux
- Vulkan compute shader codegen options are mutually exclusive: keep `lc_vk_backend_use_xir_spirv=true` for the default native SPIR-V path, or set it to `false` before enabling `lc_vk_backend_use_ast_llvm_spirv=true`.

# Xmake Target Writing Tutorial

## Overview

This tutorial covers how to write xmake targets using the standard xmake API, with examples drawn from real projects like LuisaCompute. The recommended style uses `on_load` callbacks for dynamic configuration, with static declarations outside.

---

## 1. Basic Target Structure

```lua
target("<name>", {kind = "static"})   -- Optional: pass kind inline
-- or
target("<name>")
set_kind("static")  -- "static", "shared", "binary", "object", "phony", "headeronly", "moduleonly"

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

- **`add_rules()` must be outside `on_load`** — they are target-level; cannot be set from inside `on_load`.
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
| Output | `targetdir`, `objectdir`, `dependir`, `rundir`, `installdir`, `prefixdir` |
| Sources | `files`, `headerfiles`, `extrafiles`, `remove_files`, `configfiles`, `installfiles` |
| Includes | `includedirs`, `sysincludedirs` |
| Defines | `defines`, `undefines`, `configvar` |
| Links | `links`, `syslinks`, `linkdirs`, `rpathdirs`, `linkorders`, `linkgroups`, `frameworks`, `frameworkdirs` |
| Compilation | `languages`, `optimize`, `warnings`, `symbols`, `runtimes`, `exceptions`, `fpmodels`, `encodings`, `strip`, `vectorexts`, `forceincludes`, `pcheader`, `pcxxheader` |
| Dependencies | `deps`, `packages`, `options`, `rules` |
| Misc | `values.*`, `runenv`, `runargs`, `enabled`, `default`, `toolchains`, `toolset`, `plat`, `arch`, `policy` |

Rules for config fields:

1. **Most config fields can be set either outside or inside `on_load`** using the equivalent `target:set("field", value)` / `target:add("field", value)` form.
2. **`add_rules()` must be outside `on_load`** — rules are target-level metadata and cannot be added from inside a script field.
3. **Static config goes outside** for readability; **dynamic/conditional config goes inside `on_load`**.
4. **Use `{public = true}` / `{interface = true}` / `{private = true}`** with `target:add()` to control inheritance of includedirs, defines, links, etc.
5. **`target:get("field")` reads a config field** inside a script field; `has_config("opt")` reads project-level options.

### Script Fields

Script fields are the lifecycle hooks where you write imperative Lua code. They receive the `target` object (and sometimes other arguments) and run at specific build phases.

Common script fields:

| Script field | Runs when | Typical use |
|---|---|---|
| `on_load(function(target) ... end)` | Target is loaded (early) | Dynamic config, conditional deps/files |
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
        local dest = path.join("$(buildir)", "publish")
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
| `add_rcflags(...)` | Rust compilation flags |
| `add_fcflags(...)` | Fortran compilation flags |
| `add_zcflags(...)` | Zig compilation flags |
| `add_cuflags(...)` | CUDA compilation flags |
| `add_culdflags(...)` | CUDA device link flags |
| `add_cugencodes(...)` | CUDA gencode settings (e.g., `"sm_30"`, `"native"`) |

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

---

## 5. Precompiled Headers (PCH)

```lua
target("my-target")
set_pcheader("precompiled.h")     -- C PCH
set_pcxxheader("precompiled.hpp") -- C++ PCH
```

Enable conditionally with:

```lua
if has_config("enable_pch") then
    set_pcxxheader("mypch.hpp")
end
```

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
if is_os("windows") then ... end      -- Target OS (e.g., "ios", "android")
if is_host("windows") then ... end    -- Host OS running xmake
if is_subhost("msys") then ... end    -- Subsystem (e.g., "msys", "cygwin")
if is_subarch(...) then ... end       -- Subsystem architecture
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

-- Loading phase
on_load(function(target)     -- When target is loaded (early)
end)
on_config(function(target)   -- After 'xmake config', before build
end)

-- Build preparation
on_prepare(function(target)     -- Source preprocessing/code generation
end)
on_prepare_file(func)           -- Single file preprocessing
on_prepare_files(func)          -- Batch file preprocessing

-- Build phase
on_build(function(target)       -- Override entire build
end)
on_build_file(func)             -- Replace single file compilation
on_build_files(func)            -- Replace batch file compilation
on_link(function(target)        -- Custom link process
end)

-- Clean / Package / Install / Run
on_clean(function(target)
end)
on_package(function(target)
end)
on_install(function(target)
end)
on_uninstall(function(target)
end)
on_run(function(target)         -- Override 'xmake run'
end)

-- Test hooks
on_test(function(target)        -- Custom test (return true=pass)
end)

-- Before/After variants exist for all of the above:
before_build(function(target) ... end)
after_build(function(target) ... end)
before_link(function(target) ... end)
after_link(function(target) ... end)
before_install(function(target) ... end)
after_install(function(target) ... end)
-- ... etc.
```

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
    runenvs = {PATH = "/usr/bin"},
    timeout = 30,
    group = "unit",
    pass_outputs = {"PASSED"},
    fail_outputs = {"FAILED"},
    should_fail = false,
    build_should_pass = true,
})

-- Or with custom test script
on_test(function(target)
    -- Return true for pass, false + error for fail
    local ok = os.execv("./my_test")
    if not ok then
        return false, "test failed"
    end
    return true
end)
```

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

if has_config("enable_pch") then
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
add_deps("runtime", "dsl")
add_files("main.cpp")
add_includedirs("include")

on_load(function(target)
    if has_config("enable_gui") then
        target:add("deps", "gui")
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
    add_deps("runtime", "dsl")
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
add_rules("build_cargo")   -- Rust/Cargo build
add_rules("lc_llvm")       -- LLVM integration
target_end()
```

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
| `target:is_mode("debug")` | Check build mode (alias for `is_mode()`) |
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
| `target:is_kind("kind")` | Check target kind |
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
add_rules("mode.release", "mode.debug")
add_requires("spdlog")

-- Library target
target("mylib")
set_kind("shared")
set_basename("mylib")
add_deps("core")
add_headerfiles("include/**.h")
add_rules("c++.unity_build", {batchsize = 8})

if has_config("enable_pch") then
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
    if has_config("use_xrepo_spdlog") then
        target:add("packages", "spdlog")
    else
        target:add("deps", "spdlog-bundled")
    end
end)

after_build(function(target)
    if is_plat("windows") then
        os.cp("$(buildir)/mydep.dll", target:targetdir())
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
2. **`add_rules()` stays outside** — cannot be set from inside `on_load`.
3. **Simple globs outside, conditional additions inside** — keep `add_files`/`add_headerfiles` outside for simple cases.
4. **Prefer `target:set()` / `target:add()` inside `on_load`** for most configuration — it's equivalent to outside calls.
5. **Visibility** — `{public = true}` propagates to dependents, `{interface = true}` propagates only to dependents, `{private = true}` (default) is local-only.
6. **`add_deps()` outside = `target:add("deps", ...)` inside** — choose whichever fits your style.
7. **All APIs listed here work at the target scope level** — use them outside `on_load` as `set_kind(...)` or inside as `target:set("kind", ...)`.---

# Lua Scripting in xmake

> Reference: `D:/xmake/core/sandbox/modules/`, `D:/xmake/modules/`, `D:/xmake/core/base/`

xmake scripts (in `on_load`, `on_build`, `after_install`, etc.) run in a **sandboxed Lua environment**. This section documents all available built-in modules and APIs.

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
path.join("a", "b", "c")          -- "a/b/c" (OS-aware)
path.join("a", "..", "b")         -- "b"
path.absolute("rel/path")         -- Full absolute path
path.relative("/abs/path", "/base") -- Relative path from base
path.basename("foo/bar.cpp")      -- "bar.cpp"
path.filename("foo/bar.cpp")      -- "bar"
path.extension("foo/bar.cpp")     -- ".cpp"
path.directory("foo/bar.cpp")     -- "foo"
path.normalize("a/./b/../c")     -- "a/c"
```

### 1.5 `string` — String Operations

All standard Lua string functions are available. Extended functions:

```lua
string.vformat("$(var) hello", ...) -- Format with built-in variables
string.format("hello %s", "world")  -- Standard Lua format
-- All standard: sub, gsub, find, match, gmatch, upper, lower, rep, reverse, char, byte, len, split
```

**Built-in variables** (resolved in strings via `$()` or vformat):

| Variable | Description |
|---|---|
| `$(host)` | Host OS (windows, linux, macosx) |
| `$(tmpdir)` | Temp directory |
| `$(curdir)` | Current directory |
| `$(scriptdir)` | Directory of the current xmake.lua |
| `$(projectdir)` | Project root directory |
| `$(buildir)` | Build output directory |
| `$(globaldir)` | Global xmake directory |
| `$(programdir)` | xmake installation directory |

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
os.match("src/*.c", "file")            -- Match with mode ("file", "dir", "alldir")
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
os.mclock()                            -- CPU clock (ms)
os.sleep(1000)                         -- Sleep ms (coroutine-safe)
os.nuldev()                            -- Null device path
os.xmakever()                          -- xmake version (semver)
os.args({"-a", "-b"})                  -- Format args array to string
os.getpid()                            -- Current process ID
os.cpuinfo()                           -- CPU info table
os.meminfo()                           -- Memory info table
```

### 1.8 `io` — File I/O

```lua
-- Read/write entire files
local data = io.readfile("path")       -- Read all text
io.writefile("path", "content")        -- Write text
local obj = io.load("data.json")       -- Load serialized object (JSON/Lua)
io.save("data.json", obj)              -- Save serialized object

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
io.cat("file", 10)                     -- Print first 10 lines
io.tail("file", 10)                    -- Print last 10 lines

-- stdin/stdout/stderr
io.stdin:read("*l")                    -- Read from stdin
io.stdout:write("hello")               -- Write to stdout
io.stderr:write("error")               -- Write to stderr
io.write("hello")                      -- Shortcut for stdout:write
io.print("hello")                      -- Write string to file
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
hash.uuid()                            -- Generate UUID v1
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
print("$(buildir)/$(mode)")            -- Resolves to build/debug etc.
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
on_buildcmd_file(function(target, batchcmds, sourcefile)
    batchcmds:show("compiling %s", sourcefile)
    batchcmds:vrun("gcc -c %s", sourcefile)
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