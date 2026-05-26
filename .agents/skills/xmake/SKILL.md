---
name: xmake
description: XMake build system — configuration, project options, commands, and common patterns for LuisaCompute
---

# XMake Build System

Primary build system. Requires XMake 3.0.6+. Optional: CUDA Toolkit, Vulkan SDK, LLVM 20, Rust.

## Quick Start

```bash
xmake f -m debug -c
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

### Flags
`-c` clean cache, `-m <mode>` (release/debug/releasedbg), `-p <plat>` (linux/windows/macosx), `-a <arch>` (x86_64/x64/arm64), `--check` check before building.

## Project Options

| Option | Default | Description |
|---|---|---|
| `lc_cuda_backend` | true | CUDA |
| `lc_vk_backend` | true | Vulkan |
| `lc_dx_backend` | true | DirectX 12 |
| `lc_metal_backend` | true | Metal |
| `lc_fallback_backend` | false | CPU fallback |
| `lc_toy_c_backend` | false | Toy C (testing) |
| `lc_enable_dsl` | true | C++ DSL |
| `lc_enable_gui` | true | GUI |
| `lc_enable_imgui` | true | ImGui |
| `lc_enable_tests` | true | Tests |
| `lc_enable_py` | true | Python bindings |
| `lc_enable_osl` | true | OSL support |
| `lc_enable_unity_build` | true | Unity build |
| `lc_enable_simd` | true | SSE/SSE2 |
| `lc_use_lto` | false | LTO |
| `lc_enable_mimalloc` | true | mimalloc |
| `lc_enable_custom_malloc` | false | Custom allocator |
| `lc_cxx_standard` | cxx20 | C++ standard |
| `lc_rtti` | false | RTTI |
| `lc_enable_xir` | false | XIR support |
| `lc_dx_cuda_interop` | false | DX-CUDA interop |
| `lc_vk_cuda_interop` | false | VK-CUDA interop |
| `lc_cuda_ext_lcub` | false | CUDA CUB ext |
| `lc_sdk_dir` | - | SDK download dir |
| `lc_bin_dir` | bin | Binary output dir |

## Build Examples

```bash
# Full
xmake f -p linux -m release --lc_cuda_backend=true --lc_vk_backend=true \
  --lc_enable_dsl=true --lc_enable_gui=true --lc_enable_tests=true -c
xmake

# Minimal
xmake f -m release --lc_enable_tests=false --lc_enable_gui=false --lc_enable_dsl=false -c
xmake

# Debug with tests
xmake f -m debug --lc_enable_tests=true -c && xmake
```

## Commands

| Command | Description |
|---|---|
| `xmake clean` | Clean |
| `xmake -r` | Rebuild |
| `xmake build <target>` | Build target |
| `xmake run <target>` | Run target |
| `xmake -l` | List targets |
| `xmake install -o <dir>` | Install |

## Test Scripts

Located under `scripts/test/xmake/`.

| Script | Description |
|---|---|
| `test_ast.py` | AST tests |
| `test_core.py` | Core library tests |
| `test_dsl.py` | DSL tests |
| `test_runtime.py` | Runtime tests |
| `test_xir.py` | XIR tests |
| `test_examples.py` | Example tests |

### Usage
These scripts only run tests. Build and configure the project first, then execute:

```bash
# Run all backends (default)
python scripts/test/xmake/test_runtime.py

# Run a specific backend only
python scripts/test/xmake/test_runtime.py cuda
python scripts/test/xmake/test_runtime.py dx
```

Available backends: `dx`, `vk`, `cuda`, `metal`.

## Common Issues

- `-v`, `-D`, `--diagnosis` invalid; use `--verbose`
- Boolean options: `--lc_option=true`/`=false`
- Always `-c` to clean cache before reconfiguring

### Minimal Target Skeleton
```lua
target("my-target")
set_basename("luisa-my-target")
_config_project({ project_kind = "shared", batch_size = 8 })  -- omit batch_size to disable unity
add_deps("lc-core")
add_files("**.cpp")
target_end()
```

### CMake vs XMake
| Aspect | CMake | XMake |
|---|---|---|
| Build file | `CMakeLists.txt` | `xmake.lua` |
| Backend output | `MODULE` library | `shared` target |
| Install | `install(TARGETS ...)` | `on_install()` rules |
| Rust | Custom commands + imported | `os.run()` |
| Config | `cmake -D` options | `xmake f --option=value` |
| Unity build | `LUISA_COMPUTE_ENABLE_UNITY_BUILD` | `lc_enable_unity_build` |
