---
name: xmake
---

# XMake Build System

Primary build system for this project.

## Requirements

- XMake 3.0.6+
- Optional: CUDA Toolkit, Vulkan SDK, LLVM 20, Rust

## Quick Start

```bash
xmake f -m debug -c
xmake build
```


Update `compile_commands.json`:
```bash
xmake project -k compile_commands --lsp=clangd .vscode
```

## Configuration

### Platform Examples

| Platform | Command |
|----------|---------|
| Linux GCC | `xmake f -p linux -a x86_64 --toolchain=gcc -m release -c` |
| Linux Clang | `xmake f -p linux -a x86_64 --toolchain=clang -m release -c` |
| Windows MSVC | `xmake f -p windows -a x64 --toolchain=msvc -m release -c` |
| Windows Clang-CL | `xmake f -p windows -a x64 --toolchain=clang-cl -m release -c` |
| Windows LLVM | `xmake f -p windows -a x64 --toolchain=llvm -m release -c` |

### Flags

| Flag | Description |
|------|-------------|
| `-c` | Clean configuration cache |
| `-m <mode>` | Build mode: `release`, `debug`, `releasedbg` |
| `-p <plat>` | Platform: `linux`, `windows`, `macosx` |
| `-a <arch>` | Architecture: `x86_64`, `x64`, `arm64` |
| `--check` | Check flags before building |

## Project Options

### Backends

| Option | Default | Description |
|--------|---------|-------------|
| `lc_cuda_backend` | true | NVIDIA CUDA backend |
| `lc_vk_backend` | true | Vulkan backend |
| `lc_dx_backend` | true | DirectX-12 backend |
| `lc_metal_backend` | true | Metal backend |
| `lc_fallback_backend` | false | CPU fallback backend |
| `lc_toy_c_backend` | false | Toy C backend for testing |

### Features

| Option | Default | Description |
|--------|---------|-------------|
| `lc_enable_dsl` | true | C++ DSL module |
| `lc_enable_gui` | true | GUI module |
| `lc_enable_imgui` | true | ImGui integration |
| `lc_enable_tests` | true | Tests module |
| `lc_enable_py` | true | Python bindings |
| `lc_enable_osl` | true | OpenShadingLanguage support |

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `lc_enable_unity_build` | true | Unity (jumbo) build |
| `lc_enable_simd` | true | SSE/SSE2 SIMD |
| `lc_use_lto` | false | Link Time Optimization |
| `lc_enable_mimalloc` | true | Use mimalloc allocator |
| `lc_enable_custom_malloc` | false | Use custom allocator |

### Advanced Options

| Option | Default | Description |
|--------|---------|-------------|
| `lc_cxx_standard` | cxx20 | C++ standard |
| `lc_rtti` | false | Enable RTTI |
| `lc_enable_xir` | false | XIR (IR) support |
| `lc_dx_cuda_interop` | false | DirectX-CUDA interop |
| `lc_vk_cuda_interop` | false | Vulkan-CUDA interop |
| `lc_cuda_ext_lcub` | false | CUDA CUB extension |
| `lc_sdk_dir` | false | SDK download directory |
| `lc_bin_dir` | bin | Binary output directory |

## Build Examples

```bash
# Full build
xmake f -p linux -m release --lc_cuda_backend=true --lc_vk_backend=true \
  --lc_enable_dsl=true --lc_enable_gui=true --lc_enable_tests=true -c
xmake

# Minimal build (no tests, no GUI)
xmake f -m release --lc_enable_tests=false --lc_enable_gui=false --lc_enable_dsl=false -c
xmake

# Debug with tests
xmake f -m debug --lc_enable_tests=true -c && xmake
```

## Commands

| Command | Description |
|---------|-------------|
| `xmake clean` | Clean build files |
| `xmake -r` | Rebuild |
| `xmake build <target>` | Build target |
| `xmake run <target>` | Run target |
| `xmake -l` | List targets |
| `xmake install -o <dir>` | Install to directory |

## Common Issues

### Build Flags

- `-v`, `-D`, `--diagnosis` are **invalid**; use `--verbose` for verbose output
- Boolean options: `--lc_option=true` or `=false`
- Always use `-c` to clean cache before reconfiguring

### Minimal Target Skeleton

```lua
target("my-target")
set_basename("luisa-my-target")
_config_project({
    project_kind = "shared", -- or "static", "object"
    batch_size = 8           -- unity-build batch size; omit or set 1 to disable
})
add_deps("lc-core")
add_files("**.cpp")
target_end()
```

### Key Differences from CMake

| Aspect | CMake | XMake |
|--------|-------|-------|
| Build file | `CMakeLists.txt` | `xmake.lua` |
| Backend output | `MODULE` library | `shared` target |
| Install paths | `install(TARGETS ...)` | `on_install()` rules |
| Rust integration | Custom commands + imported targets | Direct `os.run()` invocation |
| Configuration | `cmake -D` options | `xmake f --option=value` |
| Unity build | `LUISA_COMPUTE_ENABLE_UNITY_BUILD` | `lc_enable_unity_build` |
