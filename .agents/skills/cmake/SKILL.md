---
name: cmake
description: CMake build system for LuisaCompute — options, architecture, custom functions, and backend patterns
---

# CMake Build Guide

**Requirements**: CMake 3.26+, Ninja (recommended), C++20 compiler (MSVC/Clang/GCC).

## Quick Start

```bash
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix dist
```

**Platform specifics**:

Linux:
```bash
export CC=clang-20 CXX=clang++-20
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release
```

macOS:
```bash
export PATH="$PATH:/opt/homebrew/opt/llvm/bin"
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export SDKROOT=$(xcrun --show-sdk-path)
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release
```

Windows: Requires VS Developer Command Prompt. Or use Python bootstrap:
```python
import bootstrap
bootstrap.prepare_msvc_environment()
```

## Build Options

| Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | - | `Release` / `Debug` |
| `LUISA_COMPUTE_ENABLE_DSL` | ON | C++ DSL |
| `LUISA_COMPUTE_ENABLE_CUDA` | ON | CUDA backend |
| `LUISA_COMPUTE_ENABLE_METAL` | ON | Metal (macOS) |
| `LUISA_COMPUTE_ENABLE_DX` | ON | DirectX (Windows) |
| `LUISA_COMPUTE_ENABLE_VULKAN` | ON | Vulkan backend |
| `LUISA_COMPUTE_ENABLE_CPU` | ON | CPU backend |
| `LUISA_COMPUTE_ENABLE_REMOTE` | ON | Remote backend |
| `LUISA_COMPUTE_ENABLE_FALLBACK` | ON | Fallback backend |
| `LUISA_COMPUTE_ENABLE_GUI` | ON | GUI support |
| `LUISA_COMPUTE_ENABLE_UNITY_BUILD` | OFF | Unity build |
| `LUISA_COMPUTE_ENABLE_SANITIZERS` | OFF | Address/UB sanitizers |
| `LUISA_COMPUTE_USE_SYSTEM_LIBS` | OFF | Prefer system libs |

**CI minimal build**:
```bash
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release \
  -D LUISA_COMPUTE_ENABLE_RUST=OFF -D LUISA_COMPUTE_ENABLE_REMOTE=OFF \
  -D LUISA_COMPUTE_ENABLE_CPU=OFF
cmake --build build
```

## Target Naming

| Prefix | Example | Purpose |
|---|---|---|
| `luisa-compute-<module>` | `luisa-compute-core` | Internal library |
| `luisa-compute-backend-<name>` | `luisa-compute-backend-cuda` | Backend plugin (output: `luisa-backend-<name>`) |
| `luisa-compute-ext-<name>` | `luisa-compute-ext-spdlog` | Third-party ext |
| `luisa::compute` | Alias | Interface target for all core modules |

## Module Hierarchy

```
luisa-compute-include (INTERFACE, header-only)
  → luisa-compute-ext (INTERFACE, third-party deps)
    → luisa-compute-core (SHARED)
      → luisa-compute-ast (SHARED) → luisa-compute-xir (SHARED)
        → luisa-compute-runtime (SHARED)
          → luisa-compute-dsl, luisa-compute-gui, luisa-compute-ir
            → luisa-compute-backends (INTERFACE aggregator)
```

## Custom CMake Functions

### `luisa_compute_add_backend(name)`
Creates backend plugin MODULE target. Links `ast`, `runtime`, `gui`. Output name: `luisa-backend-<name>`. Installed to `bin/`, not `lib/`.
```cmake
luisa_compute_add_backend(cuda SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})
```

### `luisa_compute_install(target)`
Installs target with consistent destination paths.
```cmake
luisa_compute_install(core SOURCES ${LUISA_COMPUTE_CORE_SOURCES})
```

### `luisa_compute_add_executable(name)`
Creates executable linked to `luisa::compute`.
```cmake
luisa_compute_add_executable(my_app)
```

### `luisa_compute_test_suite(name)` / `luisa_compute_add_test(name)`
```cmake
luisa_compute_test_suite(feat)   # globs next/test/feat/**.cpp
luisa_compute_add_test(my_test)  # adds to test_main executable
```

## Backend Plugin Build

Backends built as `MODULE` (runtime-loadable shared libs):
```cmake
luisa_compute_add_backend(cuda SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})
```

Key: output renamed to `luisa-backend-<name>`, installed to `bin/`, supports `luisa_embed_device_lib` for builtin device libs.

## Rust Integration

**File**: `src/rust/CMakeLists.txt`

Custom command invokes `cargo build` (profile: `dev` for Debug, `release` for Release). CMake targets:
- `luisa-compute-rust-meta` (INTERFACE): static Rust libs
- `luisa_compute_backend_impl` (INTERFACE): shared Rust backend

## Third-Party Extension Pattern

Each `src/ext/<lib>/`:
```cmake
if (LUISA_COMPUTE_USE_SYSTEM_<LIB>)
    find_package(<LIB> REQUIRED)
    target_link_libraries(luisa-compute-ext INTERFACE <target>)
    target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_<LIB>=1)
else()
    add_subdirectory(<lib>)
    target_link_libraries(luisa-compute-ext INTERFACE <target>)
    luisa_compute_install_extension(<target> ...)
endif()
```

## Output & RPATH

```
${CMAKE_BINARY_DIR}/bin  → Runtime outputs (DLLs, executables)
${CMAKE_BINARY_DIR}/lib  → Archive outputs (static libs, PDBs)
```

- **macOS**: `@loader_path`, `@loader_path/../bin`, `@loader_path/../lib`
- **Linux**: `$ORIGIN`, `$ORIGIN/../bin`, `$ORIGIN/../lib`
