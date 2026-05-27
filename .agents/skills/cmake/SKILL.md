---
name: cmake
description: CMake build options, custom functions, and backend patterns for LuisaCompute.
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

### `scripts/agent_windows_cmake.py`

One-shot configure + build + verify on Windows. CI-matching flags (`LUISA_COMPUTE_ENABLE_RUST=OFF`, `LUISA_COMPUTE_ENABLE_REMOTE=OFF`, `LUISA_COMPUTE_ENABLE_CPU=OFF`).

```bash
# Full pipeline: configure → build → verify
python scripts/agent_windows_cmake.py

# Individual steps
python scripts/agent_windows_cmake.py --config          # configure only
python scripts/agent_windows_cmake.py --build           # build only
python scripts/agent_windows_cmake.py --verify          # check key .lib/.dll outputs
python scripts/agent_windows_cmake.py --clean           # clear CMake cache

# Options
python scripts/agent_windows_cmake.py --type Debug      # Debug build
python scripts/agent_windows_cmake.py -j 8              # limit parallel jobs
python scripts/agent_windows_cmake.py --clean --config  # clean re-configure
```

Auto-finds `cmake` and `ninja` (PATH → `.deps/` → pip). Auto-prepares MSVC environment via `vswhere`. Verifies: `SPIRV-Tools-opt.lib`, `SPIRV-Tools.lib`, `luisa-ast.dll`, `luisa-core.dll`.

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

### `luisa_compute_add_test(name source [LABELS ...] [ARGS ...])`
**File**: `src/tests/CMakeLists.txt`. Builds one standalone executable per source. With `LABELS`, registers a CTest entry (CPU-only tests). Without `LABELS`, just builds the binary (GPU-using tests are invoked manually with a backend arg).
```cmake
luisa_compute_add_test(test_basic_traits unit/core/test_basic_traits.cpp LABELS "unit;unit_core")
luisa_compute_add_test(test_my_gpu unit/runtime/test_my_gpu.cpp)  # no CTest
```

### `luisa_compute_add_example(name source... [MIRROR_AS_TEST])`
**File**: `examples/CMakeLists.txt`. Builds `example_<name>` and, when `MIRROR_AS_TEST` is set, additionally builds a `test_<name>` mirror executable from the same sources. Reserved for auto-checkable examples (reference-image comparison, deterministic sims, headless compute). GUI/interop demos must omit the flag.
```cmake
luisa_compute_add_example(example_path_tracing rendering/path_tracing.cpp MIRROR_AS_TEST)
luisa_compute_add_example(example_swapchain_qt gui/swapchain_qt.cpp)  # no mirror
```

### `luisa_example_pair_link(name <link-args>)`
Companion to `luisa_compute_add_example`. Calls `target_link_libraries` on both `example_<name>` and its `test_<name>` mirror (if any). Use whenever an example needs extra libs.
```cmake
luisa_compute_add_example(example_cuda_lcub extension/cuda_lcub.cpp MIRROR_AS_TEST)
luisa_example_pair_link(example_cuda_lcub PRIVATE CUDA::cudart CUDA::cuda_driver)
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
