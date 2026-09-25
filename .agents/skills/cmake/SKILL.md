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

One-shot configure + build + verify on Windows using the repository defaults.

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
| `CMAKE_BUILD_TYPE` | - | `Release` / `Debug` / `RelWithDebInfo` / `MinSizeRel` (single-config generators default to `Release`, `scripts/setup_output_dirs.cmake:16-18`) |
| `LUISA_COMPUTE_ENABLE_PACKAGE_DISTRIBUTION` | OFF | Relocatable CPack distribution build; also turns developer-feature defaults OFF (`CMakeLists.txt:45-58`) |
| `LUISA_COMPUTE_ENABLE_DSL` | ON | C++ DSL |
| `LUISA_COMPUTE_ENABLE_CUDA` | ON | CUDA backend |
| `LUISA_COMPUTE_ENABLE_METAL` | ON | Metal backend (macOS only) |
| `LUISA_COMPUTE_ENABLE_METAL4` | OFF | Independent Metal4 XIR→LLVM→AIR backend; requires LLVM 22 and Apple Metal 4 tools |
| `LUISA_COMPUTE_ENABLE_DX` | ON | DirectX backend (Windows only) |
| `LUISA_COMPUTE_ENABLE_VULKAN` | ON | Vulkan backend |
| `LUISA_COMPUTE_ENABLE_HIP` | OFF | HIP backend (work in progress) |
| `LUISA_COMPUTE_ENABLE_FALLBACK` | ON (OFF in package-distribution builds) | Native C++ fallback backend (requires LLVM + Embree, `src/backends/fallback/CMakeLists.txt:11-30`) |
| `LUISA_COMPUTE_ENABLE_GUI` | ON | GUI support (GLFW/ImGui) |
| `LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB` | OFF | CUDA extension: LCUB |
| `LUISA_COMPUTE_ENABLE_CLANG_CXX` | OFF | ClangTooling-based C++ shading language |
| `LUISA_COMPUTE_ENABLE_SIMD` | OFF | Experimental SIMD CPU backend (`src/backends/simd/`) |
| `LUISA_COMPUTE_ENABLE_REMOTE` | OFF | Experimental C++ remote backend (`src/backends/remote/`) |
| `LUISA_COMPUTE_ENABLE_TILE_TIRX_BRIDGE` | OFF | Native C++ TileIR to TVM TIRx bridge |
| `LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV` | ON | Native XIR-to-SPIR-V codegen path for Vulkan |
| `LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV` | OFF | Experimental AST→LLVM SPIR-V path; requires LLVM's native `SPIRV` target |
| `LUISA_COMPUTE_VULKAN_ENABLE_DXC_COMPATIBILITY` | ON | Legacy Vulkan HLSL-to-SPIR-V compatibility through DXC |
| `LUISA_COMPUTE_BUILD_TESTS` | ON in master project | Build tests, examples and tutorials |
| `LUISA_COMPUTE_BUILD_IOS_EXAMPLES` | OFF | Build the 19 opt-in UIKit example bundles (18 `luisa_compute_add_ios_rendering_example` apps plus `example_ios_path_tracing`, `examples/ios/CMakeLists.txt:25,109-161`); requires an iPhoneOS toolchain and Metal4 |
| `LUISA_COMPUTE_BUILD_IOS_TESTS` | OFF | Build the independent iOS device-conformance bundle (or its host-AOT oracle on macOS) |
| `LUISA_COMPUTE_BUILD_IOS_BENCHMARKS` | OFF | Build the opt-in iOS Metal backend comparison application |
| `LUISA_COMPUTE_ENABLE_SAFE_MODE` | OFF | Runtime safe mode |
| `LUISA_COMPUTE_ENABLE_UNITY_BUILD` | OFF | Unity build |
| `LUISA_COMPUTE_ENABLE_SANITIZERS` | OFF | Address/UB sanitizers |
| `LUISA_COMPUTE_ENABLE_LTO` | OFF | Link-time optimization (release builds only) |
| `LUISA_COMPUTE_ENABLE_SCCACHE` | ON (non-MSVC) | Use `sccache` compiler launcher |
| `LUISA_COMPUTE_ENABLE_DX_CUDA_INTEROP` / `LUISA_COMPUTE_ENABLE_VK_CUDA_INTEROP` | ON (developer builds) | CUDA interop in the DX / Vulkan backends when `CUDAToolkit` is available |
| `LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES` | ON | Auto-disable backends with missing dependencies |
| `LUISA_COMPUTE_ENABLE_WAYLAND` | OFF (Linux) | Wayland support in GUI/Vulkan swapchains |
| `LUISA_COMPUTE_USE_SYSTEM_LIBS` | OFF | Prefer system libraries; also enables per-lib `USE_SYSTEM_*` overrides |
| `LUISA_COMPUTE_DOWNLOAD_OIDN` | OFF | Download OpenImageDenoise |
| `LUISA_COMPUTE_DOWNLOAD_NVCOMP` | OFF (if CUDA) | Download nvCOMP for CUDA decompression |

`LUISA_COMPUTE_USE_SYSTEM_*` options exist for `STL`, `GLFW`, `LMDB`, `REPROC`, `SPDLOG`, `XXHASH`, `YYJSON`, `MAGIC_ENUM`, and `MARL` (`CMakeLists.txt:180-188`; `STL` defaults to ON, the rest follow `LUISA_COMPUTE_USE_SYSTEM_LIBS`).

`LUISA_COMPUTE_ENABLE_SCCACHE` and `LUISA_COMPUTE_ENABLE_LTO` are declared in `scripts/setup_compilation.cmake:60,76` (not the root file); LTO only sets `CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE` / `_RELWITHDEBINFO` / `_MINSIZEREL`.

The two Vulkan SPIR-V codegen options are mutually exclusive: `CMakeLists.txt:160-164` and
`src/backends/CMakeLists.txt:3-8` raise a `FATAL_ERROR` if both are ON. The LLVM path
also builds/links the common `luisa-compute-spirv` support library (`STATIC`,
`src/backends/common/spirv/CMakeLists.txt:18`) because the Vulkan backend still uses its
disassembler/validation utilities and its target-feature reconciliation
(`reconcile_spirv_target_features`, `src/backends/common/spirv/spirv_codegen/optimizer.cpp:657`)
in that configuration (`src/backends/vk/CMakeLists.txt:143-148`).

**Minimal build** (no GPU/fallback backends):
```bash
cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release \
  -D LUISA_COMPUTE_ENABLE_CUDA=OFF \
  -D LUISA_COMPUTE_ENABLE_VULKAN=OFF \
  -D LUISA_COMPUTE_ENABLE_FALLBACK=OFF
cmake --build build
```

CI jobs are defined in `.github/workflows/build-cmake.yml`: all three platforms configure with `-G Ninja -D LUISA_COMPUTE_ENABLE_PACKAGE_DISTRIBUTION=<ON|OFF> -D CMAKE_BUILD_TYPE=<Release|Debug>`, and the macOS job additionally passes `-D LUISA_COMPUTE_ENABLE_VULKAN=OFF -D LUISA_COMPUTE_ENABLE_UNITY_BUILD=OFF`. Release jobs then run `cmake --build build --target luisa-compute-package-e2e` (`cmake/LuisaComputeCPack.cmake:78`).

## Target Naming

| Prefix | Example | Purpose |
|---|---|---|
| `luisa-compute-<module>` | `luisa-compute-core` | Internal library |
| `luisa-compute-backend-<name>` | `luisa-compute-backend-cuda` | Backend plugin (output: `luisa-backend-<name>`) |
| `luisa-compute-ext-<name>` | `luisa-compute-ext-btree` | Third-party ext |
| `luisa::compute` | Alias | Interface target for all core modules |

## Module Hierarchy

```
luisa-compute-include (INTERFACE, header-only)
  → luisa-compute-ext (INTERFACE, third-party deps)
    → luisa-compute-core
      → luisa-compute-ast
        → luisa-compute-xir
      → luisa-compute-runtime (links ast; PRIVATE luisa-compute-vstl)
        → luisa-compute-dsl, luisa-compute-gui (gui links runtime + dsl)
luisa-compute-backends (INTERFACE aggregator: each luisa_compute_add_backend target links ast/runtime/gui/dsl and registers itself via add_dependencies)
```

These libraries follow `BUILD_SHARED_LIBS`: ordinary desktop builds force it
ON, while iOS forces it OFF so the signed app contains static core/runtime/XIR
and backend slices.

The umbrella target is `compute`, aliased as `luisa::compute` (`src/CMakeLists.txt:55-75`); it links `luisa-compute-core`, `-tile`, `-ast`, `-xir`, `-dsl`, `-osl`, `-runtime`, `-gui`, `-backends`, `-coro`, `-clangcxx`, plus `luisa-compute-cuda-ext-lcub` when `LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB` is ON. `luisa-compute-vstl` is an `OBJECT` helper (`src/vstl/CMakeLists.txt:11`) linked PRIVATE by `luisa-compute-runtime` (`src/runtime/CMakeLists.txt:44-46`), not by `compute`.

## Custom CMake Functions

### `luisa_compute_add_backend(name [SOURCES <srcs...>] [SUPPORT_DIR dir] [BUILTIN_DIR dir])`
**File**: `src/backends/CMakeLists.txt:11`.
Creates a backend `MODULE` target named `luisa-compute-backend-<name>` on desktop
platforms and a `STATIC` target on iOS (`CMAKE_SYSTEM_NAME STREQUAL "iOS"`). It
links `luisa-compute-ast`, `luisa-compute-runtime`, `luisa-compute-gui`, and (when
`LUISA_COMPUTE_ENABLE_DSL` is ON) `luisa-compute-dsl`. Desktop output is named
`luisa-backend-<name>` (`OUTPUT_NAME`, `EXPORT_NAME backend_<name>`) and runtime
artifacts are installed to `bin/`. If `SUPPORT_DIR` is given, its contents are
copied next to the runtime outputs (`$<TARGET_FILE_DIR:luisa-compute-core>`) and
installed to `bin/`. A `BUILTIN_DIR` keyword is parsed but currently unused by the
function body.
```cmake
luisa_compute_add_backend(cuda SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})          # src/backends/cuda/CMakeLists.txt:53
luisa_compute_add_backend(dx SOURCES ${LC_BACKEND_DX_SOURCES} SUPPORT_DIR ${LUISA_COMPUTE_DX_SDK_DIR})
```

### `luisa_compute_install(target [SOURCES <srcs...>])`
**File**: `src/CMakeLists.txt:14`. Installs `luisa-compute-<target>` into the
`LuisaComputeTargets` export with `LIBRARY`/`ARCHIVE` → `${CMAKE_INSTALL_LIBDIR}`
and `RUNTIME` → `${CMAKE_INSTALL_BINDIR}`. The `SOURCES` keyword is accepted by the
call sites but ignored by the function.
```cmake
luisa_compute_install(core SOURCES ${LUISA_COMPUTE_CORE_SOURCES})
```

### `luisa_compute_add_executable(name [sources...])`
**File**: `src/CMakeLists.txt:77`. Wraps `add_executable(${name} ${ARGN})` (extra
arguments are the sources) and links `luisa::compute` PRIVATE.
```cmake
luisa_compute_add_executable(test_basic_traits unit/core/test_basic_traits.cpp)
```

### `luisa_compute_add_test(name source [LABELS ...] [ARGS ...])`
**File**: `src/tests/CMakeLists.txt:19`. Builds one standalone executable per
source (via `luisa_compute_add_executable`) and adds `./` and `./common` to its
include paths. With `LABELS`, registers a CTest entry; `ARGS` becomes the test's
argv, used to pin a backend for a GPU integration test (e.g. `ARGS metal4`,
`ARGS simd`). Without `LABELS`, it just builds the binary and any
device test is invoked manually with a backend argument.
```cmake
luisa_compute_add_test(test_basic_traits unit/core/test_basic_traits.cpp LABELS "unit;unit_core")
luisa_compute_add_test(test_vk_cuda_kernel_launch integration/runtime/test_vk_cuda_kernel_launch.cpp)  # no CTest
luisa_compute_add_test(test_tile_xir_runtime_gpu_dx unit/tile/bridge/test_xir_runtime_gpu.cpp LABELS "integration;integration_tile_xir;integration_dx" ARGS dx)
```

### `luisa_compute_add_example(name [sources...] [MIRROR_AS_TEST])`
**File**: `examples/CMakeLists.txt:1`. Builds the `<name>` executable (call sites pass names already prefixed with `example_`) and, when `MIRROR_AS_TEST` is set, additionally builds a mirror executable whose name is `<name>` with the leading `example_` replaced by `test_` (only if that target does not already exist), recording it in the `LUISA_TEST_MIRROR` target property. Reserved for auto-checkable examples (reference-image comparison, deterministic sims, headless compute). GUI/interop demos must omit the flag.
```cmake
luisa_compute_add_example(example_path_tracing rendering/path_tracing.cpp MIRROR_AS_TEST)
luisa_compute_add_example(example_swapchain_qt gui/swapchain_qt.cpp)  # no mirror
```

### `luisa_example_pair_link(name <link-args>)`
**File**: `examples/CMakeLists.txt:16`. Companion to `luisa_compute_add_example`. Calls `target_link_libraries` on `<name>` and, if its `LUISA_TEST_MIRROR` property is set, on that mirror too. Use whenever an example needs extra libs.
```cmake
luisa_compute_add_example(example_cuda_lcub extension/cuda_lcub.cpp)
luisa_example_pair_link(example_cuda_lcub PRIVATE CUDA::cudart CUDA::cuda_driver)
```

## Backend Plugin Build

Desktop backends are built as `MODULE` runtime-loadable libraries:
```cmake
luisa_compute_add_backend(cuda SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})
```

Key: output renamed to `luisa-backend-<name>`, installed to `bin/`. Device-lib embedding is done by each backend's own `CMakeLists.txt` through the `luisa_embed_device_lib` host tool (`utils/CMakeLists.txt:2`; e.g. `src/backends/cuda/CMakeLists.txt:6-11`) — `luisa_compute_add_backend` itself does not embed anything.

On iOS, the same helper emits a static backend. A signed Metal4 AIR device app
also requires static arm64 iPhoneOS LLVM 22 archives; an arm64 macOS LLVM build
is not platform-compatible. Prefer the checked scripts so local and CI options
remain identical:

```bash
scripts/build_ios_llvm.sh \
  --host-llvm-prefix "$(brew --prefix llvm@22)"
scripts/build_ios_metal4.sh \
  --llvm-dir cmake-build-llvm22-ios/lib/cmake/llvm \
  --team <team-id> --mode all

# CI/link closure only; these bundles are not installable.
scripts/build_ios_metal4.sh \
  --llvm-dir cmake-build-llvm22-ios/lib/cmake/llvm \
  --mode all --unsigned
scripts/audit_ios_bundles.sh \
  --bin-dir cmake-build-ios-metal4-device-air-xcode/bin/Release
```

`build_ios_llvm.sh` downloads the official LLVM 22.1.8 source when needed and
uses CMake/Ninja with `arm64-apple-ios<deployment>` host/default triples. The
application script uses CMake's Xcode generator only because provisioning and
automatic signing are Xcode workflows. `--mode examples`, `tests`, or `all`
maps to `luisa-ios-rendering-examples`, `luisa-ios-device-tests`, or both.

Metal4 user shaders and fixed runtime builtins are LLVM/AIR. BC6H/BC7 fixed
support sources are compiled to target-specific metallibs by `xcrun metal`
and `metallib` at build time; the runtime embeds and loads only their binary
bytes.

## Third-Party Extension Pattern

Each `src/ext/<lib>/` (pattern from `src/ext/CMakeLists.txt:283-310`):
```cmake
if (LUISA_COMPUTE_USE_SYSTEM_<LIB>)
    find_package(<LIB> REQUIRED)
    target_link_libraries(luisa-compute-ext INTERFACE <target>)
    target_compile_definitions(luisa-compute-ext INTERFACE LUISA_USE_SYSTEM_<LIB>=1)
else()
    add_subdirectory(<lib>)
    target_link_libraries(luisa-compute-ext INTERFACE <target>)
    luisa_compute_install_extension(<target> INCLUDE <lib>/include/<LIB>)
endif()
```
`luisa_compute_install_extension(target [INCLUDE dir] [INCLUDES dirs...] [HEADER_FILES files...] [HEADER_DESTINATION dir])` is defined in `src/ext/CMakeLists.txt:243` and installs the target plus its headers to `${CMAKE_INSTALL_INCLUDEDIR}/luisa/ext`.

## Output & RPATH

```
${CMAKE_BINARY_DIR}/bin  → Runtime + library outputs (DLLs/.so/.dylib, executables)
${CMAKE_BINARY_DIR}/lib  → Archive outputs (static libs, import libs, PDBs)
```

Per `scripts/setup_output_dirs.cmake:1-14`; multi-config generators (e.g. Xcode, VS) write `bin/<config>` and `lib/<config>` instead.

Install layout: headers → `${CMAKE_INSTALL_INCLUDEDIR}/luisa` (`CMakeLists.txt:323-326`), libraries → `${CMAKE_INSTALL_LIBDIR}` and runtime binaries → `${CMAKE_INSTALL_BINDIR}` (`luisa_compute_install`, `src/CMakeLists.txt:14-20`), CMake package files (`LuisaComputeConfig.cmake`, `LuisaComputeTargets.cmake`) → `LUISA_COMPUTE_INSTALL_CMAKEDIR`, which defaults to `${CMAKE_INSTALL_LIBDIR}/cmake/LuisaCompute` (`CMakeLists.txt:267-270`, `367-381`). RPATH is set from `CMakeLists.txt:244-249`.

- **macOS**: `@loader_path`, `@loader_path/../bin`, `@loader_path/../lib`
- **Linux**: `$ORIGIN`, `$ORIGIN/../bin`, `$ORIGIN/../lib`
