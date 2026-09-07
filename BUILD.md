# Build

## Requirements

### General
- Only 64-bit systems are supported
- C++ compilers with C++20 support (e.g., Clang-15, GCC-11, MSVC-17)
- On Linux, `uuid-dev` is required to build the core libraries and the following libraries are required for the GUI module:
  - `libopencv-dev`
  - `libglfw3-dev`
  - `libxinerama-dev`
  - `libxcursor-dev`
  - `libxi-dev`


### C++ with XMake

- [XMake](https://xmake.io/) 3.0.4+

### C++ with CMake

- [CMake](https://cmake.org/) 3.26+
- [Ninja](https://ninja-build.org) is the recommended generator

### Backends

- Fallback
    - LLVM and Embree CMake packages are required
- CUDA
    - CUDA 12.0 or higher
    - Nvidia graphics cards with appropriate drivers (R535+ for OptiX 8).
      - To use hardware ray tracing, RTX-compatible graphics cards are needed.
    - To build with GPU compression/decompression support, you may specify `-D LUISA_COMPUTE_DOWNLOAD_NVCOMP=ON` to let CMake automatically download the [nvCOMP](https://developer.nvidia.com/nvcomp) library for you
    - To build with GPU BC texture compression support, you may download [NVTT 3](https://developer.nvidia.com/gpu-accelerated-texture-compression)
      - On Linux (or when the library is installed to a custom location on Windows), also specify `-D NVTT_DIR=<path-to-nvtt>` to the directory containing the NVTT DLLs so CMake could find it.

- DirectX
    - DirectX 12 with ray tracing support
    - DirectX-12.1 & Shader Model 6.5 compatible graphics cards with appropriate drivers
- Metal
    - macOS 13 or higher with Metal 3 support
    - Apple M1 chips are recommended (older GPUs are probably supported but not tested)

## Build Instructions

### XMake Build Commands

```bash
xmake f -m release -c
xmake
```

### XMake Options

All xmake options declared in ./xmake.lua clearly, you can create ./scripts/options.lua to save a default config for your local environment. An example of options.lua is:

```lua
-- for xmake internal arguments
lc_options = {
    toolchain = "llvm",
    lc_enable_tests = true,
    lc_fallback_backend = false,
    lc_cuda_backend = true,
    lc_dx_backend = false,
}

```
Options in options.lua can be covered by command-line config, for example:

```bash
xmake f --lc_enable_dsl=false --lc_enable_gui=false -c
xmake
```

Now both "lc_enable_dsl" and "lc_enable_gui" are false values so the DSL and GUI modules will not be built.

You can use ./scripts/write_options.lua to generate a default options.lua:
```bash
xmake lua scripts/write_options.lua
```

### XMake Config [Experimental]

When LuisaCompute is required by other xmake projects, include `config/xmake_config.lua` and call methods to add `includedirs`, `linkdirs` and `defines` to other projects.

### CMake Build Commands

```bash
cmake -S . -B <build-dir> -D CMAKE_BUILD_TYPE=Release # if you want a debug build, change to `-D CMAKE_BUILD_TYPE=Debug`; optionally followed by other flags as listed above
cmake --build <build-dir> # when building on Windows using Visual Studio Generators, add `--config=Release` in a release build
```

### CMake Install and Package Consumption

For a reusable prebuilt installation, enable distribution mode and use either
the generated CPack archive or a direct install prefix:

```bash
cmake -S . -B <build-dir> -D CMAKE_BUILD_TYPE=Release \
    -D LUISA_COMPUTE_ENABLE_PACKAGE_DISTRIBUTION=ON
cmake --build <build-dir> --config Release
cmake --build <build-dir> --config Release --target package
# The archive is written to <build-dir>/package/.

# A direct install remains supported:
cmake --install <build-dir> --config Release --prefix <install-dir>
```

Distribution mode prefers supported system dependencies and falls back to the
bundled submodule implementation when a usable system package is unavailable.
Set `LUISA_COMPUTE_PACKAGE_REQUIRE_SYSTEM_LIBS=ON` to make a missing or
toolchain-incompatible preferred dependency a configuration error. The
fallback backend has additional LLVM/Embree toolchain and runtime distribution
requirements and is opt-in in this mode. Native GPU backends remain enabled
according to platform support.
DirectX/CUDA interop is also opt-in for distributions because it introduces a
hard dependency on the NVIDIA driver DLL; enable it explicitly with
`LUISA_COMPUTE_ENABLE_DX_CUDA_INTEROP=ON` when that dependency is intentional.
These defaults do not affect ordinary developer builds.

The default CPack generator produces a `.tar.gz` archive on Linux/macOS and a
`.zip` archive on Windows. Run the same isolated package-consumer test used by
CI with:

```bash
cmake --build <build-dir> --config Release \
    --target luisa-compute-package-e2e
```

This target extracts the archive, audits it for producer paths, configures and
builds a separate `find_package` consumer, removes the extracted SDK, and then
loads the deployed native backend. System libraries selected during packaging
remain external runtime/development prerequisites and are rediscovered by the
installed package config.

Use a separate build and install prefix for each configuration. LuisaCompute's
Debug and Release runtime libraries intentionally use the same filenames and
must not be installed over one another.

The install tree contains a relocatable CMake package. Point consumers at the
prefix with `-D CMAKE_PREFIX_PATH=<install-dir>`, link the exported target, and
deploy the backend/runtime payload beside every executable that creates a
`Context`:

```cmake
find_package(LuisaCompute CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE luisa::compute)
luisa_compute_deploy_runtime(TARGET my_target)
```

The deploy helper copies the installed backend modules, support files, and
LuisaCompute shared libraries to the target's output directory. This matches
`Context`'s runtime discovery contract and makes the staged executable
independent of the SDK prefix; system dependencies are intentionally not
vendored. Consumers may require a backend as a package component, for example
`find_package(LuisaCompute CONFIG REQUIRED COMPONENTS dx)`. Optional public
distribution components are named `dsl` and `gui`; the available backend names
are reported in `LuisaCompute_BACKENDS`.

As with other prebuilt C++ SDKs, producer and consumer toolchains must use a
compatible compiler ABI and C++ runtime. The package exports LuisaCompute's
required C++ standard, definitions, include paths, link interface, and public
compile options; it intentionally does not replay arbitrary producer-wide
`CMAKE_CXX_FLAGS` in downstream projects.

### CMake Flags

All backends are enabled by default if the corresponding required
APIs/frameworks are detected available. You can override the default
settings by specifying CMake flags manually, in form of `-D FLAG=value`
behind the first cmake command.

> Note: On Windows, please remember to replace the backslashes `\\` in the paths with `/` when passing arguments to CMake.

- `LUISA_COMPUTE_ENABLE_CUDA`: Enable CUDA backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_DX`: Enable DirectX backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_METAL`: Enable Metal backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_FALLBACK`: Enable the native C++ LLVM/Embree fallback backend
- `LUISA_COMPUTE_ENABLE_GUI`: Enable GUI display in C++ tests (Default: `ON`)
- `LUISA_COMPUTE_USE_SYSTEM_STL`: Use the toolchain's standard library instead of bundled EASTL (Default: `ON`; set to `OFF` to opt into EASTL). This CMake option does not change xmake defaults.

> Note: You may also edit the `scripts/options.cmake` file (generated by `bootstrap.py` or copied by yourself from
> `scripts/options.cmake.template`) to set the default values of these flags. You can still override the default
> values by specifying the above CMake flags manually.
  
## Running the Programs

1. LuisaCompute C++ tests are output to the `<build-dir>/bin` folder.
2. All tests accept a command-line argument specifying the backend, which can be chosen from `cuda`, `dx`, `metal`,
   `vk`, `hip`, and `fallback` when built (all in lower case).
