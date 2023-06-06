# Build

## Requirements
### C++ with XMake

- [xmake](https://xmake.io/) 2.7.8+
- 64-bit OS supported only.
- C++ compilers with C++20 support (e.g., Clang-15, GCC-11, MSVC-17)
    - LLVM toolchain is recommended and well-tested

### C++ with CMake

- [CMake](https://cmake.org/) 3.20+
- Ninja is recommended and required for Rust frontend
- C++ compilers with C++20 support (e.g., Clang-15, GCC-11, MSVC-17)
    - LLVM toolchain is recommended and well-tested
- On Linux, `uuid-dev` is required to build the core libraries and the following libraries are required for the GUI module:
    - libopencv-dev
    - libglfw3-dev
    - libxinerama-dev
    - libxcursor-dev
    - libxi-dev
- On macOS with M1, you need to install `embree` since a pre-built binary is not provided by the official embree repo. We recommend using [Homebrew](https://brew.sh/) to install it. You can install it by running `brew install embree`.
- For Python Module (Python 3.10+): if you have multiple versions of Python installed, please use CMake flag `-D Python_ROOT_DIR=<Python-install-dir>` (or `-D PYTHON_EXECUTABLE=<Python-bin>`) to specific Python version

### Rust (IR module / Rust frontend)
- Rust 1.64+ (latest stable version is recommended)

### Backends

- CPU
    - `clang++` must be in `PATH`
- CUDA
    - CUDA 11.2 or higher
    - RTX-compatible graphics cards with appropriate drivers
- DirectX
    - DirectX 12 with ray tracing support
    - DirectX-12.1 & Shader Model 6.5 compatible graphics cards with appropriate drivers
- Metal
    - macOS 12 or higher
    - Apple M1 chips are recommended (older GPUs are probably supported but not tested)
<!-- - LLVM
    - x86-64 CPU with AVX256 or Apple M1 CPU with ARM Neon
    - LLVM 13+ with the corresponding targets and features enabled
        - CMake seems to have trouble with LLVM 15 on Ubuntu, so we recommend using LLVM 13/14; please install LLVM 14 via `wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 14` and use CMake flag `-D LLVM_ROOT=/usr/lib/llvm-14` to specify the LLVM installation directory if you already have LLVM 15 installed
 -->
### Python

- Packages: sourceinspect, numpy
- Backend-specific requirements are the same as above

## Build Instructions

### XMake Build Commands

```bash
xmake f -c
xmake
```

### XMake Options

All xmake options declared in ./xmake.lua clearly, you can create ./scripts/options.lua to save a default config for your local environment. An example of options.lua is:

```lua
-- for xmake internal arguments
lc_toolchain = {
	toolchain = "llvm",
}
-- for LC's custom options
function get_options()
	return {
		enable_dsl = true,
		enable_gui = true,
	}
end
```
Options in options.lua can be covered by command-line config, for example:

```bash
xmake f --enable_dsl=false --enable_gui=false -c
xmake
```

Now both "enable_dsl" and "enable_gui" are false value.

You can use ./scripts/write_options.lua to generate a default options.lua:
```bash
xmake lua scripts/write_options.lua
```

### XMake Config

When LuisaCompute is required by other xmake projects, include config/xmake_config.lua and call methods to add includedirs, linkdirs and defines to other projects.

### CMake Build Commands

```bash
cmake -S . -B <build-dir> -D CMAKE_BUILD_TYPE=Release # if you want a debug build, change to `-D CMAKE_BUILD_TYPE=Debug`; optionally followed by other flags as listed above
cmake --build <build-dir> # when building on Windows using Visual Studio Generators, add `--config=Release` in a release build
```

> Note: a typical choice of `<build-dir>` is `build`, as assumed the default in the `set_python_path.{bat|ps1|sh}`
> scripts to set `PYTHONPATH`. If it is not your case, please modify the scripts to export the correct paths.

### CMake Flags

All backends are enabled by default if the corresponding required
APIs/frameworks are detected available. You can override the default
settings by specifying CMake flags manually, in form of `-D FLAG=value`
behind the first cmake command.

Note: On Windows, please remember to replace the backslashes `\\` in the paths with `/` when passing arguments to CMake.

- `LUISA_COMPUTE_ENABLE_CUDA`: Enable CUDA backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_DX`: Enable DirectX backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_METAL`: Enable Metal backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_CPU`: Enable CPU backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_GUI`: Enable GUI display in C++ tests (Default: `ON`)
  
## Running the Programs

1. LuisaCompute C++ tests are output to the `<build-dir>/bin` folder.
2. All tests accept a command-line argument specifying backend, which can be chosen from `cuda`, `dx`, `metal`,
   and `llvm` (all in the lower case).
