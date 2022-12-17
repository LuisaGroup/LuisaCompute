# Build Instructions

## Requirements

### C++

- CMake 3.20+
- C++ compilers with C++20 support (e.g., Clang-13, GCC-11, MSVC-17)
    - MSVC and Clang (with GNU-style command-line options) are recommended and tested on Windows
- On Linux, `uuid-dev` is required to build the core libraries and the following libraries are required for the GUI module:
    - libopencv-dev
    - libglfw3-dev
    - libxinerama-dev
    - libxcursor-dev
    - libxi-dev
- On macOS with M1, you need to install `embree` since a pre-built binary is not provided by the official embree repo. We recommend using [Homebrew](https://brew.sh/) to install it. You can install it by running `brew install embree`.

### Rust (IR module / Rust frontend)
- Rust 1.56+ (latest stable version is recommended)

### Backends

- CUDA
    - CUDA 11.2 or higher
    - RTX-compatible graphics cards with appropriate drivers
- DirectX
    - DirectX 12 with ray tracing support
    - RTX-compatible graphics card with appropriate drivers
- Metal
    - macOS 12 or higher
    - Apple M1 chips are recommended (older GPUs are probably supported but not tested)
- LLVM
    - x86-64 CPU with AVX256 or Apple M1 CPU with ARM Neon
    - LLVM 13+ with the corresponding targets and features enabled
        - CMake seems to have trouble with LLVM 15 on Ubuntu, so we recommend using LLVM 13/14; please install LLVM 14 via `wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 14` and use CMake flag `-D LLVM_ROOT=/usr/lib/llvm-14` to specify the LLVM installation directory if you already have LLVM 15 installed

### Python

- Python 3.9+: if you have multiple versions of Python installed, please use CMake flag `-D Python_ROOT_DIR=<Python-install-dir>` (or `-D PYTHON_EXECUTABLE=<Python-bin>`) to specific Python version
- Packages: astpretty, dearpygui, sourceinspect, numpy, pillow
- Backend-specific requirements are the same as above

## CMake Flags

All backends are enabled by default if the corresponding required
APIs/frameworks are detected available. You can override the default
settings by specifying CMake flags manually, in form of `-D FLAG=value`
behind the first cmake command.

Note: On Windows, please remember to replace the backslashes `\\` in the paths with `/` when passing arguments to CMake.

- `LUISA_COMPUTE_ENABLE_CUDA`: Enable CUDA backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_DX`: Enable DirectX backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_METAL`: Enable Metal backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_LLVM`: Enable LLVM backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_PYTHON`: Enable LuisaCompute Python (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_GUI`: Enable GUI display in C++ tests (Default: `ON`)

## Build Commands

```bash
cmake -S . -B <build-dir> -D CMAKE_BUILD_TYPE=Release # if you want a debug build, change to `-D CMAKE_BUILD_TYPE=Debug`; optionally followed by other flags as listed above
cmake --build <build-dir> # when building on Windows using Visual Studio Generators, add `--config=Release` in a release build
```

> Note: a typical choice of `<build-dir>` is `build`, as assumed the default in the `set_python_path.{bat|ps1|sh}`
> scripts to set `PYTHONPATH`. If it is not your case, please modify the scripts to export the correct paths.

## Running the Programs

1. LuisaCompute C++ tests are output to the `<build-dir>/bin` folder.
2. LuisaCompute Python tests are in the `src/py` directory. Run `set_python_path.{bat|ps1|sh}` (or manually with shell
   commands if you prefer) to set the `PYTHONPATH` environment variable to `<build-dir>/bin` before running any python
   test, otherwise Python will fail to find and load the bindings exported from C++. To run `lrenderer.py`, copy files
   in `pyscenes` to the same directory of `lrenderer.py`.
4. All tests accept a command-line argument specifying backend, which can be chosen from `cuda`, `dx`, `metal`,
   and `llvm` (all in the lower case).
