# Build Instructions

## Requirements

- CMake 3.20+
- C++ compiler which supports C++20 (e.g., clang-13, gcc-11, msvc-17)
- MSVC compiler is recommended on Windows

### Backend Requirements

1. CUDA
    - CUDA 11.2 or higher
    - RTX-compatiable graphics card
2. DirectX
    - DirectX 12
    - RTX-compatiable graphics card
3. ISPC
    - x64 CPU with AVX256 or Apple M1 CPU
    - LLVM 12+ (for JIT executing the IR emitted by ISPC)
4. Metal
    - macOS 12 or higher
    - Discrete GPU or Apple M1 GPU (older GPUs are probably also supported, but not tested)
5. Python
    - Python 3.9+ (please use the system one because virtual environments might cause mismatches between the linked Python library and the intepreter)
    - Packages: astpretty, dearpygui, sourceinspect, numpy, pillow
    - Backend-specific requirements are the same as above

## CMake Flags

The ISPC backend is disabled by default. Other backends will automatically be enabled if the corresponding APIs are detected. You can override the default settings by supplying CMake flags manually, in form of `-DFLAG=value` behind the first cmake command.

In case you need to run the ISPC backend, download the [ISPC compiler executable](https://ispc.github.io/downloads.html) of your platform and copy it to `src/backends/ispc/ispc_support/` before compiling.

- `LUISA_COMPUTE_ENABLE_CUDA`: Enable CUDA backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_DX`: Enable DirectX backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_ISPC`: Enable ISPC backend (Default: `OFF`)
- `LUISA_COMPUTE_ENABLE_METAL`: Enable Metal backend (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_PYTHON`: Enable LuisaCompute Python (Default: `ON`)
- `LUISA_COMPUTE_ENABLE_GUI`: Enable GUI display in C++ tests (Default: `ON`)

## Build Commands

> Note: build directory should be named to be `build`, so that the set path script can set python path correctly.

```bash
cmake  -S . -B build	# optionally with CMake flags above
cmake --build build
```

## Tests

1. LuisaCompute C++ tests are in the `build/bin` folder.
2. LuisaCompute Python tests are in the `src/py` directory. Run `set_python_path.bat/ps1/sh` to set environment path to `luisa` before running any python test. To run `lrenderer.py`, copy files in `pyscenes` to the same directory of `lrenderer.py`.
3. All tests accept a command-line argument specifying backend, which can be chosen from `cuda`/`dx`/`ispc`/`metal`.
