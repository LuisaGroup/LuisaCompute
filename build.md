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
    - LLVM 12+
4. Metal
    - macOS 12 or higher
    - Discrete GPU or Apple M1 GPU
5. Python
    - Python 3.9+
    - Packages: TODO
    - Backend-specific requirements are the same as above

## CMake Flags

- `LUISA_COMPUTE_ENABLE_CUDA`: Enable CUDA backend
- `LUISA_COMPUTE_ENABLE_DX`: Enable DirectX backend
- `LUISA_COMPUTE_ENABLE_ISPC`: Enable ISPC backend
- `LUISA_COMPUTE_ENABLE_METAL`: Enable Metal backend
- `LUISA_COMPUTE_ENABLE_PYTHON`: Enable LuisaCompute Python
- `LUISA_COMPUTE_ENABLE_GUI`: Enable GUI display in C++ tests

## Build Commands

CMake flags should be add in form of `-DFLAG=value` behind first cmake command.

```bash
cmake  -S . -B build
cmake --build build
```

## Tests

1. LuisaCompute C++ tests are in build files.
2. LuisaCompute Python tests are in `src/py` directory. Test files should be copied into build directory and run there.