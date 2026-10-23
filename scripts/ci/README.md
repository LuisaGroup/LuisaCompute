# Backend and LLVM compatibility CI

`build-backends.yml` adds a Linux CPU matrix for LLVM 20, 21, 22, 23,
and 24, plus a manually selected Windows backend build. Every job completes
the full configured build before running tests. Missing dependencies fail
configuration rather than silently removing requested backends.

## Routes

| Backend | Code generation |
| --- | --- |
| fallback | LLVM CPU |
| SIMD | Schedule IR and LLVM CPU |
| HIP | LLVM AMDGPU and HIPRT bitcode |
| Vulkan | XIR to native SPIR-V by default |
| DX | HLSL and DXC |

The optional `llvm_spirv` input is experimental and defaults to false. Its
resource-descriptor limitations prevent the buffer runtime tests from running;
the helper rejects combining this option with `gpu_tests`. It is not the
default Vulkan route.

## Windows reproduction

Use an x64 Visual Studio Build Tools developer shell with Git, CMake, Ninja,
Python, and the Vulkan SDK available. A full Visual Studio IDE is unnecessary.
Clone recursively and select the requested branch:

```powershell
git clone --branch next --recurse-submodules https://github.com/LuisaGroup/LuisaCompute.git
cd LuisaCompute
git submodule update --init --recursive
./scripts/ci/prepare-windows-backends.ps1
./scripts/ci/build-windows-backends.ps1 -Fallback $true -Simd $true -Hip $true -Vulkan $true -Dx $true -RunGpuTests $true
```

The preparation script downloads pinned official ROCm, Embree, and oneTBB
archives and verifies SHA256 hashes. Existing installations can be supplied
with `-RocmRoot`, `-EmbreeRoot`, and `-TbbRoot`. It checks LLVM's actual CRT ABI
before supplying missing package metadata. The build script sets HIP_PATH,
ROCM_PATH, compiler/bitcode paths, and the process PATH, and copies the matching
runtime DLLs next to executables. Running HIP binaries from an unrelated shell
without the ROCm environment can fail device discovery.

The pinned ROCm distribution targets gfx120X; other AMD architectures require
a matching distribution and HIPRT_GPU_ARCHS. Windows hardware tests require a
prepared self-hosted runner tagged `luisa-amd-gfx120x`. Hosted runners compile
GPU backends and run CPU tests; they do not certify GPU runtime behavior.

## Compatibility scope

The version-specific source paths cover AMDGPU target-parser header changes,
OptimizationLevel, MCSubtargetInfo, and compressstore address-space overload changes in LLVM 23, and MergeFunctions
analysis-manager and conditional-branch changes in LLVM 24. LLVM 23+ SIMD
integer division/remainder uses masked intrinsics; LLVM 20–22 retains guarded
operands for inactive lanes. Numerical tail-lane tests cover signed and unsigned
division/remainder.

Local validation used ROCm LLVM 24.0.0git. LLVM 20–23 API paths were checked
against official release headers, but are not claimed to have passed local
full builds. The five-version Linux matrix must actually run before claiming
all versions validated; it covers CPU backends, not every HIP SDK/compiler
combination. Workflow syntax checking alone is not a successful CI run.

CTest XML and configure/build/runtime logs are uploaded even on failure.
Keep execution-policy failures distinct from compiler or numerical failures;
do not disable machine security controls to make a test execute.

The Windows DXC runtime remains the project's pinned DX SDK from scripts/sdks.cmake. The Vulkan SDK provides platform tooling and headers; its DXC DLL is not substituted into the build. The HLSL atomic regression compiles and validates actual SPIR-V using the project DXC runtime.
