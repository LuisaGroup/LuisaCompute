# Metal4 AIR iOS Device Conformance and Path Tracing

This application is the physical-device closure test for Luisa's Metal4 AIR
backend. It links the real static `DeviceInterface`, DSL, XIR passes, LLVM 21
AIR code generator, LLVM 14 downgrade, MTLB writer, and Metal4 runtime into one
signed arm64 iOS application. No MSL code generator is linked or used.

The iPhone performs the complete shader path at runtime:

~~~text
DSL/AST -> XIR -> XIR optimization -> LLVM 21 IR -> LLVM O2
        -> LLVM 14 downgrade -> Apple AIR -> MTLB -> MTL4 compiler/runtime
~~~

This is legal on iOS because LLVM is used as an IR builder, optimizer, verifier,
and bitcode writer. The app does not create or execute arbitrary CPU machine
code. Metal still performs the normal runtime compilation of GPU AIR.

The app must pass all of the following before it writes a successful result:

- Metal LogState shader logging with an exact callback message;
- bindless table update and typed buffer read;
- GPU-authored MTL4 indirect dispatch and kernel-ID propagation;
- offscreen AIR vertex/fragment rasterization with D32 depth and readback;
- real triangle Mesh and TLAS construction through the runtime feature guard;
- closest-hit and any-hit AIR ray tracing, shader execution reordering, and a
  seven-bounce Cornell-style path trace;
- nonempty/nondegenerate pixel checks, PNG creation, and JSON evidence.

The host repository remains a CMake/Ninja build. The signed iOS bundle is also
defined by the root CMake project, but uses CMake's Xcode generator because
physical-device provisioning and automatic signing are Xcode workflows.

## Requirements

- Xcode 26 with the matching iOS platform and Metal command-line tools.
- An iPhone running iOS 26 or newer, paired, unlocked, and connected with a
  data-capable cable or an active CoreDevice wireless tunnel.
- Developer Mode enabled on the iPhone.
- An Apple Development identity and development team. A Personal Team works;
  its provisioning profile expires after seven days.
- LLVM 21 built as arm64 iOS static libraries. Configure the Luisa iOS build
  with its `lib/cmake/llvm` directory through `LLVM_DIR`.
- The current metal-cpp headers under `src/backends/common/metal-cpp`.

The first Personal Team launch may require manual trust under Settings ->
General -> VPN & Device Management -> Developer App.

## Configure and sign the runtime-linked app

From the repository root, replace the LLVM and development-team placeholders:

~~~sh
cmake -S . -B cmake-build-ios-metal4-device-air-xcode -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=<ios-llvm21>/lib/cmake/llvm \
  -DLUISA_COMPUTE_BUILD_TESTS=OFF \
  -DLUISA_COMPUTE_ENABLE_CLANG_CXX=OFF \
  -DLUISA_COMPUTE_ENABLE_CUDA=OFF \
  -DLUISA_COMPUTE_ENABLE_DX=OFF \
  -DLUISA_COMPUTE_ENABLE_FALLBACK=OFF \
  -DLUISA_COMPUTE_ENABLE_GUI=OFF \
  -DLUISA_COMPUTE_ENABLE_HIP=OFF \
  -DLUISA_COMPUTE_ENABLE_METAL=OFF \
  -DLUISA_COMPUTE_ENABLE_METAL4=ON \
  -DLUISA_COMPUTE_ENABLE_SIMD=OFF \
  -DLUISA_COMPUTE_ENABLE_TENSOR=OFF \
  -DLUISA_COMPUTE_ENABLE_VULKAN=OFF \
  -DLUISA_IOS_DEVELOPMENT_TEAM=<team-id> \
  -DLUISA_IOS_PATH_TRACING_SPP=8

cmake --build cmake-build-ios-metal4-device-air-xcode \
  --config Release \
  --target luisa-metal4-ios-device-air-path-tracer -j 8

codesign --verify --deep --strict --verbose=2 \
  cmake-build-ios-metal4-device-air-xcode/bin/Release/luisa-metal4-ios-device-air-path-tracer.app
~~~

The resulting executable is intentionally large because it contains the iOS
LLVM 21 static libraries. That size is acceptable for this conformance runner;
shipping applications should normally cache or distribute precompiled MTLB
artifacts while keeping this dynamic path available for development.

## Install, run, and retrieve evidence

CoreDevice must report the phone as `available`, not merely remember a paired
device as `unavailable`:

~~~sh
xcrun devicectl list devices

xcrun devicectl device install app \
  --device <device-udid> \
  cmake-build-ios-metal4-device-air-xcode/bin/Release/luisa-metal4-ios-device-air-path-tracer.app

xcrun devicectl device process launch \
  --device <device-udid> \
  --terminate-existing --console \
  org.luisa-compute.metal4-path-tracer

xcrun devicectl device copy from \
  --device <device-udid> \
  --domain-type appDataContainer \
  --domain-identifier org.luisa-compute.metal4-path-tracer \
  --source Documents/luisa_metal4_path_tracing.png \
  --destination <local-output.png>

xcrun devicectl device copy from \
  --device <device-udid> \
  --domain-type appDataContainer \
  --domain-identifier org.luisa-compute.metal4-path-tracer \
  --source Documents/luisa_metal4_path_tracing.json \
  --destination <local-output.json>
~~~

Do not count installation or launch as a pass. The JSON must report at least:

- `success: true`, `metal4: true`, and `renderer` equal to the hardware RTX
  Cornell path tracer;
- `shader_generation` equal to the device AST/XIR/LLVM/downgrade/AIR path;
- the exact shader-log message `ios-metal4-air-log value=42`;
- bindless value `0x13579bdf` (decimal `324508639`) and indirect checksum
  `8084`;
- approximately 1,352 colored pixels in the 64x64 raster probe and a nonblack
  interpolated center pixel;
- a nonzero acceleration-build, raster, compile, and dispatch/readback time;
- a nondegenerate path-traced image, its raw-pixel SHA-256, and PNG path.

On Apple9 and newer, `acceleration_structure_path` must select native MTL4
address-driven AS construction. Apple7/Apple8 devices retain the synchronized
legacy AS-build bridge because they can compile and dispatch Metal4 AIR ray
tracing but cannot execute MTL4 AS build commands. This is a runtime feature
guard, not a shader-codegen fallback.

## Desktop preflight using the identical workload

The iOS conformance body is deliberately portable. The normal macOS
CMake/Ninja build registers it as `test_metal4_device_conformance`, so CodeGen,
ABI, image, and runtime failures can be found before provisioning the phone:

~~~sh
cmake --build cmake-build-metal-air-llvm21 \
  --target test_metal4_device_conformance -j 8

ctest --test-dir cmake-build-metal-air-llvm21 \
  -R '^test_metal4_device_conformance$' --output-on-failure -V

LUISA_ENABLE_VALIDATION=1 MTL_DEBUG_LAYER=1 \
ctest --test-dir cmake-build-metal-air-llvm21 \
  -R '^test_metal4_device_conformance$' --output-on-failure -V
~~~

On 2026-08-28, both normal and Metal API Validation runs passed on an M1 Max.
The guard reported `metal4_address_driven_acceleration_structures=false` and
used the synchronized compatibility AS builder while retaining Metal4 AIR for
all shader compilation and dispatch. The normal run produced:

| Check | Result |
|---|---:|
| Shader log | `ios-metal4-air-log value=42` |
| Bindless value | `0x13579bdf` |
| Indirect checksum | 8,084 |
| Raster colored pixels | 1,352 |
| Raster center RGBA | `(63, 67, 125, 255)` |
| AS build | 44.33 ms |
| RTX compile | 192.15 ms |
| RTX dispatch/readback, 256x256 at 4 spp | 8.02 ms |
| Nonblack pixels | 65,536 |
| Maximum channel | 247 |
| Mean normalized RGB | 0.340667 |
| PNG SHA-256 | `02859d00fd996b0fd3bd054de7bab6d5176828c0d62b2e6133a2a329a59e3b01` |

The whole host suite also passed 160/160 tests, including the rendering,
tutorial, Metal4 AIR, raster, ray tracing, and validation groups.

## Host-AOT baseline

`luisa-metal4-ios-path-tracer-aot` remains a small container/ABI oracle. It
generates a simple SDF kernel on macOS and is useful for
`metallib --app-store-validate`, AIR/MTLB reverse engineering, and comparing
cold/warm MTL4 pipeline creation. It is not the acceptance path for the
runtime-linked app because it does not execute XIR or LLVM on the phone and it
does not exercise Mesh/TLAS construction.

The earlier iPhone 17 Pro Max host-AOT baseline used iOS 26.6 and an A19 Pro.
At 512x512, its first 8-spp launch compiled the MTL4 pipeline in 38.538 ms and
executed in 7.257 ms. Warm 32-spp repetitions compiled in 1.100..1.205 ms and
executed in 25.853..28.005 ms with identical raw pixels. Retain those numbers
as the manual-encoder baseline only; the runtime-linked RTX run needs its own
retrieved PNG/JSON evidence.
