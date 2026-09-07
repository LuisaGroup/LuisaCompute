# Metal4 AIR iOS Device Conformance and Path Tracing

This application is the physical-device closure test for Luisa's Metal4 AIR
backend. It links the real static `DeviceInterface`, DSL, XIR passes, LLVM 22
AIR code generator, LLVM 14 downgrade, MTLB writer, and Metal4 runtime into one
signed arm64 iOS application. No MSL code generator is linked or used.

The iPhone performs the complete shader path at runtime:

~~~text
DSL/AST -> XIR -> XIR optimization -> LLVM 22 IR -> LLVM O2
        -> LLVM 14 downgrade -> Apple AIR -> MTLB -> MTL4 compiler/runtime
~~~

This is legal on iOS because LLVM is used as an IR builder, optimizer, verifier,
and bitcode writer. The app does not create or execute arbitrary CPU machine
code. Metal still performs the normal runtime compilation of GPU AIR.

The app must pass all of the following before it writes a successful result:

- Metal LogState shader logging with an exact callback message;
- four-byte `{bool, bool, bool, bool}` and `bool4` ABI, four-byte `byte4`
  load/store, device atomics, and direct texture I/O;
- `ExternalCallable` value/reference ABI with LLVM IR supplied through
  `ShaderOption::native_include`, linked before AIR optimization/downgrade;
- cross-stream timeline ordering at an unsigned fence value above `INT64_MAX`;
- bindless table update and typed buffer read;
- GPU-authored MTL4 indirect dispatch and kernel-ID propagation;
- offscreen AIR vertex/fragment rasterization with `base_instance`, D32 depth,
  D24S8/D32S8A24 stencil compare/replace, and readback;
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
- LLVM 22 built as arm64 iOS static libraries. Configure the Luisa iOS build
  with its `lib/cmake/llvm` directory through `LLVM_DIR`.
- The current metal-cpp headers under `src/backends/common/metal-cpp`.

The first Personal Team launch may require manual trust under Settings ->
General -> VPN & Device Management -> Developer App.

## Configure and sign the runtime-linked app

The reproducible shortcut is:

~~~sh
scripts/build_ios_llvm.sh \
  --host-llvm-prefix "$(brew --prefix llvm@22)"
scripts/build_ios_metal4.sh \
  --llvm-dir cmake-build-llvm22-ios/lib/cmake/llvm \
  --team <apple-development-team> --mode all
~~~

From the repository root, the expanded manual equivalent is:

~~~sh
cmake -S . -B cmake-build-ios-metal4-device-air-xcode -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=<ios-llvm22>/lib/cmake/llvm \
  -DLUISA_COMPUTE_BUILD_TESTS=OFF \
  -DLUISA_COMPUTE_BUILD_IOS_TESTS=ON \
  -DLUISA_COMPUTE_BUILD_IOS_EXAMPLES=ON \
  -DLUISA_COMPUTE_ENABLE_CLANG_CXX=OFF \
  -DLUISA_COMPUTE_ENABLE_CUDA=OFF \
  -DLUISA_COMPUTE_ENABLE_DX=OFF \
  -DLUISA_COMPUTE_ENABLE_FALLBACK=OFF \
  -DLUISA_COMPUTE_ENABLE_GUI=ON \
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
  --target example_ios_path_tracing -j 8

codesign --verify --deep --strict --verbose=2 \
  cmake-build-ios-metal4-device-air-xcode/bin/Release/example_ios_path_tracing.app
~~~

`example_ios_path_tracing` is the interactive example bundle. The independent
`LUISA_COMPUTE_BUILD_IOS_TESTS` mirror is named
`luisa-metal4-ios-device-air-path-tracer`; it uses the same shared conformance
and repository path-tracing sources but a test-specific bundle identifier.

The resulting executable is intentionally large because it contains the iOS
LLVM 22 static libraries. That size is acceptable for this conformance runner;
shipping applications should normally cache or distribute precompiled MTLB
artifacts while keeping this dynamic path available for development.

## Install, run, and retrieve evidence

CoreDevice must report the phone as `available`, not merely remember a paired
device as `unavailable`:

~~~sh
xcrun devicectl list devices

xcrun devicectl device install app \
  --device <device-udid> \
  cmake-build-ios-metal4-device-air-xcode/bin/Release/example_ios_path_tracing.app

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
- ABI checksum `166`, atomic value `64`, and timeline value `2^63`;
- native-include/`ExternalCallable` checksum `3840`;
- nonzero matrix-motion hit count and a positive time-dependent centroid
  delta; on Apple9 or newer, the SRT/component-motion probe must also be
  reported as exercised with the same invariants;
- exactly 1,352 colored pixels in the 64x64 varying/`base_instance` raster
  probe, 2,704 colored pixels across the D24S8 and D32S8A24 stencil probes,
  and a nonblack interpolated center pixel;
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
cmake --build cmake-build-metal-air-llvm22 \
  --target test_metal4_device_conformance -j 8

ctest --test-dir cmake-build-metal-air-llvm22 \
  -R '^test_metal4_device_conformance$' --output-on-failure -V

LUISA_ENABLE_VALIDATION=1 MTL_DEBUG_LAYER=1 \
ctest --test-dir cmake-build-metal-air-llvm22 \
  -R '^test_metal4_device_conformance$' --output-on-failure -V
~~~

On 2026-08-28, both normal and Metal API Validation runs passed on an M1 Max.
The guard reported `metal4_address_driven_acceleration_structures=false` and
used the synchronized compatibility AS builder while retaining Metal4 AIR for
all shader compilation and dispatch. The normal run produced:

| Check | Result |
|---|---:|
| Shader log | `ios-metal4-air-log value=42` |
| bool/byte ABI checksum | 166 |
| Device atomic result | 64 |
| Direct BYTE4 texture RGBA | `(1.0, 0.066667, 0.129412, 1.0)` |
| ExternalCallable/native-include checksum | 3,840 |
| Unsigned cross-stream timeline | `0x8000000000000000` |
| Matrix/primitive motion | 464 hits, 8.36-pixel centroid delta |
| Component motion | Correctly skipped by the Apple9 feature guard |
| Bindless value | `0x13579bdf` |
| Indirect checksum | 8,084 |
| Raster colored pixels | 1,352 |
| D24S8 plus D32S8A24 stencil colored pixels | 2,704 |
| Raster center RGBA | `(63, 67, 125, 255)` |
| AS build | 44.33 ms |
| RTX compile | 192.15 ms |
| RTX dispatch/readback, 256x256 at 4 spp | 8.02 ms |
| Nonblack pixels | 65,536 |
| Maximum channel | 247 |
| Mean normalized RGB | 0.340667 |
| PNG SHA-256 | `02859d00fd996b0fd3bd054de7bab6d5176828c0d62b2e6133a2a329a59e3b01` |

The whole host suite also passed 159/159 tests, including the rendering,
tutorial, Metal4 AIR, raster, ray tracing, and validation groups.

## Runtime-linked iPhone result (2026-08-28)

The signed `example_ios_path_tracing` bundle passed on an iPhone 17 Pro Max
running iOS 26.6. Metal reported `Apple A19 Pro GPU`, Luisa selected family
Apple10, the MTL4 runtime and address-driven acceleration-structure path were
active, and the pre-Apple9 compatibility bridge was not used. Every feature in
the emitted `exercised_features` map reported `passed`, including the static
`DeviceInterface`, device XIR/LLVM/AIR generation, LLVM 14 downgrade, MTL4
compiler/queue/command buffer/compute encoder, native shader logging,
bool/byte ABI, atomics, direct textures, external callable/native include,
unsigned timeline event, bindless and indirect dispatch, raster
`base_instance`, both stencil formats, matrix/component motion, address-driven
AS construction, closest/any-hit ray tracing, shader execution reordering,
Window/Swapchain presentation, and repository path tracing.

The A19 GPU exposes a method signature for the macOS-oriented
`isDepth24Stencil8PixelFormatSupported` query but does not respond to that
selector. The runtime now checks actual Objective-C selector responsiveness
before calling it and safely maps Luisa D24S8 storage to D32S8A24 when the
query is unavailable. Both logical depth formats then passed the real two-draw
Replace/Equal stencil probe.

| Check | A19 Pro device result |
|---|---:|
| Shader log | `ios-metal4-air-log value=42` |
| bool/byte ABI / atomic / native include | 166 / 64 / 3,840 |
| Unsigned cross-stream timeline | `0x8000000000000000` |
| Matrix motion | 464 hits, 8.357-pixel centroid delta |
| Component/SRT motion | 448 hits, 7.397-pixel centroid delta |
| Bindless / indirect | `0x13579bdf` / 8,084 |
| Raster / stencil colored pixels | 1,352 / 2,704 |
| Conformance RTX compile | 74.67 ms |
| Conformance RTX dispatch/readback, 512x512 at 8 spp | 22.67 ms |
| Conformance image | 262,144 nonblack, max 247, mean luma 0.358485 |
| Conformance raw RGBA SHA-256 | `633e3d5a62273d90f93f59c6856b0c7b1f572895fdd29622f2487c48d1a95080` |

After conformance, the same app invoked the real
`examples/rendering/path_tracing.cpp` implementation and continued presenting
through `Window -> Swapchain -> MTL4`. Its retrieved 1024x1024, 64-spp snapshot
completed in 2,002.36 ms, contained 1,046,904 nonblack pixels with maximum
channel 255 and mean luma 0.329453, and had raw RGBA SHA-256
`8e865e0ac3272b42b9b7362a6c15caf7679bd126b3eb9c6698fc2adc01769433`.
The retrieved image was visually inspected as a correctly oriented Cornell
box rather than treating the success flag alone as rendering evidence.

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
