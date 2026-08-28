# Metal4 AIR iOS Path-Tracing Probe

This app is the physical-device smoke test for Luisa's explicit iOS AIR target.
The macOS host generates the shader through
`DSL/AST -> XIR -> XIR passes -> LLVM 21 -> LLVM O2 -> LLVM 14 downgrade -> AIR -> MTLB`.
The iOS app verifies the bundled library and runs it exclusively through MTL4
compiler, queue, command-buffer, compute-encoder, argument-table, allocator,
residency-set, and commit-feedback APIs.

The host repository remains a CMake/Ninja build. This small app is also defined
by CMake, but uses CMake's Xcode generator because physical-device provisioning,
automatic signing, and installation are Xcode workflows.

## Requirements

- Xcode 26 with the matching iOS platform component installed.
- An iPhone running iOS 26 or newer, paired and connected with Developer Mode
  enabled.
- An Apple Development identity and development team. A Personal Team works;
  its provisioning profile expires after seven days.
- The current metal-cpp headers under `src/backends/common/metal-cpp`.

The first Personal Team launch may require manual trust on the phone under
Settings -> General -> VPN & Device Management -> Developer App.

## Generate the iOS AIR library on the host

From the repository root:

~~~sh
cmake --build cmake-build-metal4-air \
  --target luisa-metal4-ios-path-tracer-aot -j 8

cmake-build-metal4-air/bin/luisa-metal4-ios-path-tracer-aot \
  cmake-build-metal4-air/ios-path-tracer/luisa_ios_path_tracing.metallib \
  26.0 26.4

xcrun metallib --app-store-validate \
  cmake-build-metal4-air/ios-path-tracer/luisa_ios_path_tracing.metallib
~~~

The generator rejects an unexpected root layout. The expected root block is
32 bytes: an 8-byte output texture resource ID in the first 16-byte slot and a
32-bit sample count at offset 16 in the second slot.

## Configure and sign the device app

Replace the development-team and device placeholders:

~~~sh
cmake -S src/backends/metal4/tools/ios_path_tracer_app \
  -B cmake-build-ios-path-tracer -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
  -DLUISA_METAL_CPP_DIR="$PWD/src/backends/common/metal-cpp" \
  -DLUISA_IOS_PATH_TRACER_METALLIB="$PWD/cmake-build-metal4-air/ios-path-tracer/luisa_ios_path_tracing.metallib" \
  -DLUISA_IOS_DEVELOPMENT_TEAM=<team-id> \
  -DLUISA_IOS_PATH_TRACING_SPP=8

xcodebuild \
  -project cmake-build-ios-path-tracer/luisa_metal4_ios_path_tracer.xcodeproj \
  -scheme luisa-metal4-ios-path-tracer \
  -configuration Release \
  -destination id=<device-udid> \
  -allowProvisioningUpdates \
  -allowProvisioningDeviceRegistration build
~~~

The CMake configure step hashes the input metallib and compiles that expected
SHA-256 into the app. Runtime execution fails before library loading if the
bundled bytes do not match.

## Install, run, and retrieve evidence

~~~sh
xcrun devicectl device install app \
  --device <device-udid> \
  cmake-build-ios-path-tracer/Release-iphoneos/luisa-metal4-ios-path-tracer.app

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

Do not count installation or app launch as a pass. The JSON must report:

- `success: true`, `metal4: true`, and matching `air_sha256` and
  `bundled_air_sha256` values;
- a 32-byte root block and 16-byte dispatch-size record;
- successful MTL4 library/pipeline creation and nonzero GPU completion time;
- a nonuniform output image with the expected scene geometry.

`apple9` selects the native MTL4 acceleration-structure-build capability.
Devices below Apple9 retain the runtime's synchronized compatibility build
bridge; the probe records this boundary even though its current SDF scene does
not allocate an acceleration structure.

## Current compilation boundary

This probe is host-AOT plus device-side Metal pipeline compilation. It does not
yet run XIR or LLVM inside the iOS process. iOS prevents arbitrary CPU machine-
code JIT, but that does not prevent runtime GPU shader compilation. On-device
XIR-to-AIR generation can use the same code path once LLVM 21, the LLVM
downgrade, and Luisa's AIR codegen dependencies are available as iOS arm64
static libraries. Such a build needs no CPU JIT entitlement, but it does carry
binary-size, startup-time, memory, private-AIR-ABI, and distribution risks.

## Reference device result

On 2026-08-28, an iPhone 17 Pro Max with iOS 26.6 reported an Apple A19 Pro GPU
with Metal4, Apple9, and Apple10 support. A cold 512x512, 8-spp run created the
pipeline in 38.538 ms and executed in 7.257 ms. A second 32-spp run hit the
system pipeline cache. Two warm 32-spp repetitions created the pipeline in
1.100..1.205 ms and executed in 25.853..28.005 ms; their raw pixel hashes and
encoded PNG bytes matched exactly. The retrieved image contained 15,144
distinct RGBA values and the expected sphere, box, checker floor, lighting,
shadows, and sampling noise.
