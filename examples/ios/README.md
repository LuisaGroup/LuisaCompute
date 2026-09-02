# LuisaCompute iOS Examples

The targets in this directory run the repository's existing rendering examples
inside native UIKit application bundles. They preserve the normal Luisa path:

~~~text
example source -> Context -> create_device(argv[1])
               -> Window -> Swapchain -> Metal4 DeviceInterface
               -> XIR -> LLVM -> downgraded AIR -> MTL4 runtime
~~~

The rendering sources do not hard-code `metal4`. The small iOS host supplies it
as `argv[1]`, exactly as a desktop launch such as
`example_path_tracing metal4` does. The source `main` is renamed only while it
is included in its iOS application target. Arguments passed after the bundle
identifier are forwarded unchanged starting at `argv[2]`, so the same example
options can select an offline run or a finite performance experiment:

~~~sh
xcrun devicectl device process launch \
  --device <device-id> --terminate-existing --console \
  <bundle-id> -- \
  --offline --spp 64 --max-spp-per-dispatch 1
~~~

Launching without additional arguments retains the normal interactive
Window/Swapchain behavior.

`example_ios_path_tracing_cutout` additionally accepts
`--trace-mode direct|opaque-query|accept-query|cutout-query`,
`--ray-query-lowering pipeline|loop`, and `--capture-float4s N`. Omitting the
lowering option exercises Metal4's automatic device/payload policy; `pipeline`
forces every semantically eligible triangle query through a non-null
intersection-function table, while `loop` forces the native stateful query.
The direct and opaque-query modes remain useful as an all-opaque direct-
intersector versus stateful-query pair, but they are separate from the matched
cutout IFT/loop comparison documented in the AIR report.

## Reproducible build scripts

The shortest complete setup uses two repository scripts. Homebrew LLVM 21 is
only a host tool provider for `llvm-tblgen`; none of its macOS libraries are
linked into the phone application:

~~~sh
brew install llvm@21 ninja

# Downloads the official llvm-project-21.1.8 source release when it is not
# already cached, then builds static arm64 iPhoneOS libraries with Ninja.
scripts/build_ios_llvm.sh \
  --host-llvm-prefix "$(brew --prefix llvm@21)"

# Configure all 19 examples, the matched Metal4 benchmark, and the independent
# device-test mirror; sign the bundles through Xcode automatic signing.
scripts/build_ios_metal4.sh \
  --llvm-dir cmake-build-llvm21-ios/lib/cmake/llvm \
  --team <apple-development-team> \
  --mode all
~~~

Pass `--mode examples`, `--mode tests`, or `--mode benchmark` to build only one
opt-in group. For a CI/cross-link check without a certificate, use `--unsigned`;
those bundles cannot be installed on a device:

~~~sh
scripts/build_ios_metal4.sh \
  --llvm-dir cmake-build-llvm21-ios/lib/cmake/llvm \
  --mode all --unsigned

scripts/audit_ios_bundles.sh \
  --bin-dir cmake-build-ios-metal4-device-air-xcode/bin/Release \
  --expected-count 21
~~~

The LLVM script also accepts `--source <llvm-project>` to use an existing
official source checkout/tarball, `--llvm-version`, `--source-cache`, and
`--build-dir`. It rejects a non-LLVM-21 source or host `llvm-tblgen`, because
mixing LLVM majors would invalidate the AIR/downgrade integration.

Xcode's system toolchain is sufficient for compiling ordinary iOS C++, but it
does not expose the complete linkable LLVM 21 development archives needed by
runtime XIR-to-AIR generation. Official prebuilt arm64 macOS LLVM packages are
also tagged for the macOS Mach-O platform, not iPhoneOS, and cannot replace the
cross-built libraries. An AOT-only application could omit LLVM by shipping
precompiled MTLB files, but it would not exercise the required on-device
`XIR -> LLVM -> downgrade -> AIR` path.

The LLVM dependency is configured with CMake/Ninja. The final application
project uses CMake's Xcode generator because provisioning, entitlements, and
automatic device signing are Xcode build operations; this is not an alternate
source or backend build system.

## Build separation

Three independent options keep examples, tests, and the benchmark out of
normal builds:

- `LUISA_COMPUTE_BUILD_IOS_EXAMPLES=ON` builds the interactive examples here.
- `LUISA_COMPUTE_BUILD_IOS_TESTS=ON` builds the conformance mirror under
  `src/tests/ios`.
- `LUISA_COMPUTE_BUILD_IOS_BENCHMARKS=ON` builds the matched offline backend
  comparison under `examples/ios/benchmark`.

The examples and device tests require an iOS toolchain, Metal4, GUI support,
and an arm64 iPhoneOS LLVM 21 static build. The comparison selects exactly one
of `metal` or `metal4`; the generic benchmark build script configures those
backend requirements without linking both into one app.

The following is the expanded manual equivalent of
`scripts/build_ios_metal4.sh`; the script is preferred so local and CI flags do
not drift:

~~~sh
cmake -S . -B cmake-build-ios-metal4-device-air-xcode -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
  -DLLVM_DIR=<ios-llvm21>/lib/cmake/llvm \
  -DLUISA_COMPUTE_BUILD_TESTS=OFF \
  -DLUISA_COMPUTE_BUILD_IOS_EXAMPLES=ON \
  -DLUISA_COMPUTE_BUILD_IOS_TESTS=ON \
  -DLUISA_COMPUTE_BUILD_IOS_BENCHMARKS=ON \
  -DLUISA_COMPUTE_IOS_BENCHMARK_BACKEND=metal4 \
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
  -DLUISA_IOS_DEVELOPMENT_TEAM=<team-id>

cmake --build cmake-build-ios-metal4-device-air-xcode \
  --config Release \
  --target luisa-ios-rendering-examples luisa-ios-device-tests \
           example_ios_path_tracing_benchmark -j 8
~~~

`example_ios_path_tracing` is the primary interactive path tracer. It also runs
the device-conformance preflight and records image/JSON evidence. Every other
ported source has a target named `example_ios_<desktop-example-name>`.

The explicit target list currently covers path tracing and its camera, cutout,
HDR, nested-callable, ray-mask, spectrum, and XIR-to-AST variants; photon
mapping; both SDF renderers; black hole; voxel ray tracing; procedural ray
tracing; shader-toy variants; shader visuals; and both coroutine renderers.
`procedural` and `coro_sdf_renderer` use progressive Window/Swapchain loops on
iOS while retaining finite deterministic `--offline` behavior.

## Window, presentation, and input

UIKit owns the `UIView` and `CAMetalLayer`. The common Luisa `Window`
constructor obtains that native layer from a provider, and the example itself
creates the regular `Swapchain`. The host aspect-fits the requested render
resolution inside the safe area, so square and landscape images are not
stretched to the portrait screen.

Input is routed through the same Window callbacks/state queried by desktop
examples:

- one-finger drag is the left mouse button plus cursor motion;
- two-finger drag is the right mouse button plus cursor motion;
- pinch is a scroll event;
- a connected hardware keyboard supplies key state and key callbacks.

Native events are queued and delivered by `Window::poll_events()` on the
rendering thread. Camera accumulation is reset when touch changes the view.

## Static application layout and LLVM

Each bundle contains one arm64 Mach-O executable. Luisa runtime, DSL, XIR,
Metal4, coroutine support when needed, llvm-downgrade, and LLVM 21 component
archives are linked statically and dead-stripped. No Luisa or LLVM dynamic
library is expected in `otool -L`; only Apple system frameworks and libraries
should remain.

The unsigned arm64 audit on 2026-08-28 built all 19 rendering-example bundles,
the Metal4 benchmark, and the separate device-test mirror. All 21 executables
were arm64 and
`otool -L` found only Apple system frameworks/libraries: LLVM,
`llvm-downgrade`, Luisa, and the Metal4 backend were all inside each Mach-O.

The 2026-09-01 follow-up also completed an examples-only generated Xcode
project's unsigned Release `ALL_BUILD` with `CODE_SIGNING_ALLOWED=NO`: all 49
targets built and linked, including the 19 rendering-example bundles. Its
bundle audit passes with `--expected-count 19`. The cutout bundle was then
separately signed and installed on the physical iPhone 17 Pro Max. Automatic
ray-query lowering completed finite renders at zero, four, and sixteen mutable
`float4` captures; device logs proved the 128-byte Apple10 gate selected IFT at
zero and four, and retained the stateful loop at sixteen (512 bytes).

The physical iPhone 17 Pro Max run separately executed four representative
signed apps rather than inferring runtime support from that link audit:

- `example_ios_path_tracing` passed the complete Apple10 capability matrix and
  interactively displayed the real repository 1024x1024 path tracer;
- `example_ios_procedural` compiled and ran the procedural ray-query renderer;
- `example_ios_coro_path_tracing` compiled its nine coroutine subroutines and
  accumulated beyond 1,280 spp while presenting;
- `example_ios_shader_visuals_present` created its 1280x720
  Window/Swapchain/MTL4 presentation and remained stable during observation.

All 19 bundles are build/link/audit covered; the list above is the current
on-device execution set and deliberately does not claim that the remaining 15
apps were individually launched.

LLVM is not used as a CPU JIT on iOS. It builds, optimizes, verifies, downgrades,
and serializes GPU IR. Apple's Metal compiler consumes the resulting AIR for
GPU execution. Shader caches and generated evidence belong in Application
Support or Documents because the signed app bundle is read-only.

An unsigned build (`CODE_SIGNING_ALLOWED=NO`) proves only cross-compilation and
link closure. It is not device evidence. Rebuild with signing enabled, install
the `.app`, run it on the phone, and inspect the visible output, console, and
retrieved evidence before declaring a rendering target supported.

See [metal4_path_tracing/README.md](metal4_path_tracing/README.md) for the
physical-device conformance workflow.
See [benchmark/README.md](benchmark/README.md) for matched old-Metal/Metal4
build, execution, cache interpretation, and A19 Pro measurements.

## Continuous integration boundary

`.github/workflows/build-ios.yml` runs on GitHub's arm64 `macos-26` image with
Xcode 26.6. It downloads/caches the same official LLVM 21.1.8 source, builds the
static iPhoneOS libraries, compiles all Metal4 example/test/benchmark bundles
and a separate old-Metal benchmark without code signing, and runs
`scripts/audit_ios_bundles.sh`. CI therefore enforces target count, arm64
architecture, iPhoneOS Mach-O platform, system-only dynamic dependencies, and
static inclusion of the selected Luisa backend and LLVM/llvm-downgrade where
applicable.

A hosted runner has neither this repository's development identity nor the
physical iPhone. CI success is build/link/package evidence, not a substitute
for launching the signed app, observing progressive presentation, and
retrieving/validating its JSON and PNG artifacts.
