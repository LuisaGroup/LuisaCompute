# iOS old Metal versus Metal4 benchmark

This application runs the repository's real
`examples/rendering/path_tracing.cpp` implementation through one statically
linked backend. It exists to compare backend behavior without changing the
scene, kernel, resolution, dispatch plan, readback, or pixel validation.

The two selectable paths are:

~~~text
metal:  AST -> MSL -> Metal runtime source compiler
metal4: AST -> XIR opt -> LLVM IR -> LLVM opt/downgrade -> Metal AIR
~~~

This is an offline performance harness. The interactive iOS rendering examples
and their common Window/Swapchain presentation remain under `examples/ios`.

## Build

Build the iPhoneOS LLVM archives once for Metal4:

~~~sh
brew install llvm@21 ninja
scripts/build_ios_llvm.sh \
  --host-llvm-prefix "$(brew --prefix llvm@21)"
~~~

Then configure two independent static application builds. Distinct bundle
identifiers give each backend its own Application Support shader cache:

~~~sh
scripts/build_ios_backend_benchmark.sh \
  --backend metal \
  --team <apple-development-team>

scripts/build_ios_backend_benchmark.sh \
  --backend metal4 \
  --llvm-dir cmake-build-llvm21-ios/lib/cmake/llvm \
  --team <apple-development-team>
~~~

The bundles are:

~~~text
cmake-build-ios-metal-benchmark-xcode/bin/Release/example_ios_path_tracing_benchmark.app
cmake-build-ios-metal4-benchmark-xcode/bin/Release/example_ios_path_tracing_benchmark.app
~~~

Pass `--unsigned` for a CI link audit. An unsigned bundle cannot run on a
phone. `--bundle-id` is available when a provisioning profile does not cover
the default `org.luisa-compute.benchmark.ios.{metal,metal4}` identifiers.

If Xcode is not signed into a developer account, use an existing profile and
identity instead of `--team`. The script extracts the profile entitlements,
derives or validates its bundle identifier, builds without Xcode signing, and
then signs the complete app:

~~~sh
scripts/build_ios_backend_benchmark.sh \
  --backend metal4 \
  --llvm-dir cmake-build-llvm21-ios/lib/cmake/llvm \
  --profile ~/Library/Developer/Xcode/UserData/Provisioning\ Profiles/<id>.mobileprovision \
  --identity <codesign-identity-sha1>
~~~

The Metal4 executable contains Luisa, XIR, llvm-downgrade, and the required
arm64 iPhoneOS LLVM component archives as statically linked and dead-stripped
code. LLVM is a GPU-IR construction/optimization library here; it is not a CPU
JIT and does not make executable iOS pages.

## Run and retrieve evidence

Find the CoreDevice identifier with `xcrun devicectl list devices`, then install
and launch one backend at a time:

~~~sh
device=<coredevice-id>
bundle=org.luisa-compute.benchmark.ios.metal4
app=cmake-build-ios-metal4-benchmark-xcode/bin/Release/example_ios_path_tracing_benchmark.app

xcrun devicectl device install app --device "$device" "$app"
xcrun devicectl device process launch \
  --device "$device" --terminate-existing --console "$bundle" -- \
  --spp 64 --iterations 3 --max-spp-per-dispatch 1 --clear-cache

mkdir -p benchmark-results/metal4
xcrun devicectl device copy from \
  --device "$device" \
  --domain-type appDataContainer \
  --domain-identifier "$bundle" \
  --source Documents/luisa_ios_path_tracing_benchmark.json \
  --destination benchmark-results/metal4/cold.json
xcrun devicectl device copy from \
  --device "$device" \
  --domain-type appDataContainer \
  --domain-identifier "$bundle" \
  --source Documents/luisa_ios_path_tracing_benchmark.png \
  --destination benchmark-results/metal4/cold.png
~~~

Repeat with the old-Metal bundle and identifier. A second launch without
`--clear-cache` measures a cross-process application-cache hit. Each launch
overwrites its JSON and PNG, so retrieve or rename them before the next launch.

The default workload is 1024x1024, 64 spp, one spp per dispatch, and three
complete scene runs. Run zero is process-first; runs one and two are in-process
warm. Stage synchronization is enabled only for this opt-in timing mode, so the
ordinary interactive path tracer keeps its normal pipelined behavior.

`--clear-cache` removes only this app's
`Application Support/LuisaComputeBenchmark` directory. It cannot clear Metal
framework, compiler-service, driver, or operating-system caches. Consequently,
"first compile after app-cache clear" is not equivalent to a factory-cold
Metal compiler. Preserve run order and preferably repeat in the reverse order.

Summarize one report per backend, or repeat either option to aggregate
order-reversed trials:

~~~sh
scripts/summarize_ios_backend_benchmark.py \
  --metal benchmark-results/metal/cold.json \
  --metal4 benchmark-results/metal4/cold.json
~~~

## iPhone 17 Pro Max result (2026-08-28)

The matched physical-device run used an iPhone 17 Pro Max, Apple A19 Pro GPU,
iOS 26.6, Release builds, 1024x1024, 64 spp, and one spp per dispatch. Two
backend orders were collected. Stable render is the median of in-process-warm
runs, excluding scene setup, acceleration build, compilation, initialization,
and readback.

| order set | old Metal stable render | Metal4 stable render | Metal4 time delta |
|---|---:|---:|---:|
| A | 342.556 ms | 348.699 ms | +1.793% |
| B, reversed | 347.260 ms | 342.582 ms | -1.347% |
| aggregate median | 343.865 ms (186.120 spp/s) | 344.622 ms (185.710 spp/s) | +0.220% |

The sign changes with order and the aggregate difference is 0.22%. For this
path-tracing kernel, Metal4 AIR and old Metal MSL therefore have equivalent
steady GPU performance on this device; there is no supported claim that
switching to Metal4 alone makes the render loop faster.

Compilation/cache observations must be interpreted separately:

| observation | old Metal | Metal4 AIR |
|---|---:|---:|
| first observed fresh application sandbox | 1738.016 ms | 147.481 ms |
| later reverse-order app-cache-clear trial | 80.771 ms | 153.679 ms |
| cross-process application-cache hit | 5.096 ms | 10.734 ms |
| aggregate in-process cache lookup median | 0.323 ms | 0.018 ms |

The first two rows demonstrate the uncontrolled Apple compiler-cache effect:
an app-cache clear changed old Metal's process-first compile by more than 20x.
They are useful operational measurements, but not a stable backend speedup
ratio. The last row verifies the corrected Metal4 semantic cache lookup and
shows that a repeated shader no longer executes XIR/LLVM/AIR generation.

The final images were not bit-identical, but they were numerically equivalent:

- both had 1,046,904 nonblack pixels and maximum channel value 255;
- mean luminance differed by 0.0000004523;
- RGB PSNR was 58.893 dB and mean absolute error was 0.01118/255;
- 2,676 of 1,048,576 pixels (0.2552%) differed in at least one RGB channel;
- alpha was identical everywhere.

## Metal4 cache regression found by the benchmark

Before this comparison, the Metal4 disk key was derived from downgraded
metallib bytes. Identical optimized LLVM IR produced byte-different downgraded
containers in separate processes, so every launch reported a metadata mismatch
and regenerated AST -> XIR -> LLVM -> AIR.

The compute cache now probes before lowering with a semantic key containing
the kernel AST hash, all code-affecting `ShaderOption` state (including
`native_include` and `max_registers`), the AIR platform/OS/SDK tuple, and an
explicit lowering/ABI revision. On the phone, shader compilation changed from
roughly 142--150 ms on every in-process iteration to 0.020/0.018 ms after the
first compile; the next process loaded the archive in 10.734 ms with no cache
metadata mismatch.

This benchmark exercises the real compute path tracer, buffers, textures,
bindless resources, acceleration structures/ray tracing, shader compilation,
dispatch, and readback. It does not by itself certify every Metal4 runtime
feature such as raster stages, all coroutine renderers, indirect drawing,
motion acceleration structures, or device-specific feature guards; those
remain covered by their dedicated examples and conformance tests.
