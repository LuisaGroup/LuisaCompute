---
name: test
description: Boost.UT test layout, device helpers, adding tests, and running them.
---

# LuisaCompute Test Guide

Tests are standalone executables using [Boost.UT](https://github.com/boost-ext/ut), vendored at `src/tests/ut/ut.hpp`. Both CMake and xmake are supported (CMakeLists.txt and xmake.lua coexist in `src/tests/`).

## Layout

All test source files live in `src/tests/` under one of the directories below. Nothing else belongs at the root of `src/tests/`. Shared assets stay at the root only when they are loaded by binaries via CWD-relative paths (e.g. `SRGBToFourierEvenPacked.dat`, `genshin_start.jpg`, `logo.png`).

| Directory | Content | Needs Device |
|---|---|---|
| `unit/core/` | core library units: types/traits, math, IO, containers, hash, logging, platform utilities, fiber, dynamic module, pool, spin mutex, etc. | No (CTest-registered) |
| `unit/ext/` | external integrations (e.g. glslang/SPIR-V) | No (CTest-registered) |
| `unit/ast/` | AST construction, builtin kernels, manual AST | Yes |
| `unit/dsl/` | DSL syntax/sugar, structs, callables, SoA, polymorphic, autodiff, device math, variables, matrices, 8-bit/quantization, normal encoding, etc. | Yes |
| `unit/runtime/` | buffers, textures, streams, copy, atomics, warp operations, printer, sampler, pinned memory, mipmap, bindless, matrix multiply, softmax, buffer/byte IO, external buffers, FP4/FP8 quantization, etc. | Yes |
| `unit/xir/` | XIR builder, module, translators, and pass tests (early-cse, licm, simplify-cfg, restructure-cfg, etc.) | No (CTest-registered) |
| `integration/runtime/` | bindless, curves, RTX, motion blur, AOT, indirect, denoiser, dstorage, present/swapchain, select device, runtime, texture3d, native include, procedural callable, device debugger, mesh tests, transient resource, plus backend-specific tests (CUDA graph, raster, memory compact, HIPRT) | Yes |
| `integration/xir/` | XIR↔AST roundtrip integration coverage | Yes |
| `ios/` | Shared Metal4 device conformance, iOS path-tracing kernel, signed test bundle, and host-AOT oracle | Physical iPhone for acceptance |
| `common/` | shared headers: `test_device.h`, `ut/` (Boost.UT), `cornell_box.h`, `tinyexr.h`, `tiny_obj_loader.h`, `projection.hpp`, `spectrum_data.h`, `reference_image.h` | — |
| `python/` | Python frontend tests (run directly with `python src/tests/python/test_xxx.py [backend]`) | — |
| `cxx_shaders/` | `clangcxx` source shaders consumed by tests/extension examples | — |

Include path setup (in both CMakeLists.txt and xmake.lua) exposes `src/tests/` and `src/tests/common/`, so test sources just write `#include "test_device.h"`, `#include "ut/ut.hpp"`, `#include "reference_image.h"`, `#include "cornell_box.h"`, etc. Do **not** use `../../` relative paths and do **not** wrap includes in `__has_include` guards — `ut/ut.hpp` and the `common/` headers are vendored and always present.

Some integration and XIR tests are only built when the corresponding option is enabled (e.g. `LUISA_COMPUTE_ENABLE_GUI`, `LUISA_COMPUTE_ENABLE_XIR`, and `lc_enable_xir`).

## Adding a Test

CMake (`src/tests/CMakeLists.txt`) — use the `luisa_compute_add_test` helper:
```cmake
# Signature: luisa_compute_add_test(name source [LABELS "label1;label2"] [ARGS arg1 ...])
# Standalone GPU-using test, NOT auto-run via CTest:
luisa_compute_add_test(test_my_feature unit/runtime/test_my_feature.cpp)

# CPU-only test, auto-registered with CTest under the given labels:
luisa_compute_add_test(test_my_pure unit/core/test_my_pure.cpp LABELS "unit;unit_core")

# Passing fixed arguments to CTest (e.g. forcing the DX backend):
luisa_compute_add_test(test_raster integration/runtime/test_raster.cpp
    LABELS "integration" ARGS dx)
```

xmake (`src/tests/xmake.lua`):
```lua
-- Signature: test_proj(name, source, gui_dep, callable, kind)
--   gui_dep:  if true, built only when lc_enable_gui=true and defines LUISA_ENABLE_GUI
--   callable: optional config callback for deps/includes/defines
--   kind:     optional target kind (default "binary")
test_proj("test_my_feature", "unit/runtime/test_my_feature.cpp")

-- With GUI dependency:
test_proj("test_name", "integration/runtime/test_name.cpp", true)

-- With extra config:
test_proj("test_with_dep", "unit/ext/test_with_dep.cpp", false, function()
    add_deps("lc-glslang")
end)
```

## Example ↔ Test Mirror Targets

Auto-checkable examples in `examples/` (rendering w/ reference image, deterministic sims, headless compute) are built as **two executables sharing one source file**: `example_<name>` and `test_<name>`. Opt in with the `MIRROR_AS_TEST` flag on `luisa_compute_add_example` in `examples/CMakeLists.txt`:

```cmake
luisa_compute_add_example(example_path_tracing rendering/path_tracing.cpp MIRROR_AS_TEST)
# Produces both bin/example_path_tracing and bin/test_path_tracing.
```

When extra `target_link_libraries` are needed, use `luisa_example_pair_link` so both targets get the libs:
```cmake
luisa_compute_add_example(example_cuda_lcub extension/cuda_lcub.cpp MIRROR_AS_TEST)
luisa_example_pair_link(example_cuda_lcub PRIVATE CUDA::cudart CUDA::cuda_driver)
```

**Do NOT mirror**: GUI toolkit demos (`swapchain*`, `imgui`, `mnist`, Qt, wxWidgets, `win_hdr`) and extension/interop demos. Correctness can't be auto-checked for interactive windows.

**Mirrored set** (rendering + simulation + headless compute): all `example_path_tracing*` (including `example_path_tracing_xir2ast` when XIR is enabled), `example_sdf_renderer` and `example_sdf_renderer_xir2ast` (the latter gated by XIR), `example_photon_mapping`, `example_blackhole`, `example_voxel_raytracer`, `example_procedural`, `example_shader_toy[_spacex]`, `example_shader_visuals_present`, all simulations (`fire_simulation`, `game_of_life`, `mpm3d`, `mpm88`, `nbody_simulation`, `wave_equation`), `example_image_processing`, `example_helloworld`, `example_multi_head_attention`.

GUI toolkit demos (`imgui`, `swapchain*`, `win_hdr`, Qt, wxWidgets), extension/interop demos, and `example_bindless_mip` are **not** mirrored because they are interactive or lack deterministic offline validation.

## C++ Test Templates & Style

### Template 1: No-Device Unit Test (CTest-registered)

For tests in `unit/core/`, `unit/ext/`, and `unit/xir/` — no GPU backend needed.

**Option A — static registration with standalone test functions** (preferred for many small tests):

```cpp
// Test for <header>.h
// This test covers: <list of features>

#include "ut/ut.hpp"
#include <luisa/core/<header>.h>

using namespace boost::ut;
using namespace boost::ut::literals;

// Test functions: void test_<scenario>()
void test_basic_construction() {
    expect(true) << "description";
}

void test_edge_case() {
    expect(condition) << "message on failure";
}

// Static registration + main for CLI filtering
static auto test_<name>_registration = [] {
    "<scenario_name>"_test = [] { test_basic_construction(); };
    "<scenario_name2>"_test = [] { test_edge_case(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
```

**Option B — `reg_` functions with explicit main** (preferred for XIR/pass tests):

```cpp
#include "ut/ut.hpp"
#include <luisa/xir/module.h>

using namespace luisa;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

void reg_feature_scenario() {
    "feature_scenario"_test = [] {
        Module m;
        // ... build IR, run pass, verify
        expect(condition);
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_feature_scenario();
    // ... more reg_xxx() calls
    return 0;
}
```

**Option C — single static registration** (for single-cohesive-group tests like `test_clock.cpp`):

```cpp
#include "ut/ut.hpp"
#include <luisa/core/clock.h>

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

static auto test_clock_registration = [] {
    "test_clock"_test = [] {
        Clock clock;
        // ... multiple sub-tests in one lambda
        expect(condition);
    };
    return 0;
}();
// No main() — Boost.UT auto-generates one
```

### Template 2: Device-Needed Test (manual backend arg)

For tests in `unit/ast/`, `unit/dsl/`, `unit/runtime/`, `integration/` — GPU backend required.

```cpp
// Test for <feature>.
// Features tested:
// - <feature 1>
// - <feature 2>

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// Structs used in DSL kernels must be registered with LUISA_STRUCT
struct MyData {
    int a;
    float b;
};
LUISA_STRUCT(MyData, a, b) {};

void test_my_feature(Device &device) {
    // Create buffers, streams, compile kernels, dispatch, validate
    Buffer<float> buf = device.create_buffer<float>(1024u);
    Stream stream = device.create_stream();

    Kernel1D kernel = [&](BufferFloat b) noexcept {
        b.write(dispatch_id().x, 1.0f);
    };
    auto shader = device.compile(kernel);

    luisa::vector<float> host(1024u);
    stream << shader(buf).dispatch(1024u)
           << buf.copy_to(luisa::span{host})
           << synchronize();

    bool ok = true;
    for (auto v : host) {
        if (std::abs(v - 1.0f) > 1e-4f) { ok = false; break; }
    }
    expect(ok) << "kernel should fill buffer with 1.0f";
}

static inline const auto reg = [] {
    "my_feature"_test = [] {
        auto dc = luisa::test::create_device_from_ut();
        if (!dc) return;
        test_my_feature(dc->device);
    };
    return 0;
}();

int main() {}
```

**Alternate device pattern — `main()` calls test directly** (used by `test_callable.cpp`, `test_bindless.cpp`, `test_gemm.cpp`):

```cpp
void test_my_feature(Device &device) { /* ... */ }

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_my_feature(device);
}
```

---

## C++ Style Conventions

### Includes — canonical order

1. Test framework: `"ut/ut.hpp"` and `"test_device.h"` (when needed)
2. Project core headers: `<luisa/core/...>`
3. Project runtime/DSL headers: `<luisa/runtime/...>`, `<luisa/dsl/...>`
4. Project XIR headers: `<luisa/xir/...>`
5. Standard library: `<cmath>`, `<vector>`, `<numeric>`, etc.

Do **not** use `../../` relative paths. Include paths `src/tests/` and `src/tests/common/` are already exposed by the build system. Use `"ut/ut.hpp"`, `"test_device.h"`, `"cornell_box.h"`, `"reference_image.h"` directly.

### Using declarations — always present

```cpp
// No-device tests:
using namespace luisa;           // always
using namespace boost::ut;
using namespace boost::ut::literals;

// Device tests — add:
using namespace luisa::compute;

// XIR tests — add:
using namespace luisa::compute::xir;
```

### Naming conventions

| Element | Convention | Example |
|---|---|---|
| Test source file | `test_<feature>.cpp` | `test_buffer.cpp` |
| Test function | `test_<feature>()` or `test_<feature>(Device &)` | `test_basic_construction()` |
| Registration function | `reg_<feature>()` | `reg_alloca()` |
| Static registration lambda | `test_<feature>_registration` | `test_basic_types_registration` |
| Test name string | `"<snake_case_description>"` | `"hash64_basic"`, `"xir_builder_alloca_local"` |
| Test executable | `test_<feature>` | `test_dsl_mathematic` |

### Assertions

```cpp
// Basic
expect(condition);
expect(condition) << "descriptive message on failure";
expect(ptr != nullptr);
expect(eq(a, b)) << "values should be equal";          // Boost.UT eq()

// Complex expressions — wrap in static_cast<bool>
expect(static_cast<bool>(a == 1 && b == 2));

// Float comparison — always use epsilon, never direct ==
expect(std::abs(result - expected) < 1e-4f);

// For complex DSL validation — accumulate errors, expect once
bool all_correct = true;
for (size_t i = 0; i < n; i++) {
    if (std::abs(results[i] - expected) > 1e-4f) {
        LUISA_WARNING("Mismatch at [{}]: got {} expected {}", i, results[i], expected);
        all_correct = false;
    }
}
expect(all_correct) << "all elements must match expected values";
```

### LUISA_STRUCT registration

Any struct used in `BufferVar<T>`, `Var<T>`, or kernel/callable signatures must be registered:

```cpp
struct MyType {
    int x;
    float3 v;
};
LUISA_STRUCT(MyType, x, v) {};

// Template structs need LUISA_TEMPLATE_STRUCT:
#define MY_PAIR_TEMPLATE() template<typename K, typename V>
#define MY_PAIR() MyPair<K, V>
LUISA_TEMPLATE_STRUCT(MY_PAIR_TEMPLATE, MY_PAIR, key, value) {};
```

### File header comment

Every test file starts with a descriptive comment block:

```cpp
// Test for <module/feature>.
// This test covers:
// - <feature 1>
// - <feature 2>
```

### Test organization within a file

- Each test function covers one logical area
- Test function bodies are self-contained: create their own objects, run, validate
- Use scoped blocks `{ ... }` within a test lambda to isolate sub-tests
- Prefer many small `"name"_test` lambdas over one giant test
- Put `log_level_verbose()` at the top of device tests for debug output
- Use `LUISA_INFO("...")` for progress messages; `LUISA_WARNING("...")` for non-fatal issues

### `main()` function shape

```cpp
// For CTest-registered tests (Pattern 1):
int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}

// For device tests (Pattern 2, static reg style):
int main() {}

// For device tests (Pattern 2, explicit main style):
int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) return 0;
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    test_xxx(device);
}
```

### Build registration

**CMake** (`src/tests/CMakeLists.txt`):
```cmake
# No-device, CTest auto-run:
luisa_compute_add_test(test_name unit/core/test_name.cpp LABELS "unit;unit_core")

# Device-needed, NOT auto-run:
luisa_compute_add_test(test_name unit/runtime/test_name.cpp)

# With extra link deps:
luisa_compute_add_test(test_name unit/ext/test_name.cpp LABELS "unit;unit_ext")
target_link_libraries(test_name PRIVATE some-lib)

# Multi-source test: use luisa_compute_add_executable directly and add includes:
luisa_compute_add_executable(test_transient_resource
    integration/runtime/test_transient_resource.cpp
    integration/runtime/transient_resource_device/managed_first_fit.cpp
    integration/runtime/transient_resource_device/managed_first_fit.h
    integration/runtime/transient_resource_device/transient_resource_device.cpp
    integration/runtime/transient_resource_device/transient_resource_device.h)
target_include_directories(test_transient_resource PRIVATE ./ ./common)
```

**xmake** (`src/tests/xmake.lua`):
```lua
test_proj("test_name", "unit/core/test_name.cpp")
-- With GUI dependency:
test_proj("test_name", "integration/runtime/test_name.cpp", true)
-- With extra config:
test_proj("test_name", "unit/ext/test_name.cpp", false, function()
    add_deps("lc-glslang")
end)
```

## Device Helpers (`common/test_device.h`)

- `luisa::test::create_device(argc, argv)` — call from `main()`; prints usage and exits on missing backend arg.
- `luisa::test::create_device_from_ut()` — call from a UT registration lambda; returns `std::nullopt` when no backend was passed so the test is silently skipped.

Backend is passed as the first positional arg: `cuda`, `dx`, `cpu`, `metal`, `vk`. The exact set available depends on which backends were built (e.g. `LUISA_COMPUTE_ENABLE_CUDA`, `LUISA_COMPUTE_ENABLE_DX`, etc.).

## Coroutine Scheduler Tests

Coroutine unit tests in `src/tests/unit/coro/` use `src/tests/common/coro_test_utils.h`. They must require an explicit backend as the first positional argument, e.g. `test_coro_pipeline_1suspend vk`; do not default the backend or hard-code `vk`/`cuda` in the test source.

For scheduler-agnostic coroutine behavior, run all schedulers inside the test body (`state_machine`, `wavefront`, and `persistent`) instead of accepting a test-side `--scheduler` option. Keep scheduler-specific option matrices in scheduler-specific tests such as `test_coro_wavefront.cpp` and `test_coro_persistent_opt.cpp`.

Examples may expose scheduler selection, but tests should preserve broad coverage. If a smaller coroutine/MHA repro is useful for debugging, add it as a new focused test and keep the original mirrored example/test target intact.

## Assertions

```cpp
expect(condition);
expect(condition) << "message";
expect(a == b) << "values differ";
```
For floats: `expect(std::abs(a - b) < eps)` or use the helpers in `common/test_device.h` / individual tests.

## Running

Before running any test binary or `ctest`, complete a full build of the selected build tree:

```bash
cmake --build <build-dir> --parallel
```

A target-only build is useful for compilation diagnostics but does not satisfy this gate. If source changes after the full build starts, repeat the full build before resuming tests.

### iOS Metal4 device tests

iOS tests are opt-in application bundles, not ordinary CTest registrations.
Use the repository scripts rather than copying a desktop toolchain:

```bash
scripts/build_ios_llvm.sh \
  --host-llvm-prefix "$(brew --prefix llvm@22)"
scripts/build_ios_metal4.sh \
  --llvm-dir cmake-build-llvm22-ios/lib/cmake/llvm \
  --team <team-id> --mode tests
```

`src/tests/ios/metal4_device_conformance.cpp` is also compiled into the macOS
`test_metal4_device_conformance` preflight and the interactive
`examples/ios` path tracer. Keep that body portable and preserve exact numeric
checks for ABI, logging, native include, unsigned timelines, bindless,
indirect dispatch, raster/base-instance/stencil, motion, AS build, and RTX.

An unsigned `luisa-ios-device-tests` build plus
`scripts/audit_ios_bundles.sh` proves cross-compilation/link/package closure
only. A pass requires a signed physical-device launch, visible progressive
Window/Swapchain output, `success: true`, every supported feature marked
`passed`, and retrieved nondegenerate JSON/PNG evidence. Do not infer device
execution from installation, bundle names, GPU-family queries, or CI success.

CMake build:
```bash
cmake --build cmake-build-debug --parallel
./cmake-build-debug/bin/test_dsl_mathematic dx
ctest --test-dir cmake-build-debug -L unit_core    # run CTest-registered unit tests
```

The Vulkan native-route guard is device-dependent and therefore runs manually in both Vulkan configurations after their respective full builds:

```bash
LUISA_VULKAN_VALIDATION=1 build-cmake-ninja-xir-llvm/bin/test_vk_native_route_guard vk
LUISA_VULKAN_VALIDATION=1 build-cmake-ninja-vk-llvm-gfx1201/bin/test_vk_native_route_guard vk
```

xmake build:
```bash
xmake                              # build all enabled targets (tests included when lc_enable_tests=true)
xmake build test_dsl_mathematic
xmake run test_dsl_mathematic dx
./build/bin/test_dsl_mathematic dx
./build/bin/test_basic_types "vector*"   # filter by name (Boost.UT CLI)
```

Python tests:
```bash
python src/tests/python/test-helloworld.py dx
```

## Dependencies

Tests link `lc-runtime`, `lc-dsl`, `lc-vstl`, `stb-image`, and optionally `lc-gui`. The dummy backend `lc-backends-dummy` is added as a non-linking build dependency so all backends get rebuilt before tests run.

## Reference Image Comparison (Opt-In)

Tests and mirrored examples that produce images compare against reference PNGs using PSNR. Comparison is **opt-in via an explicit CLI arg** — there is no auto-discovery of a reference directory and no implicit reference creation. A missing reference file FAILS the comparison; it is never silently created.

CLI: pass the backend first, then offline/comparison flags: `<test_binary> <backend> --offline --compare <path.png>` or `<test_binary> <backend> --offline -c <path.png>`. Without `--compare`/`-c`, the test/example only renders and **does not validate against the reference image**.

For mirrored rendering examples that accept `--spp`, offline reference validation must use at least `--spp 1024` unless a test-specific instruction says otherwise. The expected command shape is:
```bash
LUISA_VULKAN_VALIDATION=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_DUMP_SOURCE=1 \
cmake-build-release/bin/test_path_tracing vk --offline --spp 1024 --compare docs/gallery/test_path_tracing.png
```
When the result is intended to validate Vulkan's native XIR -> SPIR-V path,
always enable this guard. It rejects a user shader that would otherwise route
through the compatibility HLSL compiler, rejects non-native Vulkan builds, and
constrains strict AOT loads to XIR-produced SPIR-V while still allowing
Vulkan's internal HLSL-generated builtins. `LUISA_DUMP_SOURCE=1` additionally
forces fresh JIT codegen; pair both with Vulkan validation for runtime coverage.
Do not apply the strict guard blindly to a mixed-route executable. In
`test_vk_spirv_codegen_path`, the typed `BUFFER_ONLY` case and two native-HLSL
interoperability cases deliberately use the compatibility route. Cover the
whole suite once under validation, and cover all remaining native cases
separately under the strict guard; never hide a fallback by locally clearing
the environment inside a nominally strict test. Keep explicit exclusions in
the strict runner synchronized instead of documenting a brittle case count.
Lower default offline sample counts may produce PSNR failures from sampling noise rather than code regressions. Do not report an offline rendering test as passing image validation unless the log contains `Reference comparison: PASSED` and exit code `0`.

For path-tracing gallery validation, run the mirrored executable with its matching reference, for example:
```bash
cmake-build-release/bin/test_path_tracing vk --offline --spp 1024 --compare docs/gallery/test_path_tracing.png
cmake-build-release/bin/test_path_tracing_cutout vk --offline --spp 1024 --compare docs/gallery/test_path_tracing_cutout.png
cmake-build-release/bin/test_path_tracing_nested_callable vk --offline --spp 1024 --compare docs/gallery/test_path_tracing_nested_callable.png
cmake-build-release/bin/test_path_tracing_hdr vk --offline --spp 1024 --compare docs/gallery/test_path_tracing_hdr.png
cmake-build-release/bin/test_path_tracing_camera vk --offline --spp 1024 --compare docs/gallery/test_path_tracing_camera.png
cmake-build-release/bin/test_path_tracing_spectrum vk --offline --spp 1024 --compare docs/gallery/test_path_tracing_spectrum.png
cmake-build-release/bin/test_path_tracing_ir vk --offline --spp 1024 --compare docs/gallery/test_path_tracing.png
```
If a reference PNG is missing, the comparison is a real failure and should be reported as `reference not found`; do not count a render-only run as a validation pass.

Examples-side header: `examples/common/reference_compare.h` (namespace `luisa::ref`).
- `luisa::ref::parse_compare_arg(argc, argv) -> std::optional<std::filesystem::path>`
- `luisa::ref::compare_with_reference_file(pixels, w, h, channels, ref_path, threshold=30.0) -> CompareResult`
- `luisa::ref::ExampleOptions::parse(argc, argv)` parses `--offline`, `--compare <path.png>` / `-c <path.png>`, `--spp <n>`, and `--out_ref write <path.png>` / `--out_ref read <path.png>`.

Tests-side header: `src/tests/common/reference_image.h` (namespace `luisa::test`) follows the same opt-in contract:
- `luisa::test::parse_compare_arg(argc, argv) -> std::optional<std::filesystem::path>`
- `luisa::test::compare_with_reference_file(..., threshold=30.0) -> ReferenceCompareResult`
- `luisa::test::ImageTestOptions::parse(argc, argv)` parses `--offline`, `--compare <path.png>`, and `--output-dir <dir>`.

Typical usage:
```cpp
if (auto ref = luisa::ref::parse_compare_arg(argc, argv)) {
    auto r = luisa::ref::compare_with_reference_file(
        host_image.data(), w, h, 4, *ref);
    LUISA_INFO("Reference: {} ({})", r.passed ? "PASSED" : "FAILED", r.message);
    if (!r.passed) return 1;
}
```

Reference PNGs live under `docs/gallery/<test_name>.png` in the repo. Always pass the absolute or repo-relative path explicitly — never rely on cwd or executable-relative walking.

**NEVER regenerate or overwrite reference images unless the user explicitly asks you to.** Reference images are ground truth — if a test fails against the reference, the code is wrong, not the reference. When regeneration IS requested, always use the `fallback` (CPU) backend for determinism across GPU vendors. Regenerating from a broken GPU backend will bake bugs into the reference. Examples support `--out_ref write <path>` for explicit regeneration; tests-side code should follow the same explicit opt-in model.

## Common Build Breaks & Fixes

### `Buffer::copy_from` / `copy_to` no longer accept raw pointers

When the runtime `Buffer` API is tightened to take `luisa::span<U>` instead of a raw pointer, existing call sites that pass `.data()` will fail to compile:

```
error: no matching member function for call to 'copy_from'
note: candidate template ignored: could not match 'luisa::span<U>' against 'pointer'
```

**Fix:** wrap the container (or pointer + size) in `luisa::span`:

```cpp
// Before — breaks after the API change
stream << buf.copy_from(host.data()) << synchronize();
stream << buf.copy_to(host.data()) << synchronize();

// After
stream << buf.copy_from(luisa::span{host}) << synchronize();
stream << buf.copy_to(luisa::span{host}) << synchronize();

// For C arrays or pre-sized pointers
stream << buf.copy_from(luisa::span{arr, std::size(arr)}) << synchronize();
stream << buf.copy_from(luisa::span{ptr, n}) << synchronize();
```

This affects both **tests** (e.g. `src/tests/unit/coro/test_coro_persistent_opt.cpp`) and **examples** (e.g. `examples/rendering/coro_path_tracing.cpp`). When patching, search the repo for existing `copy_from(luisa::span{...})` / `copy_to(luisa::span{...})` usage to match the local convention, then apply a bulk `replace_all` across the affected file(s).

## What Not to Do

- Do not put new test sources directly under `src/tests/`. Pick the right subfolder.
- Do not create ad-hoc top-level folders (e.g. `for_agent/`, `next/`, `tmp/`). The layout above is the entire test taxonomy.
- Do not reintroduce doctest. The framework is Boost.UT only.
- Do not delete or `// skip` failing tests to make a build pass — fix the code under test instead.
- Do not pass `vulkan` as a backend name; the CLI name is `vk`.
- Do not duplicate headers between `src/tests/` root and `src/tests/common/`. The canonical copy lives in `common/`.
