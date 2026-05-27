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
| `unit/core/` | basic_types, traits, io, math, logging, containers, hash | No (CTest-registered) |
| `unit/ext/` | external integrations (e.g. glslang/SPIR-V) | No (CTest-registered) |
| `unit/ast/` | AST construction, manual AST, builtin kernels | Yes |
| `unit/dsl/` | DSL sugar, structs, callables, autodiff, polymorphic, SoA, mathematics | Yes |
| `unit/runtime/` | buffer, texture, stream, warp, atomics, FP4/FP8 quantization, gemm | Yes |
| `unit/xir/` | XIR builder, translators, passes | No (CTest-registered) |
| `integration/runtime/` | bindless, curves, rtx, motion blur, swapchain, denoiser, dstorage | Yes |
| `integration/ir/` | autodiff, AST↔IR roundtrip, kernel-IR (gated `LUISA_COMPUTE_ENABLE_RUST`) | Yes |
| `common/` | shared headers: `test_device.h`, `ut/` (Boost.UT), `cornell_box.h`, `tinyexr.h`, `tiny_obj_loader.h`, `projection.hpp`, `spectrum_data.h`, `reference_image.h` | — |
| `python/` | Python frontend tests (run via `pytest` or directly with `python`) | — |
| `cxx_shaders/` | `clangcxx` source shaders consumed by tests | — |

Include path setup (in both CMakeLists.txt and xmake.lua) exposes `src/tests/` and `src/tests/common/`, so test sources just write `#include "test_device.h"`, `#include "ut/ut.hpp"`, `#include "reference_image.h"`, `#include "cornell_box.h"`, etc. Do **not** use `../../` relative paths and do **not** wrap includes in `__has_include` guards — `ut/ut.hpp` and the `common/` headers are vendored and always present.

## Adding a Test

CMake (`src/tests/CMakeLists.txt`) — use the `luisa_compute_add_test` helper:
```cmake
# Standalone GPU-using test, NOT auto-run via CTest:
luisa_compute_add_test(test_my_feature unit/runtime/test_my_feature.cpp)

# CPU-only test, auto-registered with CTest under the given labels:
luisa_compute_add_test(test_my_pure unit/core/test_my_pure.cpp LABELS "unit;unit_core")
```

xmake (`src/tests/xmake.lua`):
```lua
test_proj("test_my_feature", "unit/runtime/test_my_feature.cpp")
-- 3rd arg = gui_dep: if true, only built when lc_enable_gui=true, and defines LUISA_ENABLE_GUI
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

**Mirrored set** (rendering + simulation + headless compute): all `example_path_tracing*`, `example_sdf_renderer[_ir]`, `example_photon_mapping`, `example_blackhole`, `example_voxel_raytracer`, `example_procedural`, `example_shader_toy[_spacex]`, `example_shader_visuals_present`, all simulations (`fire_simulation`, `game_of_life`, `mpm3d`, `mpm88`, `nbody_simulation`, `wave_equation`), `example_image_processing`, `example_helloworld`.

## Test File Template

```cpp
#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/runtime/device.h>
#include <luisa/dsl/sugar.h>

using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_my_feature(Device &device) {
    expect(true);
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

## Device Helpers (`common/test_device.h`)

- `luisa::test::create_device(argc, argv)` — call from `main()`; prints usage and exits on missing backend arg.
- `luisa::test::create_device_from_ut()` — call from a UT registration lambda; returns `std::nullopt` when no backend was passed so the test is silently skipped.

Backend is passed as the first positional arg: `cuda`, `dx`, `cpu`, `metal`, `vulkan`, `hip`, `metal`, `fallback`.

## Assertions

```cpp
expect(condition);
expect(condition) << "message";
expect(a == b) << "values differ";
```
For floats: `expect(std::abs(a - b) < eps)` or use the helpers in `common/test_device.h` / individual tests.

## Running

CMake build:
```bash
cmake --build cmake-build-debug --target test_dsl_mathematic
./cmake-build-debug/bin/test_dsl_mathematic cuda
ctest --test-dir cmake-build-debug -L unit_core    # run CTest-registered unit tests
```

xmake build:
```bash
xmake -g tests                     # build all tests
xmake run test_dsl_mathematic cuda
./bin/test_dsl_mathematic dx
./bin/test_basic_types "vector*"   # filter by name (Boost.UT CLI)
```

## Dependencies

Tests link `lc-runtime`, `lc-dsl`, `lc-vstl`, `stb-image`, and optionally `lc-gui`. The dummy backend `lc-backends-dummy` is added as a non-linking build dependency so all backends get rebuilt before tests run.

## Reference Image Comparison (Opt-In)

Tests and mirrored examples that produce images compare against reference PNGs using PSNR. Comparison is **opt-in via an explicit CLI arg** — there is no auto-discovery of a reference directory and no implicit reference creation. A missing reference file FAILS the comparison; it is never silently created.

CLI: pass the backend first, then offline/comparison flags: `<test_binary> <backend> --offline --compare <path.png>` or `<test_binary> <backend> --offline -c <path.png>`. Without `--compare`/`-c`, the test/example only renders and **does not validate against the reference image**.

For mirrored rendering examples that accept `--spp`, offline reference validation must use at least `--spp 1024` unless a test-specific instruction says otherwise. The expected command shape is:
```bash
cmake-build-release/bin/test_path_tracing vk --offline --spp 1024 --compare docs/gallery/test_path_tracing.png
```
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

Tests-side header: `src/tests/common/reference_image.h` (namespace `luisa::test`) follows the same opt-in contract.

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

**NEVER regenerate or overwrite reference images unless the user explicitly asks you to.** Reference images are ground truth — if a test fails against the reference, the code is wrong, not the reference. When regeneration IS requested, always use the `fallback` (CPU) backend for determinism across GPU vendors. Regenerating from a broken GPU backend will bake bugs into the reference.

## What Not to Do

- Do not put new test sources directly under `src/tests/`. Pick the right subfolder.
- Do not create ad-hoc top-level folders (e.g. `for_agent/`, `next/`, `tmp/`). The layout above is the entire test taxonomy.
- Do not reintroduce doctest. The framework is Boost.UT only.
- Do not delete or `// skip` failing tests to make a build pass — fix the code under test instead.
- Do not duplicate headers between `src/tests/` root and `src/tests/common/`. The canonical copy lives in `common/`.
