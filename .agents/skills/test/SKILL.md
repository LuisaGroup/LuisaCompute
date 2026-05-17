---
name: test
description: LuisaCompute test guide — Boost.UT framework, test layout, adding tests, device helpers, and running
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
| `integration/gui/` | window-system swapchain interop (Qt, wxWidgets), upscaling SDK demos (XeSS, FSR3) | Yes (GUI) |
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

## What Not to Do

- Do not put new test sources directly under `src/tests/`. Pick the right subfolder.
- Do not create ad-hoc top-level folders (e.g. `for_agent/`, `next/`, `tmp/`). The layout above is the entire test taxonomy.
- Do not reintroduce doctest. The framework is Boost.UT only.
- Do not delete or `// skip` failing tests to make a build pass — fix the code under test instead.
- Do not duplicate headers between `src/tests/` root and `src/tests/common/`. The canonical copy lives in `common/`.
