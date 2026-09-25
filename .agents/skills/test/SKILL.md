---
name: test
description: Boost.UT test layout, device helpers, adding tests, and running them.
---

# LuisaCompute Test Guide

Tests are standalone executables using [Boost.UT](https://github.com/boost-ext/ut), vendored at `src/tests/ut/ut.hpp`. Both CMake and xmake are supported (CMakeLists.txt and xmake.lua coexist in `src/tests/`).

## Layout

All test source files live in `src/tests/` under one of the directories below. Nothing else belongs at the root of `src/tests/` (only `CMakeLists.txt`, `xmake.lua`, and the shared data assets). The assets are addressed relative to the repo root or embedded at build time: `examples/gui/win_hdr.cpp` loads `genshin_start.jpg` from CWD, `swapchain_static.cpp`/`swapchain_wx.cpp` load `src/tests/logo.png`, `unit/runtime/test_texture_compress.cpp` resolves `logo.png` through `__FILE__`, and `SRGBToFourierEvenPacked.dat` is compiled into `example_path_tracing_spectrum` / `test_path_tracing_spectrum` (`examples/CMakeLists.txt:36-45`).

| Directory | Content | Needs Device |
|---|---|---|
| `unit/core/` | core library units: types/traits, math, IO, containers, hash, logging, platform utilities, fiber, dynamic module, pool, spin mutex, etc. | No (CTest-registered) |
| `unit/ext/` | external integrations (e.g. glslang/SPIR-V) | No (CTest-registered) |
| `unit/ast/` | AST construction, builtin kernels, manual AST | Mixed (`test_ast`, `test_ast_basic`, `test_builtin_kernel`, `test_manual_ast`, `test_cooperative_vector` need a device; `test_ast_json_serde`, `test_async_copy_ast`, `test_bindless_write_usage`, `test_function_builder_dag` are CTest-registered) |
| `unit/dsl/` | DSL syntax/sugar, structs, callables, SoA, polymorphic, autodiff, device math, variables, matrices, 8-bit/quantization, normal encoding, coroutine front-end tests, etc. | Mostly yes |
| `unit/runtime/` | buffers, textures, streams, copy, atomics, warp operations, printer, sampler, pinned memory, mipmap, bindless, matrix multiply, softmax, buffer/byte IO, external buffers, FP4/FP8 quantization, plus per-backend (`test_hip_*`, `test_vk_*`, `test_remote_*`, `test_metal*`) suites | Mostly yes |
| `unit/coro/` | device coroutine tests via `coro_test_utils.h` (`state_machine`, `wavefront`, `persistent`, pipelines, radix sort) plus 9 host-only CFG/graph tests | Mixed — the host-only ones are CTest-registered under `unit;unit_coro` |
| `unit/xir/` | XIR builder, module, translators, and pass tests (early-cse, licm, simplify-cfg, restructure-cfg, etc.) | No (CTest-registered) |
| `unit/tile/` | Tile IR/DSL/layout/memory/values tests plus `bridge/` (XIR↔Tile, TIRx) | Mostly no (CTest-registered `unit_tile`); the device ones are `test_tile_cuda_ptx` (`ARGS cuda`), `test_tile_native_runtime` (`ARGS metal`), and the `bridge/` GPU tests `test_tile_xir_ranking` / `_llm` / `_metal` / `_runtime` / `_runtime_gpu_dx` / `_runtime_gpu_vk` |
| `unit/fallback/` | Fallback-backend host tests: command queue, coro arena, LLVM ABI/native math | No (CTest-registered; CMake-only — `src/tests/xmake.lua` has no `unit/fallback` target) |
| `unit/simd/` | SIMD CPU backend: Schedule IR / scheduler-model / reference collectives (no device) plus `simd`-backend runtime tests (`test_simd_*`, benchmarks) | Mixed; runtime tests create the `simd` device internally. CMake registers all of them; xmake builds only `benchmark_simd_gemm` |
| `integration/runtime/` | bindless, curves, RTX, motion blur, AOT, indirect, denoiser, dstorage, present/swapchain, select device, runtime, texture3d, native include, procedural callable, device debugger, mesh tests, transient resource, plus backend-specific tests (CUDA graph, DX raster, memory compact, Metal4 AIR / `test_metal_xir_air*`) | Yes |
| `integration/xir/` | XIR↔AST roundtrip integration coverage (`test_xir2ast_roundtrip.cpp`) | No (CTest-registered under `unit;unit_xir`) |
| `benchmark/` | `benchmark_*` executables (command reorder, tile migrated/native/xir/tirx/system/mpp/manual, metal4) — none are CTest-registered | Yes |
| `cuda/` | `test_cuda_tensor_dispatch.cpp` — raw CUDA driver/NVRTC host probe (xmake-only target; not in `src/tests/CMakeLists.txt`) | No (needs the CUDA SDK) |
| `ios/` | Shared Metal4 device conformance, iOS path-tracing kernel, signed test bundle, and host-AOT oracle | Physical iPhone for acceptance |
| `ut/` | vendored Boost.UT single header (`ut/ut.hpp`) — included as `"ut/ut.hpp"` | — |
| `common/` | shared headers: `test_device.h`, `reference_image.h`, `coro_test_utils.h`, `xir_cfg_test_utils.h`, `cornell_box.h`, `tinyexr.h`, `tiny_obj_loader.h`, `projection.hpp`, `spectrum_data.h`, `tile_*_test_utils.h`, plus `metal*_benchmark.h` | — |
| `python/` | Python frontend tests, hyphen-named (`test-helloworld.py`, `test-aot.py`, `test-bindless.py`, …); plain scripts, not registered by either build system — run them directly: `python src/tests/python/test-helloworld.py [backend]` (backend is `sys.argv[1]`) | — |
| `cxx_shaders/` | `luisa::shader` clangcxx sources plus their vendored `luisa/` and `std/` header shims; a parallel copy lives in `examples/extension/cxx_shaders/`. Neither copy is referenced by any CMake/xmake file (they are inputs for the clangcxx tooling, compiled on demand) | — |

Include path setup (in both CMakeLists.txt and xmake.lua) exposes `src/tests/` and `src/tests/common/`, so test sources just write `#include "test_device.h"`, `#include "ut/ut.hpp"`, `#include "reference_image.h"`, `#include "cornell_box.h"`, etc. Do **not** use `../../` relative paths and do **not** guard these vendored includes with `__has_include` — `ut/ut.hpp` and the `common/` headers are always present. (`__has_include` *is* used legitimately for optional system headers, e.g. `#if __has_include(<unistd.h>) && __has_include(<sys/wait.h>)` in `unit/core/test_type.cpp` and `<vulkan/vulkan_core.h>` in `integration/runtime/test_memory_compact.cpp`.)

The whole tree is gated by `LUISA_COMPUTE_BUILD_TESTS` (CMake, `src/CMakeLists.txt`) or `lc_enable_tests` (xmake, `src/xmake.lua`). Inside it, individual tests are additionally gated on `LUISA_COMPUTE_ENABLE_GUI` / `lc_enable_gui`, on `if has_config("lc_enable_xir")` for the XIR/coro-XIR targets (`src/tests/xmake.lua`), and — in CMake — on the backend being built, i.e. `if (TARGET luisa-compute-backend-<name>)` or `if (LUISA_COMPUTE_ENABLE_<BACKEND>)`. There is no `LUISA_COMPUTE_ENABLE_XIR` option.

## Adding a Test

Register a new test in **both** build systems — `src/tests/CMakeLists.txt` and `src/tests/xmake.lua` coexist and are maintained in parallel, and a target added to only one is invisible to the other. (Deliberate exceptions exist: `test_cuda_tensor_dispatch` is xmake-only, while `test_metal4_device_conformance`, the `test_metal_xir_air*` / `test_metal4_air_*` entries, `unit/fallback/`, and most of `unit/simd/` are CMake-only, as are the mirrored `test_<example>` targets.)

CMake (`src/tests/CMakeLists.txt`) — use the `luisa_compute_add_test` helper:
```cmake
# Signature: luisa_compute_add_test(name source [LABELS "label1;label2"] [ARGS arg1 ...])
# Standalone GPU-using test, NOT auto-run via CTest:
luisa_compute_add_test(test_my_feature unit/runtime/test_my_feature.cpp)

# CPU-only test, auto-registered with CTest under the given labels:
luisa_compute_add_test(test_my_pure unit/core/test_my_pure.cpp LABELS "unit;unit_core")

# Real example of passing fixed arguments to CTest (forcing one backend):
luisa_compute_add_test(test_tile_native_runtime
    unit/tile/test_tile_native_runtime.cpp
    LABELS "integration;integration_tile_native" ARGS metal)
```

A device test only runs meaningfully under CTest when its backend argument is supplied through `ARGS` (see the `ARGS metal4` / `ARGS vk` / `ARGS cuda` registrations in `src/tests/CMakeLists.txt`); without `LABELS` it is built but never auto-run.

xmake (`src/tests/xmake.lua`):
```lua
-- Signature: test_proj(name, source, gui_dep, callable, kind, cxx_standard)
-- gui_dep:      if true, built only when lc_enable_gui=true and defines LUISA_ENABLE_GUI
-- callable:     optional config callback for deps/includes/defines
-- kind:         optional target kind (default "binary")
-- cxx_standard: optional per-target standard, e.g. "cxx23"
test_proj("test_my_feature", "unit/runtime/test_my_feature.cpp")

-- With GUI dependency:
test_proj("test_name", "integration/runtime/test_name.cpp", true)

-- With extra config:
test_proj("test_with_dep", "unit/ext/test_with_dep.cpp", false, function()
    add_deps("lc-glslang")
end)

-- Extra config plus a non-default C++ standard (positional args after `callable`):
test_proj("test_tile_values_cpp23", "unit/tile/test_tile_values.cpp", false, function()
    add_deps("lc-tile")
end, nil, "cxx23")
```

## Example ↔ Test Mirror Targets

Auto-checkable examples in `examples/` (rendering w/ reference image, deterministic sims, headless compute) are built as **two executables sharing one source file**: `example_<name>` and `test_<name>`. Mirroring is **CMake-only** (`examples/xmake.lua` defines `example_proj` targets and no `test_` mirrors). Opt in with the `MIRROR_AS_TEST` flag on `luisa_compute_add_example` in `examples/CMakeLists.txt`:

```cmake
luisa_compute_add_example(example_path_tracing
        rendering/path_tracing.cpp
        rendering/path_tracing_test.h
        MIRROR_AS_TEST)
# Produces both bin/example_path_tracing and bin/test_path_tracing (same sources).
# The mirror name is derived by replacing the leading "example_" with "test_".
```

When extra `target_link_libraries` are needed, use `luisa_example_pair_link` so both targets get the libs (it links `name`, and also its `test_` mirror when one exists):
```cmake
luisa_compute_add_example(example_cuda_lcub extension/cuda_lcub.cpp)
luisa_example_pair_link(example_cuda_lcub PRIVATE CUDA::cudart CUDA::cuda_driver)
```

**Do NOT mirror**: GUI toolkit demos (`swapchain*`, `imgui`, `mnist`, Qt, wxWidgets, `win_hdr`) and extension/interop demos. Correctness can't be auto-checked for interactive windows.

**Mirrored set** (rendering + simulation + headless compute; `examples/CMakeLists.txt`): all `example_path_tracing*` (`_camera`, `_cutout`, `_hdr`, `_nested_callable`, `_ray_masks`, `_spectrum`, plus `example_path_tracing_xir2ast` when the XIR target exists — all inside `if (LUISA_COMPUTE_ENABLE_GUI)`), `example_sdf_renderer` and `example_sdf_renderer_xir2ast` (the latter gated by XIR), `example_photon_mapping`, `example_blackhole`, `example_voxel_raytracer`, `example_procedural`, `example_shader_toy[_spacex]`, `example_shader_visuals_present`, simulations (`fire_simulation`, `game_of_life`, `mpm3d`, `mpm88`, `nbody_simulation`, `wave_equation`), `example_image_processing`, `example_helloworld`, `example_gdeflate`, `example_cluster_launch_control`, `example_async_copy_prefetch`, `example_software_lbvh`, `example_software_lbvh_test`, `example_multi_head_attention`.

GUI toolkit demos (`imgui`, `swapchain*`, `win_hdr`, Qt, wxWidgets), extension/interop demos, and `example_bindless_mip` are **not** mirrored because they are interactive or lack deterministic offline validation.

## C++ Test Templates & Style

### Template 1: No-Device Unit Test (CTest-registered)

For tests in `unit/core/`, `unit/ext/`, `unit/xir/`, `unit/fallback/`, the host-side `unit/tile/` + `unit/simd/` tests, `integration/xir/`, and the host-only CFG tests in `unit/coro/` — no GPU backend needed.

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

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
```

The vendored `src/tests/ut/ut.hpp` defines **no** `main()`; every test *executable* needs its own `main` (the two shapes above). A translation unit may omit `main` only when it is an extra source of another test target, e.g. `unit/core/test_hip_late_inline.cpp`, which is compiled into `test_hip_callable_abi` via `target_sources` (`src/tests/CMakeLists.txt:360-365`).

### Template 2: Device-Needed Test (manual backend arg)

For tests in `unit/ast/`, `unit/dsl/`, `unit/runtime/`, `unit/coro/`, and `integration/runtime/` — GPU backend required.

The shape below (static `"name"_test` registration + `create_device_from_ut()` with no arguments + `int main() {}`) is used by exactly one device test, `unit/runtime/test_cluster_launch_control.cpp`; `int main() {}` also appears in the no-device `unit/ext/test_command_reorder_ranges.cpp` (where the static registration runs without any device). Prefer the explicit-`main` shape shown after it.

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

**Device pattern used by most device tests — `main()` creates the device and calls the tests directly** (`test_callable.cpp`, `test_bindless.cpp`, `test_gemm.cpp`; `test_gemm.cpp` omits the `parse_arg_with_fallback` line and just calls `test_gemm(dc->device)`):

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

1. Test framework: `"ut/ut.hpp"` and `"test_device.h"` (when needed) — near-universally first
2. Project core headers: `<luisa/core/...>`
3. Project runtime/DSL headers: `<luisa/runtime/...>`, `<luisa/dsl/...>`
4. Project XIR headers: `<luisa/xir/...>`
5. Standard library: `<cmath>`, `<vector>`, `<numeric>`, etc.

Only rule 1 is consistently followed in-tree; the standard-library block goes before the `<luisa/...>` headers about as often as after (e.g. `unit/core/test_hash.cpp` puts `<cstring>` first, `unit/runtime/test_gemm.cpp` puts `<cmath>` last). Do not churn existing files over it.

Do **not** use `../../` relative paths. Include paths `src/tests/` and `src/tests/common/` are already exposed by the build system. Use `"ut/ut.hpp"`, `"test_device.h"`, `"cornell_box.h"`, `"reference_image.h"` directly.

### Using declarations — always present

```cpp
// No-device tests:
using namespace luisa;           // in most tests (355/432 .cpp files)
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

// Vendored ut.hpp is Boost.UT v2_3_1: `expect`, `that`, `eq`, `throws`,
// `"name"_test`, `test("name")`, `log`, and `skip` exist; `must` does NOT
// (zero occurrences in src/tests/ut/ut.hpp) — do not write `must(...)`, use `expect`.

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

### Fatal checks without exceptions

The [repository-wide C++ rule](../cpp-style/SKILL.md#no-c-exception-raising-in-project-code)
also applies to tests, benchmarks, and shared test helpers: do not use `throw`,
rethrow, or exception-raising assertion helpers. Keep them compatible with builds
that disable C++ exceptions; do not enable exceptions on a test target to work
around a diagnostic.

- Use Boost.UT `expect` for ordinary checks that can safely continue.
- Use `LUISA_ASSERT(condition, "message")` for fatal preconditions and
  `LUISA_ERROR("message")` for unconditional setup or invariant failures.
  Include `<luisa/core/logging.h>` explicitly. For a dynamic message, use a
  literal format string: `LUISA_ERROR("{}", message)`.
- Remove `try/catch` wrappers whose only purpose was to report those fatal
  failures. Preserve required cleanup with RAII and keep all validation checks.
- For expected recoverable failures, use the API's error/status/null result
  and assert its documented outcome. Do not replace an expected failure with
  process termination or remove its coverage. For an expected fatal Luisa check,
  run the invalid operation in a separate process and check both failure and its
  diagnostic. Remove stale `expect(throws(...))` assertions when the API becomes
  fatal. Tests of third-party exception contracts must guard exception syntax
  with `__cpp_exceptions`.
- Check standalone benchmark targets also link `luisa-compute-core` when
  introducing the logging macros. Leave vendored test-framework code intact.

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

**CMake** (`src/tests/CMakeLists.txt`) — real registrations from the file:
```cmake
# No-device, CTest auto-run:
luisa_compute_add_test(test_hash unit/core/test_hash.cpp LABELS "unit;unit_core")

# Device-needed, NOT auto-run:
luisa_compute_add_test(test_buffer unit/runtime/test_buffer.cpp)

# Registered test that needs extra link deps:
luisa_compute_add_test(test_metal4_xir_preflight unit/ext/test_metal4_xir_preflight.cpp
    LABELS "unit;unit_ext;unit_metal4")
target_link_libraries(test_metal4_xir_preflight PRIVATE luisa-compute-metal4-air-codegen)

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
test_proj("test_hash", "unit/core/test_hash.cpp")
test_proj("test_buffer", "unit/runtime/test_buffer.cpp")
-- With GUI dependency:
test_proj("test_aot", "integration/runtime/test_aot.cpp", true)
-- With extra config:
test_proj("test_glslang_spirv", "unit/ext/test_glslang_spirv.cpp", false, function()
    add_deps("lc-glslang")
end)
```

**Label strings actually in use** (`src/tests/CMakeLists.txt`; `ctest -L <label>` matches them): most registrations use the family prefix (`unit` or `integration`) plus scope labels — `unit_core`, `unit_ext`, `unit_ast`, `unit_dsl`, `unit_runtime`, `unit_coro`, `unit_xir`, `unit_tile`, `unit_simd`, `unit_fallback`, `unit_metal`, `unit_metal4`, `unit_cuda`, `unit_gui`, `unit_llvm`, `integration_runtime`, `integration_coro`, `integration_render`, `integration_validation`, `integration_metal`, `integration_metal4`, `integration_codegen`, `integration_tile_cuda` / `_native` / `_xir` / `_tirx` / `_timing`, `integration_simd`, plus backend tags built by loops (`hip`, `fallback`, and the `integration_<backend>` forms for `dx`/`vk`/`metal4`/`simd`) and route tags `hlsl`, `spirv`, `spirv_llvm`, `remote`. A few registrations use their own scheme instead (e.g. `integration_simd;runtime_simd`, `runtime;runtime_<backend>`, `unit_coro_runtime`).

## Device Helpers (`common/test_device.h`)

All helpers live in `namespace luisa::test` and return/take `DeviceContext { compute::Context context; compute::Device device; }`.

- `DeviceContext create_device(int argc, char *argv[])` — call from `main()`; prints usage and `exit(1)`s when no backend arg is given.
- `std::optional<DeviceContext> create_device_from_ut()` — no-arg form for a UT registration lambda; reads the argc/argv Boost.UT stored in `boost::ut::detail::cfg::largc`/`largv` (set by `parse_arg_with_fallback`, directly or through ut.hpp's platform fallback), and returns `std::nullopt` when no backend was passed so the test is silently skipped.
- `std::optional<DeviceContext> create_device_from_ut(int argc, char *argv[], const compute::DeviceConfig *config = nullptr, bool enable_validation = false)` — explicit-args form (the one most device tests call from `main()`); `enable_validation` wraps the device in the validation layer.

All three take the backend from `argv[1]`; there is no backend environment variable. Valid names are the installed backend plugin names — `cuda`, `dx`, `fallback`, `hip`, `metal`, `metal4`, `simd`, `vk` (`src/tests/ut/ut.hpp:862-865`; `print_device_usage` advertises "cuda, dx, fallback, hip, metal, vk"). The exact set available depends on which backends were built (e.g. `LUISA_COMPUTE_ENABLE_CUDA`, `LUISA_COMPUTE_ENABLE_DX`, …). The CPU/software backend is `fallback` (or `simd` for the SIMD backend) — there is no `cpu` backend.

## Coroutine Scheduler Tests

Coroutine unit tests in `src/tests/unit/coro/` use `src/tests/common/coro_test_utils.h` (`luisa::test::coro_test::parse_options`, which errors out when `argv[1]` is missing and forwards the remaining args to Boost.UT). They must require an explicit backend as the first positional argument, e.g. `test_coro_pipeline_1suspend vk`; do not default the backend or hard-code `vk`/`cuda` in the test source. In xmake, the device-side coroutine tests inside the `if has_config("lc_enable_xir")` block are declared through the local `coro_xir_test_proj(name, source, needs_bigobj)` helper (`src/tests/xmake.lua:441`), which forwards to `test_proj` and adds `LUISA_ENABLE_XIR` + `lc-coro` (plus `/bigobj` on MSVC when needed). Four coroutine targets sit outside that helper and use plain `test_proj` (`src/tests/xmake.lua:434-437`: `test_coro_scheduler_base`, `test_coro_multisplit`, `test_coro_compaction`, `test_coro_radix_sort`).

For scheduler-agnostic coroutine behavior, run all schedulers inside the test body (`state_machine`, `wavefront`, and `persistent`) instead of accepting a test-side `--scheduler` option. Keep scheduler-specific option matrices in scheduler-specific tests such as `test_coro_wavefront.cpp` and `test_coro_persistent_opt.cpp`.

Examples may expose scheduler selection, but tests should preserve broad coverage. If a smaller coroutine/MHA repro is useful for debugging, add it as a new focused test and keep the original mirrored example/test target intact.

## Assertions

```cpp
expect(condition);
expect(condition) << "message";
expect(a == b) << "values differ";
```
For floats: `expect(std::abs(a - b) < eps)`, or a local helper defined in the test file itself — e.g. `approx_eq` in `src/tests/unit/dsl/test_dsl_mathematic.cpp:38-46` or `check_floatx_equal` in `src/tests/unit/runtime/test_buffer.cpp`. `common/test_device.h` provides no comparison helpers.

## Running

Before running any test binary or `ctest`, complete a full build of the selected build tree:

```bash
cmake --build <build-dir> --parallel
```

`<build-dir>` (and the `cmake-build-debug` / `cmake-build-release` / `build` names used in examples below) is a local convention — `.gitignore` covers `cmake-build-*/`, `/build`, `/build-*`; CI configures `build` (`.github/workflows/build-cmake.yml:72`), and the iOS scripts use `cmake-build-llvm22-ios` / `cmake-build-ios-metal4-device-air-xcode`.

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

CMake build (the `cmake-build-*` / `build-cmake-ninja-*` names below are local build-dir conventions — `.gitignore` covers `cmake-build-*/`, `/build-*`, `/build`; CI configures a plain `build`):
```bash
cmake --build cmake-build-debug --parallel
./cmake-build-debug/bin/test_dsl_mathematic dx
ctest --test-dir cmake-build-debug -L unit_core # run CTest-registered unit tests
```

The Vulkan native-route guard is device-dependent and therefore runs manually in both Vulkan configurations after their respective full builds (`build-cmake-ninja-xir-llvm` and `build-cmake-ninja-vk-llvm-gfx1201` are example local build-dir names):

```bash
LUISA_VULKAN_VALIDATION=1 build-cmake-ninja-xir-llvm/bin/test_vk_native_route_guard vk
LUISA_VULKAN_VALIDATION=1 build-cmake-ninja-vk-llvm-gfx1201/bin/test_vk_native_route_guard vk
```

xmake build:
```bash
xmake # build all enabled targets (tests included when lc_enable_tests=true)
xmake build test_dsl_mathematic
xmake run test_dsl_mathematic dx
./bin/debug/test_dsl_mathematic.exe dx # lc_bin_dir="bin" + per-mode subdir
./bin/debug/test_basic_types.exe test_vector_construction # run one named test
./bin/debug/test_basic_types.exe --list-test-names-only   # list names
```
The CLI name filter is a plain literal match: `cfg::parse` converts a positional pattern into `query_regex_pattern` (`src/tests/ut/ut.hpp:931-947`) and `utility::regex_match` (`:255-271`) has no `*` or escape handling, so `"vector*"` matches nothing and patterns containing `.`/`*` are unusable — pass the exact `"name"_test` string. (`utility::is_match` at `:174-192` does implement `*`/`?` globs, but it is only used for the tag/suite filter path, not the CLI query.) `--list-test-names-only` / `-l` list the real names.

Python tests:
```bash
python src/tests/python/test-helloworld.py dx
```

## Dependencies

xmake (`test_proj`): tests link `lc-runtime`, `lc-dsl`, `lc-vstl`, `stb-image`, and additionally `lc-gui` when `lc_enable_gui` is on. The dummy backend `lc-backends-dummy` is added as a non-linking build dependency (`add_deps("lc-backends-dummy", {inherit = false, links = false})`) so all backends get rebuilt before tests run.

CMake: `luisa_compute_add_test` → `luisa_compute_add_executable` links the aggregate `luisa::compute` interface target, which already pulls in core/ast/xir/dsl/runtime/gui/backends (`src/CMakeLists.txt:54-70,77-80`).

## Reference Image Comparison (Opt-In)

Tests and mirrored examples that produce images compare against reference PNGs using PSNR. Comparison is **opt-in via an explicit CLI arg** — there is no auto-discovery of a reference directory and no implicit reference creation. A missing reference file FAILS the comparison; it is never silently created.

A pass requires all of: `psnr >= threshold`, luminance `correlation >= 0.5` (`DEFAULT_CORRELATION_THRESHOLD`), and `contrast_ratio` inside `[0.25, 4.0]` (`MIN_CONTRAST_RATIO`/`MAX_CONTRAST_RATIO`), with all metrics finite — see `examples/common/reference_compare.h:484-529` and `src/tests/common/reference_image.h:99-145`. PSNR alone is not the gate.

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
cmake-build-release/bin/test_path_tracing_ray_masks vk --offline --spp 1024 --compare docs/gallery/test_path_tracing_ray_masks.png
```
If a reference PNG is missing, the comparison is a real failure and should be reported as `reference not found`; do not count a render-only run as a validation pass.

Examples-side header: `examples/common/reference_compare.h` (namespace `luisa::ref`).
- `luisa::ref::parse_compare_arg(argc, argv) -> std::optional<std::filesystem::path>`
- `luisa::ref::compare_with_reference_file(pixels, w, h, channels, ref_path, threshold=30.0) -> CompareResult` (`CompareResult{passed, psnr, message, correlation, contrast_ratio}`)
- `luisa::ref::ExampleOptions::parse(argc, argv)` parses `--offline`, `--compare <path.png>` / `-c <path.png>`, `--spp <n>`, `--iterations <n>`, `--max-spp-per-dispatch <n>`, and `--out_ref write <path.png>` / `--out_ref read <path.png>` (unknown flags are intentionally left in `argv` for extension parsers).

Tests-side header: `src/tests/common/reference_image.h` (namespace `luisa::test`) follows the same opt-in contract:
- `luisa::test::parse_compare_arg(argc, argv) -> std::optional<std::filesystem::path>`
- `luisa::test::compare_with_reference_file(..., threshold=30.0) -> ReferenceCompareResult`
- `luisa::test::ImageTestOptions::parse(argc, argv)` parses `--offline`, `--compare <path.png>` / `-c <path.png>`, `--output-dir <dir>`, and `--input <path>`.

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

### `Buffer::copy_from` / `copy_to` raw-pointer calls break in safe-mode builds

The raw-pointer overloads still exist, but they are compiled out when `LUISA_ENABLE_SAFE_MODE` is defined (`include/luisa/runtime/buffer.h:164-193` for `Buffer<T>`, `:300-307` for `BufferView<T>`); only the `luisa::span<U, Extent>` overloads at `:153-163` remain. The macro is set from CMake `LUISA_COMPUTE_ENABLE_SAFE_MODE` (`src/runtime/CMakeLists.txt:53-55`) or xmake `lc_safe_mode` (`src/runtime/xmake.lua:12-13`). Call sites that pass `.data()` therefore fail to compile in safe-mode builds:

```
error: no matching member function for call to 'copy_from'
note: candidate template ignored: could not match 'luisa::span<U>' against 'pointer'
```

**Fix:** wrap the container (or pointer + size) in `luisa::span`; that form compiles in both modes and is the dominant convention in the tree:

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

Note `copy_to` requires a non-const element type (`requires(!std::is_const_v<U>)`), and the command only asserts the byte-size match under `#ifndef NDEBUG`.

This affects both **tests** (e.g. `src/tests/unit/coro/test_coro_persistent_opt.cpp`) and **examples** (e.g. `examples/rendering/coro_path_tracing.cpp`). When patching, search the repo for existing `copy_from(luisa::span{...})` / `copy_to(luisa::span{...})` usage to match the local convention, then apply a bulk `replace_all` across the affected file(s).

## What Not to Do

- Do not put new test sources directly under `src/tests/`. Pick the right subfolder.
- Do not create ad-hoc top-level folders (e.g. `for_agent/`, `next/`, `tmp/`). The layout above is the entire test taxonomy.
- Do not reintroduce doctest. The framework is Boost.UT only.
- Do not delete or `// skip` failing tests to make a build pass — fix the code under test instead.
- Do not pass `vulkan` as a backend name; the CLI name is `vk`.
- Do not duplicate headers between `src/tests/` root and `src/tests/common/`. The canonical copy lives in `common/`.
