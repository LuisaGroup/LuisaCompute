---
name: test
description: LuisaCompute test guide — Boost.UT framework, test layout, adding tests, device helpers, and running
---

# LuisaCompute Test Guide

Tests are standalone binaries using [Boost.UT](https://github.com/boost-ext/ut) (`src/tests/ut/ut.hpp`). Built when `lc_enable_tests=true` (default).

## Layout

| Directory | Content | Needs Device |
|---|---|---|
| `unit/core/` | basic_types, traits, io, math, logging | No |
| `unit/ast/` | AST construction, manual AST, builtin kernels | Yes |
| `unit/dsl/` | DSL sugar, structs, callables, buffers | Yes |
| `unit/runtime/` | buffer, texture, stream, warp, atomics | Yes |
| `unit/xir/` | XIR builder, translators (gated `lc_enable_xir`) | Yes |
| `integration/runtime/` | multi-stream, swapchain, rtx, raster | Yes |
| `integration/ir/` | autodiff, AST↔IR (gated `lc_enable_ir`) | Yes |

## Adding a Test

In `src/tests/xmake.lua`:
```lua
test_proj("test_my_feature", "unit/runtime/test_my_feature.cpp")
```
Optional: `gui_dep=true` (skips if `lc_enable_gui=false`, defines `LUISA_ENABLE_GUI`), `callable` for extra target config.

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

## Device Helpers (`test_device.h`)

- `luisa::test::create_device(argc, argv)` — from `main()`; exits on missing backend
- `luisa::test::create_device_from_ut()` — from UT registration; returns `std::nullopt`

Backend passed as first positional arg: `cuda`, `dx`, `cpu`, `metal`.

## Assertions

```cpp
expect(condition);
expect(condition) << "message";
expect(bool_expr) << "msg";
```
For floats use `float_eq(a, b, eps)`.

## Running

```bash
xmake -g tests                     # build all
xmake run test_dsl cuda            # run one with backend
./bin/test_dsl dx                  # or direct
./bin/test_basic_types "vector*"   # filter by name (Boost.UT CLI)
```

## Dependencies

Tests link: `lc-runtime`, `lc-dsl`, `lc-vstl`, `stb-image`, optionally `lc-gui`. Dummy backend `lc-backends-dummy` added as non-linking dep.
