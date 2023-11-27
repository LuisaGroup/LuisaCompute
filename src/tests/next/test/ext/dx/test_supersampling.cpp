#ifdef LUISA_TEST_DX_BACKEND
/**
 * @file src/tests/next/tset/ext/dx/test_dml.cpp
 * @author sailing-innocent, on maxwell's previous work
 * @date 2023/08/01
 * @brief the directML test suite
*/

#include "common/config.h"

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/gui/window.h>
#include <luisa/backends/ext/dx_custom_cmd.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {



} // namespace luisa::test

TEST_SUITE("dx_ext") {
    TEST_CASE("supersampling") {
        REQUIRE(1==1); // TODO: no implement yet
    }

} // TEST_SUITE("dx_ext")
#endif