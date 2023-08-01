/**
 * @file: tests/next/gui/test_window.cpp
 * @author: sailing-innocent
 * @date: 2023-07-26
 * @brief: the plain window runtime
*/

#include "common/config.h"

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/swapchain.h>

#include <luisa/gui/window.h>

#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {

int test_window(Device &device, bool enable_clear = false) {
    static constexpr uint resolution = 1024u;
    Window window{"MPM3D", resolution, resolution};
    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Swapchain swap_chain{device.create_swapchain(
        window.native_handle(),
        stream,
        make_uint2(resolution),
        false, false, 3)};

    Image<float> display = device.create_image<float>(swap_chain.backend_storage(), make_uint2(resolution));

    Shader2D<> clear_display = device.compile<2>([&] {
        display->write(dispatch_id().xy(), make_float4(.1f, .2f, .3f, 1.f));
    });

    while (!window.should_close()) {
        if (enable_clear) {
            // stream << clear_display().dispatch(resolution, resolution);
            CommandList cmd_list;
            cmd_list << clear_display().dispatch(resolution, resolution);
            // stream << cmd_list.commit();
            // stream << swap_chain.present(display);
            stream << cmd_list.commit() << swap_chain.present(display);
        }

        window.poll_events();
    }
    stream << synchronize();
    return 0;
}

} // namespace luisa::test

TEST_SUITE("test_gui") {
    TEST_CASE("plain_window") {
        Context context{luisa::test::argv()[0]};
       
        for (auto i = 0; i < luisa::test::backends_to_test_count(); i++) {
            luisa::string device_name = luisa::test::backends_to_test()[i];
            SUBCASE(device_name.c_str()) {
                Device device = context.create_device(device_name.c_str());
                REQUIRE(luisa::test::test_window(device) == 0);
            }
        }
    }
    TEST_CASE("cleared_window") {
        Context context{luisa::test::argv()[0]};
       
        for (auto i = 0; i < luisa::test::backends_to_test_count(); i++) {
            luisa::string device_name = luisa::test::backends_to_test()[i];
            SUBCASE(device_name.c_str()) {
                Device device = context.create_device(device_name.c_str());
                REQUIRE(luisa::test::test_window(device, true) == 0);
            }
        }
    }
}