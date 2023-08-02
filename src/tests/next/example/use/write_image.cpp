//
// Created by Mike Smith on 2022/11/26.
// Moved by Saili from test_helloworld case on 2023/07/26
//

#include "common/config.h"

#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/syntax.h>
#include <stb/stb_image_write.h>

#include <iostream>

using namespace luisa;
using namespace luisa::compute;

namespace luisa::test {
int write_image(Device &device, luisa::string filename = "write_image.png") {
    Stream stream = device.create_stream();
    constexpr uint2 resolution = make_uint2(1024, 1024);
    Image<float> image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    luisa::vector<std::byte> host_image(image.view().size_bytes());
    Kernel2D kernel = [&]() {
        Var coord = dispatch_id().xy();
        Var size = dispatch_size().xy();
        Var uv = (make_float2(coord) + 0.5f) / make_float2(size);
        image->write(coord, make_float4(uv, 0.5f, 1.0f));
    };
    Shader2D<> shader = device.compile(kernel);
    stream << shader().dispatch(resolution)
           << image.copy_to(host_image.data())
           << synchronize();
    stbi_write_png(filename.c_str(), resolution.x, resolution.y, 4, host_image.data(), 0);
    return 0;
}
}// namespace luisa::test


TEST_SUITE("example") {
    TEST_CASE("write_image") {
        Context context{luisa::test::argv()[0]};
       
        for (auto i = 0; i < luisa::test::backends_to_test_count(); i++) {
            luisa::string device_name = luisa::test::backends_to_test()[i];
            SUBCASE(device_name.c_str()) {
                Device device = context.create_device(device_name.c_str());
                REQUIRE(luisa::test::write_image(device, "write_image_"+device_name+".png") == 0);
            }
        }
    }
}
