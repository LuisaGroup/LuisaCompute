//
// Created by ChenXin on 2022/3/22.
//

#include <iostream>

#include <core/clock.h>
#include <core/logging.h>
#include <runtime/context.h>
#include <runtime/device.h>
#include <runtime/stream.h>
#include <runtime/event.h>
#include <dsl/sugar.h>
#include <runtime/rtx/accel.h>
#include <tests/cornell_box.h>
#include <stb/stb_image_write.h>
#include <stb/stb_image.h>

#include <dsl/printer.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tests/tiny_obj_loader.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    log_level_info();

    Context context{argv[0]};
    if(argc <= 1){
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, ispc, metal", argv[0]);
        exit(1);
    }
    auto device = context.create_device(argv[1]);
    Printer printer{device};

    static constexpr auto resolution = make_uint2(1024u);
    auto image_byte4 = device.create_image<float>(PixelStorage::BYTE4, resolution);
    auto image_float4 = device.create_image<float>(PixelStorage::FLOAT4, resolution);
    std::vector<std::array<uint8_t, 4u>> array_byte4(resolution.x * resolution.y);
    std::vector<std::array<float, 4u>> array_float4(resolution.x * resolution.y);

    auto stream = device.create_stream();
    stream << printer.reset();

    for (auto i = 0u; i < resolution.x * resolution.y; ++i) {
        array_byte4[i] = {12, 34, 56, 78};
        array_float4[i] = {-12.0f, 43.121f, -89.1f, 0.f};
    }

    auto command_buffer = stream.command_buffer();

    command_buffer << image_byte4.copy_from(array_byte4.data())
                   << image_float4.copy_from(array_float4.data());

    Kernel2D display_kernel = [&](ImageFloat image) {
        auto coord = dispatch_id().xy();
        auto num = image.read(coord);
        printer.info_with_location("color = ({}, {}, {}, {})", num.x, num.y, num.z, num.w);
    };

    auto display_shader = device.compile(display_kernel);

    Clock clock;
    command_buffer << display_shader(image_byte4).dispatch(resolution)
                   << printer.retrieve()
                   << synchronize();
    LUISA_INFO("Finished in {} ms.", clock.toc());
}
