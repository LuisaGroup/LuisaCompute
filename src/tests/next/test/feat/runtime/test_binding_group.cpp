/**
 * @file test_bindless_group.cpp
 * @brief Bindless Group Test Suite
 * @author sailing-innocent
 * @date 2024-05-16
 */
#include "common/config.h"
#include "luisa/core/logging.h"
#include "luisa/dsl/binding_group.h"
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/dsl/syntax.h>
#include <stb/stb_image_write.h>
using namespace luisa;
using namespace luisa::compute;

// Group Decl
namespace luisa::test {

struct BDArguments {
    Image<float> image;
    uint2 resolution;
};

struct BDArgumentsView {
    ImageView<float> image;
    uint2 resolution;
};

struct BDNestedArguments {
    BDArgumentsView args;
    Image<float> image;
};

}// namespace luisa::test

LUISA_BINDING_GROUP(luisa::test::BDArguments, image, resolution) {
    [[nodiscard]] auto write(const UInt2 &coord, const Float4 &color) noexcept {
        image->write(coord, color);
    }
};

LUISA_BINDING_GROUP(luisa::test::BDArgumentsView, image, resolution) {
    [[nodiscard]] auto write(const UInt2 &coord, const Float4 &color) noexcept {
        image->write(coord, color);
    }
};

LUISA_BINDING_GROUP(luisa::test::BDNestedArguments, args, image) {
    void blit(const UInt2 &coord) noexcept {
        auto color = args.image.read(coord).xyz();
        image->write(coord, make_float4(1.f - color, 1.f));
    }
};

namespace luisa::test {

int test_binding_group(Device &device) {
    log_level_verbose();
    Stream stream = device.create_stream();

    Callable color = [](UInt2 coord, Var<BDArguments> args) noexcept {
        auto uv = (make_float2(coord) + .5f) / make_float2(args.resolution);
        return make_float4(uv, .5f, 1.f);
    };

    Callable color_with_view = [](UInt2 coord, Var<BDArgumentsView> args) noexcept {
        auto uv = (make_float2(coord) + .5f) / make_float2(args.resolution);
        return make_float4(uv, .5f, 1.f);
    };

    Kernel2D kernel = [&color](Var<BDArguments> args) noexcept {
        auto coord = dispatch_id().xy();
        args->write(coord, color(coord, args));
    };

    Kernel2D kernel_with_view = [&color_with_view](Var<BDArgumentsView> args) noexcept {
        auto coord = dispatch_id().xy();
        args->write(coord, color_with_view(coord, args));
    };
    Kernel2D kernel_with_nested = [](Var<BDNestedArguments> args) noexcept {
        auto coord = dispatch_id().xy();
        args->blit(coord);
    };
    auto shader = device.compile(kernel);
    auto shader_with_view = device.compile(kernel_with_view);
    auto shader_with_nested = device.compile(kernel_with_nested);
    BDArguments args{
        .image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(1024, 1024)),
        .resolution = make_uint2(1024, 1024)};

    BDArgumentsView args_view{
        .image = args.image.view(),
        .resolution = args.resolution};

    BDNestedArguments args_nested{
        .args = args_view,
        .image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(1024, 1024))};

    luisa::vector<std::byte> host_image(args.image.view().size_bytes());

    stream << shader(args).dispatch(args.resolution)
           << args.image.copy_to(host_image.data())
           << synchronize();
    stbi_write_png("test_bindless_group.png",
                   (int)args.resolution.x, (int)args.resolution.y, 4,
                   host_image.data(), 0);

    // binding group with view
    stream << shader_with_view(args_view).dispatch(args_view.resolution)
           << args.image.copy_to(host_image.data())
           << synchronize();
    stbi_write_png("test_binding_group_with_view.png",
                   (int)args.resolution.x, (int)args.resolution.y, 4,
                   host_image.data(), 0);
    // nested binding group
    stream << shader_with_nested(args_nested).dispatch(args_nested.image.view().size())
           << args_nested.image.copy_to(host_image.data())
           << synchronize();
    stbi_write_png("test_binding_group_nested.png",
                   (int)args.resolution.x, (int)args.resolution.y, 4,
                   host_image.data(), 0);
    return 0;
}

}// namespace luisa::test

TEST_SUITE("runtime") {
    using namespace luisa::test;
    LUISA_TEST_CASE_WITH_DEVICE("binding_group", test_binding_group(device) == 0);
}