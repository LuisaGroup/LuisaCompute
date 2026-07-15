#include "ut/ut.hpp"
#include "test_device.h"
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/dsl/syntax.h>
#include <stb/stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

struct Arguments {
    Image<float> image;
    uint2 resolution;
};

struct ArgumentsView {
    ImageView<float> image;
    uint2 resolution;
};

struct NestedArguments {
    ArgumentsView args;
    Image<float> image;
};

LUISA_BINDING_GROUP(Arguments, image, resolution) {
    [[nodiscard]] auto write(const UInt2 &coord, const Float4 &color) noexcept {
        image->write(coord, color);
    }
};

LUISA_BINDING_GROUP(ArgumentsView, image, resolution) {
    [[nodiscard]] auto write(const UInt2 &coord, const Float4 &color) noexcept {
        image->write(coord, color);
    }
};

LUISA_BINDING_GROUP(NestedArguments, args, image) {
    void blit(const UInt2 &coord) noexcept {
        auto color = args.image.read(coord).xyz();
        image->write(coord, make_float4(1.f - color, 1.f));
    }
};

void test_binding_group(Device &device) {

    log_level_verbose();

    auto stream = device.create_stream();

    Callable color = [](UInt2 coord, Var<Arguments> args) noexcept {
        auto uv = (make_float2(coord) + .5f) / make_float2(args.resolution);
        return make_float4(uv, .5f, 1.f);
    };

    Callable color_with_view = [](UInt2 coord, Var<ArgumentsView> args) noexcept {
        auto uv = (make_float2(coord) + .5f) / make_float2(args.resolution);
        return make_float4(uv, .5f, 1.f);
    };

    Kernel2D kernel = [&color](Var<Arguments> args) noexcept {
        auto coord = dispatch_id().xy();
        args->write(coord, color(coord, args));
    };

    Kernel2D kernel_with_view = [&color_with_view](Var<ArgumentsView> args) noexcept {
        auto coord = dispatch_id().xy();
        args->write(coord, color_with_view(coord, args));
    };

    Kernel2D kernel_with_nested = [](Var<NestedArguments> args) noexcept {
        auto coord = dispatch_id().xy();
        args->blit(coord);
    };

    auto shader = device.compile(kernel);
    auto shader_with_view = device.compile(kernel_with_view);
    auto shader_with_nested = device.compile(kernel_with_nested);
    expect(true) << "binding group shaders compiled";

    Arguments args{
        .image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(1024, 1024)),
        .resolution = make_uint2(1024, 1024)};

    ArgumentsView args_view{
        .image = args.image.view(),
        .resolution = args.resolution};

    NestedArguments args_nested{
        .args = args_view,
        .image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(1024, 1024))};

    luisa::vector<std::byte> host_image(args.image.view().size_bytes());

    // simple binding group
    stream << shader(args).dispatch(args.resolution)
           << args.image.copy_to(luisa::span{host_image})
           << synchronize();
    stbi_write_png("test_binding_group.png",
                   args.resolution.x, args.resolution.y, 4,
                   host_image.data(), 0);

    // binding group with view
    stream << shader_with_view(args_view).dispatch(args_view.resolution)
           << args.image.copy_to(luisa::span{host_image})
           << synchronize();
    stbi_write_png("test_binding_group_with_view.png",
                   args.resolution.x, args.resolution.y, 4,
                   host_image.data(), 0);

    // nested binding group
    stream << shader_with_nested(args_nested).dispatch(args_nested.image.view().size())
           << args_nested.image.copy_to(luisa::span{host_image})
           << synchronize();
    stbi_write_png("test_binding_group_nested.png",
                   args.resolution.x, args.resolution.y, 4,
                   host_image.data(), 0);
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_binding_group(device);
}
