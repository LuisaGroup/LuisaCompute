// Test for template binding groups.
// This test covers image/image-view members, uniform members, methods, and
// nested template binding groups with deterministic device-to-host validation.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>

#include <cstddef>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

template<class T>
struct TArguments {
    Image<T> image;
    uint2 resolution;
};
template<class T>
struct TArgumentsView {
    ImageView<T> image;
    uint2 resolution;
};
template<class T>
struct TNestedArguments {
    TArgumentsView<T> args;
    Image<T> image;
};

#define TEMPLATE_T()  \
    template<class T> \
        requires is_legal_image_element<T>

LUISA_BINDING_GROUP_TEMPLATE(TEMPLATE_T, TArguments<T>, image, resolution) {
    [[nodiscard]] auto write(const UInt2 &coord, const Float4 &color) noexcept {
        this->image->write(coord, color);
    }
};
LUISA_BINDING_GROUP_TEMPLATE(TEMPLATE_T, TArgumentsView<T>, image, resolution) {
    [[nodiscard]] auto write(const UInt2 &coord, const Float4 &color) noexcept {
        this->image->write(coord, color);
    }
};
LUISA_BINDING_GROUP_TEMPLATE(TEMPLATE_T, TNestedArguments<T>, args, image) {
    void blit(const UInt2 &coord) noexcept {
        auto color = this->args.image.read(coord).xyz();
        this->image->write(coord, make_float4(1.f - color, 1.f));
    }
};
using Arguments = TArguments<float>;
using ArgumentsView = TArgumentsView<float>;
using NestedArguments = TNestedArguments<float>;

[[nodiscard]] bool validate_image(const luisa::vector<std::byte> &pixels,
                                  uint2 resolution,
                                  bool inverted) noexcept {
    auto expected_size = static_cast<size_t>(resolution.x) * resolution.y * 4u;
    if (pixels.size() != expected_size) {
        LUISA_WARNING("Unexpected image size: got {}, expected {}.",
                      pixels.size(), expected_size);
        return false;
    }
    for (auto y = 0u; y < resolution.y; ++y) {
        for (auto x = 0u; x < resolution.x; ++x) {
            auto expected_r = static_cast<uint8_t>(((x + resolution.x) & 1u) * 255u);
            auto expected_g = static_cast<uint8_t>(((y + resolution.y) & 1u) * 255u);
            auto expected_b = static_cast<uint8_t>(((x + y + resolution.x + resolution.y) & 1u) * 255u);
            if (inverted) {
                expected_r = static_cast<uint8_t>(255u - expected_r);
                expected_g = static_cast<uint8_t>(255u - expected_g);
                expected_b = static_cast<uint8_t>(255u - expected_b);
            }
            auto offset = (static_cast<size_t>(y) * resolution.x + x) * 4u;
            auto r = std::to_integer<uint8_t>(pixels[offset + 0u]);
            auto g = std::to_integer<uint8_t>(pixels[offset + 1u]);
            auto b = std::to_integer<uint8_t>(pixels[offset + 2u]);
            auto a = std::to_integer<uint8_t>(pixels[offset + 3u]);
            if (r != expected_r || g != expected_g || b != expected_b || a != 255u) {
                LUISA_WARNING(
                    "Template binding-group image mismatch at ({}, {}): got ({}, {}, {}, {}), "
                    "expected ({}, {}, {}, 255).",
                    x, y, r, g, b, a, expected_r, expected_g, expected_b);
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] int test_binding_group_template(Device &device) {

    log_level_verbose();

    auto stream = device.create_stream();

    Callable color = [](UInt2 coord, Var<Arguments> args) noexcept {
        auto r = cast<float>((coord.x + args.resolution.x) & 1u);
        auto g = cast<float>((coord.y + args.resolution.y) & 1u);
        auto b = cast<float>((coord.x + coord.y + args.resolution.x + args.resolution.y) & 1u);
        return make_float4(r, g, b, 1.f);
    };

    Callable color_with_view = [](UInt2 coord, Var<ArgumentsView> args) noexcept {
        auto r = cast<float>((coord.x + args.resolution.x) & 1u);
        auto g = cast<float>((coord.y + args.resolution.y) & 1u);
        auto b = cast<float>((coord.x + coord.y + args.resolution.x + args.resolution.y) & 1u);
        return make_float4(1.f - r, 1.f - g, 1.f - b, 1.f);
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

    constexpr auto resolution = make_uint2(17u, 11u);
    Arguments args{
        .image = device.create_image<float>(PixelStorage::BYTE4, resolution),
        .resolution = resolution};

    ArgumentsView args_view{
        .image = args.image.view(),
        .resolution = args.resolution};

    NestedArguments args_nested{
        .args = args_view,
        .image = device.create_image<float>(PixelStorage::BYTE4, resolution)};

    luisa::vector<std::byte> host_image(args.image.view().size_bytes());
    auto all_correct = true;

    // simple binding group
    stream << shader(args).dispatch(args.resolution)
           << args.image.copy_to(luisa::span{host_image})
           << synchronize();
    all_correct &= validate_image(host_image, resolution, false);

    // binding group with view
    stream << shader_with_view(args_view).dispatch(args_view.resolution)
           << args.image.copy_to(luisa::span{host_image})
           << synchronize();
    all_correct &= validate_image(host_image, resolution, true);

    // nested binding group
    stream << shader_with_nested(args_nested).dispatch(args_nested.image.view().size())
           << args_nested.image.copy_to(luisa::span{host_image})
           << synchronize();
    all_correct &= validate_image(host_image, resolution, false);

    expect(all_correct) << "template binding-group image, image-view, and nested bindings must match the host oracle";
    return all_correct ? 0 : 1;
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    return test_binding_group_template(device);
}
