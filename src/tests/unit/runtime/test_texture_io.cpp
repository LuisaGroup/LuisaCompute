// Deterministic texture I/O test.
//
// This covers host upload/download plus shader reads and writes for both 2D
// images and 3D volumes. Every texel is checked against an independent host
// oracle; no generated image is used as a proxy for correctness.

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <cmath>

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] bool close(float4 actual, float4 expected) noexcept {
    constexpr auto epsilon = 1.0e-6f;
    auto d = abs(actual - expected);
    return d.x <= epsilon && d.y <= epsilon &&
           d.z <= epsilon && d.w <= epsilon;
}

}// namespace

void test_texture_io(Device &device) {
    log_level_verbose();

    constexpr auto image_size = make_uint2(32u, 24u);
    constexpr auto changed_size = make_uint2(image_size.x / 2u,
                                             image_size.y / 3u);
    auto image = device.create_image<float>(PixelStorage::FLOAT4, image_size);
    luisa::vector<float4> image_input(
        static_cast<size_t>(image_size.x) * image_size.y);
    luisa::vector<float4> image_expected(image_input.size());
    luisa::vector<float4> image_output(image_input.size());

    for (auto y = 0u; y < image_size.y; y++) {
        for (auto x = 0u; x < image_size.x; x++) {
            auto index = static_cast<size_t>(y) * image_size.x + x;
            auto value = make_float4(
                static_cast<float>(x) * (1.0f / 32.0f),
                static_cast<float>(y) * (1.0f / 16.0f),
                static_cast<float>(x + 2u * y) * (1.0f / 64.0f),
                1.0f);
            image_input[index] = value;
            value = make_float4(value.z, value.x * 2.0f,
                                value.y + 0.25f, value.w);
            if (x < changed_size.x && y < changed_size.y) {
                value += make_float4(1.0f, 0.5f, 0.25f, 0.0f);
            }
            image_expected[index] = value;
        }
    }

    Kernel2D transform_image = [](ImageFloat target) noexcept {
        auto coord = dispatch_id().xy();
        auto value = target.read(coord);
        target.write(coord, make_float4(
                                value.z, value.x * 2.0f,
                                value.y + 0.25f, value.w));
    };
    Kernel2D change_subregion = [](ImageFloat target) noexcept {
        auto coord = dispatch_id().xy();
        target.write(coord, target.read(coord) +
                                make_float4(1.0f, 0.5f, 0.25f, 0.0f));
    };

    auto transform_image_shader = device.compile(transform_image);
    auto change_subregion_shader = device.compile(change_subregion);
    auto stream = device.create_stream();
    stream << image.copy_from(luisa::span{image_input})
           << transform_image_shader(image).dispatch(image_size)
           << change_subregion_shader(image).dispatch(changed_size)
           << image.copy_to(luisa::span{image_output})
           << synchronize();

    auto image_valid = true;
    for (auto i = 0u; i < image_output.size(); i++) {
        if (!close(image_output[i], image_expected[i])) {
            LUISA_WARNING("Image texel {} mismatch: got {}, expected {}.",
                          i, image_output[i], image_expected[i]);
            image_valid = false;
            break;
        }
    }
    expect(image_valid)
        << "2D image upload, shader read/write, partial dispatch, and download must match the host oracle";

    constexpr auto volume_size = make_uint3(8u, 6u, 4u);
    auto volume = device.create_volume<float>(PixelStorage::FLOAT4, volume_size);
    auto volume_texel_count = static_cast<size_t>(volume_size.x) *
                              volume_size.y * volume_size.z;
    luisa::vector<float4> volume_input(volume_texel_count);
    luisa::vector<float4> volume_expected(volume_texel_count);
    luisa::vector<float4> volume_output(volume_texel_count);
    for (auto z = 0u; z < volume_size.z; z++) {
        for (auto y = 0u; y < volume_size.y; y++) {
            for (auto x = 0u; x < volume_size.x; x++) {
                auto index = (static_cast<size_t>(z) * volume_size.y + y) *
                                 volume_size.x +
                             x;
                auto value = make_float4(
                    static_cast<float>(x) * 0.125f,
                    static_cast<float>(y) * 0.25f,
                    static_cast<float>(z) * 0.5f,
                    2.0f);
                volume_input[index] = value;
                volume_expected[index] = make_float4(
                    value.x + value.y, value.z * 2.0f,
                    value.x - value.y, value.w + 1.0f);
            }
        }
    }

    auto transform_volume = device.compile<3>([](VolumeFloat target) noexcept {
        auto coord = dispatch_id();
        auto value = target.read(coord);
        target.write(coord, make_float4(
                                value.x + value.y, value.z * 2.0f,
                                value.x - value.y, value.w + 1.0f));
    });
    stream << volume.copy_from(luisa::span{volume_input})
           << transform_volume(volume).dispatch(volume_size)
           << volume.copy_to(luisa::span{volume_output})
           << synchronize();

    auto volume_valid = true;
    for (auto i = 0u; i < volume_output.size(); i++) {
        if (!close(volume_output[i], volume_expected[i])) {
            LUISA_WARNING("Volume texel {} mismatch: got {}, expected {}.",
                          i, volume_output[i], volume_expected[i]);
            volume_valid = false;
            break;
        }
    }
    expect(volume_valid)
        << "3D volume upload, shader read/write, and download must match the host oracle";
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_texture_io(device);
}
