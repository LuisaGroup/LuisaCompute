// Deterministic compute mipmap-generation test.
//
// A single shader dispatch recursively box-filters five mip levels through
// shared memory. The source is generated in memory and every texel of every
// level is compared with an independent host implementation, so the test has
// no working-directory dependency and never treats image dumping as success.

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto base_size = make_uint2(64u, 64u);
constexpr auto mip_level_count = 6u;
constexpr auto reduction_tile_size = 32u;

[[nodiscard]] uint2 mip_size(uint32_t level) noexcept {
    return max(base_size >> level, make_uint2(1u));
}

[[nodiscard]] auto make_source() noexcept {
    luisa::vector<float4> pixels(
        static_cast<size_t>(base_size.x) * base_size.y);
    for (auto y = 0u; y < base_size.y; y++) {
        for (auto x = 0u; x < base_size.x; x++) {
            auto index = static_cast<size_t>(y) * base_size.x + x;
            pixels[index] = make_float4(
                static_cast<float>(x + 2u * y) * (1.0f / 256.0f),
                static_cast<float>((3u * x) ^ y) * (1.0f / 256.0f),
                static_cast<float>((x & 3u) + 4u * (y & 3u)) * (1.0f / 32.0f),
                0.75f);
        }
    }
    return pixels;
}

[[nodiscard]] auto make_reference_levels() noexcept {
    std::array<luisa::vector<float4>, mip_level_count> levels;
    levels[0] = make_source();
    for (auto level = 1u; level < mip_level_count; level++) {
        auto previous_size = mip_size(level - 1u);
        auto size = mip_size(level);
        levels[level].resize(static_cast<size_t>(size.x) * size.y);
        for (auto y = 0u; y < size.y; y++) {
            for (auto x = 0u; x < size.x; x++) {
                auto p00 = static_cast<size_t>(2u * y) * previous_size.x + 2u * x;
                auto p10 = p00 + 1u;
                auto p01 = p00 + previous_size.x;
                auto p11 = p01 + 1u;
                auto value = (levels[level - 1u][p00] +
                              levels[level - 1u][p10] +
                              levels[level - 1u][p01] +
                              levels[level - 1u][p11]) *
                             0.25f;
                value.w = 1.0f;
                levels[level][static_cast<size_t>(y) * size.x + x] = value;
            }
        }
    }
    return levels;
}

[[nodiscard]] float max_error(float4 a, float4 b) noexcept {
    auto d = abs(a - b);
    return std::max(std::max(d.x, d.y), std::max(d.z, d.w));
}

}// namespace

void test_mipmap(Device &device) {
    auto reference = make_reference_levels();
    auto texture = device.create_image<float>(
        PixelStorage::FLOAT4, base_size, mip_level_count);

    auto write_level = [](ImageVar<float> **levels, UInt2 pixel,
                          Float4 value, UInt index) noexcept {
        switch_(index)
            .case_(0u, [&] { levels[0]->write(pixel, value); })
            .case_(1u, [&] { levels[1]->write(pixel, value); })
            .case_(2u, [&] { levels[2]->write(pixel, value); })
            .case_(3u, [&] { levels[3]->write(pixel, value); })
            .case_(4u, [&] { levels[4]->write(pixel, value); })
            .case_(5u, [&] { levels[5]->write(pixel, value); });
    };

    Kernel2D generate_mip_levels =
        [&](ImageVar<float> level0,
            ImageVar<float> level1,
            ImageVar<float> level2,
            ImageVar<float> level3,
            ImageVar<float> level4,
            ImageVar<float> level5) noexcept {
            set_block_size(reduction_tile_size, reduction_tile_size, 1u);
            Shared<float3> shared_array{reduction_tile_size * reduction_tile_size};
            ImageVar<float> *levels[] = {
                std::addressof(level0), std::addressof(level1),
                std::addressof(level2), std::addressof(level3),
                std::addressof(level4), std::addressof(level5)};

            auto block_coord = block_id().xy();
            auto local_coord = thread_id().xy();
            auto texture_size = dispatch_size().xy();
            auto color = level0.read(dispatch_id().xy()).xyz();
            auto active_size = def(reduction_tile_size);
            auto level = def(0u);

            $while (active_size > 1u) {
                $if (all(local_coord < make_uint2(active_size))) {
                    shared_array.write(
                        active_size * local_coord.y + local_coord.x, color);
                };
                sync_block();

                auto next_size = active_size / 2u;
                $if (all(local_coord < make_uint2(next_size))) {
                    auto source = local_coord * 2u;
                    color = (shared_array.read(active_size * source.y + source.x) +
                             shared_array.read(active_size * (source.y + 1u) + source.x) +
                             shared_array.read(active_size * source.y + source.x + 1u) +
                             shared_array.read(active_size * (source.y + 1u) + source.x + 1u)) *
                            0.25f;
                    level += 1u;
                    texture_size /= 2u;
                    auto level_coord = block_coord * next_size + local_coord;
                    $if (all(level_coord < texture_size)) {
                        write_level(levels, level_coord,
                                    make_float4(color, 1.0f), level);
                    };
                };
                sync_block();
                active_size = next_size;
            };
        };

    auto shader = device.compile(generate_mip_levels);
    auto stream = device.create_stream();
    stream << texture.view(0u).copy_from(luisa::span{reference[0]})
           << shader(texture.view(0u), texture.view(1u), texture.view(2u),
                     texture.view(3u), texture.view(4u), texture.view(5u))
                  .dispatch(base_size);

    std::array<luisa::vector<float4>, mip_level_count> actual;
    for (auto level = 0u; level < mip_level_count; level++) {
        auto size = texture.view(level).size();
        actual[level].resize(static_cast<size_t>(size.x) * size.y);
        stream << texture.view(level).copy_to(luisa::span{actual[level]});
    }
    stream << synchronize();

    constexpr auto epsilon = 1.0e-6f;
    for (auto level = 0u; level < mip_level_count; level++) {
        auto size = texture.view(level).size();
        auto expected_size = mip_size(level);
        expect(static_cast<bool>(size.x == expected_size.x &&
                                 size.y == expected_size.y))
            << "mip view must report the expected dimensions";
        auto valid = true;
        for (auto i = 0u; i < actual[level].size(); i++) {
            auto error = max_error(actual[level][i], reference[level][i]);
            if (error > epsilon) {
                LUISA_WARNING(
                    "Mip {} texel {} mismatch: got {}, expected {}, max error {}.",
                    level, i, actual[level][i], reference[level][i], error);
                valid = false;
                break;
            }
        }
        expect(valid) << "every generated mip texel must match the recursive host box filter";
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));

    auto &device = dc->device;
    test_mipmap(device);
}
