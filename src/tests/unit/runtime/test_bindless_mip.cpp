#include "ut/ut.hpp"
#include "test_device.h"

#include <cmath>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] auto make_level_pixels(uint2 size, uint8_t value) noexcept {
    luisa::vector<std::byte> pixels(size.x * size.y * 4u);
    for (auto i = 0u; i < size.x * size.y; i++) {
        pixels[i * 4u + 0u] = static_cast<std::byte>(value);
        pixels[i * 4u + 1u] = static_cast<std::byte>(value + 1u);
        pixels[i * 4u + 2u] = static_cast<std::byte>(value + 2u);
        pixels[i * 4u + 3u] = static_cast<std::byte>(255u);
    }
    return pixels;
}

[[nodiscard]] auto expected_color(uint32_t level) noexcept {
    auto value = static_cast<float>(32u + level * 48u) / 255.0f;
    return make_float4(value, value + 1.0f / 255.0f, value + 2.0f / 255.0f, 1.0f);
}

void expect_close(float4 actual, float4 expected) noexcept {
    constexpr auto eps = 1.0f / 255.0f + 1e-4f;
    expect(static_cast<bool>(std::abs(actual.x - expected.x) < eps)) << "x mismatch";
    expect(static_cast<bool>(std::abs(actual.y - expected.y) < eps)) << "y mismatch";
    expect(static_cast<bool>(std::abs(actual.z - expected.z) < eps)) << "z mismatch";
    expect(static_cast<bool>(std::abs(actual.w - expected.w) < eps)) << "w mismatch";
}

}// namespace

void test_bindless_mip(Device &device) {
    static constexpr auto level_count = 4u;
    auto texture = device.create_image<float>(PixelStorage::BYTE4, make_uint2(8u, 8u), level_count);
    auto heap = device.create_bindless_array(1u);
    auto out = device.create_buffer<float4>(level_count * 2u);
    luisa::vector<float4> host(out.size());

    auto stream = device.create_stream();
    for (auto level = 0u; level < level_count; level++) {
        auto view = texture.view(level);
        auto pixels = make_level_pixels(view.size(), static_cast<uint8_t>(32u + level * 48u));
        stream << view.copy_from(luisa::span{pixels});
    }
    stream << heap.emplace_on_update(0u, texture, Sampler::point_edge()).update();

    Kernel1D check_mips = [&](BindlessVar bindless, BufferVar<float4> output) noexcept {
        auto level = dispatch_id().x;
        auto tex = bindless.tex2d(0u);
        auto size = tex.size(level);
        auto read_value = tex.read(make_uint2(0u), level);
        auto sample_value = tex.sample(make_float2(0.5f), cast<float>(level));
        output.write(level * 2u, read_value);
        output.write(level * 2u + 1u, sample_value);
        $if (any(size != max(make_uint2(8u) >> level, make_uint2(1u)))) {
            output.write(level * 2u, make_float4(1000.0f));
        };
    };

    auto shader = device.compile(check_mips);
    stream << shader(heap, out).dispatch(level_count)
           << out.copy_to(luisa::span{host})
           << synchronize();

    for (auto level = 0u; level < level_count; level++) {
        auto expected = expected_color(level);
        expect_close(host[level * 2u], expected);
        expect_close(host[level * 2u + 1u], expected);
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    test_bindless_mip(dc->device);
}
