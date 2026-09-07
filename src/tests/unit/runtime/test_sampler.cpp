// Deterministic bindless-sampler test.
//
// The source texture is analytic and small. Stored sampler state is checked for
// point and bilinear filtering plus edge, repeat, mirror, and zero addressing.
// Results are read back and compared with an independent normalized-coordinate
// host sampler instead of being dumped to an unverified image.

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <array>
#include <cmath>

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto texture_size = 4u;

enum Output : uint32_t {
    point_interior,
    point_edge,
    point_repeat,
    point_mirror,
    point_zero,
    bilinear_center,
    bilinear_repeat_edge,
    size_query,
    explicit_2d,
    explicit_2d_level,
    explicit_2d_grad,
    explicit_2d_grad_level,
    explicit_3d,
    explicit_3d_level,
    explicit_3d_grad,
    explicit_3d_grad_level,
    output_count
};

[[nodiscard]] float4 texel(uint32_t x, uint32_t y) noexcept {
    return make_float4(
        static_cast<float>(x + 1u) * (1.0f / 16.0f),
        static_cast<float>(y + 1u) * (1.0f / 16.0f),
        static_cast<float>(x + 4u * y) * (1.0f / 32.0f),
        1.0f);
}

[[nodiscard]] int positive_mod(int x, int modulus) noexcept {
    auto value = x % modulus;
    return value < 0 ? value + modulus : value;
}

[[nodiscard]] float4 fetch(int x, int y, Sampler::Address address) noexcept {
    constexpr auto size = static_cast<int>(texture_size);
    auto address_one = [address](int value, bool &valid) noexcept {
        switch (address) {
            case Sampler::Address::EDGE:
                return std::clamp(value, 0, size - 1);
            case Sampler::Address::REPEAT:
                return positive_mod(value, size);
            case Sampler::Address::MIRROR: {
                auto mirrored = positive_mod(value, 2 * size);
                return mirrored < size ? mirrored : 2 * size - 1 - mirrored;
            }
            case Sampler::Address::ZERO:
                if (value < 0 || value >= size) { valid = false; }
                return std::clamp(value, 0, size - 1);
        }
        valid = false;
        return 0;
    };
    auto valid = true;
    x = address_one(x, valid);
    y = address_one(y, valid);
    return valid ? texel(static_cast<uint32_t>(x), static_cast<uint32_t>(y)) :
                   make_float4(0.0f);
}

[[nodiscard]] float4 sample_point(float2 uv, Sampler::Address address) noexcept {
    auto x = static_cast<int>(std::floor(uv.x * texture_size));
    auto y = static_cast<int>(std::floor(uv.y * texture_size));
    return fetch(x, y, address);
}

[[nodiscard]] float4 sample_linear(float2 uv, Sampler::Address address) noexcept {
    auto px = uv.x * texture_size - 0.5f;
    auto py = uv.y * texture_size - 0.5f;
    auto x0 = static_cast<int>(std::floor(px));
    auto y0 = static_cast<int>(std::floor(py));
    auto tx = px - static_cast<float>(x0);
    auto ty = py - static_cast<float>(y0);
    auto v00 = fetch(x0, y0, address);
    auto v10 = fetch(x0 + 1, y0, address);
    auto v01 = fetch(x0, y0 + 1, address);
    auto v11 = fetch(x0 + 1, y0 + 1, address);
    auto vx0 = v00 + (v10 - v00) * tx;
    auto vx1 = v01 + (v11 - v01) * tx;
    return vx0 + (vx1 - vx0) * ty;
}

[[nodiscard]] float max_error(float4 a, float4 b) noexcept {
    auto d = abs(a - b);
    return std::max(std::max(d.x, d.y), std::max(d.z, d.w));
}

}// namespace

void test_sampler(Device &device) {
    auto texture = device.create_image<float>(
        PixelStorage::FLOAT4, make_uint2(texture_size));
    auto volume = device.create_volume<float>(
        PixelStorage::FLOAT4, make_uint3(2u));
    luisa::vector<float4> source(texture_size * texture_size);
    for (auto y = 0u; y < texture_size; y++) {
        for (auto x = 0u; x < texture_size; x++) {
            source[y * texture_size + x] = texel(x, y);
        }
    }

    std::array<float4, 8u> volume_source{};
    for (auto i = 0u; i < volume_source.size(); ++i) {
        volume_source[i] = make_float4(
            static_cast<float>(i + 1u),
            static_cast<float>(i + 2u),
            static_cast<float>(i + 3u), 1.0f);
    }

    auto heap = device.create_bindless_array(7u);
    auto output = device.create_buffer<float4>(output_count);
    std::array<float4, output_count> actual{};
    auto stream = device.create_stream();
    stream << texture.copy_from(luisa::span{source})
           << volume.copy_from(luisa::span{volume_source})
           << heap.emplace_on_update(0u, texture, Sampler::point_edge())
                  .emplace_on_update(1u, texture, Sampler::point_repeat())
                  .emplace_on_update(2u, texture, Sampler::point_mirror())
                  .emplace_on_update(3u, texture, Sampler::point_zero())
                  .emplace_on_update(4u, texture, Sampler::linear_point_edge())
                  .emplace_on_update(5u, texture, Sampler::linear_point_repeat())
                  .emplace_on_update(6u, volume, Sampler::point_edge())
                  .update();

    constexpr auto interior_uv = make_float2(0.30f, 0.60f);
    constexpr auto outside_uv = make_float2(-0.40f, 1.40f);
    constexpr auto center_uv = make_float2(0.50f, 0.50f);
    constexpr auto repeat_edge_uv = make_float2(0.0f, 0.50f);
    constexpr auto explicit_uv = make_float2(-0.25f, 0.25f);
    constexpr auto explicit_uvw = make_float3(-0.25f, 0.25f, 0.75f);
    Kernel1D check_sampler = [&](BindlessVar bindless,
                                 BufferFloat4 result) noexcept {
        result.write(static_cast<uint32_t>(point_interior),
                     bindless.tex2d(0u).sample(interior_uv));
        result.write(static_cast<uint32_t>(point_edge),
                     bindless.tex2d(0u).sample(outside_uv));
        result.write(static_cast<uint32_t>(point_repeat),
                     bindless.tex2d(1u).sample(outside_uv));
        result.write(static_cast<uint32_t>(point_mirror),
                     bindless.tex2d(2u).sample(outside_uv));
        result.write(static_cast<uint32_t>(point_zero),
                     bindless.tex2d(3u).sample(outside_uv));
        result.write(static_cast<uint32_t>(bilinear_center),
                     bindless.tex2d(4u).sample(center_uv));
        result.write(static_cast<uint32_t>(bilinear_repeat_edge),
                     bindless.tex2d(5u).sample(repeat_edge_uv));
        auto size = bindless.tex2d(0u).size();
        result.write(static_cast<uint32_t>(size_query),
                     make_float4(cast<float>(size.x), cast<float>(size.y),
                                 0.0f, 0.0f));
        auto explicit_texture = bindless.tex2d(0u);
        auto zero2 = make_float2(0.0f);
        result.write(static_cast<uint32_t>(explicit_2d),
                     explicit_texture.sample(
                         explicit_uv, SamplerFilter::POINT,
                         SamplerAddress::REPEAT));
        result.write(static_cast<uint32_t>(explicit_2d_level),
                     explicit_texture.sample(
                         explicit_uv, 0.0f, SamplerFilter::POINT,
                         SamplerAddress::REPEAT));
        result.write(static_cast<uint32_t>(explicit_2d_grad),
                     explicit_texture.sample(
                         explicit_uv, zero2, zero2,
                         SamplerFilter::POINT, SamplerAddress::REPEAT));
        result.write(static_cast<uint32_t>(explicit_2d_grad_level),
                     explicit_texture.sample(
                         explicit_uv, zero2, zero2, 0.0f,
                         SamplerFilter::POINT, SamplerAddress::REPEAT));

        auto explicit_volume = bindless.tex3d(6u);
        auto zero3 = make_float3(0.0f);
        result.write(static_cast<uint32_t>(explicit_3d),
                     explicit_volume.sample(
                         explicit_uvw, SamplerFilter::POINT,
                         SamplerAddress::REPEAT));
        result.write(static_cast<uint32_t>(explicit_3d_level),
                     explicit_volume.sample(
                         explicit_uvw, 0.0f, SamplerFilter::POINT,
                         SamplerAddress::REPEAT));
        result.write(static_cast<uint32_t>(explicit_3d_grad),
                     explicit_volume.sample(
                         explicit_uvw, zero3, zero3,
                         SamplerFilter::POINT, SamplerAddress::REPEAT));
        result.write(static_cast<uint32_t>(explicit_3d_grad_level),
                     explicit_volume.sample(
                         explicit_uvw, zero3, zero3, 0.0f,
                         SamplerFilter::POINT, SamplerAddress::REPEAT));
    };
    auto shader = device.compile(check_sampler);
    stream << shader(heap, output).dispatch(1u)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    std::array<float4, output_count> expected{
        sample_point(interior_uv, Sampler::Address::EDGE),
        sample_point(outside_uv, Sampler::Address::EDGE),
        sample_point(outside_uv, Sampler::Address::REPEAT),
        sample_point(outside_uv, Sampler::Address::MIRROR),
        sample_point(outside_uv, Sampler::Address::ZERO),
        sample_linear(center_uv, Sampler::Address::EDGE),
        sample_linear(repeat_edge_uv, Sampler::Address::REPEAT),
        make_float4(static_cast<float>(texture_size),
                    static_cast<float>(texture_size), 0.0f, 0.0f),
        sample_point(explicit_uv, Sampler::Address::REPEAT),
        sample_point(explicit_uv, Sampler::Address::REPEAT),
        sample_point(explicit_uv, Sampler::Address::REPEAT),
        sample_point(explicit_uv, Sampler::Address::REPEAT),
        volume_source[5u], volume_source[5u], volume_source[5u],
        volume_source[5u]};

    constexpr auto epsilon = 1.0e-5f;
    for (auto i = 0u; i < output_count; i++) {
        auto error = max_error(actual[i], expected[i]);
        expect(static_cast<bool>(error <= epsilon))
            << "sampler result " << i << " must match the host oracle; max error " << error;
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    auto &device = dc->device;
    test_sampler(device);
}
