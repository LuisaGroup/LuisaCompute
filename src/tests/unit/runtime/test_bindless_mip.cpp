// Deterministic bindless 2D mip/sampler conformance test.
//
// This deliberately uploads unrelated data to every mip. It checks raw reads
// and sizes, stored and call-site samplers, bilinear spatial filtering,
// point-vs-linear mip filtering, explicit fractional LOD, gradient-derived
// LOD, minimum-LOD clamping, address modes, and out-of-range LOD clamping.

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <array>
#include <cmath>

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

constexpr auto base_size = 8u;
constexpr auto level_count = 4u;
constexpr auto packed_size = 2u;

enum Output : uint32_t {
    read_begin,
    size_begin = read_begin + level_count,
    stored_bilinear = size_begin + level_count,
    custom_point,
    stored_trilinear,
    stored_mip_point,
    custom_trilinear,
    stored_gradient,
    custom_gradient,
    minimum_mip,
    custom_repeat,
    stored_repeat,
    custom_edge,
    clamped_lod,
    output_count
};

[[nodiscard]] uint32_t mip_size(uint32_t level) noexcept {
    return std::max(base_size >> level, 1u);
}

[[nodiscard]] float4 texel(uint32_t level, uint32_t x, uint32_t y) noexcept {
    auto base = static_cast<float>(level * 100u + x * 7u + y * 13u);
    // Keep the analytic signal normalized. Hardware texture units quantize
    // interpolation weights; using values in the hundreds would amplify that
    // expected coordinate quantization beyond a meaningful absolute tolerance.
    constexpr auto scale = 1.0f / 512.0f;
    return make_float4((base + 0.25f) * scale,
                       (base * 0.5f + 1.0f) * scale,
                       (static_cast<float>(level * 11u + x * 3u + y) + 0.5f) * scale,
                       1.0f);
}

[[nodiscard]] auto make_level_pixels(uint32_t level) noexcept {
    auto size = mip_size(level);
    luisa::vector<float4> pixels(static_cast<size_t>(size) * size);
    for (auto y = 0u; y < size; y++) {
        for (auto x = 0u; x < size; x++) {
            pixels[static_cast<size_t>(y) * size + x] = texel(level, x, y);
        }
    }
    return pixels;
}

[[nodiscard]] int wrap(int x, int size) noexcept {
    auto r = x % size;
    return r < 0 ? r + size : r;
}

[[nodiscard]] int mirror(int x, int size) noexcept {
    auto period = size * 2;
    auto m = wrap(x, period);
    return m >= size ? period - 1 - m : m;
}

[[nodiscard]] float4 fetch(uint32_t level, int x, int y,
                           Sampler::Address address) noexcept {
    auto size = static_cast<int>(mip_size(level));
    if (address == Sampler::Address::REPEAT) {
        x = wrap(x, size);
        y = wrap(y, size);
    } else {
        x = std::clamp(x, 0, size - 1);
        y = std::clamp(y, 0, size - 1);
    }
    return texel(level, static_cast<uint32_t>(x), static_cast<uint32_t>(y));
}

[[nodiscard]] float4 sample_spatial(uint32_t level, float2 uv,
                                    Sampler::Filter filter,
                                    Sampler::Address address) noexcept {
    auto size = static_cast<float>(mip_size(level));
    if (filter == Sampler::Filter::POINT) {
        return fetch(level,
                     static_cast<int>(std::floor(uv.x * size)),
                     static_cast<int>(std::floor(uv.y * size)),
                     address);
    }
    auto px = uv.x * size - 0.5f;
    auto py = uv.y * size - 0.5f;
    auto x0 = static_cast<int>(std::floor(px));
    auto y0 = static_cast<int>(std::floor(py));
    auto tx = px - static_cast<float>(x0);
    auto ty = py - static_cast<float>(y0);
    auto v00 = fetch(level, x0, y0, address);
    auto v10 = fetch(level, x0 + 1, y0, address);
    auto v01 = fetch(level, x0, y0 + 1, address);
    auto v11 = fetch(level, x0 + 1, y0 + 1, address);
    auto vx0 = v00 + (v10 - v00) * tx;
    auto vx1 = v01 + (v11 - v01) * tx;
    return vx0 + (vx1 - vx0) * ty;
}

[[nodiscard]] float4 sample_lod(float2 uv, float lod,
                                Sampler::Filter filter,
                                Sampler::Address address) noexcept {
    lod = std::clamp(lod, 0.0f, static_cast<float>(level_count - 1u));
    if (filter == Sampler::Filter::LINEAR_LINEAR ||
        filter == Sampler::Filter::ANISOTROPIC) {
        auto level0 = static_cast<uint32_t>(std::floor(lod));
        auto level1 = std::min(level0 + 1u, level_count - 1u);
        auto t = lod - static_cast<float>(level0);
        auto v0 = sample_spatial(level0, uv, filter, address);
        auto v1 = sample_spatial(level1, uv, filter, address);
        return v0 + (v1 - v0) * t;
    }
    auto level = static_cast<uint32_t>(std::floor(lod + 0.5f));
    return sample_spatial(level, uv, filter, address);
}

[[nodiscard]] float gradient_lod(float2 ddx, float2 ddy) noexcept {
    auto sx = ddx.x * static_cast<float>(base_size);
    auto sy = ddx.y * static_cast<float>(base_size);
    auto tx = ddy.x * static_cast<float>(base_size);
    auto ty = ddy.y * static_cast<float>(base_size);
    auto rho2 = std::max(sx * sx + sy * sy, tx * tx + ty * ty);
    return 0.5f * std::log2(std::max(rho2, 1.0f));
}

[[nodiscard]] constexpr uint32_t pack_r10g10b10a2(
    uint32_t r, uint32_t g, uint32_t b, uint32_t a) noexcept {
    return (r & 0x3ffu) |
           ((g & 0x3ffu) << 10u) |
           ((b & 0x3ffu) << 20u) |
           ((a & 0x3u) << 30u);
}

constexpr std::array packed_texels{
    pack_r10g10b10a2(0u, 0u, 0u, 0u),
    pack_r10g10b10a2(1023u, 0u, 0u, 1u),
    pack_r10g10b10a2(0u, 1023u, 0u, 2u),
    pack_r10g10b10a2(0u, 0u, 1023u, 3u),
    pack_r10g10b10a2(1023u, 1023u, 0u, 3u),
    pack_r10g10b10a2(512u, 256u, 768u, 2u),
    pack_r10g10b10a2(1u, 1022u, 511u, 1u),
    pack_r10g10b10a2(333u, 666u, 999u, 0u)};
constexpr std::array packed_mip_texels{
    pack_r10g10b10a2(777u, 111u, 555u, 3u)};

[[nodiscard]] float4 unpack_r10g10b10a2(uint32_t packed) noexcept {
    return make_float4(
        static_cast<float>(packed & 0x3ffu) * (1.0f / 1023.0f),
        static_cast<float>((packed >> 10u) & 0x3ffu) * (1.0f / 1023.0f),
        static_cast<float>((packed >> 20u) & 0x3ffu) * (1.0f / 1023.0f),
        static_cast<float>((packed >> 30u) & 0x3u) * (1.0f / 3.0f));
}

[[nodiscard]] float4 fetch_packed(int x, int y, int z,
                                  Sampler::Address address) noexcept {
    constexpr auto size = static_cast<int>(packed_size);
    if (address == Sampler::Address::ZERO &&
        (x < 0 || x >= size || y < 0 || y >= size ||
         z < 0 || z >= size)) {
        return make_float4(0.0f);
    }
    if (address == Sampler::Address::REPEAT) {
        x = wrap(x, size);
        y = wrap(y, size);
        z = wrap(z, size);
    } else if (address == Sampler::Address::MIRROR) {
        x = mirror(x, size);
        y = mirror(y, size);
        z = mirror(z, size);
    } else {
        x = std::clamp(x, 0, size - 1);
        y = std::clamp(y, 0, size - 1);
        z = std::clamp(z, 0, size - 1);
    }
    auto index = (static_cast<size_t>(z) * packed_size +
                  static_cast<size_t>(y)) *
                     packed_size +
                 static_cast<size_t>(x);
    return unpack_r10g10b10a2(packed_texels[index]);
}

[[nodiscard]] float4 sample_packed(float3 uvw, Sampler::Filter filter,
                                   Sampler::Address address) noexcept {
    constexpr auto size = static_cast<float>(packed_size);
    if (filter == Sampler::Filter::POINT) {
        return fetch_packed(
            static_cast<int>(std::floor(uvw.x * size)),
            static_cast<int>(std::floor(uvw.y * size)),
            static_cast<int>(std::floor(uvw.z * size)), address);
    }
    auto p = uvw * size - 0.5f;
    auto x0 = static_cast<int>(std::floor(p.x));
    auto y0 = static_cast<int>(std::floor(p.y));
    auto z0 = static_cast<int>(std::floor(p.z));
    auto tx = p.x - static_cast<float>(x0);
    auto ty = p.y - static_cast<float>(y0);
    auto tz = p.z - static_cast<float>(z0);
    auto lerp = [](float4 x, float4 y, float t) noexcept {
        return x + (y - x) * t;
    };
    auto v00 = lerp(fetch_packed(x0, y0, z0, address),
                    fetch_packed(x0 + 1, y0, z0, address), tx);
    auto v10 = lerp(fetch_packed(x0, y0 + 1, z0, address),
                    fetch_packed(x0 + 1, y0 + 1, z0, address), tx);
    auto v01 = lerp(fetch_packed(x0, y0, z0 + 1, address),
                    fetch_packed(x0 + 1, y0, z0 + 1, address), tx);
    auto v11 = lerp(fetch_packed(x0, y0 + 1, z0 + 1, address),
                    fetch_packed(x0 + 1, y0 + 1, z0 + 1, address), tx);
    return lerp(lerp(v00, v10, ty), lerp(v01, v11, ty), tz);
}

void expect_close(float4 actual, float4 expected,
                  const char *description) noexcept {
    constexpr auto eps = 2.0e-3f;
    auto delta = abs(actual - expected);
    auto error = std::max({delta.x, delta.y, delta.z, delta.w});
    expect(static_cast<bool>(error < eps))
        << description << ", max error " << error;
}

}// namespace

void test_bindless_mip(Device &device) {
    auto texture = device.create_image<float>(
        PixelStorage::FLOAT4, make_uint2(base_size), level_count);
    auto heap = device.create_bindless_array(3u);
    auto out = device.create_buffer<float4>(output_count);
    luisa::vector<float4> host(output_count);

    auto stream = device.create_stream();
    for (auto level = 0u; level < level_count; level++) {
        auto pixels = make_level_pixels(level);
        stream << texture.view(level).copy_from(luisa::span{pixels});
    }
    stream << heap.emplace_on_update(
                      0u, texture, Sampler::linear_linear_edge())
                  .emplace_on_update(
                      1u, texture, Sampler::linear_point_edge())
                  .emplace_on_update(
                      2u, texture, Sampler::point_repeat())
                  .update();

    constexpr auto uv = make_float2(0.31f, 0.43f);
    constexpr auto address_uv = make_float2(-0.10f, 1.10f);
    constexpr auto gradient_scale = 0.3535533905932738f;// exp2(1.5) / 8
    Kernel1D check_mips = [&](BindlessVar bindless,
                              BufferVar<float4> output) noexcept {
        auto linear = bindless.tex2d(0u);
        auto mip_point = bindless.tex2d(1u);
        auto point_repeat = bindless.tex2d(2u);
        for (auto level = 0u; level < level_count; level++) {
            auto size = max(make_uint2(base_size) >> level, make_uint2(1u));
            auto coord = min(make_uint2(1u, 2u), size - 1u);
            output.write(read_begin + level, linear.read(coord, level));
            auto actual_size = linear.size(level);
            output.write(size_begin + level,
                         make_float4(cast<float>(actual_size.x),
                                     cast<float>(actual_size.y),
                                     0.0f, 0.0f));
        }
        output.write(static_cast<uint32_t>(stored_bilinear), linear.sample(uv));
        output.write(static_cast<uint32_t>(custom_point),
                     linear.sample(uv, 0.0f,
                                   SamplerFilter::POINT,
                                   SamplerAddress::EDGE));
        output.write(static_cast<uint32_t>(stored_trilinear), linear.sample(uv, 1.25f));
        output.write(static_cast<uint32_t>(stored_mip_point), mip_point.sample(uv, 1.75f));
        output.write(static_cast<uint32_t>(custom_trilinear),
                     mip_point.sample(uv, 1.25f,
                                      SamplerFilter::LINEAR_LINEAR,
                                      SamplerAddress::EDGE));
        auto ddx = make_float2(gradient_scale, 0.0f);
        auto ddy = make_float2(0.0f, gradient_scale);
        output.write(static_cast<uint32_t>(stored_gradient), linear.sample(uv, ddx, ddy));
        output.write(static_cast<uint32_t>(custom_gradient),
                     mip_point.sample(uv, ddx, ddy,
                                      SamplerFilter::LINEAR_LINEAR,
                                      SamplerAddress::EDGE));
        auto base_ddx = make_float2(1.0f / base_size, 0.0f);
        auto base_ddy = make_float2(0.0f, 1.0f / base_size);
        output.write(static_cast<uint32_t>(minimum_mip),
                     linear.sample(uv, base_ddx, base_ddy, 2.25f));
        output.write(static_cast<uint32_t>(custom_repeat),
                     linear.sample(address_uv, 0.0f,
                                   SamplerFilter::POINT,
                                   SamplerAddress::REPEAT));
        output.write(static_cast<uint32_t>(stored_repeat), point_repeat.sample(address_uv));
        output.write(static_cast<uint32_t>(custom_edge),
                     point_repeat.sample(address_uv, 0.0f,
                                         SamplerFilter::POINT,
                                         SamplerAddress::EDGE));
        output.write(static_cast<uint32_t>(clamped_lod), linear.sample(uv, 100.0f));
    };

    auto shader = device.compile(check_mips);
    stream << shader(heap, out).dispatch(1u)
           << out.copy_to(luisa::span{host})
           << synchronize();

    for (auto level = 0u; level < level_count; level++) {
        auto size = mip_size(level);
        auto x = std::min(1u, size - 1u);
        auto y = std::min(2u, size - 1u);
        expect_close(host[read_begin + level], texel(level, x, y),
                     "explicit mip read");
        expect_close(host[size_begin + level],
                     make_float4(static_cast<float>(size),
                                 static_cast<float>(size), 0.0f, 0.0f),
                     "mip size query");
    }

    expect_close(host[stored_bilinear],
                 sample_lod(uv, 0.0f, Sampler::Filter::LINEAR_LINEAR,
                            Sampler::Address::EDGE),
                 "stored bilinear sampler");
    expect_close(host[custom_point],
                 sample_lod(uv, 0.0f, Sampler::Filter::POINT,
                            Sampler::Address::EDGE),
                 "call-site point sampler");
    expect_close(host[stored_trilinear],
                 sample_lod(uv, 1.25f, Sampler::Filter::LINEAR_LINEAR,
                            Sampler::Address::EDGE),
                 "stored trilinear fractional LOD");
    expect_close(host[stored_mip_point],
                 sample_lod(uv, 1.75f, Sampler::Filter::LINEAR_POINT,
                            Sampler::Address::EDGE),
                 "stored point mip filtering");
    expect_close(host[custom_trilinear],
                 sample_lod(uv, 1.25f, Sampler::Filter::LINEAR_LINEAR,
                            Sampler::Address::EDGE),
                 "call-site trilinear sampler overrides stored sampler");

    auto ddx = make_float2(gradient_scale, 0.0f);
    auto ddy = make_float2(0.0f, gradient_scale);
    auto lod = gradient_lod(ddx, ddy);
    expect_close(host[stored_gradient],
                 sample_lod(uv, lod, Sampler::Filter::LINEAR_LINEAR,
                            Sampler::Address::EDGE),
                 "gradient-derived LOD");
    expect_close(host[custom_gradient],
                 sample_lod(uv, lod, Sampler::Filter::LINEAR_LINEAR,
                            Sampler::Address::EDGE),
                 "gradient-derived LOD with call-site sampler");
    expect_close(host[minimum_mip],
                 sample_lod(uv, 2.25f, Sampler::Filter::LINEAR_LINEAR,
                            Sampler::Address::EDGE),
                 "minimum mip clamp");
    expect_close(host[custom_repeat],
                 sample_lod(address_uv, 0.0f, Sampler::Filter::POINT,
                            Sampler::Address::REPEAT),
                 "call-site repeat address mode");
    expect_close(host[stored_repeat],
                 sample_lod(address_uv, 0.0f, Sampler::Filter::POINT,
                            Sampler::Address::REPEAT),
                 "stored repeat address mode");
    expect_close(host[custom_edge],
                 sample_lod(address_uv, 0.0f, Sampler::Filter::POINT,
                            Sampler::Address::EDGE),
                 "call-site edge address mode overrides stored sampler");
    expect_close(host[clamped_lod],
                 sample_lod(uv, 100.0f, Sampler::Filter::LINEAR_LINEAR,
                            Sampler::Address::EDGE),
                 "out-of-range LOD clamp");
}

void test_bindless_packed_volume(Device &device) {
    constexpr auto size = make_uint3(packed_size);
    constexpr auto sample_center = make_float3(0.5f);
    constexpr auto sample_off_center = make_float3(0.375f, 0.625f, 0.375f);
    constexpr auto sample_point = make_float3(0.25f, 0.75f, 0.75f);
    constexpr auto sample_outside = make_float3(-0.25f, 0.25f, 0.25f);
    constexpr auto sample_mirror = make_float3(-0.375f, 0.625f, 0.375f);
    enum : uint32_t {
        bindless_read,
        point_edge,
        linear_center,
        linear_off_center,
        mip_read,
        trilinear_mip,
        linear_mirror,
        point_repeat,
        point_zero,
        packed_output_count
    };

    auto volume = device.create_volume<float>(
        PixelStorage::R10G10B10A2, size, 2u);
    auto heap = device.create_bindless_array(1u);
    auto output = device.create_buffer<float4>(packed_output_count);
    std::array<float4, packed_output_count> host_output{};
    std::array<uint32_t, packed_texels.size()> host_round_trip{};
    std::array<uint32_t, packed_mip_texels.size()> host_mip_round_trip{};

    Kernel3D round_trip = [](VolumeFloat texture) noexcept {
        auto coord = dispatch_id();
        texture.write(coord, texture.read(coord));
    };
    Kernel1D sample = [&](BindlessVar bindless,
                          BufferVar<float4> output_buffer) noexcept {
        auto texture = bindless.tex3d(0u);
        output_buffer.write(static_cast<uint32_t>(bindless_read),
                            texture.read(make_uint3(1u, 0u, 1u)));
        output_buffer.write(static_cast<uint32_t>(point_edge),
                            texture.sample(sample_point,
                                           SamplerFilter::POINT,
                                           SamplerAddress::EDGE));
        output_buffer.write(static_cast<uint32_t>(linear_center),
                            texture.sample(sample_center));
        output_buffer.write(static_cast<uint32_t>(linear_off_center),
                            texture.sample(sample_off_center));
        output_buffer.write(static_cast<uint32_t>(mip_read),
                            texture.read(make_uint3(0u), 1u));
        output_buffer.write(static_cast<uint32_t>(trilinear_mip),
                            texture.sample(sample_off_center, 0.5f));
        output_buffer.write(static_cast<uint32_t>(linear_mirror),
                            texture.sample(sample_mirror));
        output_buffer.write(static_cast<uint32_t>(point_repeat),
                            texture.sample(sample_outside,
                                           SamplerFilter::POINT,
                                           SamplerAddress::REPEAT));
        output_buffer.write(static_cast<uint32_t>(point_zero),
                            texture.sample(sample_outside,
                                           SamplerFilter::POINT,
                                           SamplerAddress::ZERO));
    };

    auto round_trip_shader = device.compile(round_trip);
    auto sample_shader = device.compile(sample);
    auto stream = device.create_stream();
    stream << volume.view(0u).copy_from(luisa::span{packed_texels})
           << volume.view(1u).copy_from(luisa::span{packed_mip_texels})
           << round_trip_shader(volume.view(0u)).dispatch(size)
           << round_trip_shader(volume.view(1u)).dispatch(make_uint3(1u))
           << heap.emplace_on_update(
                      0u, volume, Sampler::linear_linear_mirror())
                  .update()
           << sample_shader(heap, output).dispatch(1u)
           << output.copy_to(luisa::span{host_output})
           << volume.view(0u).copy_to(luisa::span{host_round_trip})
           << volume.view(1u).copy_to(luisa::span{host_mip_round_trip})
           << synchronize();

    expect(std::equal(packed_texels.begin(), packed_texels.end(),
                      host_round_trip.begin()))
        << "packed R10G10B10A2 shader read/write preserves exact texels";
    expect(std::equal(packed_mip_texels.begin(), packed_mip_texels.end(),
                      host_mip_round_trip.begin()))
        << "packed R10G10B10A2 nonzero mip preserves exact texels";
    expect_close(host_output[bindless_read],
                 unpack_r10g10b10a2(packed_texels[5u]),
                 "packed bindless volume read");
    expect_close(host_output[point_edge],
                 sample_packed(sample_point, Sampler::Filter::POINT,
                               Sampler::Address::EDGE),
                 "packed bindless point sample");
    expect_close(host_output[linear_center],
                 sample_packed(sample_center, Sampler::Filter::LINEAR_LINEAR,
                               Sampler::Address::EDGE),
                 "packed bindless trilinear sample at volume center");
    expect_close(host_output[linear_off_center],
                 sample_packed(sample_off_center,
                               Sampler::Filter::LINEAR_LINEAR,
                               Sampler::Address::EDGE),
                 "packed bindless off-center trilinear sample");
    auto mip_value = unpack_r10g10b10a2(packed_mip_texels[0u]);
    expect_close(host_output[mip_read], mip_value,
                 "packed bindless nonzero mip read");
    auto base_value = sample_packed(
        sample_off_center, Sampler::Filter::LINEAR_LINEAR,
        Sampler::Address::EDGE);
    expect_close(host_output[trilinear_mip],
                 base_value + (mip_value - base_value) * 0.5f,
                 "packed bindless fractional mip interpolation");
    expect_close(host_output[linear_mirror],
                 sample_packed(sample_mirror,
                               Sampler::Filter::LINEAR_LINEAR,
                               Sampler::Address::MIRROR),
                 "packed bindless linear mirror address mode");
    expect_close(host_output[point_repeat],
                 sample_packed(sample_outside, Sampler::Filter::POINT,
                               Sampler::Address::REPEAT),
                 "packed bindless repeat address mode");
    expect_close(host_output[point_zero], make_float4(0.0f),
                 "packed bindless zero address mode");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    test_bindless_mip(dc->device);
    // This is the HIP software unpack/filter path; other backends may not
    // expose R10G10B10A2 volume resources at all.
    if (dc->device.backend_name() == "hip") {
        test_bindless_packed_volume(dc->device);
    }
}
