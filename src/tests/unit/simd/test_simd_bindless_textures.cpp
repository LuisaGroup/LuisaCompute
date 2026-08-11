#include "ut/ut.hpp"

#include <array>
#include <cmath>
#include <limits>

#include <luisa/backends/ext/simd_config_ext.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

constexpr auto thread_count = 35u;
constexpr auto output_stride = 21u;
constexpr auto image_size = make_uint2(4u, 4u);
constexpr auto volume_size = make_uint3(2u, 2u, 2u);

[[nodiscard]] float4 image_texel(
    uint32_t slot, uint32_t x, uint32_t y) noexcept {
    auto base = static_cast<float>(slot * 100u + y * image_size.x + x);
    return make_float4(base + 0.25f, base + 0.5f,
                       base + 0.75f, base + 1.0f);
}

[[nodiscard]] float4 volume_texel(
    uint32_t slot, uint32_t x, uint32_t y, uint32_t z) noexcept {
    auto linear = (z * volume_size.y + y) * volume_size.x + x;
    auto base = static_cast<float>(slot * 200u + linear);
    return make_float4(base + 0.125f, base + 0.25f,
                       base + 0.5f, base + 1.0f);
}

[[nodiscard]] float4 image_mip_texel(
    uint32_t slot, uint32_t x, uint32_t y) noexcept {
    auto base = static_cast<float>(
        slot * 100u + 50u + y * 2u + x);
    return make_float4(base + 0.25f, base + 0.5f,
                       base + 0.75f, base + 1.0f);
}

[[nodiscard]] float4 volume_mip_texel(uint32_t slot) noexcept {
    auto base = static_cast<float>(slot * 200u + 50u);
    return make_float4(base + 0.125f, base + 0.25f,
                       base + 0.5f, base + 1.0f);
}

void expect_close(float4 actual, float4 expected,
                  const char *description) noexcept {
    auto delta = abs(actual - expected);
    auto error = std::max({delta.x, delta.y, delta.z, delta.w});
    expect(static_cast<bool>(error <= 1.0e-6f))
        << description << ", max error " << error;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    Context context{argc > 0 ? argv[0] : ""};

    for (auto width : std::array{1u, 2u, 4u, 8u, 16u}) {
        DeviceConfig config{};
        config.extension =
            luisa::make_unique<SIMDDeviceConfigExt>(width);
        auto device = context.create_device("simd", &config);
        auto stream = device.create_stream();

        std::array<Image<float>, 2u> images{
            device.create_image<float>(PixelStorage::FLOAT4, image_size, 2u),
            device.create_image<float>(PixelStorage::FLOAT4, image_size, 2u)};
        std::array<Volume<float>, 2u> volumes{
            device.create_volume<float>(PixelStorage::FLOAT4, volume_size, 2u),
            device.create_volume<float>(PixelStorage::FLOAT4, volume_size, 2u)};
        std::array<luisa::vector<float4>, 2u> image_pixels;
        std::array<luisa::vector<float4>, 2u> volume_pixels;
        std::array<luisa::vector<float4>, 2u> image_mip_pixels;
        std::array<luisa::vector<float4>, 2u> volume_mip_pixels;
        for (auto slot = 0u; slot < 2u; slot++) {
            image_pixels[slot].resize(image_size.x * image_size.y);
            for (auto y = 0u; y < image_size.y; y++) {
                for (auto x = 0u; x < image_size.x; x++) {
                    image_pixels[slot][y * image_size.x + x] =
                        image_texel(slot, x, y);
                }
            }
            volume_pixels[slot].resize(
                volume_size.x * volume_size.y * volume_size.z);
            for (auto z = 0u; z < volume_size.z; z++) {
                for (auto y = 0u; y < volume_size.y; y++) {
                    for (auto x = 0u; x < volume_size.x; x++) {
                        auto index =
                            (z * volume_size.y + y) * volume_size.x + x;
                        volume_pixels[slot][index] =
                            volume_texel(slot, x, y, z);
                    }
                }
            }
            image_mip_pixels[slot].resize(4u);
            for (auto y = 0u; y < 2u; y++) {
                for (auto x = 0u; x < 2u; x++) {
                    image_mip_pixels[slot][y * 2u + x] =
                        image_mip_texel(slot, x, y);
                }
            }
            volume_mip_pixels[slot] = {volume_mip_texel(slot)};
        }

        auto bindless = device.create_bindless_array(2u);
        bindless.emplace_on_update(
            0u, images[0u], Sampler::point_edge());
        bindless.emplace_on_update(
            1u, images[1u], Sampler::linear_point_mirror());
        bindless.emplace_on_update(
            0u, volumes[0u], Sampler::point_edge());
        bindless.emplace_on_update(
            1u, volumes[1u], Sampler::linear_point_mirror());

        auto output = device.create_buffer<float4>(
            thread_count * output_stride);
        Kernel1D kernel = [width](
                              BindlessVar array,
                              BufferVar<float4> result) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto gid = dispatch_x();
            auto slot = gid & 1u;

            auto x2 = gid & 3u;
            auto y2 = (gid >> 2u) & 3u;
            auto coord2 = make_uint2(x2, y2);
            auto uv = (make_float2(coord2) + 0.5f) / 4.0f;
            auto texture2d = array.tex2d(slot);
            auto size2 = texture2d.size();
            result.write(gid * output_stride + 0u,
                         texture2d.sample(uv));
            result.write(
                gid * output_stride + 1u,
                texture2d.sample(
                    uv, SamplerFilter::POINT,
                    SamplerAddress::EDGE));
            result.write(gid * output_stride + 2u,
                         texture2d.sample(uv, 0.0f));
            result.write(gid * output_stride + 3u,
                         texture2d.read(coord2));
            result.write(
                gid * output_stride + 4u,
                make_float4(make_float2(size2), 0.0f, 0.0f));

            auto x3 = gid & 1u;
            auto y3 = (gid >> 1u) & 1u;
            auto z3 = (gid >> 2u) & 1u;
            auto coord3 = make_uint3(x3, y3, z3);
            auto uvw = (make_float3(coord3) + 0.5f) / 2.0f;
            auto texture3d = array.tex3d(slot);
            auto size3 = texture3d.size();
            result.write(gid * output_stride + 5u,
                         texture3d.sample(uvw));
            result.write(
                gid * output_stride + 6u,
                texture3d.sample(
                    uvw, SamplerFilter::POINT,
                    SamplerAddress::EDGE));
            result.write(gid * output_stride + 7u,
                         texture3d.sample(uvw, 0.0f));
            result.write(gid * output_stride + 8u,
                         texture3d.read(coord3));
            result.write(
                gid * output_stride + 9u,
                make_float4(make_float3(size3), 0.0f));
            auto transformed =
                make_float2x2(2.0f, 0.0f, 0.0f, 3.0f) *
                make_float2(cast<float>(x2), cast<float>(y2));
            result.write(
                gid * output_stride + 10u,
                make_float4(transformed, 0.0f, 0.0f));
            result.write(
                gid * output_stride + 11u,
                make_float4(
                    make_float2(texture2d.size(99u)), 0.0f, 0.0f));
            result.write(
                gid * output_stride + 12u,
                texture2d.read(coord2, 99u));
            result.write(
                gid * output_stride + 13u,
                array.tex2d(0u).sample(make_float2(0.125f)));
            auto coord2_mip = make_uint2(x2 & 1u, y2 & 1u);
            auto uv_mip = (make_float2(coord2_mip) + 0.5f) / 2.0f;
            result.write(
                gid * output_stride + 14u,
                make_float4(
                    make_float2(texture2d.size(1u)), 0.0f, 0.0f));
            result.write(
                gid * output_stride + 15u,
                texture2d.read(coord2_mip, 1u));
            result.write(
                gid * output_stride + 16u,
                texture2d.sample(
                    uv_mip, 1.0f, SamplerFilter::LINEAR_POINT,
                    SamplerAddress::EDGE));
            result.write(
                gid * output_stride + 17u,
                texture3d.sample(
                    make_float3(0.5f), 1.0f,
                    SamplerFilter::LINEAR_POINT,
                    SamplerAddress::EDGE));
            result.write(
                gid * output_stride + 18u,
                texture2d.sample(
                    make_float2(0.0f),
                    std::numeric_limits<float>::quiet_NaN(),
                    SamplerFilter::LINEAR_POINT,
                    SamplerAddress::EDGE));
            result.write(
                gid * output_stride + 19u,
                texture2d.sample(
                    make_float2(0.0f),
                    -std::numeric_limits<float>::infinity(),
                    SamplerFilter::LINEAR_POINT,
                    SamplerAddress::EDGE));
            result.write(
                gid * output_stride + 20u,
                texture2d.sample(
                    make_float2(0.0f),
                    std::numeric_limits<float>::infinity(),
                    SamplerFilter::LINEAR_POINT,
                    SamplerAddress::EDGE));
        };

        auto shader = device.compile(kernel);
        luisa::vector<float4> host(thread_count * output_stride);
        stream << images[0u].copy_from(luisa::span{image_pixels[0u]})
               << images[1u].copy_from(luisa::span{image_pixels[1u]})
               << images[0u].view(1u).copy_from(
                      luisa::span{image_mip_pixels[0u]})
               << images[1u].view(1u).copy_from(
                      luisa::span{image_mip_pixels[1u]})
               << volumes[0u].copy_from(luisa::span{volume_pixels[0u]})
               << volumes[1u].copy_from(luisa::span{volume_pixels[1u]})
               << volumes[0u].view(1u).copy_from(
                      luisa::span{volume_mip_pixels[0u]})
               << volumes[1u].view(1u).copy_from(
                      luisa::span{volume_mip_pixels[1u]})
               << bindless.update()
               << shader(bindless, output).dispatch(thread_count)
               << output.copy_to(luisa::span{host})
               << synchronize();

        for (auto gid = 0u; gid < thread_count; gid++) {
            auto slot = gid & 1u;
            auto x2 = gid & 3u;
            auto y2 = (gid >> 2u) & 3u;
            auto expected2 = image_texel(slot, x2, y2);
            for (auto field = 0u; field < 4u; field++) {
                expect_close(
                    host[gid * output_stride + field], expected2,
                    "SIMD bindless 2D texture result mismatch");
            }
            expect_close(
                host[gid * output_stride + 4u],
                make_float4(4.0f, 4.0f, 0.0f, 0.0f),
                "SIMD bindless 2D texture size mismatch");

            auto x3 = gid & 1u;
            auto y3 = (gid >> 1u) & 1u;
            auto z3 = (gid >> 2u) & 1u;
            auto expected3 = volume_texel(slot, x3, y3, z3);
            for (auto field = 5u; field < 9u; field++) {
                expect_close(
                    host[gid * output_stride + field], expected3,
                    "SIMD bindless 3D texture result mismatch");
            }
            expect_close(
                host[gid * output_stride + 9u],
                make_float4(2.0f, 2.0f, 2.0f, 0.0f),
                "SIMD bindless 3D texture size mismatch");
            expect_close(
                host[gid * output_stride + 10u],
                make_float4(
                    static_cast<float>(x2) * 2.0f,
                    static_cast<float>(y2) * 3.0f,
                    0.0f, 0.0f),
                "SIMD matrix-vector multiply mismatch");
            expect_close(
                host[gid * output_stride + 11u],
                make_float4(1.0f, 1.0f, 0.0f, 0.0f),
                "SIMD bindless 2D out-of-range mip size mismatch");
            expect_close(
                host[gid * output_stride + 12u], make_float4(0.0f),
                "SIMD bindless 2D out-of-range mip read mismatch");
            expect_close(
                host[gid * output_stride + 13u],
                image_texel(0u, 0u, 0u),
                "SIMD uniform bindless texture sample mismatch");
            auto x2_mip = x2 & 1u;
            auto y2_mip = y2 & 1u;
            expect_close(
                host[gid * output_stride + 14u],
                make_float4(2.0f, 2.0f, 0.0f, 0.0f),
                "SIMD bindless 2D mip size mismatch");
            expect_close(
                host[gid * output_stride + 15u],
                image_mip_texel(slot, x2_mip, y2_mip),
                "SIMD bindless 2D mip read mismatch");
            expect_close(
                host[gid * output_stride + 16u],
                image_mip_texel(slot, x2_mip, y2_mip),
                "SIMD bindless 2D mip sample mismatch");
            expect_close(
                host[gid * output_stride + 17u],
                volume_mip_texel(slot),
                "SIMD bindless 3D mip sample mismatch");
            expect_close(
                host[gid * output_stride + 18u],
                image_texel(slot, 0u, 0u),
                "SIMD bindless NaN mip sample mismatch");
            expect_close(
                host[gid * output_stride + 19u],
                image_texel(slot, 0u, 0u),
                "SIMD bindless negative-infinity mip sample mismatch");
            expect_close(
                host[gid * output_stride + 20u],
                image_mip_texel(slot, 0u, 0u),
                "SIMD bindless positive-infinity mip sample mismatch");
        }
    }
}
