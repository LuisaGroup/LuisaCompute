#include "ut/ut.hpp"

#include <array>
#include <limits>

#include <luisa/ast/function_builder.h>
#include <luisa/backends/ext/simd_config_ext.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

constexpr auto thread_count = 35u;
constexpr auto output_stride = 30u;
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
            0u, images[0u], Sampler::linear_point_edge());
        bindless.emplace_on_update(
            1u, images[1u], Sampler::linear_point_mirror());
        bindless.emplace_on_update(
            0u, volumes[0u], Sampler::linear_point_edge());
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
            auto ddx2 = make_float2(0.5f, 0.0f);
            auto ddy2 = make_float2(0.0f, 0.5f);
            result.write(
                gid * output_stride + 21u,
                texture2d.sample(uv_mip, ddx2, ddy2));
            auto base_ddx2 = make_float2(0.25f, 0.0f);
            auto base_ddy2 = make_float2(0.0f, 0.25f);
            result.write(
                gid * output_stride + 22u,
                texture2d.sample(
                    uv_mip, base_ddx2, base_ddy2, 1.0f,
                    SamplerFilter::LINEAR_POINT,
                    SamplerAddress::EDGE));
            auto ddx3 = make_float3(1.0f, 0.0f, 0.0f);
            auto ddy3 = make_float3(0.0f, 1.0f, 0.0f);
            result.write(
                gid * output_stride + 23u,
                texture3d.sample(make_float3(0.5f), ddx3, ddy3));
            auto base_ddx3 = make_float3(0.5f, 0.0f, 0.0f);
            auto base_ddy3 = make_float3(0.0f, 0.5f, 0.0f);
            result.write(
                gid * output_stride + 24u,
                texture3d.sample(
                    make_float3(0.5f), base_ddx3, base_ddy3, 1.0f,
                    SamplerFilter::LINEAR_POINT,
                    SamplerAddress::EDGE));
            auto zero_gradient = make_float2(0.0f);
            result.write(
                gid * output_stride + 25u,
                texture2d.sample(uv, zero_gradient, zero_gradient));
            auto nan_gradient = make_float2(
                std::numeric_limits<float>::quiet_NaN(), 0.0f);
            result.write(
                gid * output_stride + 26u,
                texture2d.sample(uv, nan_gradient, zero_gradient));
            auto inf_gradient = make_float2(
                std::numeric_limits<float>::infinity(), 0.0f);
            result.write(
                gid * output_stride + 27u,
                texture2d.sample(uv_mip, inf_gradient, zero_gradient));
            auto large_gradient = make_float2(1.0f, 0.0f);
            result.write(
                gid * output_stride + 28u,
                texture2d.sample(uv, nan_gradient, large_gradient));
            result.write(
                gid * output_stride + 29u,
                array.tex2d(0u).sample(uv_mip, ddx2, ddy2));
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
            expect_close(
                host[gid * output_stride + 21u],
                image_mip_texel(slot, x2_mip, y2_mip),
                "SIMD bindless 2D gradient sample mismatch");
            expect_close(
                host[gid * output_stride + 22u],
                image_mip_texel(slot, x2_mip, y2_mip),
                "SIMD bindless 2D minimum-mip gradient mismatch");
            expect_close(
                host[gid * output_stride + 23u],
                volume_mip_texel(slot),
                "SIMD bindless 3D gradient sample mismatch");
            expect_close(
                host[gid * output_stride + 24u],
                volume_mip_texel(slot),
                "SIMD bindless 3D minimum-mip gradient mismatch");
            expect_close(
                host[gid * output_stride + 25u], expected2,
                "SIMD bindless zero-gradient sample mismatch");
            expect_close(
                host[gid * output_stride + 26u], expected2,
                "SIMD bindless NaN-gradient sample mismatch");
            expect_close(
                host[gid * output_stride + 27u],
                image_mip_texel(slot, x2_mip, y2_mip),
                "SIMD bindless infinite-gradient sample mismatch");
            expect_close(
                host[gid * output_stride + 28u], expected2,
                "SIMD bindless mixed-NaN gradient sample mismatch");
            expect_close(
                host[gid * output_stride + 29u],
                image_mip_texel(0u, x2_mip, y2_mip),
                "SIMD bindless uniform-gradient LOD sample mismatch");
        }

        // Direct sampled-image CallOps use the same packet sampler, but the
        // resource descriptor carries a bound base mip and an explicit
        // per-lane sampler. Exercise all eight 2D/3D variants at every
        // runtime width, including the three-lane W16 tail.
        constexpr auto direct_stride = 9u;
        auto direct_output = device.create_buffer<float4>(
            thread_count * direct_stride);
        Kernel1D direct_kernel = [width, direct_stride](
                                     ImageFloat image,
                                     VolumeFloat volume,
                                     BufferVar<float4> result) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto gid = dispatch_x();
            auto parity = gid & 1u;
            auto dynamic_filter = ite(
                parity != 0u,
                static_cast<uint32_t>(Sampler::Filter::LINEAR_POINT),
                static_cast<uint32_t>(Sampler::Filter::POINT));
            auto dynamic_address = ite(
                (gid & 2u) != 0u,
                static_cast<uint32_t>(Sampler::Address::REPEAT),
                static_cast<uint32_t>(Sampler::Address::EDGE));

            auto x2 = gid & 3u;
            auto y2 = (gid >> 2u) & 3u;
            auto coord2 = make_uint2(x2, y2);
            auto uv2 = (make_float2(coord2) + 0.5f) / 4.0f;
            auto mip_coord2 = make_uint2(x2 & 1u, y2 & 1u);
            auto mip_uv2 =
                (make_float2(mip_coord2) + 0.5f) / 2.0f;
            auto varying_level = cast<float>(parity);
            auto gradient_scale2 =
                0.5f + cast<float>(parity) * 0.25f;
            auto ddx2 = make_float2(gradient_scale2, 0.0f);
            auto ddy2 = make_float2(0.0f, gradient_scale2);
            auto minimum_gradient_scale2 =
                cast<float>(parity) * 0.125f;
            auto minimum_ddx2 = make_float2(
                minimum_gradient_scale2, 0.0f);
            auto minimum_ddy2 = make_float2(
                0.0f, minimum_gradient_scale2);

            auto x3 = gid & 1u;
            auto y3 = (gid >> 1u) & 1u;
            auto z3 = (gid >> 2u) & 1u;
            auto coord3 = make_uint3(x3, y3, z3);
            auto uvw3 = (make_float3(coord3) + 0.5f) / 2.0f;
            auto gradient_scale3 =
                1.0f + cast<float>(parity) * 0.5f;
            auto ddx3 = make_float3(gradient_scale3, 0.0f, 0.0f);
            auto ddy3 = make_float3(0.0f, gradient_scale3, 0.0f);
            auto minimum_gradient_scale3 =
                cast<float>(parity) * 0.25f;
            auto minimum_ddx3 = make_float3(
                minimum_gradient_scale3, 0.0f, 0.0f);
            auto minimum_ddy3 = make_float3(
                0.0f, minimum_gradient_scale3, 0.0f);

            auto &builder =
                *luisa::compute::detail::FunctionBuilder::current();
            auto literal = [&](auto value) noexcept {
                return builder.literal(
                    Type::of<decltype(value)>(), value);
            };
            auto write_sample = [&](uint32_t field,
                                    const Expression *sample) noexcept {
                result.write(
                    gid * direct_stride + field,
                    def<float4>(sample));
            };
            write_sample(
                0u,
                builder.call(
                    Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE,
                    {image.expression(), uv2.expression(),
                     dynamic_filter.expression(),
                     dynamic_address.expression()}));
            write_sample(
                1u,
                builder.call(
                    Type::of<float4>(),
                    CallOp::TEXTURE2D_SAMPLE_LEVEL,
                    {image.expression(), mip_uv2.expression(),
                     varying_level.expression(),
                     literal(static_cast<uint32_t>(
                         Sampler::Filter::POINT)),
                     dynamic_address.expression()}));
            write_sample(
                2u,
                builder.call(
                    Type::of<float4>(),
                    CallOp::TEXTURE2D_SAMPLE_GRAD,
                    {image.expression(), mip_uv2.expression(),
                     ddx2.expression(), ddy2.expression(),
                     literal(static_cast<uint32_t>(
                         Sampler::Filter::LINEAR_POINT)),
                     dynamic_address.expression()}));
            write_sample(
                3u,
                builder.call(
                    Type::of<float4>(),
                    CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL,
                    {image.expression(), mip_uv2.expression(),
                     minimum_ddx2.expression(),
                     minimum_ddy2.expression(), literal(1.0f),
                     literal(static_cast<uint32_t>(
                         Sampler::Filter::LINEAR_POINT)),
                     dynamic_address.expression()}));
            write_sample(
                4u,
                builder.call(
                    Type::of<float4>(), CallOp::TEXTURE3D_SAMPLE,
                    {volume.expression(), uvw3.expression(),
                     dynamic_filter.expression(),
                     dynamic_address.expression()}));
            write_sample(
                5u,
                builder.call(
                    Type::of<float4>(),
                    CallOp::TEXTURE3D_SAMPLE_LEVEL,
                    {volume.expression(), literal(make_float3(0.5f)),
                     literal(1.0f),
                     literal(static_cast<uint32_t>(
                         Sampler::Filter::LINEAR_POINT)),
                     dynamic_address.expression()}));
            write_sample(
                6u,
                builder.call(
                    Type::of<float4>(),
                    CallOp::TEXTURE3D_SAMPLE_GRAD,
                    {volume.expression(), literal(make_float3(0.5f)),
                     ddx3.expression(), ddy3.expression(),
                     literal(static_cast<uint32_t>(
                         Sampler::Filter::LINEAR_POINT)),
                     dynamic_address.expression()}));
            write_sample(
                7u,
                builder.call(
                    Type::of<float4>(),
                    CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL,
                    {volume.expression(), literal(make_float3(0.5f)),
                     minimum_ddx3.expression(),
                     minimum_ddy3.expression(), literal(1.0f),
                     literal(static_cast<uint32_t>(
                         Sampler::Filter::LINEAR_POINT)),
                     dynamic_address.expression()}));
            write_sample(
                8u,
                builder.call(
                    Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE,
                    {image.expression(), literal(make_float2(0.125f)),
                     literal(static_cast<uint32_t>(
                         Sampler::Filter::POINT)),
                     literal(static_cast<uint32_t>(
                         Sampler::Address::EDGE))}));
        };
        auto direct_shader = device.compile(direct_kernel);
        luisa::vector<float4> direct_host(
            thread_count * direct_stride);
        stream << direct_shader(
                      images[0u], volumes[0u], direct_output)
                      .dispatch(thread_count)
               << direct_output.copy_to(luisa::span{direct_host})
               << synchronize();
        for (auto gid = 0u; gid < thread_count; gid++) {
            auto x2 = gid & 3u;
            auto y2 = (gid >> 2u) & 3u;
            auto mip_x2 = x2 & 1u;
            auto mip_y2 = y2 & 1u;
            expect_close(
                direct_host[gid * direct_stride],
                image_texel(0u, x2, y2),
                "SIMD direct 2D implicit sample mismatch");
            auto expected_level2 = (gid & 1u) == 0u ?
                                       image_texel(
                                           0u, mip_x2 * 2u + 1u,
                                           mip_y2 * 2u + 1u) :
                                       image_mip_texel(
                                           0u, mip_x2, mip_y2);
            expect_close(
                direct_host[gid * direct_stride + 1u],
                expected_level2,
                "SIMD direct 2D varying LOD sample mismatch");
            for (auto field : {2u, 3u}) {
                expect_close(
                    direct_host[gid * direct_stride + field],
                    image_mip_texel(0u, mip_x2, mip_y2),
                    "SIMD direct 2D gradient sample mismatch");
            }
            auto x3 = gid & 1u;
            auto y3 = (gid >> 1u) & 1u;
            auto z3 = (gid >> 2u) & 1u;
            expect_close(
                direct_host[gid * direct_stride + 4u],
                volume_texel(0u, x3, y3, z3),
                "SIMD direct 3D implicit sample mismatch");
            for (auto field : {5u, 6u, 7u}) {
                expect_close(
                    direct_host[gid * direct_stride + field],
                    volume_mip_texel(0u),
                    "SIMD direct 3D mip sample mismatch");
            }
            expect_close(
                direct_host[gid * direct_stride + 8u],
                image_texel(0u, 0u, 0u),
                "SIMD uniform direct sample mismatch");
        }

        // The common uniform-slot BYTE1/mip-zero/stored-mirror path executes
        // directly in fixed-vector JIT IR. Keep an explicit-sampler packet
        // call beside it as an independent runtime oracle, including a
        // partial dispatch tail and non-finite coordinates.
        constexpr auto byte_image_size = make_uint2(7u, 5u);
        luisa::vector<uint8_t> byte_pixels(
            byte_image_size.x * byte_image_size.y);
        for (auto i = 0u; i < byte_pixels.size(); i++) {
            byte_pixels[i] = static_cast<uint8_t>(i * 37u + 11u);
        }
        auto byte_image = device.create_image<float>(
            PixelStorage::BYTE1, byte_image_size);
        auto byte_bindless = device.create_bindless_array(1u);
        byte_bindless.emplace_on_update(
            0u, byte_image, Sampler::linear_point_mirror());
        auto byte_output = device.create_buffer<float4>(
            thread_count * 2u);
        Kernel1D byte_kernel = [width](
                                   BindlessVar array,
                                   BufferVar<float4> result) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto gid = dispatch_x();
            auto lane = cast<float>(gid);
            auto u = lane * 0.371f - 3.25f;
            auto v = lane * -0.217f + 1.75f;
            u = ite(
                gid == 0u,
                std::numeric_limits<float>::quiet_NaN(), u);
            u = ite(
                gid == 1u,
                std::numeric_limits<float>::infinity(), u);
            v = ite(
                gid == 2u,
                -std::numeric_limits<float>::infinity(), v);
            u = ite(
                gid == 3u,
                std::numeric_limits<float>::max(), u);
            v = ite(
                gid == 4u,
                std::numeric_limits<float>::lowest(), v);
            auto uv = make_float2(u, v);
            auto texture = array.tex2d(0u);
            result.write(gid * 2u, texture.sample(uv));
            result.write(
                gid * 2u + 1u,
                texture.sample(
                    uv, SamplerFilter::LINEAR_POINT,
                    SamplerAddress::MIRROR));
        };
        auto byte_shader = device.compile(byte_kernel);
        luisa::vector<float4> byte_host(thread_count * 2u);
        stream << byte_image.copy_from(luisa::span{byte_pixels})
               << byte_bindless.update()
               << byte_shader(byte_bindless, byte_output)
                      .dispatch(thread_count)
               << byte_output.copy_to(luisa::span{byte_host})
               << synchronize();
        for (auto gid = 0u; gid < thread_count; gid++) {
            expect_close(
                byte_host[gid * 2u], byte_host[gid * 2u + 1u],
                "SIMD IR-native BYTE1 texture sample mismatch");
        }
    }
}
