// Deterministic direct 2D texture sampling conformance test.
//
// The public DSL intentionally exposes sampled textures through bindless
// arrays, while the C++ shader frontend emits the direct texture CallOps.
// This test constructs those calls with FunctionBuilder so the production
// AST -> XIR -> backend path is tested without depending on clangcxx.
// It checks surface reads, view-relative sizes, spatial/address filtering,
// explicit fractional LOD, gradient LOD, minimum-LOD clamping, and a bound
// nonzero base mip against an independent CPU oracle. It also forces one
// texture argument through sampled-image and storage-image paths in one kernel.

#include "ut/ut.hpp"
#include "test_device.h"

#include <algorithm>
#include <array>
#include <cmath>

#include <luisa/ast/function_builder.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

constexpr auto base_size = 8u;
constexpr auto level_count = 4u;
constexpr auto result_count_per_view = 9u;
constexpr auto divergent_lane_count = 64u;

enum Result : uint32_t {
    surface_read,
    view_size,
    implicit_point_repeat,
    explicit_point_level,
    explicit_trilinear,
    gradient_trilinear,
    gradient_minimum_level,
    explicit_mip_point,
    linear_zero_border
};

[[nodiscard]] uint32_t mip_size(uint32_t level) noexcept {
    return std::max(base_size >> level, 1u);
}

[[nodiscard]] float4 texel(uint32_t level, uint32_t x,
                           uint32_t y) noexcept {
    auto v = static_cast<float>(level) * 0.2f +
             static_cast<float>(x) * 0.01f +
             static_cast<float>(y) * 0.02f;
    return make_float4(
        v + 0.01f,
        v * 0.5f + 0.03f,
        static_cast<float>(level) * 0.1f +
            static_cast<float>(x) * 0.005f +
            static_cast<float>(y) * 0.003f,
        1.0f);
}

[[nodiscard]] auto make_level_pixels(uint32_t level) noexcept {
    auto size = mip_size(level);
    luisa::vector<float4> pixels(static_cast<size_t>(size) * size);
    for (auto y = 0u; y < size; y++) {
        for (auto x = 0u; x < size; x++) {
            pixels[static_cast<size_t>(y) * size + x] =
                texel(level, x, y);
        }
    }
    return pixels;
}

[[nodiscard]] int wrap(int x, int size) noexcept {
    auto r = x % size;
    return r < 0 ? r + size : r;
}

[[nodiscard]] float4 fetch(uint32_t level, int x, int y,
                           Sampler::Address address) noexcept {
    auto size = static_cast<int>(mip_size(level));
    switch (address) {
        case Sampler::Address::EDGE:
            x = std::clamp(x, 0, size - 1);
            y = std::clamp(y, 0, size - 1);
            break;
        case Sampler::Address::REPEAT:
            x = wrap(x, size);
            y = wrap(y, size);
            break;
        case Sampler::Address::ZERO:
            if (x < 0 || x >= size || y < 0 || y >= size) {
                return make_float4(0.0f);
            }
            break;
        case Sampler::Address::MIRROR:
            LUISA_ERROR_WITH_LOCATION(
                "Mirror addressing is not used by this oracle.");
    }
    return texel(level, static_cast<uint32_t>(x),
                 static_cast<uint32_t>(y));
}

[[nodiscard]] float4 sample_spatial(
    uint32_t level, float2 uv, Sampler::Filter filter,
    Sampler::Address address) noexcept {
    auto size = static_cast<float>(mip_size(level));
    if (filter == Sampler::Filter::POINT) {
        return fetch(
            level,
            static_cast<int>(std::floor(uv.x * size)),
            static_cast<int>(std::floor(uv.y * size)),
            address);
    }
    auto x = uv.x * size - 0.5f;
    auto y = uv.y * size - 0.5f;
    auto x0 = static_cast<int>(std::floor(x));
    auto y0 = static_cast<int>(std::floor(y));
    auto tx = x - static_cast<float>(x0);
    auto ty = y - static_cast<float>(y0);
    auto v00 = fetch(level, x0, y0, address);
    auto v10 = fetch(level, x0 + 1, y0, address);
    auto v01 = fetch(level, x0, y0 + 1, address);
    auto v11 = fetch(level, x0 + 1, y0 + 1, address);
    auto vx0 = v00 + (v10 - v00) * tx;
    auto vx1 = v01 + (v11 - v01) * tx;
    return vx0 + (vx1 - vx0) * ty;
}

[[nodiscard]] float4 sample_lod(
    uint32_t base_level, float2 uv, float lod,
    Sampler::Filter filter, Sampler::Address address) noexcept {
    auto max_level = level_count - base_level - 1u;
    lod = std::clamp(lod, 0.0f, static_cast<float>(max_level));
    if (filter == Sampler::Filter::LINEAR_LINEAR ||
        filter == Sampler::Filter::ANISOTROPIC) {
        auto relative0 = static_cast<uint32_t>(std::floor(lod));
        auto relative1 = std::min(relative0 + 1u, max_level);
        auto t = lod - static_cast<float>(relative0);
        auto v0 = sample_spatial(
            base_level + relative0, uv, filter, address);
        auto v1 = sample_spatial(
            base_level + relative1, uv, filter, address);
        return v0 + (v1 - v0) * t;
    }
    auto relative = static_cast<uint32_t>(std::floor(lod + 0.5f));
    return sample_spatial(
        base_level + relative, uv, filter, address);
}

[[nodiscard]] float gradient_lod(
    uint32_t base_level, float2 ddx, float2 ddy) noexcept {
    auto size = static_cast<float>(mip_size(base_level));
    auto dx = ddx * size;
    auto dy = ddy * size;
    auto rho2 = std::max(dot(dx, dx), dot(dy, dy));
    return 0.5f * std::log2(std::max(rho2, 1.0f));
}

void expect_close(float4 actual, float4 expected,
                  uint32_t base_level,
                  const char *description) noexcept {
    constexpr auto epsilon = 1.0e-3f;
    auto delta = abs(actual - expected);
    auto error = std::max({delta.x, delta.y, delta.z, delta.w});
    expect(static_cast<bool>(error < epsilon))
        << description << " for base mip " << base_level
        << ", max error " << error;
}

}// namespace

void test_direct_texture_sampling(Device &device) {
    auto texture = device.create_image<float>(
        PixelStorage::FLOAT4, make_uint2(base_size), level_count);
    auto output = device.create_buffer<float4>(
        result_count_per_view * 2u);
    luisa::vector<float4> host(output.size());

    auto stream = device.create_stream();
    for (auto level = 0u; level < level_count; level++) {
        auto pixels = make_level_pixels(level);
        stream << texture.view(level).copy_from(luisa::span{pixels});
    }

    constexpr auto uv = make_float2(0.31f, 0.43f);
    constexpr auto address_uv = make_float2(-0.10f, 1.10f);
    // Grid-exact fractions isolate ZERO-address blending from the native
    // sampler's finite interpolation precision on both tested mip sizes.
    constexpr auto border_uv = make_float2(-1.0f / 32.0f, 3.0f / 32.0f);
    constexpr auto gradient = make_float2(0.3535533905932738f, 0.0f);
    constexpr auto gradient_y = make_float2(0.0f, 0.3535533905932738f);
    constexpr auto base_gradient = make_float2(1.0f / base_size, 0.0f);
    constexpr auto base_gradient_y = make_float2(0.0f, 1.0f / base_size);

    Kernel1D sample_kernel = [&](ImageFloat image, BufferFloat4 results,
                                 UInt result_base) noexcept {
        auto &builder =
            *luisa::compute::detail::FunctionBuilder::current();
        auto texture_expression = image.expression();
        auto literal = [&](auto value) noexcept {
            return builder.literal(Type::of<decltype(value)>(), value);
        };
        auto filter = [&](Sampler::Filter value) noexcept {
            return literal(static_cast<uint32_t>(value));
        };
        auto address = [&](Sampler::Address value) noexcept {
            return literal(static_cast<uint32_t>(value));
        };
        auto write_call = [&](uint32_t index,
                              const Expression *value) noexcept {
            results.write(result_base + index, def<float4>(value));
        };

        results.write(result_base + static_cast<uint32_t>(surface_read),
                      image.read(make_uint2(1u, 2u)));
        auto size = image.size();
        results.write(
            result_base + static_cast<uint32_t>(view_size),
            make_float4(cast<float>(size.x), cast<float>(size.y),
                        0.0f, 0.0f));

        write_call(
            implicit_point_repeat,
            builder.call(
                Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE,
                {texture_expression, literal(address_uv),
                 filter(Sampler::Filter::POINT),
                 address(Sampler::Address::REPEAT)}));
        write_call(
            explicit_point_level,
            builder.call(
                Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE_LEVEL,
                {texture_expression, literal(uv), literal(2.0f),
                 filter(Sampler::Filter::POINT),
                 address(Sampler::Address::EDGE)}));
        write_call(
            explicit_trilinear,
            builder.call(
                Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE_LEVEL,
                {texture_expression, literal(uv), literal(1.25f),
                 filter(Sampler::Filter::LINEAR_LINEAR),
                 address(Sampler::Address::EDGE)}));
        write_call(
            gradient_trilinear,
            builder.call(
                Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE_GRAD,
                {texture_expression, literal(uv), literal(gradient),
                 literal(gradient_y),
                 filter(Sampler::Filter::LINEAR_LINEAR),
                 address(Sampler::Address::EDGE)}));
        write_call(
            gradient_minimum_level,
            builder.call(
                Type::of<float4>(),
                CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL,
                {texture_expression, literal(uv), literal(base_gradient),
                 literal(base_gradient_y), literal(1.25f),
                 filter(Sampler::Filter::LINEAR_LINEAR),
                 address(Sampler::Address::EDGE)}));
        write_call(
            explicit_mip_point,
            builder.call(
                Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE_LEVEL,
                {texture_expression, literal(uv), literal(1.75f),
                 filter(Sampler::Filter::LINEAR_POINT),
                 address(Sampler::Address::EDGE)}));
        write_call(
            linear_zero_border,
            builder.call(
                Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE_LEVEL,
                {texture_expression, literal(border_uv), literal(0.0f),
                 filter(Sampler::Filter::LINEAR_LINEAR),
                 address(Sampler::Address::ZERO)}));
    };

    auto shader = device.compile(sample_kernel);
    stream << shader(texture.view(0u), output, 0u).dispatch(1u)
           << shader(texture.view(1u), output,
                     result_count_per_view)
                  .dispatch(1u)
           << output.copy_to(luisa::span{host})
           << synchronize();

    for (auto base_level : {0u, 1u}) {
        auto base = base_level * result_count_per_view;
        expect_close(host[base + surface_read],
                     texel(base_level, 1u, 2u), base_level,
                     "surface read");
        auto size = static_cast<float>(mip_size(base_level));
        expect_close(host[base + view_size],
                     make_float4(size, size, 0.0f, 0.0f),
                     base_level, "view-relative size");
        expect_close(
            host[base + implicit_point_repeat],
            sample_lod(base_level, address_uv, 0.0f,
                       Sampler::Filter::POINT,
                       Sampler::Address::REPEAT),
            base_level, "implicit point/repeat sampling");
        expect_close(
            host[base + explicit_point_level],
            sample_lod(base_level, uv, 2.0f,
                       Sampler::Filter::POINT,
                       Sampler::Address::EDGE),
            base_level, "explicit point LOD");
        expect_close(
            host[base + explicit_trilinear],
            sample_lod(base_level, uv, 1.25f,
                       Sampler::Filter::LINEAR_LINEAR,
                       Sampler::Address::EDGE),
            base_level, "fractional trilinear LOD");
        auto lod = gradient_lod(base_level, gradient, gradient_y);
        expect_close(
            host[base + gradient_trilinear],
            sample_lod(base_level, uv, lod,
                       Sampler::Filter::LINEAR_LINEAR,
                       Sampler::Address::EDGE),
            base_level, "gradient-derived LOD");
        auto minimum_lod = std::max(
            gradient_lod(base_level, base_gradient,
                         base_gradient_y),
            1.25f);
        expect_close(
            host[base + gradient_minimum_level],
            sample_lod(base_level, uv, minimum_lod,
                       Sampler::Filter::LINEAR_LINEAR,
                       Sampler::Address::EDGE),
            base_level, "minimum gradient LOD");
        expect_close(
            host[base + explicit_mip_point],
            sample_lod(base_level, uv, 1.75f,
                       Sampler::Filter::LINEAR_POINT,
                       Sampler::Address::EDGE),
            base_level, "point mip with linear spatial filtering");
        expect_close(
            host[base + linear_zero_border],
            sample_lod(base_level, border_uv, 0.0f,
                       Sampler::Filter::LINEAR_LINEAR,
                       Sampler::Address::ZERO),
            base_level, "linear zero-border sampling");
    }

    // Direct texture descriptors are immutable and may be loaded through the
    // AMDGPU constant address space. Keep the image and sampler indices
    // lane-dependent here to ensure that only uniform descriptor fields are
    // scalarized: explicit mip selection and runtime-combined samplers must
    // continue to follow each lane independently.
    auto divergent_output = device.create_buffer<float4>(
        divergent_lane_count * 2u);
    luisa::vector<float4> divergent_host(divergent_output.size());
    constexpr auto divergent_uv = make_float2(-0.10f, 1.10f);
    Kernel1D divergent_kernel = [&](ImageFloat image, BufferFloat4 results,
                                    UInt result_base) noexcept {
        auto lane = dispatch_id().x;
        auto lod = cast<float>(lane & 1u);
        auto dynamic_filter = ite(
            (lane & 2u) != 0u,
            static_cast<uint32_t>(Sampler::Filter::LINEAR_POINT),
            static_cast<uint32_t>(Sampler::Filter::POINT));
        auto dynamic_address = ite(
            (lane & 4u) != 0u,
            static_cast<uint32_t>(Sampler::Address::REPEAT),
            static_cast<uint32_t>(Sampler::Address::EDGE));
        auto &builder =
            *luisa::compute::detail::FunctionBuilder::current();
        auto sampled = builder.call(
            Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE_LEVEL,
            {image.expression(),
             builder.literal(Type::of<float2>(), divergent_uv),
             lod.expression(), dynamic_filter.expression(),
             dynamic_address.expression()});
        results.write(result_base + lane, def<float4>(sampled));
    };
    auto divergent_shader = device.compile(divergent_kernel);
    stream << divergent_shader(texture.view(0u), divergent_output, 0u)
                  .dispatch(divergent_lane_count)
           << divergent_shader(texture.view(1u), divergent_output,
                               divergent_lane_count)
                  .dispatch(divergent_lane_count)
           << divergent_output.copy_to(luisa::span{divergent_host})
           << synchronize();

    for (auto base_level : {0u, 1u}) {
        auto result_base = base_level * divergent_lane_count;
        for (auto lane = 0u; lane < divergent_lane_count; lane++) {
            auto lod = static_cast<float>(lane & 1u);
            auto filter = (lane & 2u) != 0u ?
                              Sampler::Filter::LINEAR_POINT :
                              Sampler::Filter::POINT;
            auto address = (lane & 4u) != 0u ?
                               Sampler::Address::REPEAT :
                               Sampler::Address::EDGE;
            expect_close(
                divergent_host[result_base + lane],
                sample_lod(base_level, divergent_uv, lod, filter, address),
                base_level, "lane-divergent mip/sampler selection");
        }
    }

    // A read/write texture needs two descriptor roles: sampling is illegal on
    // a storage image, while image writes are illegal on a sampled image. The
    // two views deliberately overlap and therefore use GENERAL layout. The
    // following result buffer makes a one-vs-two descriptor count error
    // observable instead of merely relying on validation diagnostics.
    constexpr auto read_write_base_level = 1u;
    constexpr auto read_write_uv = make_float2(1.5f / 4.0f, 2.5f / 4.0f);
    constexpr auto write_delta = make_float4(0.125f, 0.25f, 0.5f, 0.0f);
    auto read_write_output = device.create_buffer<float4>(1u);
    Kernel1D read_write_kernel = [&](ImageFloat image,
                                     BufferFloat4 result) noexcept {
        auto &builder =
            *luisa::compute::detail::FunctionBuilder::current();
        auto literal = [&](auto value) noexcept {
            return builder.literal(Type::of<decltype(value)>(), value);
        };
        auto sampled_expression = builder.call(
            Type::of<float4>(), CallOp::TEXTURE2D_SAMPLE_LEVEL,
            {image.expression(), literal(read_write_uv), literal(0.0f),
             literal(static_cast<uint32_t>(Sampler::Filter::POINT)),
             literal(static_cast<uint32_t>(Sampler::Address::EDGE))});
        auto sampled = def<float4>(sampled_expression);
        result.write(0u, sampled);
        image.write(make_uint2(0u), sampled + write_delta);
    };
    auto read_write_shader = device.compile(read_write_kernel);
    std::array<float4, 1u> sampled_result{};
    auto written_level = make_level_pixels(read_write_base_level);
    stream << read_write_shader(
                  texture.view(read_write_base_level), read_write_output)
                  .dispatch(1u)
           << read_write_output.copy_to(luisa::span{sampled_result})
           << texture.view(read_write_base_level)
                  .copy_to(luisa::span{written_level})
           << synchronize();
    auto expected_sample = texel(read_write_base_level, 1u, 2u);
    expect_close(sampled_result[0], expected_sample,
                 read_write_base_level,
                 "combined sampled/storage texture read");
    expect_close(written_level[0], expected_sample + write_delta,
                 read_write_base_level,
                 "combined sampled/storage texture write");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    test_direct_texture_sampling(dc->device);
}
