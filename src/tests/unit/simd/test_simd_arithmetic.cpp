#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <luisa/backends/ext/simd_config_ext.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

constexpr auto thread_count = 35u;
constexpr auto output_stride = 7u;

[[nodiscard]] float smoothstep_reference(
    float edge0, float edge1, float x) noexcept {
    auto t = std::clamp(
        (x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * std::fma(-2.0f, t, 3.0f);
}

void expect_close(float4 actual, float4 expected,
                  const char *description) noexcept {
    auto delta = abs(actual - expected);
    auto error = std::max({delta.x, delta.y, delta.z, delta.w});
    expect(static_cast<bool>(error <= 2.0e-6f))
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
        auto output = device.create_buffer<float4>(
            thread_count * output_stride);

        Kernel1D kernel = [width](BufferVar<float4> result) noexcept {
            set_block_size(32u, 1u, 1u);
            set_warp_size(static_cast<uint8_t>(width));
            auto gid = dispatch_x();
            auto x = (cast<float>(gid % 13u) - 6.0f) * 0.375f;
            auto values = make_float4(
                x, x + 0.25f, x - 0.5f, x + 1.0f);
            result.write(
                gid * output_stride,
                smoothstep(-0.75f, 1.25f, values));
            result.write(
                gid * output_stride + 1u,
                smoothstep(
                    make_float4(-1.0f, -0.5f, 0.0f, 0.5f),
                    make_float4(1.0f, 0.75f, 1.5f, 2.5f),
                    values));
            Float uniform_x = 0.25f;
            auto uniform = smoothstep(-1.0f, 2.0f, uniform_x);
            result.write(
                gid * output_stride + 2u,
                make_float4(uniform));
            auto incident = make_float3(
                x, x + 0.25f, 1.0f - 0.125f * x);
            auto normal = make_float3(0.0f, 0.6f, 0.8f);
            result.write(
                gid * output_stride + 3u,
                make_float4(reflect(incident, normal), 0.0f));
            auto matrix2_a = make_float2x2(
                make_float2(x, x + 1.0f),
                make_float2(2.0f, 3.0f));
            auto matrix2_b = make_float2x2(
                make_float2(4.0f, 5.0f),
                make_float2(6.0f, 7.0f));
            auto matrix2_product = matrix2_a * matrix2_b;
            result.write(
                gid * output_stride + 4u,
                make_float4(matrix2_product[0u], matrix2_product[1u]));
            auto matrix3 = make_float3x3(
                make_float3(1.0f, 2.0f, 3.0f),
                make_float3(4.0f, 5.0f, 6.0f),
                make_float3(7.0f, 8.0f, 10.0f));
            result.write(
                gid * output_stride + 5u,
                make_float4(
                    matrix3 * make_float3(x, 2.0f, -1.0f), 0.0f));
            auto matrix4 = make_float4x4(
                make_float4(1.0f, 2.0f, 0.0f, 0.0f),
                make_float4(0.0f, 3.0f, 4.0f, 0.0f),
                make_float4(0.0f, 0.0f, 5.0f, 6.0f),
                make_float4(7.0f, 0.0f, 0.0f, 8.0f));
            result.write(
                gid * output_stride + 6u,
                matrix4 * make_float4(x, 1.0f, 2.0f, -1.0f));
        };

        auto shader = device.compile(kernel);
        luisa::vector<float4> host(thread_count * output_stride);
        stream << shader(output).dispatch(thread_count)
               << output.copy_to(luisa::span{host})
               << synchronize();

        for (auto gid = 0u; gid < thread_count; gid++) {
            auto x = (static_cast<float>(gid % 13u) - 6.0f) * 0.375f;
            auto values = make_float4(
                x, x + 0.25f, x - 0.5f, x + 1.0f);
            auto scalar_edges = make_float4(
                smoothstep_reference(-0.75f, 1.25f, values.x),
                smoothstep_reference(-0.75f, 1.25f, values.y),
                smoothstep_reference(-0.75f, 1.25f, values.z),
                smoothstep_reference(-0.75f, 1.25f, values.w));
            auto vector_edges = make_float4(
                smoothstep_reference(-1.0f, 1.0f, values.x),
                smoothstep_reference(-0.5f, 0.75f, values.y),
                smoothstep_reference(0.0f, 1.5f, values.z),
                smoothstep_reference(0.5f, 2.5f, values.w));
            auto uniform = smoothstep_reference(-1.0f, 2.0f, 0.25f);
            auto incident = make_float3(
                x, x + 0.25f, 1.0f - 0.125f * x);
            auto normal = make_float3(0.0f, 0.6f, 0.8f);
            auto projection = dot(normal, incident);
            auto reflected = incident - 2.0f * projection * normal;
            expect_close(
                host[gid * output_stride], scalar_edges,
                "SIMD smoothstep scalar-edge result mismatch");
            expect_close(
                host[gid * output_stride + 1u], vector_edges,
                "SIMD smoothstep vector-edge result mismatch");
            expect_close(
                host[gid * output_stride + 2u], make_float4(uniform),
                "SIMD smoothstep uniform result mismatch");
            expect_close(
                host[gid * output_stride + 3u],
                make_float4(reflected, 0.0f),
                "SIMD reflect result mismatch");
            expect_close(
                host[gid * output_stride + 4u],
                make_float4(
                    4.0f * x + 10.0f, 4.0f * x + 19.0f,
                    6.0f * x + 14.0f, 6.0f * x + 27.0f),
                "SIMD matrix-matrix multiply mismatch");
            expect_close(
                host[gid * output_stride + 5u],
                make_float4(
                    x + 1.0f, 2.0f * x + 2.0f,
                    3.0f * x + 2.0f, 0.0f),
                "SIMD float3x3-vector multiply mismatch");
            expect_close(
                host[gid * output_stride + 6u],
                make_float4(x - 7.0f, 2.0f * x + 3.0f, 14.0f, 4.0f),
                "SIMD float4x4-vector multiply mismatch");
        }
    }
}
