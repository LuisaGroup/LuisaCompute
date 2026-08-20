// Test for SIMD Schedule arithmetic lowering.
// This test covers:
// - uniform and varying scalar/vector arithmetic at W1/W2/W4/W8/W16
// - inactive tails and matrix/vector aggregate layouts

#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include <luisa/backends/ext/simd_config_ext.h>
#include <luisa/dsl/sugar.h>
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

constexpr auto thread_count = 35u;
constexpr auto output_stride = 32u;
constexpr auto integer_output_stride = 6u;

[[nodiscard]] float smoothstep_reference(
    float edge0, float edge1, float x) noexcept {
    auto t = std::clamp(
        (x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * std::fma(-2.0f, t, 3.0f);
}

void expect_close(float4 actual, float4 expected,
                  const char *description,
                  float tolerance = 2.0e-6f) noexcept {
    auto delta = abs(actual - expected);
    auto error = std::max({delta.x, delta.y, delta.z, delta.w});
    expect(static_cast<bool>(error <= tolerance))
        << description << ", max error " << error;
}

[[nodiscard]] float pow_int_reference(
    float base, int32_t exponent) noexcept {
    auto negative = exponent < 0;
    auto magnitude = negative ?
                         uint32_t{0u} - static_cast<uint32_t>(exponent) :
                         static_cast<uint32_t>(exponent);
    auto factor = negative ? 1.0f / base : base;
    auto result = 1.0f;
    while (magnitude != 0u) {
        if ((magnitude & 1u) != 0u) { result *= factor; }
        magnitude >>= 1u;
        factor *= factor;
    }
    return result;
}

[[nodiscard]] float pow_int_reference(
    float base, uint32_t exponent) noexcept {
    auto result = 1.0f;
    while (exponent != 0u) {
        if ((exponent & 1u) != 0u) { result *= base; }
        exponent >>= 1u;
        base *= base;
    }
    return result;
}

[[nodiscard]] uint32_t reverse_bits(uint32_t value) noexcept {
    auto result = uint32_t{0u};
    for (auto i = 0u; i < 32u; i++) {
        result = (result << 1u) | (value & 1u);
        value >>= 1u;
    }
    return result;
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
        auto integer_output = device.create_buffer<uint4>(
            thread_count * integer_output_stride);

        Kernel1D kernel = [width](BufferVar<float4> result,
                                  BufferVar<uint4> integer_result) noexcept {
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
            auto faceforward_normal = make_float3(
                x + 0.5f, 1.0f - 0.25f * x, -0.75f);
            auto faceforward_incident = make_float3(
                ite((gid & 1u) == 0u, -1.0f, 1.0f),
                0.0f, 0.0f);
            result.write(
                gid * output_stride + 7u,
                make_float4(
                    faceforward(
                        faceforward_normal,
                        faceforward_incident,
                        make_float3(1.0f, 0.0f, 0.0f)),
                    0.0f));
            Float3 uniform_faceforward_normal =
                make_float3(0.25f, -0.5f, 1.0f);
            Float3 uniform_faceforward_incident = make_float3(0.0f);
            Float3 uniform_faceforward_reference =
                make_float3(1.0f, 0.0f, 0.0f);
            auto uniform_faceforward = faceforward(
                uniform_faceforward_normal,
                uniform_faceforward_incident,
                uniform_faceforward_reference);
            result.write(
                gid * output_stride + 8u,
                make_float4(uniform_faceforward, 0.0f));
            result.write(
                gid * output_stride + 9u,
                step(
                    make_float4(-0.75f, -0.25f, 0.25f, 0.75f),
                    values));

            auto power_base =
                0.75f + cast<float>(gid % 7u) * 0.125f;
            auto signed_exponent = cast<int>(gid % 9u) - 4;
            auto unsigned_exponent = gid % 6u;
            auto vector_power_base = make_float4(
                power_base, -power_base, 2.0f, -1.0f);
            auto vector_power_exponent = make_int4(
                signed_exponent, 3, -3,
                std::numeric_limits<int32_t>::min());
            auto vector_power = def<float4>(
                luisa::compute::detail::FunctionBuilder::current()->call(
                    Type::of<float4>(), CallOp::POW,
                    {vector_power_base.expression(),
                     vector_power_exponent.expression()}));
            auto scalar_power = def<float>(
                luisa::compute::detail::FunctionBuilder::current()->call(
                    Type::of<float>(), CallOp::POW,
                    {power_base.expression(),
                     signed_exponent.expression()}));
            auto unsigned_power = def<float>(
                luisa::compute::detail::FunctionBuilder::current()->call(
                    Type::of<float>(), CallOp::POW,
                    {power_base.expression(),
                     unsigned_exponent.expression()}));
            Float uniform_power_base = 1.5f;
            Int uniform_power_exponent = -3;
            auto uniform_power = def<float>(
                luisa::compute::detail::FunctionBuilder::current()->call(
                    Type::of<float>(), CallOp::POW,
                    {uniform_power_base.expression(),
                     uniform_power_exponent.expression()}));
            UInt uniform_nan_bits = 0x7fc12345u;
            Float uniform_nan = as<float>(uniform_nan_bits);
            result.write(
                gid * output_stride + 10u,
                vector_power);
            result.write(
                gid * output_stride + 11u,
                make_float4(
                    scalar_power, unsigned_power,
                    uniform_power, step(0.0f, uniform_nan)));

            result.write(
                gid * output_stride + 12u,
                make_float4(
                    reduce_sum(values), reduce_prod(values),
                    reduce_min(values), reduce_max(values)));

            auto outer_lhs = make_float2(x, x + 1.0f);
            Float2 outer_rhs = make_float2(2.0f, -3.0f);
            auto vector_outer = def<float2x2>(
                luisa::compute::detail::FunctionBuilder::current()->call(
                    Type::of<float2x2>(), CallOp::OUTER_PRODUCT,
                    {outer_lhs.expression(), outer_rhs.expression()}));
            Float2x2 matrix2_b_dsl = matrix2_b;
            auto matrix_outer = def<float2x2>(
                luisa::compute::detail::FunctionBuilder::current()->call(
                    Type::of<float2x2>(), CallOp::OUTER_PRODUCT,
                    {matrix2_a.expression(), matrix2_b_dsl.expression()}));
            result.write(
                gid * output_stride + 13u,
                make_float4(vector_outer[0u], vector_outer[1u]));
            result.write(
                gid * output_stride + 14u,
                make_float4(matrix_outer[0u], matrix_outer[1u]));

            auto invertible2 = make_float2x2(
                make_float2(x + 3.0f, 1.0f),
                make_float2(2.0f, 4.0f));
            auto transposed2 = transpose(invertible2);
            auto inverse2 = inverse(invertible2);
            result.write(
                gid * output_stride + 15u,
                make_float4(transposed2[0u], transposed2[1u]));
            result.write(
                gid * output_stride + 16u,
                make_float4(inverse2[0u], inverse2[1u]));

            auto invertible3 = make_float3x3(
                make_float3(2.0f + 0.1f * x, 1.0f, 0.0f),
                make_float3(0.0f, 3.0f, 1.0f),
                make_float3(1.0f, 0.0f, 4.0f));
            auto transposed3 = transpose(invertible3);
            auto inverse3 = inverse(invertible3);
            result.write(
                gid * output_stride + 18u,
                make_float4(transposed3[0u], 0.0f));
            result.write(
                gid * output_stride + 19u,
                make_float4(transposed3[1u], 0.0f));
            result.write(
                gid * output_stride + 20u,
                make_float4(transposed3[2u], 0.0f));
            result.write(
                gid * output_stride + 21u,
                make_float4(inverse3[0u], 0.0f));
            result.write(
                gid * output_stride + 22u,
                make_float4(inverse3[1u], 0.0f));
            result.write(
                gid * output_stride + 23u,
                make_float4(inverse3[2u], 0.0f));

            auto invertible4 = make_float4x4(
                make_float4(4.0f + 0.1f * x, 1.0f, 0.0f, 0.0f),
                make_float4(0.0f, 5.0f, 1.0f, 0.0f),
                make_float4(0.0f, 0.0f, 6.0f, 1.0f),
                make_float4(1.0f, 0.0f, 0.0f, 7.0f));
            auto transposed4 = transpose(invertible4);
            auto inverse4 = inverse(invertible4);
            for (auto i = 0u; i < 4u; i++) {
                result.write(
                    gid * output_stride + 24u + i,
                    transposed4[i]);
                result.write(
                    gid * output_stride + 28u + i,
                    inverse4[i]);
            }
            Float2x2 uniform_matrix = make_float2x2(
                make_float2(3.0f, 1.0f),
                make_float2(2.0f, 5.0f));
            result.write(
                gid * output_stride + 17u,
                make_float4(
                    determinant(invertible2),
                    determinant(invertible3),
                    determinant(invertible4),
                    determinant(uniform_matrix)));

            auto bits = make_uint4(
                gid, 0x80000000u | gid,
                1u << (gid % 32u),
                0x01234567u ^ gid);
            integer_result.write(
                gid * integer_output_stride, clz(bits));
            integer_result.write(
                gid * integer_output_stride + 1u, ctz(bits));
            integer_result.write(
                gid * integer_output_stride + 2u, popcount(bits));
            integer_result.write(
                gid * integer_output_stride + 3u, reverse(bits));
            integer_result.write(
                gid * integer_output_stride + 4u,
                make_uint4(
                    reduce_sum(bits), reduce_prod(bits),
                    reduce_min(bits), reduce_max(bits)));
            auto signed_values = make_int4(
                cast<int>(gid % 7u) - 3, -5, 2, 9);
            integer_result.write(
                gid * integer_output_stride + 5u,
                cast<uint4>(make_int4(
                    reduce_sum(signed_values),
                    reduce_prod(signed_values),
                    reduce_min(signed_values),
                    reduce_max(signed_values))));
        };

        auto shader = device.compile(kernel);
        luisa::vector<float4> host(thread_count * output_stride);
        luisa::vector<uint4> integer_host(
            thread_count * integer_output_stride);
        stream << shader(output, integer_output).dispatch(thread_count)
               << output.copy_to(luisa::span{host})
               << integer_output.copy_to(luisa::span{integer_host})
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
            auto faceforward_normal = make_float3(
                x + 0.5f, 1.0f - 0.25f * x, -0.75f);
            auto faceforward_result =
                (gid & 1u) == 0u ?
                    faceforward_normal :
                    -faceforward_normal;
            expect_close(
                host[gid * output_stride + 7u],
                make_float4(faceforward_result, 0.0f),
                "SIMD varying faceforward mismatch");
            expect_close(
                host[gid * output_stride + 8u],
                make_float4(-0.25f, 0.5f, -1.0f, 0.0f),
                "SIMD uniform faceforward zero-orientation mismatch");
            expect_close(
                host[gid * output_stride + 9u],
                make_float4(
                    values.x < -0.75f ? 0.0f : 1.0f,
                    values.y < -0.25f ? 0.0f : 1.0f,
                    values.z < 0.25f ? 0.0f : 1.0f,
                    values.w < 0.75f ? 0.0f : 1.0f),
                "SIMD step mismatch");

            auto power_base =
                0.75f + static_cast<float>(gid % 7u) * 0.125f;
            auto signed_exponent = static_cast<int32_t>(gid % 9u) - 4;
            auto unsigned_exponent = gid % 6u;
            expect_close(
                host[gid * output_stride + 10u],
                make_float4(
                    pow_int_reference(power_base, signed_exponent),
                    pow_int_reference(-power_base, int32_t{3}),
                    pow_int_reference(2.0f, int32_t{-3}),
                    pow_int_reference(
                        -1.0f, std::numeric_limits<int32_t>::min())),
                "SIMD vector pow_int mismatch");
            expect_close(
                host[gid * output_stride + 11u],
                make_float4(
                    pow_int_reference(power_base, signed_exponent),
                    pow_int_reference(power_base, unsigned_exponent),
                    pow_int_reference(1.5f, int32_t{-3}),
                    1.0f),
                "SIMD scalar/uniform pow_int mismatch");
            expect_close(
                host[gid * output_stride + 12u],
                make_float4(
                    values.x + values.y + values.z + values.w,
                    values.x * values.y * values.z * values.w,
                    std::min({values.x, values.y, values.z, values.w}),
                    std::max({values.x, values.y, values.z, values.w})),
                "SIMD floating vector reductions mismatch");

            auto outer_lhs = make_float2(x, x + 1.0f);
            auto outer_rhs = make_float2(2.0f, -3.0f);
            auto vector_outer = make_float2x2(
                outer_lhs * outer_rhs.x,
                outer_lhs * outer_rhs.y);
            auto matrix2_a = make_float2x2(
                make_float2(x, x + 1.0f),
                make_float2(2.0f, 3.0f));
            auto matrix2_b = make_float2x2(
                make_float2(4.0f, 5.0f),
                make_float2(6.0f, 7.0f));
            auto matrix_outer = matrix2_a * transpose(matrix2_b);
            expect_close(
                host[gid * output_stride + 13u],
                make_float4(vector_outer[0u], vector_outer[1u]),
                "SIMD vector outer product mismatch");
            expect_close(
                host[gid * output_stride + 14u],
                make_float4(matrix_outer[0u], matrix_outer[1u]),
                "SIMD matrix outer product mismatch");

            auto invertible2 = make_float2x2(
                make_float2(x + 3.0f, 1.0f),
                make_float2(2.0f, 4.0f));
            auto invertible3 = make_float3x3(
                make_float3(2.0f + 0.1f * x, 1.0f, 0.0f),
                make_float3(0.0f, 3.0f, 1.0f),
                make_float3(1.0f, 0.0f, 4.0f));
            auto invertible4 = make_float4x4(
                make_float4(4.0f + 0.1f * x, 1.0f, 0.0f, 0.0f),
                make_float4(0.0f, 5.0f, 1.0f, 0.0f),
                make_float4(0.0f, 0.0f, 6.0f, 1.0f),
                make_float4(1.0f, 0.0f, 0.0f, 7.0f));
            auto transposed2 = transpose(invertible2);
            auto inverse2 = inverse(invertible2);
            expect_close(
                host[gid * output_stride + 15u],
                make_float4(transposed2[0u], transposed2[1u]),
                "SIMD float2x2 transpose mismatch");
            expect_close(
                host[gid * output_stride + 16u],
                make_float4(inverse2[0u], inverse2[1u]),
                "SIMD float2x2 inverse mismatch", 1.0e-5f);
            expect_close(
                host[gid * output_stride + 17u],
                make_float4(
                    determinant(invertible2),
                    determinant(invertible3),
                    determinant(invertible4), 13.0f),
                "SIMD matrix determinants mismatch", 2.0e-4f);
            auto transposed3 = transpose(invertible3);
            auto inverse3 = inverse(invertible3);
            for (auto i = 0u; i < 3u; i++) {
                expect_close(
                    host[gid * output_stride + 18u + i],
                    make_float4(transposed3[i], 0.0f),
                    "SIMD float3x3 transpose mismatch");
                expect_close(
                    host[gid * output_stride + 21u + i],
                    make_float4(inverse3[i], 0.0f),
                    "SIMD float3x3 inverse mismatch", 2.0e-5f);
            }
            auto transposed4 = transpose(invertible4);
            auto inverse4 = inverse(invertible4);
            for (auto i = 0u; i < 4u; i++) {
                expect_close(
                    host[gid * output_stride + 24u + i],
                    transposed4[i],
                    "SIMD float4x4 transpose mismatch");
                expect_close(
                    host[gid * output_stride + 28u + i],
                    inverse4[i],
                    "SIMD float4x4 inverse mismatch", 5.0e-5f);
            }

            auto bit_values = std::array{
                gid, 0x80000000u | gid,
                1u << (gid % 32u), 0x01234567u ^ gid};
            auto expected_clz = make_uint4(
                std::countl_zero(bit_values[0u]),
                std::countl_zero(bit_values[1u]),
                std::countl_zero(bit_values[2u]),
                std::countl_zero(bit_values[3u]));
            auto expected_ctz = make_uint4(
                std::countr_zero(bit_values[0u]),
                std::countr_zero(bit_values[1u]),
                std::countr_zero(bit_values[2u]),
                std::countr_zero(bit_values[3u]));
            auto expected_popcount = make_uint4(
                std::popcount(bit_values[0u]),
                std::popcount(bit_values[1u]),
                std::popcount(bit_values[2u]),
                std::popcount(bit_values[3u]));
            auto expected_reverse = make_uint4(
                reverse_bits(bit_values[0u]),
                reverse_bits(bit_values[1u]),
                reverse_bits(bit_values[2u]),
                reverse_bits(bit_values[3u]));
            expect(all(integer_host[gid * integer_output_stride] ==
                       expected_clz))
                << "SIMD clz mismatch";
            expect(all(integer_host[gid * integer_output_stride + 1u] ==
                       expected_ctz))
                << "SIMD ctz mismatch";
            expect(all(integer_host[gid * integer_output_stride + 2u] ==
                       expected_popcount))
                << "SIMD popcount mismatch";
            expect(all(integer_host[gid * integer_output_stride + 3u] ==
                       expected_reverse))
                << "SIMD bit reverse mismatch";
            auto unsigned_sum = bit_values[0u] + bit_values[1u] +
                                bit_values[2u] + bit_values[3u];
            auto unsigned_product = bit_values[0u] * bit_values[1u] *
                                    bit_values[2u] * bit_values[3u];
            auto expected_unsigned_reductions = make_uint4(
                unsigned_sum, unsigned_product,
                *std::min_element(
                    bit_values.begin(), bit_values.end()),
                *std::max_element(
                    bit_values.begin(), bit_values.end()));
            expect(all(integer_host[gid * integer_output_stride + 4u] ==
                       expected_unsigned_reductions))
                << "SIMD unsigned vector reductions mismatch";
            auto signed_values = std::array{
                static_cast<int32_t>(gid % 7u) - 3,
                int32_t{-5}, int32_t{2}, int32_t{9}};
            auto expected_signed_reductions = make_uint4(
                static_cast<uint32_t>(signed_values[0u] +
                                      signed_values[1u] +
                                      signed_values[2u] +
                                      signed_values[3u]),
                static_cast<uint32_t>(signed_values[0u] *
                                      signed_values[1u] *
                                      signed_values[2u] *
                                      signed_values[3u]),
                static_cast<uint32_t>(*std::min_element(
                    signed_values.begin(), signed_values.end())),
                static_cast<uint32_t>(*std::max_element(
                    signed_values.begin(), signed_values.end())));
            expect(all(integer_host[gid * integer_output_stride + 5u] ==
                       expected_signed_reductions))
                << "SIMD signed vector reductions mismatch";
        }
    }
}
