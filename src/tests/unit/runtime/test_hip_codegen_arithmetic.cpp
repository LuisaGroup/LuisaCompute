// Test for HIP XIR-to-LLVM arithmetic lowering.
// This test covers:
// - Scalar operands broadcast into vector lerp, step, and smoothstep
// - Signed, unsigned-wide, scalar-broadcast, and per-lane integer powers
// - Matrix-vector, vector-matrix, and matrix-determinant lowering

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/sugar.h>

#include <array>
#include <cmath>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

[[nodiscard]] bool approx(float actual, float expected,
                          float epsilon = 1e-5f) noexcept {
    return std::isfinite(actual) &&
           std::abs(actual - expected) <= epsilon;
}

void test_scalar_vector_broadcast(Device &device) {
    auto stream = device.create_stream();
    auto lower = device.create_buffer<float2>(1u);
    auto upper = device.create_buffer<float2>(1u);
    auto edge0 = device.create_buffer<float>(1u);
    auto edge1 = device.create_buffer<float>(1u);
    auto lerp_output = device.create_buffer<float2>(1u);
    auto step_output = device.create_buffer<float2>(1u);
    auto smoothstep_output = device.create_buffer<float2>(1u);

    Kernel1D kernel = [](BufferFloat2 lower_buffer,
                         BufferFloat2 upper_buffer,
                         BufferFloat edge0_buffer,
                         BufferFloat edge1_buffer,
                         BufferFloat2 lerp_out,
                         BufferFloat2 step_out,
                         BufferFloat2 smoothstep_out) noexcept {
        auto lower_value = lower_buffer.read(0u);
        auto upper_value = upper_buffer.read(0u);
        auto edge0_value = edge0_buffer.read(0u);
        auto edge1_value = edge1_buffer.read(0u);
        auto builder = luisa::compute::detail::FunctionBuilder::current();
        auto lerp_value = def<float2>(builder->call(
            Type::of<float2>(), CallOp::LERP,
            {lower_value.expression(), upper_value.expression(),
             edge0_value.expression()}));
        auto step_value = def<float2>(builder->call(
            Type::of<float2>(), CallOp::STEP,
            {edge0_value.expression(), lower_value.expression()}));
        auto smoothstep_value = def<float2>(builder->call(
            Type::of<float2>(), CallOp::SMOOTHSTEP,
            {edge0_value.expression(), edge1_value.expression(),
             lower_value.expression()}));
        lerp_out.write(0u, lerp_value);
        step_out.write(0u, step_value);
        smoothstep_out.write(0u, smoothstep_value);
    };

    std::array lower_source{float2{0.1f, 0.5f}};
    std::array upper_source{float2{2.1f, 2.5f}};
    std::array edge0_source{0.25f};
    std::array edge1_source{0.75f};
    std::array<float2, 1u> lerp_result{};
    std::array<float2, 1u> step_result{};
    std::array<float2, 1u> smoothstep_result{};
    auto shader = device.compile(kernel, ShaderOption{.enable_fast_math = false});
    stream << lower.copy_from(luisa::span{lower_source})
           << upper.copy_from(luisa::span{upper_source})
           << edge0.copy_from(luisa::span{edge0_source})
           << edge1.copy_from(luisa::span{edge1_source})
           << shader(lower, upper, edge0, edge1,
                     lerp_output, step_output, smoothstep_output)
                  .dispatch(1u)
           << lerp_output.copy_to(luisa::span{lerp_result})
           << step_output.copy_to(luisa::span{step_result})
           << smoothstep_output.copy_to(luisa::span{smoothstep_result})
           << synchronize();

    expect(approx(lerp_result[0].x, 0.6f));
    expect(approx(lerp_result[0].y, 1.0f));
    expect(approx(step_result[0].x, 0.0f));
    expect(approx(step_result[0].y, 1.0f));
    expect(approx(smoothstep_result[0].x, 0.0f));
    expect(approx(smoothstep_result[0].y, 0.5f));
}

void test_integer_power(Device &device) {
    auto stream = device.create_stream();
    auto scalar_base = device.create_buffer<float>(2u);
    auto signed_exponent = device.create_buffer<int32_t>(1u);
    auto unsigned_exponent = device.create_buffer<luisa::ulong>(1u);
    auto vector_base = device.create_buffer<float2>(2u);
    auto vector_exponent = device.create_buffer<int2>(1u);
    auto scalar_output = device.create_buffer<float>(2u);
    auto vector_output = device.create_buffer<float2>(2u);

    Kernel1D kernel = [](BufferFloat scalar_base_buffer,
                         BufferInt signed_exponent_buffer,
                         BufferULong unsigned_exponent_buffer,
                         BufferFloat2 vector_base_buffer,
                         BufferInt2 vector_exponent_buffer,
                         BufferFloat scalar_out,
                         BufferFloat2 vector_out) noexcept {
        auto builder = luisa::compute::detail::FunctionBuilder::current();
        auto scalar_signed_base = scalar_base_buffer.read(0u);
        auto scalar_unsigned_base = scalar_base_buffer.read(1u);
        auto signed_value = signed_exponent_buffer.read(0u);
        auto unsigned_value = unsigned_exponent_buffer.read(0u);
        auto broadcast_base = vector_base_buffer.read(0u);
        auto per_lane_base = vector_base_buffer.read(1u);
        auto per_lane_exponent = vector_exponent_buffer.read(0u);

        auto scalar_signed_result = def<float>(builder->call(
            Type::of<float>(), CallOp::POW,
            {scalar_signed_base.expression(), signed_value.expression()}));
        auto scalar_unsigned_result = def<float>(builder->call(
            Type::of<float>(), CallOp::POW,
            {scalar_unsigned_base.expression(), unsigned_value.expression()}));
        auto broadcast_result = def<float2>(builder->call(
            Type::of<float2>(), CallOp::POW,
            {broadcast_base.expression(), signed_value.expression()}));
        auto per_lane_result = def<float2>(builder->call(
            Type::of<float2>(), CallOp::POW,
            {per_lane_base.expression(), per_lane_exponent.expression()}));
        scalar_out.write(0u, scalar_signed_result);
        scalar_out.write(1u, scalar_unsigned_result);
        vector_out.write(0u, broadcast_result);
        vector_out.write(1u, per_lane_result);
    };

    std::array scalar_base_source{2.0f, -1.0f};
    std::array signed_exponent_source{int32_t{-3}};
    std::array unsigned_exponent_source{luisa::ulong{1} << 32u};
    std::array vector_base_source{
        float2{2.0f, -2.0f},
        float2{2.0f, 3.0f}};
    std::array vector_exponent_source{int2{-2, 3}};
    std::array<float, 2u> scalar_result{};
    std::array<float2, 2u> vector_result{};
    auto shader = device.compile(kernel, ShaderOption{.enable_fast_math = false});
    stream << scalar_base.copy_from(luisa::span{scalar_base_source})
           << signed_exponent.copy_from(luisa::span{signed_exponent_source})
           << unsigned_exponent.copy_from(luisa::span{unsigned_exponent_source})
           << vector_base.copy_from(luisa::span{vector_base_source})
           << vector_exponent.copy_from(luisa::span{vector_exponent_source})
           << shader(scalar_base, signed_exponent, unsigned_exponent,
                     vector_base, vector_exponent,
                     scalar_output, vector_output)
                  .dispatch(1u)
           << scalar_output.copy_to(luisa::span{scalar_result})
           << vector_output.copy_to(luisa::span{vector_result})
           << synchronize();

    expect(approx(scalar_result[0], 0.125f));
    expect(approx(scalar_result[1], 1.0f));
    expect(approx(vector_result[0].x, 0.125f));
    expect(approx(vector_result[0].y, -0.125f));
    expect(approx(vector_result[1].x, 0.25f));
    expect(approx(vector_result[1].y, 27.0f));
}

void test_matrix_operations(Device &device) {
    auto stream = device.create_stream();
    auto matrix_input = device.create_buffer<float2x2>(1u);
    auto vector_input = device.create_buffer<float2>(1u);
    auto vector_output = device.create_buffer<float2>(2u);
    auto scalar_output = device.create_buffer<float>(1u);

    Kernel1D kernel = [](BufferFloat2x2 matrix_buffer,
                         BufferFloat2 vector_buffer,
                         BufferFloat2 vector_out,
                         BufferFloat scalar_out) noexcept {
        auto matrix = matrix_buffer.read(0u);
        auto vector = vector_buffer.read(0u);
        auto builder = luisa::compute::detail::FunctionBuilder::current();
        auto vector_times_matrix = def<float2>(builder->binary(
            Type::of<float2>(), BinaryOp::MUL,
            vector.expression(), matrix.expression()));
        vector_out.write(0u, matrix * vector);
        vector_out.write(1u, vector_times_matrix);
        scalar_out.write(0u, determinant(matrix));
    };

    std::array matrix_source{
        make_float2x2(1.0f, 2.0f, 3.0f, 4.0f)};
    std::array vector_source{float2{5.0f, 6.0f}};
    std::array<float2, 2u> vector_result{};
    std::array<float, 1u> scalar_result{};
    auto shader = device.compile(kernel, ShaderOption{.enable_fast_math = false});
    stream << matrix_input.copy_from(luisa::span{matrix_source})
           << vector_input.copy_from(luisa::span{vector_source})
           << shader(matrix_input, vector_input,
                     vector_output, scalar_output)
                  .dispatch(1u)
           << vector_output.copy_to(luisa::span{vector_result})
           << scalar_output.copy_to(luisa::span{scalar_result})
           << synchronize();

    expect(approx(vector_result[0].x, 23.0f));
    expect(approx(vector_result[0].y, 34.0f));
    expect(approx(vector_result[1].x, 17.0f));
    expect(approx(vector_result[1].y, 39.0f));
    expect(approx(scalar_result[0], -2.0f));
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    test_scalar_vector_broadcast(dc->device);
    test_integer_power(dc->device);
    test_matrix_operations(dc->device);
}
