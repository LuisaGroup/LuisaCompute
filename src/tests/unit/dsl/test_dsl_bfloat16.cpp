// Test for bfloat16 DSL and Vulkan/CUDA code generation.
// This test covers:
// - BF16 buffer storage, scalar/vector construction, conversions and operators
// - arithmetic, comparison, selection and compound operators
// - scalar mathematical intrinsics, classification and rounding
// - vector math, reductions and unsupported-device compatibility

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;

namespace {

enum class BfOp : uint32_t {
    UNARY_PLUS,
    UNARY_MINUS,
    ADD,
    SUB,
    MUL,
    DIV,
    LESS,
    GREATER,
    LESS_EQUAL,
    GREATER_EQUAL,
    EQUAL,
    NOT_EQUAL,
    COMPOUND_ADD,
    COMPOUND_SUB,
    COMPOUND_MUL,
    COMPOUND_DIV,
    SELECT,
    ABS,
    MIN,
    MAX,
    MIN_NAN_LHS,
    MIN_NAN_RHS,
    MAX_NAN_LHS,
    MAX_NAN_RHS,
    CLAMP,
    SATURATE,
    LERP,
    SMOOTHSTEP,
    STEP,
    FMA,
    COPYSIGN,
    SIN,
    COS,
    TAN,
    ASIN,
    ACOS,
    ATAN,
    ATAN2,
    SINH,
    COSH,
    TANH,
    ASINH,
    ACOSH,
    ATANH,
    EXP,
    EXP2,
    EXP10,
    LOG,
    LOG2,
    LOG10,
    POW,
    SQRT,
    RSQRT,
    CEIL,
    FLOOR,
    FRACT,
    TRUNC,
    ROUND,
    RINT,
    ISNAN,
    ISINF,
    LOGICAL_NOT,
    LOGICAL_AND,
    LOGICAL_OR,
    DOT,
    CROSS_X,
    LENGTH,
    LENGTH_SQUARED,
    NORMALIZE_X,
    REFLECT_X,
    FACEFORWARD_X,
    REDUCE_SUM,
    REDUCE_PRODUCT,
    REDUCE_MIN,
    REDUCE_MAX,
    VECTOR_ADD_X,
    VECTOR_MUL_X,
    DISTANCE,
    SIGN,
    CAST_FLOAT,
    CAST_HALF,
    CAST_INT,
    CAST_UINT,
    CAST_BOOL,
    COUNT
};

constexpr auto bf_op_count = static_cast<size_t>(BfOp::COUNT);

[[nodiscard]] float narrow(float value) noexcept {
    return static_cast<float>(bfloat16{value});
}

[[nodiscard]] bool test_fp32_compatibility(Device &device) noexcept {
    auto output = device.create_buffer<float>(1u);
    Stream stream = device.create_stream();
    Kernel1D kernel = [](BufferFloat output) noexcept {
        output.write(0u, 1.0f);
    };
    auto shader = device.compile(kernel);
    float result = 0.0f;
    stream << shader(output).dispatch(1u)
           << output.copy_to(luisa::span{&result, 1u})
           << synchronize();
    return result == 1.0f;
}

[[nodiscard]] bool test_vector_and_bit_coverage(Device &device) noexcept {
    static constexpr auto value_count = 11u;
    static constexpr auto bit_count = 7u;
    static constexpr std::array constant_scalars{
        bfloat16::from_bits(0x3f80u), bfloat16::from_bits(0xbf80u),
        bfloat16::from_bits(0x7fc1u), bfloat16::from_bits(0x4000u)};
    static constexpr std::array constant_vectors{
        bfloat163{bfloat16{0.5f}, bfloat16{2.0f}, bfloat16{-1.5f}}};
    auto values = device.create_buffer<bfloat16>(value_count);
    auto bits = device.create_buffer<ushort>(bit_count);
    Stream stream = device.create_stream();
    Kernel1D kernel = [&](BufferBFloat16 values, BufferVar<ushort> bits) noexcept {
        Constant<bfloat16> scalar_constants{constant_scalars};
        Constant<bfloat163> vector_constants{constant_vectors};
        BFloat16 a = 0.5f;
        BFloat16 b = 2.0f;
        BFloat16 c = -1.5f;
        BFloat16 d = 4.0f;
        BFloat16 negative_zero = bfloat16::from_bits(0x8000u);
        BFloat16 nan_payload = bfloat16::from_bits(0x7fc1u);
        BFloat162 v2 = make_bfloat162(a, b);
        BFloat163 v3 = make_bfloat163(v2, c);
        BFloat164 v4 = make_bfloat164(v2, make_bfloat162(c, d));
        auto reversed = v4.wzyx();
        Float3 float_input = make_float3(0.5f, 2.0f, -1.5f);
        auto from_float = cast<bfloat163>(float_input);
        auto to_float = cast<float3>(v3);
        auto to_half = cast<half2>(v2);

        values.write(0u, v2.x);
        values.write(1u, v2.y);
        values.write(2u, v3.z);
        values.write(3u, v4.w);
        values.write(4u, reversed.x);
        values.write(5u, reversed.y);
        values.write(6u, from_float.z);
        values.write(7u, cast<bfloat16>(to_float.y));
        values.write(8u, cast<bfloat16>(to_half.y));
        values.write(9u, cast<bfloat16>(all(!make_bfloat162(a - a))));
        values.write(10u, cast<bfloat16>(any(!make_bfloat162(a - a))));

        bits.write(0u, as<ushort>(negative_zero));
        bits.write(1u, as<ushort>(nan_payload));
        bits.write(2u, as<ushort>(as<bfloat16>(ushort{0x3f80u})));
        bits.write(3u, as<ushort2>(v2).y);
        bits.write(4u, as<ushort4>(v4).z);
        bits.write(5u, as<ushort>(scalar_constants.read(dispatch_x() + 2u)));
        bits.write(6u, as<ushort>(vector_constants.read(dispatch_x()).z));
    };
    auto shader = device.compile(kernel);
    stream << shader(values, bits).dispatch(1u) << synchronize();
    std::array<bfloat16, value_count> value_results{};
    std::array<ushort, bit_count> bit_results{};
    stream << values.copy_to(luisa::span{value_results})
           << bits.copy_to(luisa::span{bit_results})
           << synchronize();

    static constexpr std::array expected_values{
        0.5f, 2.0f, -1.5f, 4.0f, 4.0f, -1.5f, -1.5f, 2.0f, 2.0f, 1.0f, 1.0f};
    static constexpr std::array expected_bits{
        0x8000u, 0x7fc1u, 0x3f80u, 0x4000u, 0xbfc0u, 0x7fc1u, 0xbfc0u};
    auto all_correct = true;
    for (auto i = 0u; i < value_count; i++) {
        auto correct = static_cast<float>(value_results[i]) == expected_values[i];
        expect(correct) << "BF16 vector/conversion result " << i;
        all_correct = all_correct && correct;
    }
    for (auto i = 0u; i < bit_count; i++) {
        auto correct = bit_results[i] == expected_bits[i];
        if (!correct) {
            LUISA_WARNING("BF16 bits[{}]: got 0x{:04x}, expected 0x{:04x}.",
                          i, bit_results[i], expected_bits[i]);
        }
        expect(correct) << "BF16 exact bits " << i;
        all_correct = all_correct && correct;
    }
    return all_correct;
}

[[nodiscard]] bool test_cuda_kernel_argument_abi(Device &device) noexcept {
    auto output = device.create_buffer<bfloat16>(2u);
    Stream stream = device.create_stream();
    Kernel1D kernel = [](BufferBFloat16 output,
                         BFloat16 scalar, BFloat163 vector) noexcept {
        output.write(0u, scalar);
        output.write(1u, vector.z);
    };
    auto shader = device.compile(kernel);
    auto scalar = bfloat16{3.0f};
    auto vector = bfloat163{bfloat16{0.5f}, bfloat16{2.0f}, bfloat16{-1.5f}};
    stream << shader(output, scalar, vector).dispatch(1u) << synchronize();
    std::array<bfloat16, 2u> results{};
    stream << output.copy_to(luisa::span{results}) << synchronize();
    auto scalar_correct = static_cast<float>(results[0]) == 3.0f;
    auto vector_correct = static_cast<float>(results[1]) == -1.5f;
    expect(scalar_correct) << "BF16 scalar kernel argument must preserve its value";
    expect(vector_correct) << "BF16 vector kernel argument must preserve its layout";
    return scalar_correct && vector_correct;
}

}// namespace

int main(int argc, char *argv[]) {
    log_level_verbose();
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;

    if (device.query("shader_bfloat16") != "true") {
        LUISA_INFO("BF16 is unavailable on this backend. Running FP32 compatibility probe.");
        auto compatible = test_fp32_compatibility(device);
        expect(compatible) << "A device without BF16 support must retain ordinary FP32 operation";
        return compatible ? 0 : 1;
    }

    std::array<bfloat16, 6u> host_input{
        bfloat16{0.5f},
        bfloat16{2.0f},
        bfloat16{-1.5f},
        bfloat16{4.0f},
        bfloat16{std::numeric_limits<float>::quiet_NaN()},
        bfloat16{std::numeric_limits<float>::infinity()}};
    auto input = device.create_buffer<bfloat16>(host_input.size());
    auto output = device.create_buffer<bfloat16>(bf_op_count);
    Stream stream = device.create_stream();
    stream << input.copy_from(luisa::span{host_input}) << synchronize();

    Kernel1D kernel = [](BufferBFloat16 input, BufferBFloat16 output) noexcept {
        BFloat16 a = input.read(0u);
        BFloat16 b = input.read(1u);
        BFloat16 c = input.read(2u);
        BFloat16 zero = 0.0f;
        BFloat16 one = 1.0f;
        BFloat16 two = 2.0f;
        Bool ab = a < b;

        output.write(static_cast<uint>(BfOp::UNARY_PLUS), +a);
        output.write(static_cast<uint>(BfOp::UNARY_MINUS), -a);
        output.write(static_cast<uint>(BfOp::ADD), a + b);
        output.write(static_cast<uint>(BfOp::SUB), a - b);
        output.write(static_cast<uint>(BfOp::MUL), a * b);
        output.write(static_cast<uint>(BfOp::DIV), a / b);
        output.write(static_cast<uint>(BfOp::LESS), cast<bfloat16>(a < b));
        output.write(static_cast<uint>(BfOp::GREATER), cast<bfloat16>(a > b));
        output.write(static_cast<uint>(BfOp::LESS_EQUAL), cast<bfloat16>(a <= b));
        output.write(static_cast<uint>(BfOp::GREATER_EQUAL), cast<bfloat16>(a >= b));
        output.write(static_cast<uint>(BfOp::EQUAL), cast<bfloat16>(a == b));
        output.write(static_cast<uint>(BfOp::NOT_EQUAL), cast<bfloat16>(a != b));

        BFloat16 compound = a;
        compound += b;
        output.write(static_cast<uint>(BfOp::COMPOUND_ADD), compound);
        compound -= b;
        output.write(static_cast<uint>(BfOp::COMPOUND_SUB), compound);
        compound *= b;
        output.write(static_cast<uint>(BfOp::COMPOUND_MUL), compound);
        compound /= b;
        output.write(static_cast<uint>(BfOp::COMPOUND_DIV), compound);
        output.write(static_cast<uint>(BfOp::SELECT), ite(ab, a, b));

        output.write(static_cast<uint>(BfOp::ABS), abs(c));
        output.write(static_cast<uint>(BfOp::MIN), min(a, b));
        output.write(static_cast<uint>(BfOp::MAX), max(a, b));
        output.write(static_cast<uint>(BfOp::MIN_NAN_LHS), min(input.read(4u), a));
        output.write(static_cast<uint>(BfOp::MIN_NAN_RHS), min(a, input.read(4u)));
        output.write(static_cast<uint>(BfOp::MAX_NAN_LHS), max(input.read(4u), a));
        output.write(static_cast<uint>(BfOp::MAX_NAN_RHS), max(a, input.read(4u)));
        output.write(static_cast<uint>(BfOp::CLAMP), clamp(c, zero, one));
        output.write(static_cast<uint>(BfOp::SATURATE), saturate(c));
        output.write(static_cast<uint>(BfOp::LERP), lerp(a, b, a));
        output.write(static_cast<uint>(BfOp::SMOOTHSTEP), smoothstep(zero, one, b));
        output.write(static_cast<uint>(BfOp::STEP), step(one, a));
        output.write(static_cast<uint>(BfOp::FMA), fma(a, b, c));
        output.write(static_cast<uint>(BfOp::COPYSIGN), copysign(a, c));

        output.write(static_cast<uint>(BfOp::SIN), sin(a));
        output.write(static_cast<uint>(BfOp::COS), cos(a));
        output.write(static_cast<uint>(BfOp::TAN), tan(a));
        output.write(static_cast<uint>(BfOp::ASIN), asin(a));
        output.write(static_cast<uint>(BfOp::ACOS), acos(a));
        output.write(static_cast<uint>(BfOp::ATAN), atan(a));
        output.write(static_cast<uint>(BfOp::ATAN2), atan2(a, b));
        output.write(static_cast<uint>(BfOp::SINH), sinh(a));
        output.write(static_cast<uint>(BfOp::COSH), cosh(a));
        output.write(static_cast<uint>(BfOp::TANH), tanh(a));
        output.write(static_cast<uint>(BfOp::ASINH), asinh(a));
        output.write(static_cast<uint>(BfOp::ACOSH), acosh(b));
        output.write(static_cast<uint>(BfOp::ATANH), atanh(a));

        output.write(static_cast<uint>(BfOp::EXP), exp(a));
        output.write(static_cast<uint>(BfOp::EXP2), exp2(a));
        output.write(static_cast<uint>(BfOp::EXP10), exp10(a));
        output.write(static_cast<uint>(BfOp::LOG), luisa::compute::log(a));
        output.write(static_cast<uint>(BfOp::LOG2), log2(a));
        output.write(static_cast<uint>(BfOp::LOG10), log10(a));
        output.write(static_cast<uint>(BfOp::POW), pow(a, b));
        output.write(static_cast<uint>(BfOp::SQRT), sqrt(a));
        output.write(static_cast<uint>(BfOp::RSQRT), rsqrt(a));
        output.write(static_cast<uint>(BfOp::CEIL), ceil(a));
        output.write(static_cast<uint>(BfOp::FLOOR), floor(a));
        output.write(static_cast<uint>(BfOp::FRACT), fract(c));
        output.write(static_cast<uint>(BfOp::TRUNC), trunc(c));
        output.write(static_cast<uint>(BfOp::ROUND), round(a));
        output.write(static_cast<uint>(BfOp::RINT), rint(a));

        BFloat16 nan_value = input.read(4u);
        BFloat16 inf_value = input.read(5u);
        output.write(static_cast<uint>(BfOp::ISNAN), cast<bfloat16>(luisa::compute::isnan(nan_value)));
        output.write(static_cast<uint>(BfOp::ISINF), cast<bfloat16>(luisa::compute::isinf(inf_value)));
        output.write(static_cast<uint>(BfOp::LOGICAL_NOT), cast<bfloat16>(!c));
        output.write(static_cast<uint>(BfOp::LOGICAL_AND), cast<bfloat16>(ab & (b > zero)));
        output.write(static_cast<uint>(BfOp::LOGICAL_OR), cast<bfloat16>((a > b) | (c < zero)));

        BFloat163 v = make_bfloat163(a, b, c);
        BFloat163 w = make_bfloat163(b, a, one);
        BFloat163 normal = make_bfloat163(zero, zero, one);
        BFloat163 face_normal = make_bfloat163(one, zero, zero);
        BFloat163 reflected = reflect(v, normal);
        BFloat163 normalized = normalize(v);
        BFloat163 crossed = cross(v, w);
        BFloat163 faceforwarded = faceforward(face_normal, v, normal);
        BFloat163 vector_sum = v + w;
        BFloat163 vector_product = v * w;

        output.write(static_cast<uint>(BfOp::DOT), dot(v, w));
        output.write(static_cast<uint>(BfOp::CROSS_X), crossed.x);
        output.write(static_cast<uint>(BfOp::LENGTH), length(v));
        output.write(static_cast<uint>(BfOp::LENGTH_SQUARED), length_squared(v));
        output.write(static_cast<uint>(BfOp::NORMALIZE_X), normalized.x);
        output.write(static_cast<uint>(BfOp::REFLECT_X), reflected.x);
        output.write(static_cast<uint>(BfOp::FACEFORWARD_X), faceforwarded.x);
        output.write(static_cast<uint>(BfOp::REDUCE_SUM), reduce_sum(v));
        output.write(static_cast<uint>(BfOp::REDUCE_PRODUCT), reduce_prod(v));
        output.write(static_cast<uint>(BfOp::REDUCE_MIN), reduce_min(v));
        output.write(static_cast<uint>(BfOp::REDUCE_MAX), reduce_max(v));
        output.write(static_cast<uint>(BfOp::VECTOR_ADD_X), vector_sum.x);
        output.write(static_cast<uint>(BfOp::VECTOR_MUL_X), vector_product.x);
        output.write(static_cast<uint>(BfOp::DISTANCE), distance(v, w));
        output.write(static_cast<uint>(BfOp::SIGN), sign(c));

        output.write(static_cast<uint>(BfOp::CAST_FLOAT), cast<bfloat16>(cast<float>(c)));
        output.write(static_cast<uint>(BfOp::CAST_HALF), cast<bfloat16>(cast<half>(a)));
        output.write(static_cast<uint>(BfOp::CAST_INT), cast<bfloat16>(cast<int>(c)));
        output.write(static_cast<uint>(BfOp::CAST_UINT), cast<bfloat16>(cast<uint>(b)));
        output.write(static_cast<uint>(BfOp::CAST_BOOL), cast<bfloat16>(cast<bool>(c)));
    };

    auto shader = device.compile(kernel);
    stream << shader(input, output).dispatch(1u) << synchronize();
    luisa::vector<bfloat16> results(bf_op_count);
    stream << output.copy_to(luisa::span{results}) << synchronize();

    auto all_correct = true;
    auto check = [&](BfOp op, const char *name, float expected) noexcept {
        expected = narrow(expected);
        auto actual = static_cast<float>(results[static_cast<size_t>(op)]);
        auto correct = (std::isnan(actual) && std::isnan(expected)) ||
                       std::abs(actual - expected) <=
                           std::max(0.02f, 0.025f * std::max(std::abs(actual), std::abs(expected)));
        if (!correct) {
            LUISA_WARNING("BF16 {}: got {} (0x{:04x}), expected {}.", name, actual,
                          results[static_cast<size_t>(op)].bits(), expected);
        }
        expect(correct) << "BF16 " << name;
        all_correct = all_correct && correct;
    };

    auto a = 0.5f;
    auto b = 2.0f;
    auto c = -1.5f;
    check(BfOp::UNARY_PLUS, "unary +", a);
    check(BfOp::UNARY_MINUS, "unary -", -a);
    check(BfOp::ADD, "operator +", a + b);
    check(BfOp::SUB, "operator -", a - b);
    check(BfOp::MUL, "operator *", a * b);
    check(BfOp::DIV, "operator /", a / b);
    check(BfOp::LESS, "operator <", 1.0f);
    check(BfOp::GREATER, "operator >", 0.0f);
    check(BfOp::LESS_EQUAL, "operator <=", 1.0f);
    check(BfOp::GREATER_EQUAL, "operator >=", 0.0f);
    check(BfOp::EQUAL, "operator ==", 0.0f);
    check(BfOp::NOT_EQUAL, "operator !=", 1.0f);
    check(BfOp::COMPOUND_ADD, "operator +=", 2.5f);
    check(BfOp::COMPOUND_SUB, "operator -=", 0.5f);
    check(BfOp::COMPOUND_MUL, "operator *=", 1.0f);
    check(BfOp::COMPOUND_DIV, "operator /=", 0.5f);
    check(BfOp::SELECT, "select", a);

    check(BfOp::ABS, "abs", std::abs(c));
    check(BfOp::MIN, "min", std::min(a, b));
    check(BfOp::MAX, "max", std::max(a, b));
    check(BfOp::MIN_NAN_LHS, "min with lhs NaN", a);
    check(BfOp::MIN_NAN_RHS, "min with rhs NaN", a);
    check(BfOp::MAX_NAN_LHS, "max with lhs NaN", a);
    check(BfOp::MAX_NAN_RHS, "max with rhs NaN", a);
    check(BfOp::CLAMP, "clamp", std::clamp(c, 0.0f, 1.0f));
    check(BfOp::SATURATE, "saturate", std::clamp(c, 0.0f, 1.0f));
    check(BfOp::LERP, "lerp", std::lerp(a, b, a));
    check(BfOp::SMOOTHSTEP, "smoothstep", 1.0f);
    check(BfOp::STEP, "step", 0.0f);
    check(BfOp::FMA, "fma", std::fma(a, b, c));
    check(BfOp::COPYSIGN, "copysign", std::copysign(a, c));

    check(BfOp::SIN, "sin", std::sin(a));
    check(BfOp::COS, "cos", std::cos(a));
    check(BfOp::TAN, "tan", std::tan(a));
    check(BfOp::ASIN, "asin", std::asin(a));
    check(BfOp::ACOS, "acos", std::acos(a));
    check(BfOp::ATAN, "atan", std::atan(a));
    check(BfOp::ATAN2, "atan2", std::atan2(a, b));
    check(BfOp::SINH, "sinh", std::sinh(a));
    check(BfOp::COSH, "cosh", std::cosh(a));
    check(BfOp::TANH, "tanh", std::tanh(a));
    check(BfOp::ASINH, "asinh", std::asinh(a));
    check(BfOp::ACOSH, "acosh", std::acosh(b));
    check(BfOp::ATANH, "atanh", std::atanh(a));

    check(BfOp::EXP, "exp", std::exp(a));
    check(BfOp::EXP2, "exp2", std::exp2(a));
    check(BfOp::EXP10, "exp10", std::pow(10.0f, a));
    check(BfOp::LOG, "log", std::log(a));
    check(BfOp::LOG2, "log2", std::log2(a));
    check(BfOp::LOG10, "log10", std::log10(a));
    check(BfOp::POW, "pow", std::pow(a, b));
    check(BfOp::SQRT, "sqrt", std::sqrt(a));
    check(BfOp::RSQRT, "rsqrt", 1.0f / std::sqrt(a));
    check(BfOp::CEIL, "ceil", std::ceil(a));
    check(BfOp::FLOOR, "floor", std::floor(a));
    check(BfOp::FRACT, "fract", c - std::floor(c));
    check(BfOp::TRUNC, "trunc", std::trunc(c));
    check(BfOp::ROUND, "round", std::round(a));
    check(BfOp::RINT, "rint", std::nearbyint(a));
    check(BfOp::ISNAN, "isnan", 1.0f);
    check(BfOp::ISINF, "isinf", 1.0f);
    check(BfOp::LOGICAL_NOT, "logical not", 0.0f);
    check(BfOp::LOGICAL_AND, "logical and", 1.0f);
    check(BfOp::LOGICAL_OR, "logical or", 1.0f);

    std::array<float, 3u> v{a, b, c};
    std::array<float, 3u> w{b, a, 1.0f};
    auto dot = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
    auto length_squared = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    auto length = std::sqrt(length_squared);
    auto cross_x = v[1] * w[2] - v[2] * w[1];
    auto distance_squared = (v[0] - w[0]) * (v[0] - w[0]) +
                            (v[1] - w[1]) * (v[1] - w[1]) +
                            (v[2] - w[2]) * (v[2] - w[2]);
    check(BfOp::DOT, "dot", dot);
    check(BfOp::CROSS_X, "cross", cross_x);
    check(BfOp::LENGTH, "length", length);
    check(BfOp::LENGTH_SQUARED, "length_squared", length_squared);
    check(BfOp::NORMALIZE_X, "normalize", v[0] / length);
    check(BfOp::REFLECT_X, "reflect", v[0]);
    check(BfOp::FACEFORWARD_X, "faceforward", 1.0f);
    check(BfOp::REDUCE_SUM, "reduce_sum", v[0] + v[1] + v[2]);
    check(BfOp::REDUCE_PRODUCT, "reduce_prod", v[0] * v[1] * v[2]);
    check(BfOp::REDUCE_MIN, "reduce_min", std::min({v[0], v[1], v[2]}));
    check(BfOp::REDUCE_MAX, "reduce_max", std::max({v[0], v[1], v[2]}));
    check(BfOp::VECTOR_ADD_X, "vector operator +", v[0] + w[0]);
    check(BfOp::VECTOR_MUL_X, "vector operator *", v[0] * w[0]);
    check(BfOp::DISTANCE, "distance", std::sqrt(distance_squared));
    check(BfOp::SIGN, "sign", -1.0f);

    check(BfOp::CAST_FLOAT, "float conversion", c);
    check(BfOp::CAST_HALF, "half conversion", a);
    check(BfOp::CAST_INT, "int conversion", -1.0f);
    check(BfOp::CAST_UINT, "uint conversion", 2.0f);
    check(BfOp::CAST_BOOL, "bool conversion", 1.0f);

    auto vector_and_bit_correct = test_vector_and_bit_coverage(device);
    auto cuda_abi_correct = device.backend_name() == "cuda" ?
                                test_cuda_kernel_argument_abi(device) :
                                true;
    LUISA_INFO("BF16 device math coverage completed ({} results).", bf_op_count);
    return all_correct && vector_and_bit_correct && cuda_abi_correct ? 0 : 1;
}
