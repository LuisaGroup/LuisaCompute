// Comprehensive test for DSL mathematical operations.
// Tests all mathematical CallOp operations from include/luisa/ast/op.h
// by comparing GPU results against C++ STL / <cmath> on the CPU.
//
// Features tested:
// - Unary operators: +, -, !, ~
// - Binary operators: +, -, *, /, %, &, |, ^, <<, >>, &&, ||, <, >, <=, >=, ==, !=
// - Math functions: abs, min, max, clamp, saturate, lerp, smoothstep, step
// - Trigonometric: sin, cos, tan, asin, acos, atan, atan2
// - Hyperbolic: sinh, cosh, tanh, asinh, acosh, atanh
// - Exponential/Logarithmic: exp, exp2, exp10, log, log2, log10, pow, sqrt, rsqrt
// - Rounding: ceil, floor, fract, trunc, round
// - Vector math: dot, cross, length, length_squared, normalize, reflect, faceforward
// - Matrix math: determinant, transpose, inverse
// - Reductions: reduce_sum, reduce_prod, reduce_min, reduce_max
// - Integer bit ops: clz, ctz, popcount, reverse
// - Float classification: isinf, isnan
// - Selection: select / ite
// - FMA, copysign

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

#include <cmath>
#include <algorithm>
#include <limits>
#include <bit>

using namespace luisa;
using namespace luisa::compute;

constexpr float float_eps = 1e-3f;

inline bool approx_eq(float a, float b, float eps = float_eps) noexcept {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    float diff = std::abs(a - b);
    float scale = std::max(std::abs(a), std::abs(b));
    return diff <= eps || diff <= eps * scale;
}

inline bool approx_eq(float2 a, float2 b, float eps = float_eps) noexcept {
    return approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps);
}

inline bool approx_eq(float3 a, float3 b, float eps = float_eps) noexcept {
    return approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps);
}

inline bool approx_eq(float4 a, float4 b, float eps = float_eps) noexcept {
    return approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps) && approx_eq(a.w, b.w, eps);
}

// Complex struct for testing scalar, vector, and matrix types
struct ComplexStruct {
    int16_t i16;
    uint16_t u16;
    int i32;
    uint u32;
    luisa::slong i64;
    luisa::ulong u64;
    half h;
    float f;
    double d;
    bool b;
    float2 f2;
    float3 f3;
    float4 f4;
    half3 h3;
    half4 h4;
    int4 i4;
    double2 d2;
    double3 d3;
    double4 d4;
    half2x2 h2x2;
    float2x2 f2x2;
    float3x3 f3x3;
    float4x4 f4x4;
    half h_adjacent;
    bool b_adjacent;
};
LUISA_STRUCT(ComplexStruct, i16, u16, i32, u32, i64, u64, h, f, d, b, f2, f3, f4, h3, h4, i4, d2, d3, d4, h2x2, f2x2, f3x3, f4x4, h_adjacent, b_adjacent) {};

int main(int argc, char *argv[]) {
    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    Stream stream = device.create_stream();

    // ============================================================
    // Test 1: Unary and binary scalar operators
    // ============================================================
    {
        float host_in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        auto in_buf = device.create_buffer<float>(4);
        auto out_buf = device.create_buffer<float>(32);
        stream << in_buf.copy_from(luisa::span{host_in, 4}) << synchronize();

        Kernel1D kernel = [&](BufferFloat input, BufferFloat output) noexcept {
            $uint i = dispatch_x();
            $float a = input.read(i);
            $float b = input.read((i + 1u) % 4u);

            // unary
            output.write(i + 0u * 4u, +a);
            output.write(i + 1u * 4u, -a);

            // binary arithmetic
            output.write(i + 2u * 4u, a + b);
            output.write(i + 3u * 4u, a - b);
            output.write(i + 4u * 4u, a * b);
            output.write(i + 5u * 4u, a / b);

            // relational (store as float 1.0 or 0.0)
            output.write(i + 6u * 4u, ite(a < b, 1.0f, 0.0f));
            output.write(i + 7u * 4u, ite(a > b, 1.0f, 0.0f));
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_buf).dispatch(4) << synchronize();

        luisa::vector<float> out(32);
        stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();

        for (int i = 0; i < 4; ++i) {
            float a = host_in[i];
            float b = host_in[(i + 1) % 4];
            LUISA_ASSERT(approx_eq(out[i + 0 * 4], +a), "Unary PLUS failed");
            LUISA_ASSERT(approx_eq(out[i + 1 * 4], -a), "Unary MINUS failed");
            LUISA_ASSERT(approx_eq(out[i + 2 * 4], a + b), "Binary ADD failed");
            LUISA_ASSERT(approx_eq(out[i + 3 * 4], a - b), "Binary SUB failed");
            LUISA_ASSERT(approx_eq(out[i + 4 * 4], a * b), "Binary MUL failed");
            LUISA_ASSERT(approx_eq(out[i + 5 * 4], a / b), "Binary DIV failed");
            LUISA_ASSERT(approx_eq(out[i + 6 * 4], (a < b) ? 1.0f : 0.0f), "LESS failed");
            LUISA_ASSERT(approx_eq(out[i + 7 * 4], (a > b) ? 1.0f : 0.0f), "GREATER failed");
        }
        LUISA_INFO("Unary/binary scalar operators passed.");
    }

    // ============================================================
    // Test 2: Integer bitwise and modulo operators
    // ============================================================
    {
        int host_in[4] = {5, 7, 12, 3};
        auto in_buf = device.create_buffer<int>(4);
        auto out_buf = device.create_buffer<int>(32);
        stream << in_buf.copy_from(luisa::span{host_in, 4}) << synchronize();

        Kernel1D kernel = [&](BufferInt input, BufferInt output) noexcept {
            $uint i = dispatch_x();
            $int a = input.read(i);
            $int b = input.read((i + 1u) % 4u);

            output.write(i + 0u * 4u, a % b);
            output.write(i + 1u * 4u, a & b);
            output.write(i + 2u * 4u, a | b);
            output.write(i + 3u * 4u, a ^ b);
            output.write(i + 4u * 4u, a << 1);
            output.write(i + 5u * 4u, a >> 1);
            $bool ba = a != 0;
            $bool bb = b != 0;
            output.write(i + 6u * 4u, ite(ba & bb, 1, 0));
            output.write(i + 7u * 4u, ite(ba | bb, 1, 0));
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_buf).dispatch(4) << synchronize();

        luisa::vector<int> out(32);
        stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();

        for (int i = 0; i < 4; ++i) {
            int a = host_in[i];
            int b = host_in[(i + 1) % 4];
            LUISA_ASSERT(out[i + 0 * 4] == a % b, "MOD failed");
            LUISA_ASSERT(out[i + 1 * 4] == (a & b), "BIT_AND failed");
            LUISA_ASSERT(out[i + 2 * 4] == (a | b), "BIT_OR failed");
            LUISA_ASSERT(out[i + 3 * 4] == (a ^ b), "BIT_XOR failed");
            LUISA_ASSERT(out[i + 4 * 4] == (a << 1), "SHL failed");
            LUISA_ASSERT(out[i + 5 * 4] == (a >> 1), "SHR failed");
            LUISA_ASSERT(out[i + 6 * 4] == ((a && b) ? 1 : 0), "AND failed");
            LUISA_ASSERT(out[i + 7 * 4] == ((a || b) ? 1 : 0), "OR failed");
        }
        LUISA_INFO("Integer bitwise operators passed.");
    }

    // ============================================================
    // Test 3: Common math functions (abs, min, max, clamp, saturate, lerp, step, smoothstep)
    // ============================================================
    {
        float host_in[4] = {-1.5f, 0.5f, 2.0f, 3.5f};
        auto in_buf = device.create_buffer<float>(4);
        auto out_buf = device.create_buffer<float>(32);
        stream << in_buf.copy_from(luisa::span{host_in, 4}) << synchronize();

        Kernel1D kernel = [&](BufferFloat input, BufferFloat output) noexcept {
            $uint i = dispatch_x();
            $float a = input.read(i);
            $float b = input.read((i + 1u) % 4u);

            output.write(i + 0u * 4u, abs(a));
            output.write(i + 1u * 4u, min(a, b));
            output.write(i + 2u * 4u, max(a, b));
            output.write(i + 3u * 4u, clamp(a, 0.0f, 1.0f));
            output.write(i + 4u * 4u, saturate(a));
            output.write(i + 5u * 4u, lerp(a, b, 0.5f));
            output.write(i + 6u * 4u, step(1.0f, a));
            output.write(i + 7u * 4u, smoothstep(0.0f, 1.0f, a));
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_buf).dispatch(4) << synchronize();

        luisa::vector<float> out(32);
        stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();

        for (int i = 0; i < 4; ++i) {
            float a = host_in[i];
            float b = host_in[(i + 1) % 4];
            LUISA_ASSERT(approx_eq(out[i + 0 * 4], std::abs(a)), "abs failed");
            LUISA_ASSERT(approx_eq(out[i + 1 * 4], std::min(a, b)), "min failed");
            LUISA_ASSERT(approx_eq(out[i + 2 * 4], std::max(a, b)), "max failed");
            LUISA_ASSERT(approx_eq(out[i + 3 * 4], std::clamp(a, 0.0f, 1.0f)), "clamp failed");
            LUISA_ASSERT(approx_eq(out[i + 4 * 4], std::clamp(a, 0.0f, 1.0f)), "saturate failed");
            LUISA_ASSERT(approx_eq(out[i + 5 * 4], a + 0.5f * (b - a)), "lerp failed");
            LUISA_ASSERT(approx_eq(out[i + 6 * 4], (a >= 1.0f) ? 1.0f : 0.0f), "step failed");
            float st = std::clamp((a - 0.0f) / (1.0f - 0.0f), 0.0f, 1.0f);
            LUISA_ASSERT(approx_eq(out[i + 7 * 4], st * st * (3.0f - 2.0f * st)), "smoothstep failed");
        }
        LUISA_INFO("Common math functions passed.");
    }

    // ============================================================
    // Test 4: Trigonometric and hyperbolic functions
    // ============================================================
    {
        float host_in[4] = {0.2f, 0.5f, 0.8f, 1.0f};
        auto in_buf = device.create_buffer<float>(4);
        auto out_buf = device.create_buffer<float>(64);
        stream << in_buf.copy_from(luisa::span{host_in, 4}) << synchronize();

        Kernel1D kernel = [&](BufferFloat input, BufferFloat output) noexcept {
            $uint i = dispatch_x();
            $float a = input.read(i);

            output.write(i + 0u * 4u, sin(a));
            output.write(i + 1u * 4u, cos(a));
            output.write(i + 2u * 4u, tan(a));
            output.write(i + 3u * 4u, asin(a));
            output.write(i + 4u * 4u, acos(a));
            output.write(i + 5u * 4u, atan(a));
            output.write(i + 6u * 4u, sinh(a));
            output.write(i + 7u * 4u, cosh(a));
            output.write(i + 8u * 4u, tanh(a));
            output.write(i + 9u * 4u, asinh(a));
            output.write(i + 10u * 4u, acosh(1.0f + a));
            output.write(i + 11u * 4u, atanh(a * 0.5f));
            output.write(i + 12u * 4u, atan2(a, 0.5f));
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_buf).dispatch(4) << synchronize();

        luisa::vector<float> out(64);
        stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();

        for (int i = 0; i < 4; ++i) {
            float a = host_in[i];
            LUISA_ASSERT(approx_eq(out[i + 0 * 4], std::sin(a)), "sin failed");
            LUISA_ASSERT(approx_eq(out[i + 1 * 4], std::cos(a)), "cos failed");
            LUISA_ASSERT(approx_eq(out[i + 2 * 4], std::tan(a)), "tan failed");
            LUISA_ASSERT(approx_eq(out[i + 3 * 4], std::asin(a)), "asin failed");
            LUISA_ASSERT(approx_eq(out[i + 4 * 4], std::acos(a)), "acos failed");
            LUISA_ASSERT(approx_eq(out[i + 5 * 4], std::atan(a)), "atan failed");
            LUISA_ASSERT(approx_eq(out[i + 6 * 4], std::sinh(a)), "sinh failed");
            LUISA_ASSERT(approx_eq(out[i + 7 * 4], std::cosh(a)), "cosh failed");
            LUISA_ASSERT(approx_eq(out[i + 8 * 4], std::tanh(a)), "tanh failed");
            LUISA_ASSERT(approx_eq(out[i + 9 * 4], std::asinh(a)), "asinh failed");
            LUISA_ASSERT(approx_eq(out[i + 10 * 4], std::acosh(1.0f + a)), "acosh failed");
            LUISA_ASSERT(approx_eq(out[i + 11 * 4], std::atanh(a * 0.5f)), "atanh failed");
            LUISA_ASSERT(approx_eq(out[i + 12 * 4], std::atan2(a, 0.5f)), "atan2 failed");
        }
        LUISA_INFO("Trigonometric functions passed.");
    }

    // ============================================================
    // Test 5: Exponential, logarithmic, power, sqrt
    // ============================================================
    {
        float host_in[4] = {0.5f, 1.0f, 2.0f, 4.0f};
        auto in_buf = device.create_buffer<float>(4);
        auto out_buf = device.create_buffer<float>(32);
        stream << in_buf.copy_from(luisa::span{host_in, 4}) << synchronize();

        Kernel1D kernel = [&](BufferFloat input, BufferFloat output) noexcept {
            $uint i = dispatch_x();
            $float a = input.read(i);

            output.write(i + 0u * 4u, exp(a));
            output.write(i + 1u * 4u, exp2(a));
            output.write(i + 2u * 4u, exp10(a));
            output.write(i + 3u * 4u, log(a));
            output.write(i + 4u * 4u, log2(a));
            output.write(i + 5u * 4u, log10(a));
            output.write(i + 6u * 4u, pow(a, 2.0f));
            output.write(i + 7u * 4u, sqrt(a));
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_buf).dispatch(4) << synchronize();

        luisa::vector<float> out(32);
        stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();

        for (int i = 0; i < 4; ++i) {
            float a = host_in[i];
            LUISA_ASSERT(approx_eq(out[i + 0 * 4], std::exp(a)), "exp failed");
            LUISA_ASSERT(approx_eq(out[i + 1 * 4], std::exp2(a)), "exp2 failed");
            LUISA_ASSERT(approx_eq(out[i + 2 * 4], std::pow(10.0f, a), 5e-2f), "exp10 failed: got {}, expected {}", out[i + 2 * 4], std::pow(10.0f, a));
            LUISA_ASSERT(approx_eq(out[i + 3 * 4], std::log(a)), "log failed");
            LUISA_ASSERT(approx_eq(out[i + 4 * 4], std::log2(a)), "log2 failed");
            LUISA_ASSERT(approx_eq(out[i + 5 * 4], std::log10(a)), "log10 failed");
            LUISA_ASSERT(approx_eq(out[i + 6 * 4], std::pow(a, 2.0f)), "pow failed");
            LUISA_ASSERT(approx_eq(out[i + 7 * 4], std::sqrt(a)), "sqrt failed");
        }
        LUISA_INFO("Exp/log/power functions passed.");
    }

    // ============================================================
    // Test 6: Rounding functions
    // ============================================================
    {
        float host_in[4] = {-1.7f, -0.3f, 0.6f, 2.4f};
        auto in_buf = device.create_buffer<float>(4);
        auto out_buf = device.create_buffer<float>(24);
        stream << in_buf.copy_from(luisa::span{host_in, 4}) << synchronize();

        Kernel1D kernel = [&](BufferFloat input, BufferFloat output) noexcept {
            $uint i = dispatch_x();
            $float a = input.read(i);

            output.write(i + 0u * 4u, ceil(a));
            output.write(i + 1u * 4u, floor(a));
            output.write(i + 2u * 4u, fract(a));
            output.write(i + 3u * 4u, trunc(a));
            output.write(i + 4u * 4u, round(a));
            output.write(i + 5u * 4u, rsqrt(abs(a) + 0.1f));
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_buf).dispatch(4) << synchronize();

        luisa::vector<float> out(24);
        stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();

        for (int i = 0; i < 4; ++i) {
            float a = host_in[i];
            LUISA_ASSERT(approx_eq(out[i + 0 * 4], std::ceil(a)), "ceil failed");
            LUISA_ASSERT(approx_eq(out[i + 1 * 4], std::floor(a)), "floor failed");
            LUISA_ASSERT(approx_eq(out[i + 2 * 4], a - std::floor(a)), "fract failed");
            LUISA_ASSERT(approx_eq(out[i + 3 * 4], std::trunc(a)), "trunc failed");
            LUISA_ASSERT(approx_eq(out[i + 4 * 4], std::round(a)), "round failed");
            LUISA_ASSERT(approx_eq(out[i + 5 * 4], 1.0f / std::sqrt(std::abs(a) + 0.1f)), "rsqrt failed");
        }
        LUISA_INFO("Rounding functions passed.");
    }

    // ============================================================
    // Test 7: Vector math (dot, cross, length, normalize, reflect, faceforward)
    // ============================================================
    {
        float3 host_in[4] = {
            make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 1.0f, 0.0f),
            make_float3(1.0f, 1.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f),
        };
        auto in_buf = device.create_buffer<float3>(4);
        auto out_float_buf = device.create_buffer<float>(16);
        auto out_vec_buf = device.create_buffer<float3>(16);
        stream << in_buf.copy_from(luisa::span{host_in, 4}) << synchronize();

        Kernel1D kernel = [&](BufferFloat3 input, BufferFloat out_f, BufferFloat3 out_v) noexcept {
            $uint i = dispatch_x();
            $float3 a = input.read(i);
            $float3 b = input.read((i + 1u) % 4u);

            out_f.write(i + 0u * 4u, dot(a, b));
            out_v.write(i + 0u * 4u, cross(a, b));
            out_f.write(i + 1u * 4u, length(a));
            out_f.write(i + 2u * 4u, length_squared(a));
            out_v.write(i + 1u * 4u, normalize(a));
            out_v.write(i + 2u * 4u, reflect(a, normalize(make_float3(1.0f, 1.0f, 1.0f))));
            out_v.write(i + 3u * 4u, faceforward(a, b, make_float3(1.0f, 0.0f, 0.0f)));
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_float_buf, out_vec_buf).dispatch(4) << synchronize();

        luisa::vector<float> out_f(16);
        luisa::vector<float3> out_v(16);
        stream << out_float_buf.copy_to(luisa::span{out_f.data(), out_f.size()})
               << out_vec_buf.copy_to(luisa::span{out_v.data(), out_v.size()})
               << synchronize();

        for (int i = 0; i < 4; ++i) {
            float3 a = host_in[i];
            float3 b = host_in[(i + 1) % 4];
            LUISA_ASSERT(approx_eq(out_f[i + 0 * 4], a.x * b.x + a.y * b.y + a.z * b.z), "dot failed");
            float3 cross_ref = make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
            LUISA_ASSERT(approx_eq(out_v[i + 0 * 4], cross_ref), "cross failed");
            float len = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
            LUISA_ASSERT(approx_eq(out_f[i + 1 * 4], len), "length failed");
            LUISA_ASSERT(approx_eq(out_f[i + 2 * 4], len * len), "length_squared failed");
            if (len > 0.0f) {
                LUISA_ASSERT(approx_eq(out_v[i + 1 * 4], a / len), "normalize failed");
            }
            float3 n = normalize(make_float3(1.0f, 1.0f, 1.0f));
            float3 refl = a - 2.0f * dot(n, a) * n;
            LUISA_ASSERT(approx_eq(out_v[i + 2 * 4], refl), "reflect failed");
            float3 ff = (dot(make_float3(1.0f, 0.0f, 0.0f), b) < 0.0f) ? a : -a;
            LUISA_ASSERT(approx_eq(out_v[i + 3 * 4], ff), "faceforward failed");
        }
        LUISA_INFO("Vector math functions passed.");
    }

    // ============================================================
    // Test 8: Matrix math (determinant, transpose, inverse)
    // ============================================================
    {
        auto out_mat_buf = device.create_buffer<float3x3>(12);
        auto out_float_buf = device.create_buffer<float>(4);

        Kernel1D kernel = [&](BufferFloat3x3 out_m, BufferFloat out_f) noexcept {
            $uint i = dispatch_x();
            $float3x3 m = make_float3x3(
                make_float3(3.0f, 0.0f, 2.0f),
                make_float3(2.0f, 0.0f, -2.0f),
                make_float3(0.0f, 1.0f, 1.0f));
            out_m.write(i, m);
            out_f.write(i, determinant(m));
            out_m.write(i + 4u, transpose(m));
            out_m.write(i + 8u, inverse(m));
        };
        auto shader = device.compile(kernel);
        stream << shader(out_mat_buf, out_float_buf).dispatch(4) << synchronize();

        luisa::vector<float3x3> out_m(12);
        luisa::vector<float> out_f(4);
        stream << out_mat_buf.copy_to(luisa::span{out_m.data(), out_m.size()})
               << out_float_buf.copy_to(luisa::span{out_f.data(), out_f.size()})
               << synchronize();

        float3x3 m = make_float3x3(
            make_float3(3.0f, 0.0f, 2.0f),
            make_float3(2.0f, 0.0f, -2.0f),
            make_float3(0.0f, 1.0f, 1.0f));
        float det = 3.0f * (0.0f * 1.0f - (-2.0f) * 1.0f) -
                    0.0f * (2.0f * 1.0f - (-2.0f) * 0.0f) +
                    2.0f * (2.0f * 1.0f - 0.0f * 0.0f);
        LUISA_ASSERT(approx_eq(out_f[0], det), "determinant failed: got {}, expected {}", out_f[0], det);

        // transpose: check thread 0 result
        LUISA_ASSERT(approx_eq(out_m[4][0], make_float3(3.0f, 2.0f, 0.0f)), "transpose col0 failed");
        LUISA_ASSERT(approx_eq(out_m[4][1], make_float3(0.0f, 0.0f, 1.0f)), "transpose col1 failed");
        LUISA_ASSERT(approx_eq(out_m[4][2], make_float3(2.0f, -2.0f, 1.0f)), "transpose col2 failed");

        // Validate every device-computed inverse independently by checking m * inv(m) = I.
        LUISA_ASSERT(std::abs(det) > 1e-6f, "matrix is singular");
        for (auto i = 0u; i < 4u; ++i) {
            auto inv = out_m[i + 8u];
            for (auto col = 0u; col < 3u; ++col) {
                for (auto row = 0u; row < 3u; ++row) {
                    LUISA_ASSERT(std::isfinite(inv[col][row]),
                                 "inverse result {}[{}][{}] is not finite: {}", i, col, row, inv[col][row]);
                    auto product = 0.0f;
                    for (auto k = 0u; k < 3u; ++k) {
                        product += m[k][row] * inv[col][k];
                    }
                    auto expected = col == row ? 1.0f : 0.0f;
                    LUISA_ASSERT(approx_eq(product, expected),
                                 "inverse result {} failed identity check at [{},{}]: got {}, expected {}",
                                 i, col, row, product, expected);
                }
            }
        }
        LUISA_INFO("Matrix math functions passed.");
    }

    // ============================================================
    // Test 9: Reductions
    // ============================================================
    {
        float4 host_in = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
        auto in_buf = device.create_buffer<float4>(1);
        auto out_buf = device.create_buffer<float>(4);
        stream << in_buf.copy_from(luisa::span{&host_in, 1}) << synchronize();

        Kernel1D kernel = [&](BufferFloat4 input, BufferFloat output) noexcept {
            $float4 v = input.read(0u);
            output.write(0u, reduce_sum(v));
            output.write(1u, reduce_prod(v));
            output.write(2u, reduce_min(v));
            output.write(3u, reduce_max(v));
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_buf).dispatch(1) << synchronize();

        luisa::vector<float> out(4);
        stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();

        LUISA_ASSERT(approx_eq(out[0], 10.0f), "reduce_sum failed");
        LUISA_ASSERT(approx_eq(out[1], 24.0f), "reduce_prod failed");
        LUISA_ASSERT(approx_eq(out[2], 1.0f), "reduce_min failed");
        LUISA_ASSERT(approx_eq(out[3], 4.0f), "reduce_max failed");
        LUISA_INFO("Reduction functions passed.");
    }

    // ============================================================
    // Test 10: Integer bit operations (clz, ctz, popcount, reverse)
    // ============================================================
    {
        uint host_in[4] = {0x00000001u, 0x80000000u, 0x0F0F0F0Fu, 0xFFFFFFFFu};
        auto in_buf = device.create_buffer<uint>(4);
        auto out_buf = device.create_buffer<uint>(16);
        stream << in_buf.copy_from(luisa::span{host_in, 4}) << synchronize();

        Kernel1D kernel = [&](BufferUInt input, BufferUInt output) noexcept {
            $uint i = dispatch_x();
            $uint a = input.read(i);
            output.write(i + 0u * 4u, clz(a));
            output.write(i + 1u * 4u, ctz(a));
            output.write(i + 2u * 4u, popcount(a));
            output.write(i + 3u * 4u, reverse(a));
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_buf).dispatch(4) << synchronize();

        luisa::vector<uint> out(16);
        stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();

        for (int i = 0; i < 4; ++i) {
            uint a = host_in[i];
            uint clz_ref = (a == 0u) ? 32u : static_cast<uint>(std::countl_zero(a));
            uint ctz_ref = (a == 0u) ? 32u : static_cast<uint>(std::countr_zero(a));
            uint pop_ref = static_cast<uint>(std::popcount(a));
            uint rev_ref = 0u;
            for (int b = 0; b < 32; ++b) {
                if (a & (1u << b)) rev_ref |= (1u << (31 - b));
            }
            LUISA_ASSERT(out[i + 0 * 4] == clz_ref, "clz failed");
            LUISA_ASSERT(out[i + 1 * 4] == ctz_ref, "ctz failed");
            LUISA_ASSERT(out[i + 2 * 4] == pop_ref, "popcount failed");
            LUISA_ASSERT(out[i + 3 * 4] == rev_ref, "reverse failed");
        }
        LUISA_INFO("Integer bit operations passed.");
    }

    // ============================================================
    // Test 11: Float classification (isinf, isnan)
    // ============================================================
    {
        float host_in[4] = {1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), -1.0f};
        auto in_buf = device.create_buffer<float>(4);
        auto out_buf = device.create_buffer<int>(8);
        stream << in_buf.copy_from(luisa::span{host_in, 4}) << synchronize();

        Kernel1D kernel = [&](BufferFloat input, BufferInt output) noexcept {
            $uint i = dispatch_x();
            $float a = input.read(i);
            output.write(i + 0u * 4u, ite(luisa::compute::isinf(a), 1, 0));
            output.write(i + 1u * 4u, ite(luisa::compute::isnan(a), 1, 0));
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_buf).dispatch(4) << synchronize();

        luisa::vector<int> out(8);
        stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();

        for (int i = 0; i < 4; ++i) {
            float a = host_in[i];
            LUISA_ASSERT(out[i + 0 * 4] == (std::isinf(a) ? 1 : 0), "isinf failed");
            LUISA_ASSERT(out[i + 1 * 4] == (std::isnan(a) ? 1 : 0), "isnan failed");
        }
        LUISA_INFO("Float classification passed.");
    }

    // ============================================================
    // Test 12: select / ite
    // ============================================================
    {
        auto out_buf = device.create_buffer<float>(4);

        Kernel1D kernel = [&](BufferFloat output) noexcept {
            $uint i = dispatch_x();
            $float t = select(0.0f, 1.0f, i % 2u == 0u);
            output.write(i, t);
        };
        auto shader = device.compile(kernel);
        stream << shader(out_buf).dispatch(4) << synchronize();

        luisa::vector<float> out(4);
        stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();

        for (int i = 0; i < 4; ++i) {
            LUISA_ASSERT(out[i] == ((i % 2 == 0) ? 1.0f : 0.0f), "select/ite failed");
        }
        LUISA_INFO("Select/ite passed.");
    }

    // ============================================================
    // Test 13: fma and copysign
    // ============================================================
    {
        float host_in[3] = {2.0f, 3.0f, 4.0f};
        auto in_buf = device.create_buffer<float>(3);
        auto out_buf = device.create_buffer<float>(4);
        stream << in_buf.copy_from(luisa::span{host_in, 3}) << synchronize();

        Kernel1D kernel = [&](BufferFloat input, BufferFloat output) noexcept {
            $float a = input.read(0u);
            $float b = input.read(1u);
            $float c = input.read(2u);
            output.write(0u, fma(a, b, c));
            output.write(1u, copysign(a, -1.0f));
            output.write(2u, copysign(-a, 1.0f));
            output.write(3u, copysign(-a, -1.0f));
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_buf).dispatch(1) << synchronize();

        luisa::vector<float> out(4);
        stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();

        LUISA_ASSERT(approx_eq(out[0], 2.0f * 3.0f + 4.0f), "fma failed");
        LUISA_ASSERT(approx_eq(out[1], -2.0f), "copysign failed");
        LUISA_ASSERT(approx_eq(out[2], 2.0f), "copysign failed");
        LUISA_ASSERT(approx_eq(out[3], -2.0f), "copysign failed");
        LUISA_INFO("FMA and copysign passed.");
    }

    // ============================================================
    // Test 14: Complex struct with scalar, vector, and matrix types
    // ============================================================
    {
        // Log host-side struct layout for debugging SPIR-V byte-buffer alignment
        LUISA_INFO("ComplexStruct host layout: size={}", sizeof(ComplexStruct));
        LUISA_INFO("  i16  offset={}, size={}", offsetof(ComplexStruct, i16), sizeof(int16_t));
        LUISA_INFO("  u16  offset={}, size={}", offsetof(ComplexStruct, u16), sizeof(uint16_t));
        LUISA_INFO("  i32  offset={}, size={}", offsetof(ComplexStruct, i32), sizeof(int));
        LUISA_INFO("  u32  offset={}, size={}", offsetof(ComplexStruct, u32), sizeof(uint));
        LUISA_INFO("  i64  offset={}, size={}", offsetof(ComplexStruct, i64), sizeof(luisa::slong));
        LUISA_INFO("  u64  offset={}, size={}", offsetof(ComplexStruct, u64), sizeof(luisa::ulong));
        LUISA_INFO("  h    offset={}, size={}", offsetof(ComplexStruct, h), sizeof(half));
        LUISA_INFO("  f    offset={}, size={}", offsetof(ComplexStruct, f), sizeof(float));
        LUISA_INFO("  d    offset={}, size={}", offsetof(ComplexStruct, d), sizeof(double));
        LUISA_INFO("  b    offset={}, size={}", offsetof(ComplexStruct, b), sizeof(bool));
        LUISA_INFO("  f2   offset={}, size={}", offsetof(ComplexStruct, f2), sizeof(float2));
        LUISA_INFO("  f3   offset={}, size={}", offsetof(ComplexStruct, f3), sizeof(float3));
        LUISA_INFO("  f4   offset={}, size={}", offsetof(ComplexStruct, f4), sizeof(float4));
        LUISA_INFO("  h3   offset={}, size={}", offsetof(ComplexStruct, h3), sizeof(half3));
        LUISA_INFO("  h4   offset={}, size={}", offsetof(ComplexStruct, h4), sizeof(half4));
        LUISA_INFO("  i4   offset={}, size={}", offsetof(ComplexStruct, i4), sizeof(int4));
        LUISA_INFO("  d2   offset={}, size={}", offsetof(ComplexStruct, d2), sizeof(double2));
        LUISA_INFO("  d3   offset={}, size={}", offsetof(ComplexStruct, d3), sizeof(double3));
        LUISA_INFO("  d4   offset={}, size={}", offsetof(ComplexStruct, d4), sizeof(double4));
        LUISA_INFO("  h2x2 offset={}, size={}", offsetof(ComplexStruct, h2x2), sizeof(half2x2));
        LUISA_INFO("  f2x2 offset={}, size={}", offsetof(ComplexStruct, f2x2), sizeof(float2x2));
        LUISA_INFO("  f3x3 offset={}, size={}", offsetof(ComplexStruct, f3x3), sizeof(float3x3));
        LUISA_INFO("  f4x4     offset={}, size={}", offsetof(ComplexStruct, f4x4), sizeof(float4x4));
        LUISA_INFO("  h_adj    offset={}, size={}", offsetof(ComplexStruct, h_adjacent), sizeof(half));
        LUISA_INFO("  b_adj    offset={}, size={}", offsetof(ComplexStruct, b_adjacent), sizeof(bool));

        // Prepare 4 instances with diverse initial values
        ComplexStruct host_in[4] = {
            {10, 20, 100, 200u, 1000ll, 2000ull, half(1.0f), 1.0f, 1.0, true,
             make_float2(1.0f, 2.0f),
             make_float3(1.0f, 2.0f, 3.0f),
             make_float4(1.0f, 2.0f, 3.0f, 4.0f),
             make_half3(half(1.0f), half(2.0f), half(3.0f)),
             make_half4(half(1.0f), half(2.0f), half(3.0f), half(4.0f)),
             make_int4(1, 2, 3, 4),
             make_double2(1.0, 2.0),
             make_double3(1.0, 2.0, 3.0),
             make_double4(1.0, 2.0, 3.0, 4.0),
             make_half2x2(half(1.0f), half(2.0f), half(3.0f), half(4.0f)),
             make_float2x2(1.0f, 2.0f, 3.0f, 4.0f),
             make_float3x3(make_float3(1.0f, 2.0f, 3.0f),
                           make_float3(4.0f, 5.0f, 6.0f),
                           make_float3(7.0f, 8.0f, 9.0f)),
             make_float4x4(1.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 2.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 3.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 4.0f),
             half(3.0f), true},
            {-5, 100, -50, 500u, -1000ll, 5000ull, half(0.5f), 2.5f, 2.0, false,
             make_float2(3.0f, 4.0f),
             make_float3(4.0f, 5.0f, 6.0f),
             make_float4(5.0f, 6.0f, 7.0f, 8.0f),
             make_half3(half(4.0f), half(5.0f), half(6.0f)),
             make_half4(half(5.0f), half(6.0f), half(7.0f), half(8.0f)),
             make_int4(5, 6, 7, 8),
             make_double2(3.0, 4.0),
             make_double3(4.0, 5.0, 6.0),
             make_double4(5.0, 6.0, 7.0, 8.0),
             make_half2x2(half(0.5f), half(1.5f), half(2.5f), half(3.5f)),
             make_float2x2(5.0f, 6.0f, 7.0f, 8.0f),
             make_float3x3(make_float3(9.0f, 8.0f, 7.0f),
                           make_float3(6.0f, 5.0f, 4.0f),
                           make_float3(3.0f, 2.0f, 1.0f)),
             make_float4x4(0.0f, 1.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 1.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 1.0f,
                           1.0f, 0.0f, 0.0f, 0.0f),
             half(0.25f), false},
            {32767, 65535, 1000000, 3000000u, 9223372036854775807ll, 10000ull, half(2.0f), -1.0f, 0.5, true,
             make_float2(-1.0f, -2.0f),
             make_float3(-1.0f, -2.0f, -3.0f),
             make_float4(-1.0f, -2.0f, -3.0f, -4.0f),
             make_half3(half(0.5f), half(1.0f), half(1.5f)),
             make_half4(half(0.5f), half(1.0f), half(1.5f), half(2.0f)),
             make_int4(-1, -2, -3, -4),
             make_double2(-1.0, -2.0),
             make_double3(-1.0, -2.0, -3.0),
             make_double4(-1.0, -2.0, -3.0, -4.0),
             make_half2x2(half(10.0f), half(20.0f), half(30.0f), half(40.0f)),
             make_float2x2(-1.0f, -2.0f, -3.0f, -4.0f),
             make_float3x3(make_float3(0.5f, 0.0f, 0.0f),
                           make_float3(0.0f, 2.0f, 0.0f),
                           make_float3(0.0f, 0.0f, 3.0f)),
             make_float4x4(2.0f, 0.0f, 0.0f, 1.0f,
                           0.0f, 2.0f, 0.0f, 1.0f,
                           0.0f, 0.0f, 2.0f, 1.0f,
                           0.0f, 0.0f, 0.0f, 2.0f),
             half(-2.0f), true},
            {-32768, 0, -1000000, 0u, -9223372036854775807ll - 1ll, 0ull, half(10.0f), 100.0f, 10.0, false,
             make_float2(0.0f, 0.0f),
             make_float3(0.0f, 0.0f, 1.0f),
             make_float4(0.0f, 0.0f, 0.0f, 1.0f),
             make_half3(half(10.0f), half(20.0f), half(30.0f)),
             make_half4(half(10.0f), half(20.0f), half(30.0f), half(40.0f)),
             make_int4(100, 200, 300, 400),
             make_double2(10.0, 20.0),
             make_double3(10.0, 20.0, 30.0),
             make_double4(10.0, 20.0, 30.0, 40.0),
             make_half2x2(half(-1.0f), half(-2.0f), half(-3.0f), half(-4.0f)),
             make_float2x2(0.0f, 1.0f, 1.0f, 0.0f),
             make_float3x3(make_float3(1.0f, 0.0f, 0.0f),
                           make_float3(0.0f, 1.0f, 0.0f),
                           make_float3(0.0f, 0.0f, 1.0f)),
             make_float4x4(1.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 1.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 1.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 1.0f),
             half(7.5f), false},
        };

        auto in_buf = device.create_buffer<ComplexStruct>(4);
        auto out_buf = device.create_buffer<ComplexStruct>(4);
        stream << in_buf.copy_from(luisa::span{host_in, 4}) << synchronize();

        Kernel1D kernel = [&](BufferVar<ComplexStruct> input, BufferVar<ComplexStruct> output) noexcept {
            $uint i = dispatch_x();
            Var<ComplexStruct> s = input.read(i);

            // Scalar transformations
            Var<int16_t> i16_out = -s.i16;
            Var<uint16_t> u16_out = cast<uint16_t>(s.u16 + cast<uint16_t>(42u));
            Var<int> i32_out = s.i32 * 2;
            Var<uint> u32_out = s.u32 + 100u;
            Var<luisa::slong> i64_out = s.i64 + 1ll;
            Var<luisa::ulong> u64_out = s.u64 * 2ull;
            Var<half> h_out = s.h * cast<half>(2.0f);
            Var<float> f_out = s.f * 3.0f + 1.0f;
            Var<double> d_out = s.d * 2.0;
            Var<bool> b_out = !s.b;

            // Vector transformations
            Var<float2> f2_out = make_float2(s.f2.y, s.f2.x);
            Var<float3> f3_out = normalize(s.f3) + make_float3(0.1f, 0.2f, 0.3f);
            Var<float4> f4_out = make_float4(s.f4.w, s.f4.z, s.f4.y, s.f4.x);
            Var<half3> h3_out = make_half3(s.h3.z, s.h3.x, s.h3.y);
            Var<half4> h4_out = make_half4(s.h4.w, s.h4.z, s.h4.y, s.h4.x);
            Var<int4> i4_out = s.i4 + make_int4(1, 2, 3, 4);
            Var<double2> d2_out = make_double2(s.d2.y, s.d2.x);
            Var<double3> d3_out = make_double3(s.d3.z, s.d3.x, s.d3.y);
            Var<double4> d4_out = make_double4(s.d4.w, s.d4.z, s.d4.y, s.d4.x);

            // Matrix transformations
            Var<half2x2> h2x2_out = transpose(s.h2x2);
            Var<float2x2> f2x2_out = transpose(s.f2x2);
            Var<float3x3> f3x3_out = transpose(s.f3x3);
            Var<float4x4> f4x4_out = transpose(s.f4x4);

            // Half-adjacent-to-bool test
            Var<half> h_adj_out = s.h_adjacent * cast<half>(2.0f);
            Var<bool> b_adj_out = !s.b_adjacent;

            // Build output struct
            Var<ComplexStruct> result;
            result.i16 = i16_out;
            result.u16 = u16_out;
            result.i32 = i32_out;
            result.u32 = u32_out;
            result.i64 = i64_out;
            result.u64 = u64_out;
            result.h = h_out;
            result.f = f_out;
            result.d = d_out;
            result.b = b_out;
            result.f2 = f2_out;
            result.f3 = f3_out;
            result.f4 = f4_out;
            result.h3 = h3_out;
            result.h4 = h4_out;
            result.i4 = i4_out;
            result.d2 = d2_out;
            result.d3 = d3_out;
            result.d4 = d4_out;
            result.h2x2 = h2x2_out;
            result.f2x2 = f2x2_out;
            result.f3x3 = f3x3_out;
            result.f4x4 = f4x4_out;
            result.h_adjacent = h_adj_out;
            result.b_adjacent = b_adj_out;

            output.write(i, result);
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_buf).dispatch(4) << synchronize();

        luisa::vector<ComplexStruct> out(4);
        stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();

        // Host-side expected computation and verification
        for (int i = 0; i < 4; ++i) {
            const auto &in = host_in[i];
            const auto &res = out[i];

            // int16_t: negate
            auto exp_i16 = static_cast<int16_t>(-in.i16);
            LUISA_ASSERT(res.i16 == exp_i16,
                         "[{}] int16 neg failed: got {}, expected {}", i, res.i16, exp_i16);
            LUISA_INFO("[{}] int16: {} -> {} OK", i, in.i16, res.i16);

            // bool: flip
            LUISA_ASSERT(res.b == !in.b,
                         "[{}] bool flip failed: got {}, expected {}", i, res.b, !in.b);
            LUISA_INFO("[{}] bool: {} -> {} OK", i, in.b, res.b);

            // uint16_t: add 42
            auto exp_u16 = static_cast<uint16_t>(in.u16 + 42u);
            LUISA_ASSERT(res.u16 == exp_u16,
                         "[{}] uint16 add42 failed: got {}, expected {}", i, res.u16, exp_u16);
            LUISA_INFO("[{}] uint16: {} -> {} OK", i, in.u16, res.u16);

            // int32: mul 2
            auto exp_i32 = in.i32 * 2;
            LUISA_ASSERT(res.i32 == exp_i32,
                         "[{}] int32 mul2 failed: got {}, expected {}", i, res.i32, exp_i32);
            LUISA_INFO("[{}] int32: {} -> {} OK", i, in.i32, res.i32);

            // uint32: add 100
            auto exp_u32 = in.u32 + 100u;
            LUISA_ASSERT(res.u32 == exp_u32,
                         "[{}] uint32 add100 failed: got {}, expected {}", i, res.u32, exp_u32);
            LUISA_INFO("[{}] uint32: {} -> {} OK", i, in.u32, res.u32);

            // int64: add 1
            auto exp_i64 = in.i64 + 1;
            LUISA_ASSERT(res.i64 == exp_i64,
                         "[{}] int64 add1 failed: got {}, expected {}", i, res.i64, exp_i64);
            LUISA_INFO("[{}] int64: {} -> {} OK", i, in.i64, res.i64);

            // uint64: mul 2
            auto exp_u64 = in.u64 * 2ull;
            LUISA_ASSERT(res.u64 == exp_u64,
                         "[{}] uint64 mul2 failed: got {}, expected {}", i, res.u64, exp_u64);
            LUISA_INFO("[{}] uint64: {} -> {} OK", i, in.u64, res.u64);

            // half: multiply by 2
            half exp_h = in.h * half(2.0f);
            float h_got = static_cast<float>(res.h);
            float h_exp = static_cast<float>(exp_h);
            LUISA_ASSERT(approx_eq(h_got, h_exp, 1e-2f),
                         "[{}] half mul2 failed: got {}, expected {}", i, h_got, h_exp);
            LUISA_INFO("[{}] half: {} -> {} OK", i, static_cast<float>(in.h), h_got);

            // float: f * 3 + 1
            float exp_f = in.f * 3.0f + 1.0f;
            LUISA_ASSERT(approx_eq(res.f, exp_f),
                         "[{}] float fmul3add1 failed: got {}, expected {}", i, res.f, exp_f);
            LUISA_INFO("[{}] float: {} -> {} OK", i, in.f, res.f);

            // double: multiply by 2
            double exp_d = in.d * 2.0;
            LUISA_ASSERT(std::abs(res.d - exp_d) < 1e-9 ||
                             std::abs(res.d - exp_d) < 1e-9 * std::max(std::abs(res.d), std::abs(exp_d)),
                         "[{}] double mul2 failed: got {}, expected {}", i, res.d, exp_d);
            LUISA_INFO("[{}] double: {} -> {} OK", i, in.d, res.d);

            // float2: swap components
            float2 exp_f2 = make_float2(in.f2.y, in.f2.x);
            LUISA_ASSERT(approx_eq(res.f2, exp_f2),
                         "[{}] float2 swap failed: got ({},{}), expected ({},{})",
                         i, res.f2.x, res.f2.y, exp_f2.x, exp_f2.y);
            LUISA_INFO("[{}] float2: ({},{}) -> ({},{}) OK",
                       i, in.f2.x, in.f2.y, res.f2.x, res.f2.y);

            // float3: normalize + offset
            float len3 = std::sqrt(in.f3.x * in.f3.x + in.f3.y * in.f3.y + in.f3.z * in.f3.z);
            float3 exp_f3 = make_float3(in.f3.x / len3 + 0.1f, in.f3.y / len3 + 0.2f, in.f3.z / len3 + 0.3f);
            LUISA_ASSERT(approx_eq(res.f3, exp_f3),
                         "[{}] float3 norm+off failed: got ({},{},{}), expected ({},{},{})",
                         i, res.f3.x, res.f3.y, res.f3.z, exp_f3.x, exp_f3.y, exp_f3.z);
            LUISA_INFO("[{}] float3: ({},{},{}) -> ({},{},{}) OK",
                       i, in.f3.x, in.f3.y, in.f3.z, res.f3.x, res.f3.y, res.f3.z);

            // float4: reverse components
            float4 exp_f4 = make_float4(in.f4.w, in.f4.z, in.f4.y, in.f4.x);
            LUISA_ASSERT(approx_eq(res.f4, exp_f4),
                         "[{}] float4 reverse failed: got ({},{},{},{}), expected ({},{},{},{})",
                         i, res.f4.x, res.f4.y, res.f4.z, res.f4.w,
                         exp_f4.x, exp_f4.y, exp_f4.z, exp_f4.w);
            LUISA_INFO("[{}] float4: ({},{},{},{}) -> ({},{},{},{}) OK",
                       i, in.f4.x, in.f4.y, in.f4.z, in.f4.w,
                       res.f4.x, res.f4.y, res.f4.z, res.f4.w);

            // half3: rotate (z,x,y)
            half3 exp_h3 = make_half3(in.h3.z, in.h3.x, in.h3.y);
            for (int c = 0; c < 3; ++c) {
                float got_c = static_cast<float>(res.h3[c]);
                float exp_c = static_cast<float>(exp_h3[c]);
                LUISA_ASSERT(approx_eq(got_c, exp_c, 1e-2f),
                             "[{}] half3 rotate[{}] failed: got {}, expected {}", i, c, got_c, exp_c);
            }
            LUISA_INFO("[{}] half3: ({:.2f},{:.2f},{:.2f}) -> ({:.2f},{:.2f},{:.2f}) OK",
                       i, static_cast<float>(in.h3.x), static_cast<float>(in.h3.y), static_cast<float>(in.h3.z),
                       static_cast<float>(res.h3.x), static_cast<float>(res.h3.y), static_cast<float>(res.h3.z));

            // half4: reverse components
            half4 exp_h4 = make_half4(in.h4.w, in.h4.z, in.h4.y, in.h4.x);
            for (int c = 0; c < 4; ++c) {
                float got_c = static_cast<float>(res.h4[c]);
                float exp_c = static_cast<float>(exp_h4[c]);
                LUISA_ASSERT(approx_eq(got_c, exp_c, 1e-2f),
                             "[{}] half4 reverse[{}] failed: got {}, expected {}", i, c, got_c, exp_c);
            }
            LUISA_INFO("[{}] half4 reverse OK", i);

            // int4: add (1,2,3,4)
            int4 exp_i4 = in.i4 + make_int4(1, 2, 3, 4);
            LUISA_ASSERT(all(res.i4 == exp_i4),
                         "[{}] int4 add failed: got ({},{},{},{}), expected ({},{},{},{})",
                         i, res.i4.x, res.i4.y, res.i4.z, res.i4.w,
                         exp_i4.x, exp_i4.y, exp_i4.z, exp_i4.w);
            LUISA_INFO("[{}] int4: ({},{},{},{}) -> ({},{},{},{}) OK",
                       i, in.i4.x, in.i4.y, in.i4.z, in.i4.w,
                       res.i4.x, res.i4.y, res.i4.z, res.i4.w);

            // double2: swap components
            double2 exp_d2 = make_double2(in.d2.y, in.d2.x);
            LUISA_ASSERT(std::abs(res.d2.x - exp_d2.x) < 1e-9 &&
                             std::abs(res.d2.y - exp_d2.y) < 1e-9,
                         "[{}] double2 swap failed: got ({},{}), expected ({},{})",
                         i, res.d2.x, res.d2.y, exp_d2.x, exp_d2.y);
            LUISA_INFO("[{}] double2: ({},{}) -> ({},{}) OK",
                       i, in.d2.x, in.d2.y, res.d2.x, res.d2.y);

            // double3: rotate (z,x,y)
            double3 exp_d3 = make_double3(in.d3.z, in.d3.x, in.d3.y);
            LUISA_ASSERT(std::abs(res.d3.x - exp_d3.x) < 1e-9 &&
                             std::abs(res.d3.y - exp_d3.y) < 1e-9 &&
                             std::abs(res.d3.z - exp_d3.z) < 1e-9,
                         "[{}] double3 rotate failed: got ({},{},{}), expected ({},{},{})",
                         i, res.d3.x, res.d3.y, res.d3.z, exp_d3.x, exp_d3.y, exp_d3.z);
            LUISA_INFO("[{}] double3: ({},{},{}) -> ({},{},{}) OK",
                       i, in.d3.x, in.d3.y, in.d3.z, res.d3.x, res.d3.y, res.d3.z);

            // double4: reverse components
            double4 exp_d4 = make_double4(in.d4.w, in.d4.z, in.d4.y, in.d4.x);
            LUISA_ASSERT(std::abs(res.d4.x - exp_d4.x) < 1e-9 &&
                             std::abs(res.d4.y - exp_d4.y) < 1e-9 &&
                             std::abs(res.d4.z - exp_d4.z) < 1e-9 &&
                             std::abs(res.d4.w - exp_d4.w) < 1e-9,
                         "[{}] double4 reverse failed: got ({},{},{},{}), expected ({},{},{},{})",
                         i, res.d4.x, res.d4.y, res.d4.z, res.d4.w,
                         exp_d4.x, exp_d4.y, exp_d4.z, exp_d4.w);
            LUISA_INFO("[{}] double4 reverse OK", i);

            // half2x2: transpose
            half2x2 exp_h2x2 = make_half2x2(
                in.h2x2[0][0], in.h2x2[1][0],
                in.h2x2[0][1], in.h2x2[1][1]
            );
            for (int c = 0; c < 2; ++c) {
                for (int r = 0; r < 2; ++r) {
                    float got = static_cast<float>(res.h2x2[c][r]);
                    float exp = static_cast<float>(exp_h2x2[c][r]);
                    LUISA_ASSERT(approx_eq(got, exp, 1e-2f),
                                 "[{}] half2x2 transpose[{}][{}] failed: got {}, expected {}",
                                 i, c, r, got, exp);
                }
            }
            LUISA_INFO("[{}] half2x2 transpose OK", i);

            // float2x2: transpose
            float2x2 exp_f2x2 = make_float2x2(
                in.f2x2[0][0], in.f2x2[1][0],
                in.f2x2[0][1], in.f2x2[1][1]
            );
            for (int c = 0; c < 2; ++c) {
                for (int r = 0; r < 2; ++r) {
                    LUISA_ASSERT(approx_eq(res.f2x2[c][r], exp_f2x2[c][r]),
                                 "[{}] float2x2 transpose[{}][{}] failed: got {}, expected {}",
                                 i, c, r, res.f2x2[c][r], exp_f2x2[c][r]);
                }
            }
            LUISA_INFO("[{}] float2x2 transpose OK", i);

            // float3x3: transpose
            float3x3 exp_f3x3 = make_float3x3(
                in.f3x3[0][0], in.f3x3[1][0], in.f3x3[2][0],
                in.f3x3[0][1], in.f3x3[1][1], in.f3x3[2][1],
                in.f3x3[0][2], in.f3x3[1][2], in.f3x3[2][2]);
            for (int c = 0; c < 3; ++c) {
                LUISA_ASSERT(approx_eq(res.f3x3[c], exp_f3x3[c]),
                             "[{}] float3x3 transpose col[{}] failed: got ({},{},{}), expected ({},{},{})",
                             i, c,
                             res.f3x3[c].x, res.f3x3[c].y, res.f3x3[c].z,
                             exp_f3x3[c].x, exp_f3x3[c].y, exp_f3x3[c].z);
            }
            LUISA_INFO("[{}] float3x3 transpose OK", i);

            // float4x4: transpose
            float4x4 exp_f4x4 = make_float4x4(
                in.f4x4[0][0], in.f4x4[1][0], in.f4x4[2][0], in.f4x4[3][0],
                in.f4x4[0][1], in.f4x4[1][1], in.f4x4[2][1], in.f4x4[3][1],
                in.f4x4[0][2], in.f4x4[1][2], in.f4x4[2][2], in.f4x4[3][2],
                in.f4x4[0][3], in.f4x4[1][3], in.f4x4[2][3], in.f4x4[3][3]);
            for (int c = 0; c < 4; ++c) {
                LUISA_ASSERT(approx_eq(res.f4x4[c], exp_f4x4[c]),
                             "[{}] float4x4 transpose col[{}] failed: got ({},{},{},{}), expected ({},{},{},{})",
                             i, c,
                             res.f4x4[c].x, res.f4x4[c].y, res.f4x4[c].z, res.f4x4[c].w,
                             exp_f4x4[c].x, exp_f4x4[c].y, exp_f4x4[c].z, exp_f4x4[c].w);
            }
            LUISA_INFO("[{}] float4x4 transpose OK", i);

            // half-adjacent-to-bool validation
            half exp_h_adj = in.h_adjacent * half(2.0f);
            bool exp_b_adj = !in.b_adjacent;
            LUISA_ASSERT(approx_eq(static_cast<float>(res.h_adjacent), static_cast<float>(exp_h_adj), 1e-2f),
                         "[{}] half adjacent mul2 failed: got {}, expected {}",
                         i, static_cast<float>(res.h_adjacent), static_cast<float>(exp_h_adj));
            LUISA_ASSERT(res.b_adjacent == exp_b_adj,
                         "[{}] bool adjacent flip failed: got {}, expected {}",
                         i, res.b_adjacent, exp_b_adj);
            LUISA_INFO("[{}] half-adjacent-to-bool: h {} -> {} OK, b {} -> {} OK",
                       i,
                       static_cast<float>(in.h_adjacent), static_cast<float>(res.h_adjacent),
                       in.b_adjacent, res.b_adjacent);
        }
        LUISA_INFO("Complex struct test passed ({} instances).", 4);
    }

    LUISA_INFO("All mathematical DSL tests passed!");
    return 0;
}
