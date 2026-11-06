// Tests for the elementary special functions added to the LuisaCompute DSL.
//
// Feature under test:
//   - `erf(x)`, `erfc(x)`, `expm1(x)`      unary, scalar / float2 / float4
//   - `ldexp(x, n)`                        mixed-type (floatN, intN): scalar,
//                                          broadcast and same-shape exponents,
//                                          32- and 64-bit, signed and unsigned,
//                                          including the saturation contract for
//                                          exponents that do not fit the 32-bit
//                                          `int` the target entry points take
//   - the double-precision path (`double`, `double2`) of the CUDA device library,
//     i.e. the regression guard for the DP->SP demotion that these overloads had
//
// Every expected value comes from the host `<cmath>` implementation, never from a
// DSL expression, so the two sides are independent.
//
// Backend coverage: these ops are native on cuda (both codegen routes), hip,
// metal, metal4, fallback and simd; `ldexp` is also native on the HLSL route (dx
// and the Vulkan compatibility route) and on the native XIR-to-SPIR-V route.
// Those two routes have no erf/erfc/expm1 instruction and reject them while
// generating code, so the three unary ops are skipped on dx/vk instead of being
// attempted (the rejection is process-fatal; see the TODO below).
//
// The fail-closed rejection of erf/erfc/expm1 on dx and on the native Vulkan
// SPIR-V route is asserted in test_backend_fatal_rejections (separate process,
// because LuisaCompute reports it through LUISA_ERROR: log + abort).
//
// TODO(platform): metal, metal4, hip, simd and the fallback backend are not built
// in the reference configuration, so their lowering of these four ops is
// compile-checked only. Re-run this test on those backends where they are built.
// TODO(platform): f16 / bf16 overloads of `lc_erf`/`lc_erfc`/`lc_expm1`/`lc_ldexp`
// are not covered: the DSL has no half/bf16 arithmetic path in this build, so only
// the compiler sees them.
// TODO(platform): `ldexp` with a 64-bit exponent is exercised through the CUDA AST
// route and the HLSL/SPIR-V routes here; the 64-bit clamp of the CUDA *LLVM* route
// (cuda_codegen_llvm_impl_arith.cpp), HIP, simd and metal4 is compile-checked only,
// because those routes are not built in this configuration.
// TODO(backend): test_device_math - the only other suite that covers this MathOp
// surface - currently aborts during AST construction on all three backends, so it
// cannot cross-check the numeric tables below. It is out of scope for this round
// (see RESEARCH_TESTS.md section 5 for the two discriminating recipes).
// TODO(platform): the denormal contract asserted by this file was measured on one
// platform only (RTX 4060, driver 13020, CUDA toolkit 12.9) with the default compile
// options: cuda flushes subnormal float results (the backend passes `-use_fast_math`
// to NVRTC, which implies `--ftz=true`; `ShaderOption::enable_fast_math` defaults to
// true), dx and vk preserve them. Re-measure it on any new GPU, driver or backend
// before trusting those assertions.

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace test_dsl_special_functions_detail {

constexpr auto kFloatEps = 1e-4f;
constexpr auto kRelativeEps = 1e-4f;

[[nodiscard]] bool approx_eq(float got, float expected, float eps = kFloatEps) noexcept {
    if (got == expected) { return true; }
    if (!std::isfinite(got) || !std::isfinite(expected)) { return false; }
    auto scale = std::max(std::abs(got), std::abs(expected));
    return std::abs(got - expected) <= eps * std::max(1.0f, scale);
}

/// Results below 1e-3 need a *relative* comparison: at `erfc(5)` (~1.5e-12),
/// `erfc(27)` (underflows to zero in float) or `erf(1e-8)` (~1.1e-8) an absolute
/// 1e-4 tolerance would accept a completely wrong implementation.
[[nodiscard]] bool accurate_enough(float got, float expected) noexcept {
    if (got == expected) { return true; }
    if (!std::isfinite(got) || !std::isfinite(expected)) { return false; }
    if (std::abs(expected) < 1e-3f) {
        return std::abs(got - expected) <= kRelativeEps * std::abs(expected);
    }
    return approx_eq(got, expected);
}

/// The three unary ops are unavailable on the HLSL route and on the native
/// XIR-to-SPIR-V route (both fail closed at code-generation time).
[[nodiscard]] bool supports_unary_special_functions(luisa::string_view backend) noexcept {
    return backend != "dx" && backend != "vk";
}

/// Measured denormal behaviour of the reference platform (RTX 4060, driver 13020,
/// CUDA toolkit 12.9) with the default `ShaderOption::enable_fast_math == true`, for
/// the three backends that are built here:
///   cuda    - the backend passes `-use_fast_math` to NVRTC, which implies
///             `--ftz=true`, so a subnormal float *result* may be flushed to zero.
///             That is a compile option, not a device property: test_ldexp therefore
///             accepts both the exact subnormal and a zero there, and requires the
///             two expected subnormals to agree;
///   dx, vk  - a subnormal float result is preserved bit-exactly.
/// test_ldexp asserts this both ways below; this predicate is only the tolerance
/// that the per-case table applies (a zero where a subnormal was expected is the
/// one accepted deviation, and only where the compiler flushes).
[[nodiscard]] bool flushes_denormal_results(luisa::string_view backend) noexcept {
    return backend == "cuda";
}

/// `ldexp` is exact by definition, so results are compared bit for bit; the only
/// accepted deviation is a flushed subnormal on a backend that demonstrably
/// flushes (`flushes_denormal_results`).
[[nodiscard]] bool ldexp_matches(float got, float expected,
                                 luisa::string_view backend) noexcept {
    if (std::bit_cast<uint32_t>(got) == std::bit_cast<uint32_t>(expected)) { return true; }
    return got == 0.0f &&
           flushes_denormal_results(backend) &&
           std::fpclassify(expected) == FP_SUBNORMAL;
}

/// Inputs shared by the unary-op tables. They cover the linear region, the
/// saturation of `erf`/`expm1`, the far tail of `erfc`, and negative arguments.
constexpr std::array kUnaryInputs{
    0.0f, 1e-8f, -1e-8f, 0.5f, -0.5f, 1.0f, -1.0f, 2.0f, 27.0f};

template<typename Op>
void check_unary(Device &device, Stream &stream, const char *name,
                 Op &&op, float (*reference)(float)) noexcept {
    constexpr auto n = kUnaryInputs.size();
    auto out_buf = device.create_buffer<float>(n);
    Kernel1D kernel = [&](BufferFloat output) noexcept {
        for (auto i = 0u; i < n; i++) {
            output.write(i, op(def(kUnaryInputs[i])));
        }
    };
    auto shader = device.compile(kernel);
    stream << shader(out_buf).dispatch(1u) << synchronize();
    luisa::vector<float> out(n);
    stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();
    for (auto i = 0u; i < n; i++) {
        auto x = kUnaryInputs[i];
        auto expected = reference(x);
        expect(accurate_enough(out[i], expected))
            << name << "(" << x << ") = " << out[i] << ", expected " << expected;
    }
    LUISA_INFO("{}: scalar passed ({} inputs).", name, n);
}

template<typename Op>
void check_unary_float2(Device &device, Stream &stream, const char *name,
                        Op &&op, float (*reference)(float)) noexcept {
    constexpr auto n = kUnaryInputs.size() / 2u;
    auto out_buf = device.create_buffer<float2>(n);
    Kernel1D kernel = [&](BufferFloat2 output) noexcept {
        for (auto i = 0u; i < n; i++) {
            output.write(i, op(make_float2(def(kUnaryInputs[2u * i]),
                                           def(kUnaryInputs[2u * i + 1u]))));
        }
    };
    auto shader = device.compile(kernel);
    stream << shader(out_buf).dispatch(1u) << synchronize();
    luisa::vector<float2> out(n);
    stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();
    for (auto i = 0u; i < n; i++) {
        auto x = kUnaryInputs[2u * i];
        auto y = kUnaryInputs[2u * i + 1u];
        expect(accurate_enough(out[i].x, reference(x)))
            << name << "(float2(" << x << ", " << y << ")).x = " << out[i].x
            << ", expected " << reference(x);
        expect(accurate_enough(out[i].y, reference(y)))
            << name << "(float2(" << x << ", " << y << ")).y = " << out[i].y
            << ", expected " << reference(y);
    }
    LUISA_INFO("{}: float2 passed ({} lanes).", name, 2u * n);
}

template<typename Op>
void check_unary_float4(Device &device, Stream &stream, const char *name,
                        Op &&op, float (*reference)(float)) noexcept {
    constexpr auto n = kUnaryInputs.size() / 4u;
    auto out_buf = device.create_buffer<float4>(n);
    Kernel1D kernel = [&](BufferFloat4 output) noexcept {
        for (auto i = 0u; i < n; i++) {
            output.write(i, op(make_float4(def(kUnaryInputs[4u * i]),
                                           def(kUnaryInputs[4u * i + 1u]),
                                           def(kUnaryInputs[4u * i + 2u]),
                                           def(kUnaryInputs[4u * i + 3u]))));
        }
    };
    auto shader = device.compile(kernel);
    stream << shader(out_buf).dispatch(1u) << synchronize();
    luisa::vector<float4> out(n);
    stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();
    for (auto i = 0u; i < n; i++) {
        auto lanes = std::array{out[i].x, out[i].y, out[i].z, out[i].w};
        for (auto lane = 0u; lane < 4u; lane++) {
            auto x = kUnaryInputs[4u * i + lane];
            expect(accurate_enough(lanes[lane], reference(x)))
                << name << "(float4(...)).lane" << lane << " = " << lanes[lane]
                << ", expected " << reference(x) << " for input " << x;
        }
    }
    LUISA_INFO("{}: float4 passed ({} lanes).", name, 4u * n);
}

void test_special_functions(Device &device, Stream &stream) noexcept {
    auto backend = luisa::string_view{device.backend_name()};
    if (!supports_unary_special_functions(backend)) {
        LUISA_INFO("Skipping erf/erfc/expm1 on '{}': this backend has no such "
                   "instruction and rejects them while generating code.",
                   backend);
        return;
    }
    auto erf_op = [](auto x) noexcept { return erf(x); };
    auto erfc_op = [](auto x) noexcept { return erfc(x); };
    auto expm1_op = [](auto x) noexcept { return expm1(x); };
    check_unary(device, stream, "erf", erf_op, [](float x) { return std::erf(x); });
    check_unary(device, stream, "erfc", erfc_op, [](float x) { return std::erfc(x); });
    check_unary(device, stream, "expm1", expm1_op, [](float x) { return std::expm1(x); });
    check_unary_float2(device, stream, "erf", erf_op, [](float x) { return std::erf(x); });
    check_unary_float2(device, stream, "erfc", erfc_op, [](float x) { return std::erfc(x); });
    check_unary_float2(device, stream, "expm1", expm1_op, [](float x) { return std::expm1(x); });
    check_unary_float4(device, stream, "erf", erf_op, [](float x) { return std::erf(x); });
    check_unary_float4(device, stream, "erfc", erfc_op, [](float x) { return std::erfc(x); });
    check_unary_float4(device, stream, "expm1", expm1_op, [](float x) { return std::expm1(x); });
    // The small-argument cancellation is the whole point of expm1: `exp(x) - 1`
    // loses every significant digit at 1e-8, so a relative assertion catches a
    // naive implementation that the absolute tolerance above would accept.
    {
        auto out_buf = device.create_buffer<float>(2u);
        Kernel1D kernel = [](BufferFloat output) noexcept {
            output.write(0u, expm1(def(1e-8f)));
            output.write(1u, expm1(def(-1e-8f)));
        };
        auto shader = device.compile(kernel);
        stream << shader(out_buf).dispatch(1u) << synchronize();
        luisa::vector<float> out(2u);
        stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();
        for (auto i = 0u; i < 2u; i++) {
            auto x = i == 0u ? 1e-8f : -1e-8f;
            auto expected = std::expm1(x);
            expect(std::abs(out[i] - expected) <= 1e-6f * std::abs(expected))
                << "expm1(" << x << ") = " << out[i] << ", expected " << expected
                << " (relative 1e-6: exp(x) - 1 is not an acceptable implementation)";
        }
        LUISA_INFO("expm1: small-argument cancellation passed.");
    }
}

void test_ldexp(Device &device, Stream &stream) noexcept {
    auto backend = luisa::string_view{device.backend_name()};
    constexpr std::array host_mantissa{1.0f, 1.5f, -1.5f, 3.0f};
    constexpr std::array host_exponent{0, 1, -1, 10, -10, 24, 127, 128, -127, -149};
    constexpr auto n = host_mantissa.size() * host_exponent.size();
    auto out_buf = device.create_buffer<float>(n);
    Kernel1D kernel = [&](BufferFloat output) noexcept {
        for (auto i = 0u; i < host_mantissa.size(); i++) {
            for (auto j = 0u; j < host_exponent.size(); j++) {
                output.write(i * host_exponent.size() + j,
                             ldexp(def(host_mantissa[i]), def(host_exponent[j])));
            }
        }
    };
    auto shader = device.compile(kernel);
    stream << shader(out_buf).dispatch(1u) << synchronize();
    luisa::vector<float> out(n);
    stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();
    for (auto i = 0u; i < host_mantissa.size(); i++) {
        for (auto j = 0u; j < host_exponent.size(); j++) {
            auto x = host_mantissa[i];
            auto e = host_exponent[j];
            auto expected = std::ldexp(x, e);
            auto got = out[i * host_exponent.size() + j];
            expect(ldexp_matches(got, expected, backend))
                << "ldexp(" << x << ", " << e << ") = " << got
                << " (0x" << std::hex << std::bit_cast<uint32_t>(got) << std::dec
                << "), expected " << expected;
        }
    }
    LUISA_INFO("ldexp: scalar table passed ({} cases).", n);

    // Broadcast (float2 mantissa, scalar int exponent) and same-shape
    // (float2 / int2) exponents. The mantissa is read from a buffer so that the
    // expression is a DSL value rather than an all-constant call.
    {
        luisa::vector<float2> host_in{{1.0f, 1.5f}, {-1.5f, 3.0f}};
        auto in_buf = device.create_buffer<float2>(host_in.size());
        auto out_buf2 = device.create_buffer<float2>(3u * host_in.size());
        stream << in_buf.copy_from(luisa::span{host_in}) << synchronize();
        Kernel1D vec_kernel = [](BufferFloat2 input, BufferFloat2 output) noexcept {
            for (auto i = 0u; i < 2u; i++) {
                output.write(i, ldexp(input.read(i), def(10)));
                output.write(2u + i, ldexp(input.read(i), make_int2(10, -10)));
                // XIR explicitly permits a scalar exponent beside a vector
                // mantissa, including calls built without the DSL broadcast
                // helper. Every AST backend must lower that shape as well.
                auto builder = luisa::compute::detail::FunctionBuilder::current();
                auto mantissa = input.read(i);
                auto exponent = def(10);
                auto direct = def<float2>(builder->call(
                    Type::of<float2>(), CallOp::LDEXP,
                    {mantissa.expression(), exponent.expression()}));
                output.write(4u + i, direct);
            }
        };
        auto vec_shader = device.compile(vec_kernel);
        stream << vec_shader(in_buf, out_buf2).dispatch(1u) << synchronize();
        luisa::vector<float2> out2(3u * host_in.size());
        stream << out_buf2.copy_to(luisa::span{out2.data(), out2.size()}) << synchronize();
        for (auto i = 0u; i < host_in.size(); i++) {
            expect(out2[i].x == std::ldexp(host_in[i].x, 10) &&
                   out2[i].y == std::ldexp(host_in[i].y, 10))
                << "ldexp(float2, scalar int) lane mismatch at " << i;
            expect(out2[2u + i].x == std::ldexp(host_in[i].x, 10) &&
                   out2[2u + i].y == std::ldexp(host_in[i].y, -10))
                << "ldexp(float2, int2) lane mismatch at " << i;
            expect(out2[4u + i].x == std::ldexp(host_in[i].x, 10) &&
                   out2[4u + i].y == std::ldexp(host_in[i].y, 10))
                << "direct AST ldexp(float2, scalar int) lane mismatch at " << i;
        }
        LUISA_INFO("ldexp: float2 broadcast, int2 and direct scalar-exponent AST calls passed.");
    }

    // Saturation: the exponent is an unlimited-range integer in the reference
    // semantics, but every target entry point takes a 32-bit `int`, so an
    // out-of-range exponent must saturate (an exponent beyond the 32-bit range
    // already saturates the mantissa to +inf or +0). A truncating or
    // reinterpreting implementation is caught here on the 32-bit as well as on the
    // 64-bit path.
    {
        constexpr std::array<const char *, 6u> case_name{
            "(uint)0x80000000", "(uint)4000000000", "(uint)200",
            "(long)5000000000", "(ulong)0x80000000", "(long)-5000000000"};
        constexpr std::array<bool, 6u> case_is_inf{true, true, true, true, true, false};
        auto out_buf3 = device.create_buffer<float>(case_name.size());
        Kernel1D sat_kernel = [](BufferFloat output) noexcept {
            output.write(0u, ldexp(def(1.0f), def(0x80000000u)));
            output.write(1u, ldexp(def(1.0f), def(4000000000u)));
            output.write(2u, ldexp(def(1.0f), def(200u)));
            output.write(3u, ldexp(def(1.0f), def(static_cast<int64_t>(5000000000LL))));
            output.write(4u, ldexp(def(1.0f), def(static_cast<uint64_t>(0x80000000ULL))));
            output.write(5u, ldexp(def(1.0f), def(static_cast<int64_t>(-5000000000LL))));
        };
        auto sat_shader = device.compile(sat_kernel);
        stream << sat_shader(out_buf3).dispatch(1u) << synchronize();
        luisa::vector<float> out3(case_name.size());
        stream << out_buf3.copy_to(luisa::span{out3.data(), out3.size()}) << synchronize();
        for (auto i = 0u; i < case_name.size(); i++) {
            if (case_is_inf[i]) {
                expect(std::isinf(out3[i]) && out3[i] > 0.0f)
                    << "ldexp(1.0f, " << case_name[i] << ") = " << out3[i]
                    << ", expected +inf";
            } else {
                expect(out3[i] == 0.0f)
                    << "ldexp(1.0f, " << case_name[i] << ") = " << out3[i]
                    << ", expected +0";
            }
        }
        LUISA_INFO("ldexp: 32/64-bit exponent saturation passed.");
    }

    // Same rule on the vector path: a uint2 exponent is clamped lane by lane.
    {
        luisa::vector<float2> host_in{{1.0f, 1.0f}};
        auto in_buf = device.create_buffer<float2>(1u);
        auto out_buf4 = device.create_buffer<float2>(1u);
        stream << in_buf.copy_from(luisa::span{host_in}) << synchronize();
        Kernel1D vec_sat_kernel = [](BufferFloat2 input, BufferFloat2 output) noexcept {
            output.write(0u, ldexp(input.read(0u), make_uint2(0x80000000u, 200u)));
        };
        auto vec_sat_shader = device.compile(vec_sat_kernel);
        stream << vec_sat_shader(in_buf, out_buf4).dispatch(1u) << synchronize();
        luisa::vector<float2> out4(1u);
        stream << out_buf4.copy_to(luisa::span{out4.data(), out4.size()}) << synchronize();
        expect(std::isinf(out4[0].x) && out4[0].x > 0.0f &&
               std::isinf(out4[0].y) && out4[0].y > 0.0f)
            << "ldexp(float2, uint2(0x80000000, 200)) = (" << out4[0].x << ", "
            << out4[0].y << "), expected (+inf, +inf)";
        LUISA_INFO("ldexp: uint2 exponent saturation passed.");
    }

    // The denormal contract, as a *required* behaviour instead of the per-case
    // tolerance above. Measured on the reference platform (RTX 4060, driver 13020,
    // CUDA toolkit 12.9) with the default compile options, for the three backends
    // that are built here:
    //   cuda : a subnormal float result is flushed to zero, while a normal result
    //          stays exact. Measured on this box; the *cause* is unresolved
    //          (`ShaderOption::enable_fast_math` is on by default and the backend
    //          passes `-use_fast_math`, which implies `--ftz=true`, but removing the
    //          flag did not change the observed behaviour). The contract asserted
    //          here is therefore consistency rather than flushing: each expected
    //          subnormal is either returned exactly or flushed to zero, and all of
    //          them agree. An accuracy fix (fast math off, a non-ftz ldexp path)
    //          stays green; a partial flush does not.
    //   dx, vk  : the subnormal result is preserved bit-exactly (measured), so a
    //             flush there would be a regression.
    // Both directions are asserted, so neither a silent switch to flush-to-zero nor
    // one away from it can pass unnoticed.
    auto slot_of = [&](float mantissa, int exponent) noexcept {
        for (auto i = 0u; i < host_mantissa.size(); i++) {
            for (auto j = 0u; j < host_exponent.size(); j++) {
                if (host_mantissa[i] == mantissa && host_exponent[j] == exponent) {
                    return i * host_exponent.size() + j;
                }
            }
        }
        LUISA_ERROR("Missing ldexp table entry ({}, {}).", mantissa, exponent);
    };
    auto subnormal_149 = out[slot_of(1.0f, -149)];
    auto subnormal_127 = out[slot_of(1.0f, -127)];
    auto normal_scaled = out[slot_of(3.0f, -127)];// 1.5 * FLT_MIN, a normal result
    auto is_exact = [](float got, float expected) noexcept {
        return std::bit_cast<uint32_t>(got) == std::bit_cast<uint32_t>(expected);
    };
    if (flushes_denormal_results(backend)) {
        // cuda today (see the predicate). `-use_fast_math` implies `--ftz=true`, but
        // that is a compile option, so both outcomes are accepted as long as they are
        // *uniform*: a mixed result would mean two calls of the same ldexp path
        // disagree, which no compile option explains.
        auto exact_149 = is_exact(subnormal_149, std::ldexp(1.0f, -149));
        auto exact_127 = is_exact(subnormal_127, std::ldexp(1.0f, -127));
        expect(exact_149 || std::bit_cast<uint32_t>(subnormal_149) == 0u)
            << "ldexp(1.0f, -149) must be either the exact subnormal 2^-149 or +0 on '"
            << backend << "' (fast math flushes denormal results), got "
            << subnormal_149;
        expect(exact_127 || std::bit_cast<uint32_t>(subnormal_127) == 0u)
            << "ldexp(1.0f, -127) must be either the exact subnormal 2^-127 or +0 on '"
            << backend << "' (fast math flushes denormal results), got "
            << subnormal_127;
        expect(exact_149 == exact_127)
            << "the denormal mode must be uniform: both expected subnormals must be "
               "returned exactly or both flushed, got "
            << subnormal_149 << " (exact: " << exact_149 << ") and "
            << subnormal_127 << " (exact: " << exact_127 << ")";
        expect(is_exact(normal_scaled, std::ldexp(3.0f, -127)))
            << "a normal ldexp result must stay exact on '" << backend
            << "', got " << normal_scaled << ", expected "
            << std::ldexp(3.0f, -127);
        LUISA_INFO("ldexp: denormal consistency contract asserted on '{}' "
                   "(subnormals {}).",
                   backend, exact_149 ? "returned exactly" : "flushed to zero");
    } else if (backend == "dx" || backend == "vk") {
        // Measured: both backends preserve the subnormal result bit-exactly, so a
        // flush here would be the regression.
        expect(is_exact(subnormal_149, std::ldexp(1.0f, -149)))
            << "ldexp(1.0f, -149) must stay the exact subnormal on '" << backend
            << "', got " << subnormal_149;
        expect(is_exact(subnormal_127, std::ldexp(1.0f, -127)))
            << "ldexp(1.0f, -127) must stay the exact subnormal on '" << backend
            << "', got " << subnormal_127;
        expect(is_exact(normal_scaled, std::ldexp(3.0f, -127)))
            << "a normal ldexp result must stay exact on '" << backend
            << "', got " << normal_scaled << ", expected "
            << std::ldexp(3.0f, -127);
        LUISA_INFO("ldexp: denormal-preserving contract asserted on '{}'.", backend);
    } else {
        LUISA_INFO("ldexp: unknown backend '{}', the denormal contract is not "
                   "asserted (only exact results are accepted).",
                   backend);
    }
}

/// The CUDA device library implements every transcendental for `double` and for
/// the `double2/3/4` vectors; a vector overload that calls the single-precision
/// function silently narrows every lane, which is invisible at float precision and
/// wrong at ~1e-7 relative. Every argument below is chosen so that a narrowed
/// implementation is off by far more than the 1e-12 relative tolerance.
void test_double_precision(Device &device, Stream &stream) noexcept {
    auto backend = luisa::string_view{device.backend_name()};
    if (backend != "cuda") {
        // TODO(platform): the vector DP overloads that carry the demotion bug live
        // in the CUDA device library, so this guard is CUDA-only by construction.
        // The equivalent checks on hip/metal/metal4/simd/fallback need their own
        // DP reference tables and are not implemented here.
        LUISA_INFO("Skipping the double-precision device-library guard on '{}' "
                   "(cuda-only device library).",
                   backend);
        return;
    }
    auto rel_eq = [](double got, double expected) noexcept {
        if (got == expected) { return true; }
        if (!std::isfinite(got) || !std::isfinite(expected)) { return false; }
        auto scale = std::max(std::abs(got), std::abs(expected));
        return std::abs(got - expected) <= 1e-12 * scale;
    };
    constexpr std::array<double, 3u> host_in{30.0, 30.0, 1e-8};
    auto in_buf = device.create_buffer<double>(host_in.size());
    auto out_buf = device.create_buffer<double>(host_in.size() * 5u);
    stream << in_buf.copy_from(luisa::span{host_in.data(), host_in.size()}) << synchronize();
    Kernel1D kernel = [](BufferVar<double> input, BufferVar<double> output) noexcept {
        $uint i = dispatch_x();
        Var<double> a = input.read(i);
        output.write(i + 0u * 3u, cosh(a));
        output.write(i + 1u * 3u, exp(a));
        output.write(i + 2u * 3u, log2(a));
        output.write(i + 3u * 3u, erf(a));
        output.write(i + 4u * 3u, expm1(a));
    };
    auto shader = device.compile(kernel);
    stream << shader(in_buf, out_buf).dispatch(host_in.size()) << synchronize();
    luisa::vector<double> out(host_in.size() * 5u);
    stream << out_buf.copy_to(luisa::span{out.data(), out.size()}) << synchronize();
    for (auto i = 0u; i < host_in.size(); i++) {
        auto a = host_in[i];
        expect(rel_eq(out[i + 0u * 3u], std::cosh(a)))
            << "double cosh(" << a << ") = " << out[i + 0u * 3u] << ", expected " << std::cosh(a);
        expect(rel_eq(out[i + 1u * 3u], std::exp(a)))
            << "double exp(" << a << ") = " << out[i + 1u * 3u] << ", expected " << std::exp(a);
        expect(rel_eq(out[i + 2u * 3u], std::log2(a)))
            << "double log2(" << a << ") = " << out[i + 2u * 3u] << ", expected " << std::log2(a);
        expect(rel_eq(out[i + 3u * 3u], std::erf(a)))
            << "double erf(" << a << ") = " << out[i + 3u * 3u] << ", expected " << std::erf(a);
        expect(rel_eq(out[i + 4u * 3u], std::expm1(a)))
            << "double expm1(" << a << ") = " << out[i + 4u * 3u] << ", expected " << std::expm1(a);
    }
    LUISA_INFO("double scalar transcendentals passed (relative 1e-12).");

    // The vector form is the one that used to narrow to float.
    auto in_buf2 = device.create_buffer<double2>(host_in.size());
    auto out_buf2 = device.create_buffer<double2>(host_in.size() * 3u);
    luisa::vector<double2> host_in2(host_in.size());
    for (auto i = 0u; i < host_in.size(); i++) {
        host_in2[i] = make_double2(host_in[i], -host_in[i]);
    }
    stream << in_buf2.copy_from(luisa::span{host_in2}) << synchronize();
    Kernel1D kernel2 = [](BufferVar<double2> input, BufferVar<double2> output) noexcept {
        $uint i = dispatch_x();
        Var<double2> a = input.read(i);
        output.write(i + 0u * 3u, cosh(a));
        output.write(i + 1u * 3u, exp(a));
        output.write(i + 2u * 3u, erf(a));
    };
    auto shader2 = device.compile(kernel2);
    stream << shader2(in_buf2, out_buf2).dispatch(host_in.size()) << synchronize();
    luisa::vector<double2> out2(host_in.size() * 3u);
    stream << out_buf2.copy_to(luisa::span{out2.data(), out2.size()}) << synchronize();
    for (auto i = 0u; i < host_in.size(); i++) {
        for (auto lane = 0u; lane < 2u; lane++) {
            auto a = lane == 0u ? host_in2[i].x : host_in2[i].y;
            auto cosh_got = lane == 0u ? out2[i + 0u * 3u].x : out2[i + 0u * 3u].y;
            auto exp_got = lane == 0u ? out2[i + 1u * 3u].x : out2[i + 1u * 3u].y;
            auto erf_got = lane == 0u ? out2[i + 2u * 3u].x : out2[i + 2u * 3u].y;
            expect(rel_eq(cosh_got, std::cosh(a)))
                << "double2 cosh lane " << lane << "(" << a << ") = " << cosh_got
                << ", expected " << std::cosh(a) << " (a narrowed implementation fails here)";
            expect(rel_eq(exp_got, std::exp(a)))
                << "double2 exp lane " << lane << "(" << a << ") = " << exp_got
                << ", expected " << std::exp(a);
            expect(rel_eq(erf_got, std::erf(a)))
                << "double2 erf lane " << lane << "(" << a << ") = " << erf_got
                << ", expected " << std::erf(a);
        }
    }
    LUISA_INFO("double2 transcendentals passed (DP->SP demotion guard).");
}

/// Relative 1e-12 comparison for the double tables: a float-narrowed overload is
/// ~1e-7 off, so the tolerance is what separates "double" from "float".
[[nodiscard]] bool dp_rel_eq(double got, double expected) noexcept {
    if (got == expected) { return true; }
    if (!std::isfinite(got) || !std::isfinite(expected)) { return false; }
    auto scale = std::max(std::abs(got), std::abs(expected));
    return std::abs(got - expected) <= 1e-12 * scale;
}

/// One row per demoted name. Every argument is exactly representable in float, so
/// a narrowed implementation gets the same input and differs only in the precision
/// of the result (~1e-7 relative), and every result is finite and well away from
/// zero, so the relative comparison cannot be satisfied by a float evaluation.
struct DemotedUnaryCase {
    const char *name;
    double (*reference)(double) noexcept;
    double argument_x;///< lane 0 of the double2 buffer / the scalar input
    double argument_y;///< lane 1 of the double2 buffer
};

/// The 17 unary names of the demotion set. The order matches the kernel bodies in
/// test_double_precision_demoted_names.
constexpr std::array kDemotedUnaryCases{
    DemotedUnaryCase{"acos", [](double x) noexcept { return std::acos(x); }, 0.5, 0.25},
    DemotedUnaryCase{"acosh", [](double x) noexcept { return std::acosh(x); }, 1.5, 2.0},
    DemotedUnaryCase{"asin", [](double x) noexcept { return std::asin(x); }, 0.5, 0.25},
    DemotedUnaryCase{"asinh", [](double x) noexcept { return std::asinh(x); }, 2.5, 1.5},
    DemotedUnaryCase{"atan", [](double x) noexcept { return std::atan(x); }, 2.5, 1.5},
    DemotedUnaryCase{"atanh", [](double x) noexcept { return std::atanh(x); }, 0.5, 0.25},
    DemotedUnaryCase{"cos", [](double x) noexcept { return std::cos(x); }, 0.5, 0.25},
    DemotedUnaryCase{"cosh", [](double x) noexcept { return std::cosh(x); }, 30.0, 20.0},
    DemotedUnaryCase{"exp", [](double x) noexcept { return std::exp(x); }, 0.5, 0.25},
    DemotedUnaryCase{"exp10", [](double x) noexcept { return std::pow(10.0, x); }, 0.5, 0.25},
    DemotedUnaryCase{"log", [](double x) noexcept { return std::log(x); }, 0.5, 0.25},
    DemotedUnaryCase{"log2", [](double x) noexcept { return std::log2(x); }, 3.0, 5.0},
    DemotedUnaryCase{"log10", [](double x) noexcept { return std::log10(x); }, 3.0, 5.0},
    DemotedUnaryCase{"sin", [](double x) noexcept { return std::sin(x); }, 0.5, 0.25},
    DemotedUnaryCase{"sinh", [](double x) noexcept { return std::sinh(x); }, 40.0, 20.0},
    DemotedUnaryCase{"tan", [](double x) noexcept { return std::tan(x); }, 0.5, 0.25},
    DemotedUnaryCase{"tanh", [](double x) noexcept { return std::tanh(x); }, 2.5, 0.25}};

static_assert(kDemotedUnaryCases.size() + 2u == 19u,
              "the demotion set is the 17 unary names plus atan2 and copysign");

/// All 19 demoted overloads on `double` and on `double2`. Only the three
/// (cosh/exp/log2) plus erf/expm1 were covered before; a float-narrowed body for any
/// of the other 16 was invisible to the whole suite.
void test_double_precision_demoted_names(Device &device, Stream &stream) noexcept {
    auto backend = luisa::string_view{device.backend_name()};
    if (backend != "cuda") {
        LUISA_INFO("Skipping the 19-name double table on '{}': the demoted overloads "
                   "live in the CUDA device library.",
                   backend);
        return;
    }
    constexpr auto n = kDemotedUnaryCases.size();

    // Scalar `double` form.
    {
        luisa::vector<double> host_in(n);
        for (auto i = 0u; i < n; i++) { host_in[i] = kDemotedUnaryCases[i].argument_x; }
        auto in_buf = device.create_buffer<double>(n);
        auto out_buf = device.create_buffer<double>(n);
        stream << in_buf.copy_from(luisa::span{host_in}) << synchronize();
        Kernel1D kernel = [](BufferVar<double> input, BufferVar<double> output) noexcept {
            output.write(0u, acos(input.read(0u)));
            output.write(1u, acosh(input.read(1u)));
            output.write(2u, asin(input.read(2u)));
            output.write(3u, asinh(input.read(3u)));
            output.write(4u, atan(input.read(4u)));
            output.write(5u, atanh(input.read(5u)));
            output.write(6u, cos(input.read(6u)));
            output.write(7u, cosh(input.read(7u)));
            output.write(8u, exp(input.read(8u)));
            output.write(9u, exp10(input.read(9u)));
            // `log` also names a host helper (luisa::log / ::log), so qualify it.
            output.write(10u, luisa::compute::log(input.read(10u)));
            output.write(11u, log2(input.read(11u)));
            output.write(12u, log10(input.read(12u)));
            output.write(13u, sin(input.read(13u)));
            output.write(14u, sinh(input.read(14u)));
            output.write(15u, tan(input.read(15u)));
            output.write(16u, tanh(input.read(16u)));
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_buf).dispatch(1u) << synchronize();
        luisa::vector<double> out(n);
        stream << out_buf.copy_to(luisa::span{out}) << synchronize();
        for (auto i = 0u; i < n; i++) {
            auto x = host_in[i];
            auto expected = kDemotedUnaryCases[i].reference(x);
            expect(dp_rel_eq(out[i], expected))
                << "double " << kDemotedUnaryCases[i].name << "(" << x << ") = "
                << out[i] << ", expected " << expected
                << " (relative 1e-12: a float-narrowed overload is ~1e-7 off)";
        }
        LUISA_INFO("double: all 17 unary demoted names passed (relative 1e-12).");
    }

    // `double2` form - the vector overloads are the ones that used to narrow.
    {
        luisa::vector<double2> host_in(n);
        for (auto i = 0u; i < n; i++) {
            host_in[i] = make_double2(kDemotedUnaryCases[i].argument_x,
                                      kDemotedUnaryCases[i].argument_y);
        }
        auto in_buf = device.create_buffer<double2>(n);
        auto out_buf = device.create_buffer<double2>(n);
        stream << in_buf.copy_from(luisa::span{host_in}) << synchronize();
        Kernel1D kernel = [](BufferVar<double2> input, BufferVar<double2> output) noexcept {
            output.write(0u, acos(input.read(0u)));
            output.write(1u, acosh(input.read(1u)));
            output.write(2u, asin(input.read(2u)));
            output.write(3u, asinh(input.read(3u)));
            output.write(4u, atan(input.read(4u)));
            output.write(5u, atanh(input.read(5u)));
            output.write(6u, cos(input.read(6u)));
            output.write(7u, cosh(input.read(7u)));
            output.write(8u, exp(input.read(8u)));
            output.write(9u, exp10(input.read(9u)));
            // `log` also names a host helper (luisa::log / ::log), so qualify it.
            output.write(10u, luisa::compute::log(input.read(10u)));
            output.write(11u, log2(input.read(11u)));
            output.write(12u, log10(input.read(12u)));
            output.write(13u, sin(input.read(13u)));
            output.write(14u, sinh(input.read(14u)));
            output.write(15u, tan(input.read(15u)));
            output.write(16u, tanh(input.read(16u)));
        };
        auto shader = device.compile(kernel);
        stream << shader(in_buf, out_buf).dispatch(1u) << synchronize();
        luisa::vector<double2> out(n);
        stream << out_buf.copy_to(luisa::span{out}) << synchronize();
        for (auto i = 0u; i < n; i++) {
            for (auto lane = 0u; lane < 2u; lane++) {
                auto x = lane == 0u ? host_in[i].x : host_in[i].y;
                auto got = lane == 0u ? out[i].x : out[i].y;
                auto expected = kDemotedUnaryCases[i].reference(x);
                expect(dp_rel_eq(got, expected))
                    << "double2 " << kDemotedUnaryCases[i].name << " lane " << lane
                    << "(" << x << ") = " << got << ", expected " << expected
                    << " (a narrowed lane fails here)";
            }
        }
        LUISA_INFO("double2: all 17 unary demoted names passed on both lanes.");
    }

    // The two binary members of the set: atan2 and copysign. copysign needs an
    // argument whose magnitude is not representable in float at all - 1e-300
    // flushes to zero through a float copy - so it is the one name where an exact
    // argument would hide the demotion.
    {
        luisa::vector<double> lhs{2.5, 1e-300};
        luisa::vector<double> rhs{3.5, -1.0};
        auto lhs_buf = device.create_buffer<double>(lhs.size());
        auto rhs_buf = device.create_buffer<double>(rhs.size());
        auto out_buf = device.create_buffer<double>(lhs.size());
        stream << lhs_buf.copy_from(luisa::span{lhs}) << synchronize();
        stream << rhs_buf.copy_from(luisa::span{rhs}) << synchronize();
        Kernel1D kernel = [](BufferVar<double> a, BufferVar<double> b,
                             BufferVar<double> output) noexcept {
            output.write(0u, atan2(a.read(0u), b.read(0u)));
            output.write(1u, copysign(a.read(1u), b.read(1u)));
        };
        auto shader = device.compile(kernel);
        stream << shader(lhs_buf, rhs_buf, out_buf).dispatch(1u) << synchronize();
        luisa::vector<double> out(lhs.size());
        stream << out_buf.copy_to(luisa::span{out}) << synchronize();
        expect(dp_rel_eq(out[0], std::atan2(2.5, 3.5)))
            << "double atan2(2.5, 3.5) = " << out[0] << ", expected "
            << std::atan2(2.5, 3.5);
        expect(dp_rel_eq(out[1], std::copysign(1e-300, -1.0)))
            << "double copysign(1e-300, -1.0) = " << out[1] << ", expected "
            << std::copysign(1e-300, -1.0)
            << " (a float copy flushes the magnitude to zero)";
        LUISA_INFO("double: atan2 and copysign passed.");
    }
    {
        luisa::vector<double2> lhs{make_double2(2.5, 1e-300), make_double2(1e300, 1e-300)};
        luisa::vector<double2> rhs{make_double2(3.5, -1.0), make_double2(-1.0, -1.0)};
        auto lhs_buf = device.create_buffer<double2>(lhs.size());
        auto rhs_buf = device.create_buffer<double2>(rhs.size());
        auto out_buf = device.create_buffer<double2>(lhs.size());
        stream << lhs_buf.copy_from(luisa::span{lhs}) << synchronize();
        stream << rhs_buf.copy_from(luisa::span{rhs}) << synchronize();
        Kernel1D kernel = [](BufferVar<double2> a, BufferVar<double2> b,
                             BufferVar<double2> output) noexcept {
            output.write(0u, atan2(a.read(0u), b.read(0u)));
            output.write(1u, copysign(a.read(1u), b.read(1u)));
        };
        auto shader = device.compile(kernel);
        stream << shader(lhs_buf, rhs_buf, out_buf).dispatch(1u) << synchronize();
        luisa::vector<double2> out(lhs.size());
        stream << out_buf.copy_to(luisa::span{out}) << synchronize();
        expect(dp_rel_eq(out[0].x, std::atan2(lhs[0].x, rhs[0].x)))
            << "double2 atan2 lane 0 = " << out[0].x;
        expect(dp_rel_eq(out[0].y, std::atan2(lhs[0].y, rhs[0].y)))
            << "double2 atan2 lane 1 = " << out[0].y << ", expected "
            << std::atan2(lhs[0].y, rhs[0].y);
        expect(dp_rel_eq(out[1].x, std::copysign(lhs[1].x, rhs[1].x)))
            << "double2 copysign lane 0 = " << out[1].x << ", expected "
            << std::copysign(lhs[1].x, rhs[1].x);
        expect(std::signbit(out[1].y) && dp_rel_eq(out[1].y, std::copysign(lhs[1].y, rhs[1].y)))
            << "double2 copysign lane 1 = " << out[1].y << ", expected "
            << std::copysign(lhs[1].y, rhs[1].y);
        LUISA_INFO("double2: atan2 and copysign passed on both lanes.");
    }
    // Non-integral exponents take the non-powi branch of the generated
    // powf_impl(lc_double, lc_double). A float intrinsic there loses far more
    // than the double-precision tolerance, and a generator/header drift must
    // not silently reintroduce that demotion.
    {
        luisa::vector<double> base{2.5, 7.25};
        luisa::vector<double> exponent{0.5, 1.25};
        auto base_buffer = device.create_buffer<double>(base.size());
        auto exponent_buffer = device.create_buffer<double>(exponent.size());
        auto output = device.create_buffer<double>(base.size());
        stream << base_buffer.copy_from(luisa::span{base})
               << exponent_buffer.copy_from(luisa::span{exponent})
               << synchronize();
        Kernel1D kernel = [](BufferVar<double> x, BufferVar<double> y,
                             BufferVar<double> out) noexcept {
            auto i = dispatch_x();
            out.write(i, pow(x.read(i), y.read(i)));
        };
        auto shader = device.compile(kernel);
        stream << shader(base_buffer, exponent_buffer, output).dispatch(base.size())
               << synchronize();
        luisa::vector<double> result(base.size());
        stream << output.copy_to(luisa::span{result}) << synchronize();
        for (auto i = 0u; i < result.size(); i++) {
            auto expected = std::pow(base[i], exponent[i]);
            expect(dp_rel_eq(result[i], expected))
                << "double pow(" << base[i] << ", " << exponent[i] << ") = "
                << result[i] << ", expected " << expected
                << " (the non-integral branch must not call powf)";
        }
        LUISA_INFO("double: non-integral pow stayed in double precision.");
    }
}

/// ldexp on `double` / `double2`: the same 16-overload generator emits
/// `lc_ldexp(lc_double, lc_int)`, and nothing exercised it before. ldexp is exact
/// by definition, so every case is compared bit for bit, including one exponent
/// that is a *normal* double but a float subnormal (2^-149) and one that is a
/// double subnormal (2^-1074, DBL_TRUE_MIN) - both are zero through a float copy.
void test_ldexp_double(Device &device, Stream &stream) noexcept {
    auto backend = luisa::string_view{device.backend_name()};
    if (backend != "cuda") {
        LUISA_INFO("Skipping ldexp on double on '{}' (CUDA device-library overload).",
                   backend);
        return;
    }
    constexpr std::array host_mantissa{1.0, 1.5, -1.5, 3.0};
    constexpr std::array host_exponent{0, 1, -1, 10, -127, -149, -1074, 1024};
    constexpr auto n = host_mantissa.size() * host_exponent.size();
    auto in_buf = device.create_buffer<double>(n);
    auto out_buf = device.create_buffer<double>(n);
    Kernel1D kernel = [&](BufferVar<double> output) noexcept {
        for (auto i = 0u; i < host_mantissa.size(); i++) {
            for (auto j = 0u; j < host_exponent.size(); j++) {
                output.write(i * host_exponent.size() + j,
                             ldexp(def(host_mantissa[i]), def(host_exponent[j])));
            }
        }
    };
    auto shader = device.compile(kernel);
    stream << shader(out_buf).dispatch(1u) << synchronize();
    luisa::vector<double> out(n);
    stream << out_buf.copy_to(luisa::span{out}) << synchronize();
    for (auto i = 0u; i < host_mantissa.size(); i++) {
        for (auto j = 0u; j < host_exponent.size(); j++) {
            auto x = host_mantissa[i];
            auto e = host_exponent[j];
            auto expected = std::ldexp(x, e);
            auto got = out[i * host_exponent.size() + j];
            expect(std::bit_cast<uint64_t>(got) == std::bit_cast<uint64_t>(expected))
                << "ldexp(double " << x << ", " << e << ") = " << got
                << " (0x" << std::hex << std::bit_cast<uint64_t>(got) << std::dec
                << "), expected " << expected << " (bit-exact)";
        }
    }
    LUISA_INFO("ldexp: double scalar table passed ({} cases).", n);

    // The two cases a float-narrowed body cannot pass, named explicitly.
    constexpr std::array named_exponents{-149, -1074};
    for (auto e : named_exponents) {
        auto expected = std::ldexp(1.0, e);
        auto slot = e == -149 ? 5u : 6u;// host_exponent index of -149 / -1074
        auto got = out[0u * host_exponent.size() + slot];
        expect(std::bit_cast<uint64_t>(got) == std::bit_cast<uint64_t>(expected))
            << "ldexp(1.0, " << e << ") must stay a double (2^-149 is a normal "
            << "double, 2^-1074 is DBL_TRUE_MIN): got " << got << ", expected "
            << expected;
    }

    // The `double2` / `int2` overload.
    {
        luisa::vector<double2> host_in{make_double2(1.0, 1.0)};
        auto vec_in = device.create_buffer<double2>(1u);
        auto vec_out = device.create_buffer<double2>(1u);
        stream << vec_in.copy_from(luisa::span{host_in}) << synchronize();
        Kernel1D vec_kernel = [](BufferVar<double2> input, BufferVar<double2> output) noexcept {
            output.write(0u, ldexp(input.read(0u), make_int2(-149, -1074)));
        };
        auto vec_shader = device.compile(vec_kernel);
        stream << vec_shader(vec_in, vec_out).dispatch(1u) << synchronize();
        luisa::vector<double2> out2(1u);
        stream << vec_out.copy_to(luisa::span{out2}) << synchronize();
        expect(std::bit_cast<uint64_t>(out2[0].x) ==
               std::bit_cast<uint64_t>(std::ldexp(1.0, -149)))
            << "ldexp(double2, int2).x = " << out2[0].x << ", expected "
            << std::ldexp(1.0, -149);
        expect(std::bit_cast<uint64_t>(out2[0].y) ==
               std::bit_cast<uint64_t>(std::ldexp(1.0, -1074)))
            << "ldexp(double2, int2).y = " << out2[0].y << ", expected DBL_TRUE_MIN "
            << std::ldexp(1.0, -1074);
        LUISA_INFO("ldexp: double2/int2 overload passed.");
    }
}

}// namespace test_dsl_special_functions_detail

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    auto stream = device.create_stream();
    LUISA_INFO("Special-function test device backend: {}.", device.backend_name());

    using namespace test_dsl_special_functions_detail;
    // The unary group logs and returns early on a backend that has no native
    // erf/erfc/expm1 (see the file header for why the rejection itself is not
    // asserted here).
    "special_functions_unary_scalar"_test = [&] { test_special_functions(device, stream); };
    "special_functions_ldexp"_test = [&] { test_ldexp(device, stream); };
    "special_functions_double_precision"_test = [&] { test_double_precision(device, stream); };
    "special_functions_double_precision_demoted_names"_test = [&] {
        test_double_precision_demoted_names(device, stream);
    };
    "special_functions_ldexp_double"_test = [&] { test_ldexp_double(device, stream); };
    return 0;
}
