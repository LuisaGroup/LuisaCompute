// Floating-point remainder lowering contract for the native SPIR-V route.
//
// `test_hip_fmod` and `test_hip_codegen_arithmetic` execute the remainder on a
// device; this test pins the same contract at emission time so a regression is
// caught without a GPU. The contract is:
//
//   * a floating-point MOD never lowers to OpFRem for binary16, binary32, or
//     binary64, in precise and in fast-math mode alike (OpFRem is only
//     "x - y * trunc(x / y)" up to the implementation, and drivers that form
//     the rounded quotient lose the residue);
//   * binary16 and binary32 use the exact integer-significand algorithm, so
//     the module contains 64-bit OpUMod instructions and no floating division;
//   * binary16 widens to binary32 first and converts the exact remainder back;
//   * binary64 uses the chunked exact reduction: one Fma and one Floor per
//     chunk, 41 chunks for the whole binary64 exponent range.
#include "ut/ut.hpp"
#include <luisa/dsl/sugar.h>
#include <spirv-tools/libspirv.hpp>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>
#include "spirv_codegen/entry.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto all_target_features =
    lc::spirv::SpirvTargetFeatures::from_enabled_mask(
        lc::spirv::target_feature::known_mask);

void set_environment_variable(const char *name, const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

class ScopedEnvironmentVariable {
private:
    const char *_name;
    std::string _previous;
    bool _has_previous{false};

public:
    ScopedEnvironmentVariable(const char *name, const char *value) noexcept
        : _name{name} {
        if (auto previous = std::getenv(name)) {
            _previous = previous;
            _has_previous = true;
        }
        set_environment_variable(name, value);
    }
    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(_name, _has_previous ? _previous.c_str() : nullptr);
    }
    ScopedEnvironmentVariable(const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(const ScopedEnvironmentVariable &) = delete;
};

struct RemainderSpirv {
    std::string text;
    bool validated{false};
};

[[nodiscard]] RemainderSpirv compile_remainder_kernel(
    Function kernel, bool fast_math,
    const char *optimization_level = "0") noexcept {
    ScopedEnvironmentVariable optimization{
        "LUISA_SPIRV_OPT_LEVEL", optimization_level};
    auto result = lc::spirv::SpirvCodegenEntry::compile_spirv(
        kernel, ShaderOption{.enable_cache = false, .enable_fast_math = fast_math},
        all_target_features);
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    RemainderSpirv compiled;
    compiled.validated = tools.Validate(result.spv_bin.data(), result.spv_bin.size());
    expect(tools.Disassemble(result.spv_bin.data(), result.spv_bin.size(),
                             &compiled.text))
        << "failed to disassemble the remainder fixture";
    return compiled;
}

[[nodiscard]] size_t count_occurrences(std::string_view text,
                                       std::string_view needle) noexcept {
    size_t count = 0u;
    for (auto position = text.find(needle);
         position != std::string_view::npos;
         position = text.find(needle, position + needle.size())) {
        count++;
    }
    return count;
}

// The exact binary16/binary32 path performs its modulus with 64-bit integer
// arithmetic and never divides in floating point.
void check_integer_significand_reduction(const RemainderSpirv &compiled,
                                         luisa::string_view label) noexcept {
    expect(compiled.validated) << label << " fixture must validate";
    expect(count_occurrences(compiled.text, "OpFRem") == 0u)
        << label << " must not lower a remainder to the inexact OpFRem";
    expect(count_occurrences(compiled.text, "OpUMod") != 0u)
        << label << " must use the exact integer-significand modulus";
    expect(count_occurrences(compiled.text, "OpFDiv") == 0u)
        << label << " must not form a floating-point quotient";
}

void check_chunked_binary64_reduction(const RemainderSpirv &compiled,
                                      luisa::string_view label) noexcept {
    // One Fma and one Floor per exact reduction chunk. Cross-checking both
    // counts keeps a truncated or duplicated sequence from passing.
    constexpr auto chunk_count = 41u;
    auto fma_count = count_occurrences(compiled.text, "Fma");
    auto floor_count = count_occurrences(compiled.text, "Floor");
    auto division_count = count_occurrences(compiled.text, "OpFDiv");
    expect(compiled.validated) << label << " fixture must validate";
    expect(count_occurrences(compiled.text, "OpFRem") == 0u)
        << label << " must not lower a remainder to the inexact OpFRem";
    // Two fused residuals per chunk (quotient and quotient + 1), one floored
    // quotient, and one division.
    expect(fma_count == 2u * chunk_count)
        << label << " fused residual count " << fma_count;
    expect(floor_count == chunk_count)
        << label << " floored quotient count " << floor_count;
    expect(division_count == chunk_count)
        << label << " quotient division count " << division_count;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_float32_remainder_is_exact_in_every_mode"_test = [] {
        Kernel1D kernel = [](BufferFloat lhs, BufferFloat rhs,
                             BufferFloat output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        Kernel1D vector_kernel = [](BufferFloat4 lhs, BufferFloat4 rhs,
                                    BufferFloat4 output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        for (auto fast_math : {false, true}) {
            check_integer_significand_reduction(
                compile_remainder_kernel(kernel.function()->function(), fast_math),
                "precise float remainder");
            check_integer_significand_reduction(
                compile_remainder_kernel(vector_kernel.function()->function(), fast_math),
                "precise float4 remainder");
        }
    };

    "spirv_float16_remainder_is_exact_in_every_mode"_test = [] {
        Kernel1D kernel = [](BufferHalf lhs, BufferHalf rhs,
                             BufferHalf output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        for (auto fast_math : {false, true}) {
            auto compiled =
                compile_remainder_kernel(kernel.function()->function(), fast_math);
            check_integer_significand_reduction(compiled, "half remainder");
            // The binary32 reduction is exact for binary16 because both the
            // operands and the exact remainder convert without rounding.
            auto convert_count = count_occurrences(compiled.text, "OpFConvert");
            expect(convert_count == 3u)
                << "half remainder must widen both operands and narrow once, "
                << "observed " << convert_count;
        }
    };

    "spirv_float64_remainder_is_exact_in_every_mode"_test = [] {
        Kernel1D kernel = [](BufferVar<double> lhs, BufferVar<double> rhs,
                             BufferVar<double> output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        for (auto fast_math : {false, true}) {
            check_chunked_binary64_reduction(
                compile_remainder_kernel(kernel.function()->function(), fast_math),
                "double remainder");
        }
    };

    "spirv_float_remainder_survives_optimization"_test = [] {
        // The DXC-parity optimizer must not reintroduce OpFRem or collapse the
        // exact reduction, in either numeric mode.
        Kernel1D half_kernel = [](BufferHalf lhs, BufferHalf rhs,
                                  BufferHalf output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        Kernel1D double_kernel = [](BufferVar<double> lhs, BufferVar<double> rhs,
                                    BufferVar<double> output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        for (auto fast_math : {false, true}) {
            auto half_compiled = compile_remainder_kernel(
                half_kernel.function()->function(), fast_math, "2");
            expect(half_compiled.validated);
            expect(count_occurrences(half_compiled.text, "OpFRem") == 0u)
                << "optimized half remainder must stay exact";
            auto double_compiled = compile_remainder_kernel(
                double_kernel.function()->function(), fast_math, "2");
            expect(double_compiled.validated);
            expect(count_occurrences(double_compiled.text, "OpFRem") == 0u)
                << "optimized double remainder must stay exact";
        }
    };
}
