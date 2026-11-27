// Floating-point remainder lowering contract for the native SPIR-V route.
//
// `test_device_math` and `test_hip_fmod` verify runtime semantics; this test
// pins the SPIR-V emission contract without a GPU:
//
//   * under strict math (`enable_fast_math = false`), binary16/binary32 use the
//     pure 32-bit structured-loop exact significand reduction (zero
//     `OpCapability Int64`, zero `OpFRem`, zero `OpFDiv`), and binary64 uses the
//     structured-loop chunked exact reduction with `NoContraction` on `Fma`,
//     `OpFMul`, and `OpFAdd`;
//   * fast math (`enable_fast_math = true`) does *not* switch to `OpFRem`:
//     that instruction is not the C `fmod` on every driver (NVIDIA lowers it to
//     a reciprocal range reduction whose residue is wrong for large |x / y|),
//     so both modes share the exact reduction and the fixtures must agree;
//   * both paths survive `-O2` optimization cleanly.
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

constexpr auto features_without_int64 =
    lc::spirv::SpirvTargetFeatures::from_enabled_mask(
        lc::spirv::target_feature::known_mask & ~lc::spirv::target_feature::shader_int64);

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
    lc::spirv::SpirvTargetFeatures target_features = all_target_features,
    const char *optimization_level = "0") noexcept {
    ScopedEnvironmentVariable optimization{
        "LUISA_SPIRV_OPT_LEVEL", optimization_level};
    auto result = lc::spirv::SpirvCodegenEntry::compile_spirv(
        kernel, ShaderOption{.enable_cache = false, .enable_fast_math = fast_math},
        target_features);
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

[[nodiscard]] bool has_no_contraction_on_opcode(
    std::string_view text, std::string_view opcode_fragment) noexcept {
    bool found_any = false;
    size_t pos = 0u;
    while (pos < text.size()) {
        auto end = text.find('\n', pos);
        if (end == std::string_view::npos) { end = text.size(); }
        auto line = text.substr(pos, end - pos);
        if (line.find(opcode_fragment) != std::string_view::npos) {
            auto eq = line.find('=');
            if (eq != std::string_view::npos) {
                auto id_tok = line.substr(0u, eq);
                while (!id_tok.empty() && (id_tok.back() == ' ' || id_tok.back() == '\t')) {
                    id_tok.remove_suffix(1u);
                }
                while (!id_tok.empty() && (id_tok.front() == ' ' || id_tok.front() == '\t')) {
                    id_tok.remove_prefix(1u);
                }
                if (!id_tok.empty()) {
                    found_any = true;
                    auto needle = std::string{"OpDecorate "} + std::string{id_tok} + " NoContraction";
                    if (text.find(needle) == std::string_view::npos) {
                        return false;
                    }
                }
            }
        }
        pos = end + 1u;
    }
    return found_any;
}

// The binary16/binary32 remainder performs its modulus in pure 32-bit
// arithmetic using a structured loop, without OpCapability Int64 or OpFDiv.
// Fast math must satisfy the same invariants: `OpFRem` is not the C `fmod` on
// every driver (NVIDIA's reciprocal-multiply range reduction loses the residue
// once |x / y| is large), so the mode never selects it.
void check_integer_significand_reduction(const RemainderSpirv &compiled,
                                         luisa::string_view label) noexcept {
    expect(compiled.validated) << label << " fixture must validate";
    expect(count_occurrences(compiled.text, "OpCapability Int64") == 0u)
        << label << " must not require OpCapability Int64";
    expect(count_occurrences(compiled.text, "OpFRem") == 0u)
        << label << " must not lower a remainder to the inexact OpFRem";
    expect(count_occurrences(compiled.text, "OpUMod") != 0u)
        << label << " must use the exact integer-significand modulus";
    expect(count_occurrences(compiled.text, "OpLoopMerge") == 1u)
        << label << " must lower exact significand modulus as one structured loop";
    expect(count_occurrences(compiled.text, "OpFDiv") == 0u)
        << label << " must not form a floating-point quotient";
}

void check_chunked_binary64_reduction(const RemainderSpirv &compiled,
                                      luisa::string_view label) noexcept {
    // One loop iteration body: two Fmas (q and q + 1), one Floor, one OpFDiv,
    // and NoContraction protecting Fma, OpFMul, and OpFAdd.
    expect(compiled.validated) << label << " fixture must validate";
    expect(count_occurrences(compiled.text, "OpFRem") == 0u)
        << label << " must not lower a remainder to the inexact OpFRem";
    expect(count_occurrences(compiled.text, "OpLoopMerge") == 1u)
        << label << " must emit a single structured reduction loop instead of unrolling 41 chunks";
    auto fma_count = count_occurrences(compiled.text, "Fma");
    auto floor_count = count_occurrences(compiled.text, "Floor");
    auto division_count = count_occurrences(compiled.text, "OpFDiv");
    expect(fma_count == 2u)
        << label << " fused residual count " << fma_count;
    expect(floor_count == 1u)
        << label << " floored quotient count " << floor_count;
    expect(division_count == 1u)
        << label << " quotient division count " << division_count;
    expect(has_no_contraction_on_opcode(compiled.text, " Fma "))
        << label << " GLSL.std.450 Fma must carry NoContraction";
    expect(has_no_contraction_on_opcode(compiled.text, " OpFMul "))
        << label << " exact double reduction OpFMul must carry NoContraction";
    expect(has_no_contraction_on_opcode(compiled.text, " OpFAdd "))
        << label << " exact double reduction OpFAdd must carry NoContraction";
}


}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_float32_remainder_strict_and_fast_math_contracts"_test = [] {
        Kernel1D kernel = [](BufferFloat lhs, BufferFloat rhs,
                             BufferFloat output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        Kernel1D vector_kernel = [](BufferFloat4 lhs, BufferFloat4 rhs,
                                    BufferFloat4 output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        // Must compile cleanly even on targets without shader_int64!
        check_integer_significand_reduction(
            compile_remainder_kernel(
                kernel.function()->function(), false, features_without_int64),
            "strict float remainder without Int64");
        check_integer_significand_reduction(
            compile_remainder_kernel(
                vector_kernel.function()->function(), false, features_without_int64),
            "strict float4 remainder without Int64");

        check_integer_significand_reduction(
            compile_remainder_kernel(
                kernel.function()->function(), true, features_without_int64),
            "fast-math float remainder");
        check_integer_significand_reduction(
            compile_remainder_kernel(
                vector_kernel.function()->function(), true, features_without_int64),
            "fast-math float4 remainder");
    };

    "spirv_float16_remainder_strict_and_fast_math_contracts"_test = [] {
        Kernel1D kernel = [](BufferHalf lhs, BufferHalf rhs,
                             BufferHalf output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        auto strict_compiled = compile_remainder_kernel(
            kernel.function()->function(), false, features_without_int64);
        check_integer_significand_reduction(
            strict_compiled, "strict half remainder without Int64");
        auto convert_count = count_occurrences(strict_compiled.text, "OpFConvert");
        expect(convert_count == 3u)
            << "strict half remainder must widen both operands and narrow once, "
            << "observed " << convert_count;

        check_integer_significand_reduction(
            compile_remainder_kernel(
                kernel.function()->function(), true, features_without_int64),
            "fast-math half remainder");
    };

    "spirv_float64_remainder_strict_and_fast_math_contracts"_test = [] {
        Kernel1D kernel = [](BufferVar<double> lhs, BufferVar<double> rhs,
                             BufferVar<double> output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        Kernel1D vec_kernel = [](BufferVar<double4> lhs, BufferVar<double4> rhs,
                                 BufferVar<double4> output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        check_chunked_binary64_reduction(
            compile_remainder_kernel(kernel.function()->function(), false),
            "strict double remainder");
        check_chunked_binary64_reduction(
            compile_remainder_kernel(vec_kernel.function()->function(), false),
            "strict double4 remainder");

        check_chunked_binary64_reduction(
            compile_remainder_kernel(kernel.function()->function(), true),
            "fast-math double remainder");
        check_chunked_binary64_reduction(
            compile_remainder_kernel(vec_kernel.function()->function(), true),
            "fast-math double4 remainder");
    };

    "spirv_float_remainder_survives_optimization"_test = [] {
        Kernel1D half_kernel = [](BufferHalf lhs, BufferHalf rhs,
                                  BufferHalf output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        Kernel1D float_kernel = [](BufferFloat lhs, BufferFloat rhs,
                                   BufferFloat output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        Kernel1D double_kernel = [](BufferVar<double> lhs, BufferVar<double> rhs,
                                    BufferVar<double> output) noexcept {
            output.write(0u, fmod(lhs.read(0u), rhs.read(0u)));
        };
        auto half_strict = compile_remainder_kernel(
            half_kernel.function()->function(), false, features_without_int64, "2");
        expect(half_strict.validated);
        expect(count_occurrences(half_strict.text, "OpCapability Int64") == 0u);
        expect(count_occurrences(half_strict.text, "OpFRem") == 0u)
            << "optimized strict half remainder must stay exact";

        auto float_strict = compile_remainder_kernel(
            float_kernel.function()->function(), false, features_without_int64, "2");
        expect(float_strict.validated);
        expect(count_occurrences(float_strict.text, "OpCapability Int64") == 0u);
        expect(count_occurrences(float_strict.text, "OpFRem") == 0u)
            << "optimized strict float remainder must stay exact";

        auto double_strict = compile_remainder_kernel(
            double_kernel.function()->function(), false, all_target_features, "2");
        expect(double_strict.validated);
        expect(count_occurrences(double_strict.text, "OpFRem") == 0u)
            << "optimized strict double remainder must stay exact";

        auto float_fast = compile_remainder_kernel(
            float_kernel.function()->function(), true, features_without_int64, "2");
        check_integer_significand_reduction(
            float_fast, "optimized fast-math float remainder");

        auto double_fast = compile_remainder_kernel(
            double_kernel.function()->function(), true, all_target_features, "2");
        check_chunked_binary64_reduction(
            double_fast, "optimized fast-math double remainder");
    };
}
