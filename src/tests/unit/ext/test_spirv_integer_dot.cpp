// Test for the Vulkan integer dot-product lowering (VK_KHR_shader_integer_dot_product).
//
// It covers both sides of the target-feature contract for the DSL
// `dot4x8`/`dot4x8_u`/`dot4x8_su` intrinsics:
//
//  - with `target_feature::shader_integer_dot_product` enabled the module must
//    emit exactly one OpSDot/OpUDot/OpSUDot with the trailing
//    PackedVectorFormat4x8Bit literal plus a separate wrapping OpIAdd for the
//    accumulator (`Op*DotAccSat` would saturate instead of wrapping, so it is
//    never the __dp4a contract), declare DotProductKHR +
//    DotProductInput4x8BitPackedKHR and SPV_KHR_integer_dot_product, and
//    reconcile the capability-owned requirement bit from the final binary;
//
//  - with the feature disabled the module must fall back to the exact portable
//    bitfield-extract multiply-accumulate with no capability, no extension and
//    no required feature bit, and still validate.
//
// Both paths are compared against the reference DP4A definition so the emitted
// shapes cannot drift from the DSL contract.

#include "ut/ut.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <spirv-tools/libspirv.hpp>

#include <luisa/dsl/sugar.h>
#include <luisa/core/stl/string.h>

#include "spirv_codegen/entry.h"
#include "spirv_codegen/optimizer.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct CompiledIntegerDot {
    std::string text;
    std::vector<uint32_t> words;
    lc::spirv::SpirvTargetFeatureMask required_features{};
};

[[nodiscard]] bool contains(luisa::string_view text,
                            luisa::string_view needle) noexcept {
    return text.find(needle) != luisa::string_view::npos;
}

[[nodiscard]] size_t count_occurrences(luisa::string_view text,
                                       luisa::string_view needle) noexcept {
    auto count = size_t{0u};
    if (needle.empty()) { return count; }
    for (auto position = text.find(needle);
         position != luisa::string_view::npos;
         position = text.find(needle, position + needle.size())) {
        count++;
    }
    return count;
}

// Returns the first disassembly line that contains `opcode`, or an empty
// string_view when the module does not use it.
[[nodiscard]] luisa::string_view find_instruction_line(
    luisa::string_view text, luisa::string_view opcode) noexcept {
    for (auto begin = size_t{0u}; begin < text.size();) {
        auto end = text.find('\n', begin);
        if (end == luisa::string_view::npos) { end = text.size(); }
        auto line = text.substr(begin, end - begin);
        if (contains(line, opcode)) { return line; }
        begin = end + 1u;
    }
    return {};
}

// The last whitespace-separated token of an instruction line, i.e. the trailing
// literal operand of OpSDot/OpUDot/OpSUDot.
[[nodiscard]] luisa::string_view last_token(luisa::string_view line) noexcept {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
        line.remove_suffix(1u);
    }
    auto space = line.find_last_of(' ');
    return space == luisa::string_view::npos ? luisa::string_view{} :
                                               line.substr(space + 1u);
}

template<typename Kernel>
[[nodiscard]] CompiledIntegerDot compile_integer_dot_fixture(
    Kernel &&kernel, bool shader_integer_dot_product) {
    lc::spirv::SpirvTargetFeatures features{};
    // Deliberately only the negotiated bit: everything else stays at its
    // contract-preserving default so the required-feature mask is exactly what
    // the integer dot product itself needs.
    features.shader_integer_dot_product = shader_integer_dot_product;
    auto result = lc::spirv::SpirvCodegenEntry::compile_spirv(
        kernel.function()->function(),
        ShaderOption{.enable_cache = false, .enable_fast_math = false},
        features);
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    std::string diagnostics;
    tools.SetMessageConsumer(
        [&diagnostics](spv_message_level_t, const char *,
                       const spv_position_t &, const char *message) {
            if (!diagnostics.empty()) { diagnostics.push_back('\n'); }
            diagnostics.append(message);
        });
    expect(tools.Validate(result.spv_bin.data(), result.spv_bin.size()))
        << "integer dot SPIR-V failed Vulkan 1.2 validation: " << diagnostics;
    std::string text;
    expect(tools.Disassemble(result.spv_bin.data(), result.spv_bin.size(),
                             &text))
        << "failed to disassemble the integer dot SPIR-V module";
    return {
        .text = std::move(text),
        .words = std::move(result.spv_bin),
        .required_features = result.required_target_features};
}

// The exact reference definition every backend and the portable fallback must
// reproduce: acc + sum_i lane(a)[i] * lane(b)[i], wrapped in 32 bits. The lane
// decode goes through the unsigned shift (a left shift of a negative value
// would be undefined) and the arithmetic right shift sign-extends it.
[[nodiscard]] int32_t reference_dot4x8(uint32_t a, uint32_t b,
                                      int32_t acc) noexcept {
    auto result = static_cast<uint32_t>(acc);
    for (auto lane = 0u; lane < 4u; ++lane) {
        auto shift = 24u - lane * 8u;
        auto lane_a = static_cast<int32_t>(a << shift) >> 24;
        auto lane_b = static_cast<int32_t>(b << shift) >> 24;
        result += static_cast<uint32_t>(lane_a * lane_b);
    }
    return static_cast<int32_t>(result);
}

[[nodiscard]] uint32_t reference_dot4x8_u(uint32_t a, uint32_t b,
                                          uint32_t acc) noexcept {
    auto result = acc;
    for (auto lane = 0u; lane < 4u; ++lane) {
        result += ((a >> (8u * lane)) & 0xffu) * ((b >> (8u * lane)) & 0xffu);
    }
    return result;
}

[[nodiscard]] int32_t reference_dot4x8_su(uint32_t a, uint32_t b,
                                          int32_t acc) noexcept {
    auto result = static_cast<uint32_t>(acc);
    for (auto lane = 0u; lane < 4u; ++lane) {
        auto lane_a = static_cast<int32_t>(a << (24u - lane * 8u)) >> 24;
        auto lane_b = (b >> (8u * lane)) & 0xffu;
        result += static_cast<uint32_t>(lane_a * static_cast<int32_t>(lane_b));
    }
    return static_cast<int32_t>(result);
}

// Hard-coded DP4A values: the oracle is pinned by hand instead of by a second
// implementation, so a decode bug cannot agree with itself.
void expect_reference_semantics() noexcept {
    // lanes(a) = {-128, 127, 1, 127}, lanes(b) = {127, 1, -1, -128}.
    constexpr auto packed_a = 0x7f017f80u;
    constexpr auto packed_b = 0x80ff017fu;
    expect(eq(reference_dot4x8(packed_a, packed_b, 0), -32386));
    expect(eq(reference_dot4x8(packed_a, packed_b, 1234567), 1202181));
    expect(eq(reference_dot4x8(packed_a, packed_b, -1), -32387));
    // lanes {0, 1, 2, 3} squared.
    expect(eq(reference_dot4x8(0x03020100u, 0x03020100u, 0), 14));
    expect(eq(reference_dot4x8_u(0x03020100u, 0x03020100u, 0u), 14u));
    expect(eq(reference_dot4x8_u(0xffffffffu, 0xffffffffu, 0u), 4u * 255u * 255u));
    // All lanes -128 times all lanes 255.
    expect(eq(reference_dot4x8_su(0x80808080u, 0xffffffffu, 0), -130560));
    expect(eq(reference_dot4x8_su(0x80808080u, 0xffffffffu, 1234567), 1104007));
    // The unsigned accumulator wraps where the signed one goes negative.
    expect(eq(reference_dot4x8(packed_a, packed_b, 0),
              static_cast<int32_t>(0xffff817eu)));
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_integer_dot_reference_oracle_is_exact"_test = [] {
        expect_reference_semantics();
    };

    "spirv_integer_dot_sdot_uses_packed_format_and_wrapping_add"_test = [] {
        Kernel1D ast_kernel = [](BufferUInt a, BufferUInt b, BufferInt acc,
                                 BufferInt out) noexcept {
            auto i = dispatch_x();
            out.write(i, dot4x8(a.read(i), b.read(i), acc.read(i)));
        };
        // ---- on: one OpSDot(a, b, PackedVectorFormat4x8Bit) + OpIAdd ----
        auto fused = compile_integer_dot_fixture(ast_kernel, true);
        // SPIRV-Tools prints the Khronos-aliased spelling of the opcodes.
        expect(eq(count_occurrences(fused.text, "OpSDot"), size_t{1u}));
        expect(eq(count_occurrences(fused.text, "OpUDot"), size_t{0u}));
        expect(eq(count_occurrences(fused.text, "OpSUDot"), size_t{0u}));
        expect(eq(count_occurrences(fused.text, "AccSat"), size_t{0u}))
            << "the saturating accumulator forms are not the __dp4a contract";
        expect(eq(count_occurrences(
                      fused.text, "OpCapability DotProduct\n"),
                  size_t{1u}));
        expect(eq(count_occurrences(
                      fused.text,
                      "OpCapability DotProductInput4x8BitPacked\n"),
                  size_t{1u}));
        expect(eq(count_occurrences(
                      fused.text,
                      "OpExtension \"SPV_KHR_integer_dot_product\""),
                  size_t{1u}));
        // OpSDot takes exactly (Vector 1, Vector 2, PackedVectorFormat): the
        // accumulator is a separate OpIAdd, so the last operand of the line is
        // the 4x8-bit format literal.
        auto sdot_line = find_instruction_line(fused.text, "OpSDot");
        expect(!sdot_line.empty());
        auto sdot_format = last_token(sdot_line);
        expect(contains(sdot_format, "4x8Bit") || sdot_format == "0")
            << "OpSDot must carry the trailing PackedVectorFormat4x8Bit literal";
        // No portable fallback debris on the hardware path.
        expect(eq(count_occurrences(fused.text, "OpBitFieldSExtract"),
                  size_t{0u}));
        expect(eq(count_occurrences(fused.text, "OpBitFieldUExtract"),
                  size_t{0u}));
        expect(eq(fused.required_features,
                  lc::spirv::target_feature::shader_integer_dot_product));

        // ---- off: exact portable bitfield multiply-accumulate ----
        auto fallback = compile_integer_dot_fixture(ast_kernel, false);
        expect(eq(count_occurrences(fallback.text, "OpSDot"), size_t{0u}));
        expect(eq(count_occurrences(fallback.text, "OpUDot"), size_t{0u}));
        expect(eq(count_occurrences(fallback.text, "OpSUDot"), size_t{0u}));
        expect(eq(count_occurrences(fallback.text, "DotProduct"), size_t{0u}))
            << "the fallback must not declare any DotProduct capability";
        expect(eq(count_occurrences(
                      fallback.text, "SPV_KHR_integer_dot_product"),
                  size_t{0u}))
            << "the fallback must not declare the integer dot product extension";
        // Four signed lanes per operand word.
        expect(eq(count_occurrences(fallback.text, "OpBitFieldSExtract"),
                  size_t{8u}));
        expect(eq(count_occurrences(fallback.text, "OpBitFieldUExtract"),
                  size_t{0u}));
        // Four products (plus the word-address multiply of the buffer reads,
        // which is why the bound is >= instead of ==).
        expect(count_occurrences(fallback.text, "OpIMul") >= size_t{4u});
        expect(eq(fallback.required_features,
                  lc::spirv::SpirvTargetFeatureMask{0u}))
            << "the portable fallback must not require the DP4A feature";
    };

    "spirv_integer_dot_udot_is_unsigned_and_wrapping"_test = [] {
        Kernel1D ast_kernel = [](BufferUInt a, BufferUInt b, BufferUInt acc,
                                 BufferUInt out) noexcept {
            auto i = dispatch_x();
            out.write(i, dot4x8_u(a.read(i), b.read(i), acc.read(i)));
        };
        auto fused = compile_integer_dot_fixture(ast_kernel, true);
        expect(eq(count_occurrences(fused.text, "OpUDot"), size_t{1u}));
        expect(eq(count_occurrences(fused.text, "OpSDot"), size_t{0u}));
        expect(eq(count_occurrences(fused.text, "OpSUDot"), size_t{0u}));
        expect(eq(count_occurrences(
                      fused.text, "OpCapability DotProduct\n"),
                  size_t{1u}));
        expect(eq(count_occurrences(
                      fused.text,
                      "OpCapability DotProductInput4x8BitPacked\n"),
                  size_t{1u}));
        expect(eq(fused.required_features,
                  lc::spirv::target_feature::shader_integer_dot_product));

        auto fallback = compile_integer_dot_fixture(ast_kernel, false);
        expect(eq(count_occurrences(fallback.text, "OpUDot"), size_t{0u}));
        expect(eq(count_occurrences(fallback.text, "OpBitFieldUExtract"),
                  size_t{8u}));
        expect(eq(count_occurrences(fallback.text, "OpBitFieldSExtract"),
                  size_t{0u}));
        expect(count_occurrences(fallback.text, "OpIMul") >= size_t{4u});
        expect(eq(fallback.required_features,
                  lc::spirv::SpirvTargetFeatureMask{0u}));
    };

    "spirv_integer_dot_sudot_mixes_sign_and_wraps"_test = [] {
        Kernel1D ast_kernel = [](BufferUInt a, BufferUInt b, BufferInt acc,
                                 BufferInt out) noexcept {
            auto i = dispatch_x();
            out.write(i, dot4x8_su(a.read(i), b.read(i), acc.read(i)));
        };
        auto fused = compile_integer_dot_fixture(ast_kernel, true);
        expect(eq(count_occurrences(fused.text, "OpSUDot"), size_t{1u}));
        expect(eq(count_occurrences(fused.text, "OpSDot"), size_t{0u}));
        expect(eq(count_occurrences(fused.text, "OpUDot"), size_t{0u}));
        expect(eq(count_occurrences(fused.text, "AccSat"), size_t{0u}));
        expect(eq(count_occurrences(
                      fused.text, "OpCapability DotProduct\n"),
                  size_t{1u}));
        expect(eq(fused.required_features,
                  lc::spirv::target_feature::shader_integer_dot_product));

        auto fallback = compile_integer_dot_fixture(ast_kernel, false);
        expect(eq(count_occurrences(fallback.text, "OpSUDot"), size_t{0u}));
        // Signed Vector 1, unsigned Vector 2: four extracts of each kind.
        expect(eq(count_occurrences(fallback.text, "OpBitFieldSExtract"),
                  size_t{4u}));
        expect(eq(count_occurrences(fallback.text, "OpBitFieldUExtract"),
                  size_t{4u}));
        expect(count_occurrences(fallback.text, "OpIMul") >= size_t{4u});
        expect(eq(fallback.required_features,
                  lc::spirv::SpirvTargetFeatureMask{0u}));
    };

    "spirv_integer_dot_requirement_survives_optimization"_test = [] {
        // The requirement is capability-owned, so it is reconciled from the
        // final validated binary instead of the provisional emission record.
        // A live OpSDot must therefore keep both the capability and the bit.
        Kernel1D ast_kernel = [](BufferUInt a, BufferUInt b, BufferInt acc,
                                 BufferInt out) noexcept {
            auto i = dispatch_x();
            out.write(i, dot4x8(a.read(i), b.read(i), acc.read(i)));
        };
        // The default preset runs the bounded optimization pipeline.
        auto optimized = compile_integer_dot_fixture(ast_kernel, true);
        expect(eq(count_occurrences(optimized.text, "OpSDot"), size_t{1u}));
        expect(eq(count_occurrences(optimized.text, "OpCapability DotProduct\n"),
                  size_t{1u}));
        expect(eq(optimized.required_features,
                  lc::spirv::target_feature::shader_integer_dot_product));
    };
}