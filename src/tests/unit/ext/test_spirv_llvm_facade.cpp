// Test for the LLVM AST-to-SPIR-V facade.
// This test covers:
// - no-resource/no-argument kernel emission through the public facade
// - independent Vulkan 1.2 validation of the returned SPIR-V module

#include "ut/ut.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <spirv-tools/libspirv.hpp>

#include <luisa/dsl/sugar.h>

#include <spirv_llvm/spirv_llvm.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] bool contains(std::string_view text,
                            std::string_view needle) noexcept {
    return text.find(needle) != std::string_view::npos;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_llvm_facade_emits_valid_no_argument_kernel"_test = [] {
        Kernel1D kernel = []() noexcept {};
        ShaderOption option{};
        option.enable_cache = false;
        auto result = lc::llvm_codegen::compile_spirv(
            kernel.function()->function(), option);

        expect(eq(result.properties.size(), size_t{1u}))
            << "a no-argument kernel must expose only the immutable sampler "
               "table";
        if (!result.properties.empty()) {
            auto &&sampler = result.properties.front();
            expect(sampler.type ==
                   lc::hlsl::ShaderVariableType::SamplerHeap);
            expect(eq(sampler.space_index, 1u));
            expect(eq(sampler.register_index, 0u));
            expect(eq(sampler.array_size, 16u));
        }
        expect(!result.useTex2DBindless);
        expect(!result.useTex3DBindless);
        expect(!result.useBufferBindless);
        expect(!result.spv_bin.empty())
            << "the LLVM facade must return a SPIR-V module";
        if (result.spv_bin.empty()) { return; }

        constexpr uint32_t spirv_magic = 0x07230203u;
        expect(eq(result.spv_bin.front(), spirv_magic));

        spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
        std::string diagnostics;
        tools.SetMessageConsumer(
            [&diagnostics](spv_message_level_t, const char *,
                           const spv_position_t &, const char *message) {
                if (!diagnostics.empty()) { diagnostics.push_back('\n'); }
                diagnostics.append(message);
            });
        auto valid = tools.Validate(result.spv_bin.data(),
                                    result.spv_bin.size());
        expect(valid)
            << "LLVM facade output failed Vulkan 1.2 validation: "
            << diagnostics;
        if (!valid) { return; }

        std::string assembly;
        auto disassembled = tools.Disassemble(
            result.spv_bin.data(), result.spv_bin.size(), &assembly);
        expect(disassembled)
            << "failed to disassemble LLVM facade output";
        if (!disassembled) { return; }
        expect(contains(assembly, "OpEntryPoint GLCompute"));
        expect(contains(assembly, "OpExecutionMode"));
        expect(contains(assembly, "LocalSize 256 1 1"));
    };
}
