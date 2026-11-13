// Tests for the optional shader-capability extensions of the SPIR-V target
// (`lc::hlsl::SpirvShaderCapabilities`: VK_KHR_shader_fma, VK_KHR_shader_abort).
//
// This test covers:
// - `_fma` becomes a fused multiply-add (`mad()` -> OpFmaKHR) only on a device that
//   negotiated VK_KHR_shader_fma, and stays the unfused `a*b+c` otherwise;
// - `$debug_break` becomes OpAbortKHR only on a device that negotiated
//   VK_KHR_shader_abort;
// - both generated modules still pass SPIR-V validation.
//
// The codegen is driven directly (no device): the assertions are made on the
// generated HLSL and on the SPIR-V the bundled dxc produces for it, which is
// exactly what the Vulkan compatibility HLSL-to-SPIR-V route does at runtime.
#include "ut/ut.hpp"
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>
#include <luisa/dsl/sugar.h>
#include "hlsl_codegen.h"
#include "shader_compiler.h"
#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] bool contains(luisa::string_view text,
                            luisa::string_view needle) noexcept {
    return text.find(needle) != luisa::string_view::npos;
}

struct SpirvFacts {
    bool compiled{false};
    bool validated{false};
    size_t fma_count{0u};
    size_t abort_count{0u};
    bool declares_fma{false};
    bool declares_abort{false};
    vstd::string error;
};

// Compiles `source` for SPIR-V and reports the instructions under test.  The
// opcode/capability values are the SPIR-V ones (OpFmaKHR 4427, OpAbortKHR 5121,
// CapabilityFMAKHR 6030, CapabilityAbortKHR 5120); the enumerators of the bundled
// headers are used so that a mismatch is a compile error rather than a silent
// miss.
[[nodiscard]] SpirvFacts compile_and_scan(
    luisa::string_view source,
    const luisa::filesystem::path &runtime_directory,
    lc::hlsl::SpirvShaderCapabilities capabilities) noexcept {
    SpirvFacts facts;
    lc::hlsl::ShaderCompiler compiler{runtime_directory, true};
    // The extension flags follow the capability set, exactly as the backend passes
    // them: dxc rejects an extension name it does not know, so nothing is enabled
    // unless the device asked for it.
    auto compiled = compiler.compile_compute(
        source, true, 65u, true, true, false,
        /*entry_point=*/{}, /*include_dirs=*/{}, capabilities);
    compiled.multi_visit(
        [&](const lc::hlsl::ComUniquePtr<IDxcBlob> &blob) {
            facts.compiled = true;
            auto byte_size = blob->GetBufferSize();
            if (byte_size % sizeof(uint32_t) != 0u) {
                facts.error = "DXC returned a non-word-aligned SPIR-V blob";
                return;
            }
            auto *words = static_cast<const uint32_t *>(blob->GetBufferPointer());
            auto word_count = byte_size / sizeof(uint32_t);
            spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
            facts.validated = tools.Validate(words, word_count);
            if (!facts.validated) {
                spvtools::SpirvTools quiet{SPV_ENV_VULKAN_1_2};
                vstd::string messages;
                quiet.SetMessageConsumer(
                    [&](spv_message_level_t, const char *, const spv_position_t &,
                        const char *message) { messages << message << '\n'; });
                static_cast<void>(quiet.Validate(words, word_count));
                facts.error = "the generated SPIR-V does not validate:\n" + messages;
                return;
            }
            for (auto offset = size_t{5u}; offset < word_count;) {
                auto instruction_word_count =
                    static_cast<size_t>(words[offset] >> 16u);
                if (instruction_word_count == 0u ||
                    instruction_word_count > word_count - offset) {
                    facts.error = "DXC returned malformed SPIR-V";
                    return;
                }
                auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
                if (opcode == spv::Op::OpFmaKHR) {
                    ++facts.fma_count;
                } else if (opcode == spv::Op::OpAbortKHR) {
                    ++facts.abort_count;
                } else if (opcode == spv::Op::OpCapability &&
                           instruction_word_count >= 2u) {
                    auto capability = words[offset + 1u];
                    facts.declares_fma |= capability ==
                                          static_cast<uint32_t>(spv::Capability::CapabilityFMAKHR);
                    facts.declares_abort |= capability ==
                                            static_cast<uint32_t>(spv::Capability::CapabilityAbortKHR);
                }
                offset += instruction_word_count;
            }
        },
        [&](const vstd::string &error) { facts.error = error; });
    return facts;
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "hlsl_fma_is_fused_only_with_the_shader_fma_capability"_test = [argv] {
        // `luisa::compute::fma` promises a single rounding.  SPIR-V has no fused
        // multiply-add except OpFmaKHR, so the helper only becomes `mad()` - which
        // dxc lowers to that instruction - for a device that enabled VK_KHR_shader_fma.
        Kernel1D kernel = [](BufferFloat x, BufferFloat y, BufferFloat z,
                             BufferFloat out) noexcept {
            auto i = dispatch_x();
            out.write(i, fma(x.read(i), y.read(i), z.read(i)));
        };
        auto function = kernel.function()->function();
        // The view points into the CodegenResult, so both results have to outlive it.
        auto fused_codegen = lc::hlsl::CodegenUtility{}
                                 .Codegen(function, {}, 0u, /*isSpirV=*/true, false,
                                          false, false, false, false,
                                          lc::hlsl::SpirvShaderCapabilities{.fma = true});
        auto unfused_codegen = lc::hlsl::CodegenUtility{}
                                   .Codegen(function, {}, 0u, /*isSpirV=*/true, false, false);
        auto native = fused_codegen.result.view();
        auto emulated = unfused_codegen.result.view();
        expect(contains(native, "=mad(l9,l8,l7)"))
            << "the 32-bit float fma is not routed to the fused helper";
        expect(!contains(native, "= _fma(l9,l8,l7)"))
            << "the fused helper is not selected for a shaderFmaFloat32 device";
        expect(contains(native, "#define LUISA_SPV_FMA 1"));
        expect(contains(emulated, "=_fma(l9,l8,l7)"))
            << "the unfused helper is not selected without the capability";
        expect(contains(emulated, "return a*b+c;"))
            << "the helper must stay unfused without the capability";
        expect(!contains(emulated, "#define LUISA_SPV_FMA 1"))
            << "the macro must not be defined for a device without the extension";

        // dxc has to turn the fused helper into the instruction itself; `mad()`
        // without the extension flag is an unfused OpFMul/OpFAdd pair.
        auto const &runtime_directory = luisa::filesystem::path{argv[0]}.parent_path();
        auto fused = compile_and_scan(native, runtime_directory, lc::hlsl::SpirvShaderCapabilities{.fma = true});
        expect(fused.error.empty()) << fused.error;
        expect(fused.compiled);
        expect(fused.validated);
        expect(eq(fused.fma_count, 1u))
            << "expected exactly one OpFmaKHR for the fma() of the kernel";
        expect(fused.declares_fma) << "OpCapability FMAKHR is missing";

        auto unfused = compile_and_scan(emulated, runtime_directory, lc::hlsl::SpirvShaderCapabilities{});
        expect(unfused.error.empty()) << unfused.error;
        expect(unfused.compiled);
        expect(unfused.validated);
        expect(eq(unfused.fma_count, 0u))
            << "an unfused multiply-add must not declare OpFmaKHR";
        expect(!unfused.declares_fma);
    };

    "hlsl_debug_break_lowers_to_op_abort_khr_with_the_shader_abort_capability"_test =
        [argv] {
        // A debug break has no DXIL equivalent; on SPIR-V it becomes OpAbortKHR,
        // which ends the invoking thread and reports one 32-bit word to the host.
        Kernel1D kernel = [](BufferFloat out) noexcept {
            $if (dispatch_x() == 1000u) {
                $debug_break();
            };
            out.write(dispatch_x(), 1.0f);
        };
        auto codegen = lc::hlsl::CodegenUtility{}
                           .Codegen(kernel.function()->function(), {}, 0u,
                                    /*isSpirV=*/true, false, false,
                                    false, false, false,
                                    lc::hlsl::SpirvShaderCapabilities{.abort = true});
        auto program = codegen.result.view();
        expect(contains(program, "#define LUISA_SPV_ABORT 1"));
        expect(contains(program, "vk::Abort("))
            << "the debug break did not become a vk::Abort call";

        auto const &runtime_directory = luisa::filesystem::path{argv[0]}.parent_path();
        auto facts = compile_and_scan(program, runtime_directory, lc::hlsl::SpirvShaderCapabilities{.abort = true});
        expect(facts.error.empty()) << facts.error;
        expect(facts.compiled);
        expect(facts.validated);
        expect(eq(facts.abort_count, 1u))
            << "expected exactly one OpAbortKHR for the debug break";
        expect(facts.declares_abort) << "OpCapability AbortKHR is missing";
    };
}
