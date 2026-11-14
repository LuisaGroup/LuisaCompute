// Tests for the optional shader-capability extensions of the SPIR-V target
// (`lc::hlsl::SpirvShaderCapabilities`: VK_KHR_shader_fma, VK_KHR_shader_abort,
// VK_KHR_maintenance9, VK_VALVE_shader_mixed_float_dot_product,
// VK_EXT_shader_split_barrier).
//
// This test covers:
// - `_fma` becomes a fused multiply-add (`mad()` -> OpFmaKHR) only on a device that
//   negotiated VK_KHR_shader_fma, and stays the unfused `a*b+c` otherwise;
// - `$debug_break` becomes OpAbortKHR only on a device that negotiated
//   VK_KHR_shader_abort;
// - `reverse()` on 16/64-bit values and `popcount()` on a 64-bit value reach
//   OpBitReverse/OpBitCount at their natural operand width only with
//   maintenance9_bitwise, and keep the legacy 32-bit emulation otherwise, while
//   the 8-bit `reverse()` expression stays byte-identical either way;
// - `dot2add(half2, half2, float)` is always emitted as the HLSL intrinsic; dxc
//   fuses it into OpFDot2MixAcc32VALVE only with the VALVE capability and keeps
//   its unfused half-product/FP32-accumulation sequence without it, and the same
//   source also compiles to DXIL (a platform without the VALVE device feature);
// - `split_barrier_arrive()`/`split_barrier_wait()` become
//   OpControlBarrierArriveEXT/OpControlBarrierWaitEXT (Workgroup scopes,
//   Release/Acquire WorkgroupMemory semantics) only with split_barrier, while
//   an ordinary `sync_block()` keeps GroupMemoryBarrierWithGroupSync;
// - all generated modules still pass SPIR-V validation.
//
// The codegen is driven directly (no device): the assertions are made on the
// generated HLSL and on the SPIR-V the bundled dxc produces for it, which is
// exactly what the Vulkan compatibility HLSL-to-SPIR-V route does at runtime.
#include "ut/ut.hpp"
#include <algorithm>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
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
    size_t dot2add_count{0u};
    bool declares_mixed_float_dot_product{false};
    // Per-operand-width instruction counts, indexed [16-bit, 32-bit, 64-bit];
    // any other width folds into the 32-bit bucket (DXC always widens 8-bit
    // operands to 32 bits before a bitwise op, so only these widths occur).
    size_t bit_reverse_count[3]{};
    size_t bit_count_count[3]{};
    // Split-barrier instructions: counts and the operand triples
    // (execution scope, memory scope, memory semantics) of the first hit,
    // resolved from the OpConstant ids dxc passes.
    size_t barrier_arrive_count{0u};
    size_t barrier_wait_count{0u};
    uint32_t barrier_arrive_operands[3]{};
    uint32_t barrier_wait_operands[3]{};
    bool declares_split_barrier{false};
    vstd::string error;
};

// Compiles `source` for SPIR-V and reports the instructions under test.  The
// opcode/capability values are the SPIR-V ones (OpFmaKHR 4427, OpAbortKHR 5121,
// OpFDot2MixAcc32VALVE 6916, OpControlBarrierArriveEXT 6142,
// OpControlBarrierWaitEXT 6143, CapabilityFMAKHR 6030, CapabilityAbortKHR 5120,
// CapabilityDotProductFloat16AccFloat32VALVE 6912, CapabilitySplitBarrierEXT
// 6141); the enumerators of the bundled headers are used so that a mismatch is
// a compile error rather than a silent miss.
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
            // VK_KHR_maintenance9 lets OpBitReverse/OpBitCount take a non-32-bit
            // integer operand; the validator only accepts that with this option.
            spvtools::ValidatorOptions options;
            options.SetAllowVulkan32BitBitwise(true);
            facts.validated = tools.Validate(words, word_count, options);
            if (!facts.validated) {
                spvtools::SpirvTools quiet{SPV_ENV_VULKAN_1_2};
                vstd::string messages;
                quiet.SetMessageConsumer(
                    [&](spv_message_level_t, const char *, const spv_position_t &,
                        const char *message) { messages << message << '\n'; });
                static_cast<void>(quiet.Validate(words, word_count, options));
                facts.error = "the generated SPIR-V does not validate:\n" + messages;
                return;
            }
            auto id_bound = words[3u];
            luisa::vector<uint32_t> integer_width(id_bound, 0u);
            luisa::vector<uint32_t> value_type(id_bound, 0u);
            // OpConstant values: dxc passes the barrier scopes/semantics as
            // constant ids, so the operand scan resolves them through this.
            luisa::vector<uint32_t> constant_value(id_bound, 0u);
            auto width_index = [&](uint32_t type_id) noexcept -> size_t {
                auto width = type_id < id_bound ? integer_width[type_id] : 0u;
                return width == 16u ? 0u : (width == 64u ? 2u : 1u);
            };
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
                } else if (opcode == spv::Op::OpFDot2MixAcc32VALVE) {
                    ++facts.dot2add_count;
                } else if (opcode == spv::Op::OpControlBarrierArriveEXT &&
                           instruction_word_count >= 4u) {
                    if (facts.barrier_arrive_count == 0u) {
                        for (auto i = 0u; i < 3u; ++i) {
                            auto operand = words[offset + 1u + i];
                            facts.barrier_arrive_operands[i] =
                                operand < id_bound ? constant_value[operand] : 0u;
                        }
                    }
                    ++facts.barrier_arrive_count;
                } else if (opcode == spv::Op::OpControlBarrierWaitEXT &&
                           instruction_word_count >= 4u) {
                    if (facts.barrier_wait_count == 0u) {
                        for (auto i = 0u; i < 3u; ++i) {
                            auto operand = words[offset + 1u + i];
                            facts.barrier_wait_operands[i] =
                                operand < id_bound ? constant_value[operand] : 0u;
                        }
                    }
                    ++facts.barrier_wait_count;
                } else if (opcode == spv::Op::OpConstant &&
                           instruction_word_count >= 4u &&
                           words[offset + 2u] < id_bound) {
                    constant_value[words[offset + 2u]] = words[offset + 3u];
                } else if (opcode == spv::Op::OpTypeInt &&
                           instruction_word_count >= 3u &&
                           words[offset + 1u] < id_bound) {
                    integer_width[words[offset + 1u]] = words[offset + 2u];
                } else if ((opcode == spv::Op::OpLoad ||
                            opcode == spv::Op::OpUConvert ||
                            opcode == spv::Op::OpSConvert ||
                            opcode == spv::Op::OpBitcast ||
                            opcode == spv::Op::OpShiftRightLogical ||
                            opcode == spv::Op::OpShiftRightArithmetic) &&
                           instruction_word_count >= 4u &&
                           words[offset + 2u] < id_bound) {
                    value_type[words[offset + 2u]] = words[offset + 1u];
                } else if (opcode == spv::Op::OpBitReverse &&
                           instruction_word_count >= 4u) {
                    if (words[offset + 2u] < id_bound) {
                        value_type[words[offset + 2u]] = words[offset + 1u];
                    }
                    // OpBitReverse has the same scalar/vector integer type for
                    // its result and operand, so its result type is the operand width.
                    ++facts.bit_reverse_count[width_index(words[offset + 1u])];
                } else if (opcode == spv::Op::OpBitCount &&
                           instruction_word_count >= 4u) {
                    auto operand = words[offset + 3u];
                    auto operand_type = operand < id_bound ? value_type[operand] : 0u;
                    if (operand_type == 0u || integer_width[operand_type] == 0u) {
                        facts.error = "failed to determine an OpBitCount operand type";
                        return;
                    }
                    ++facts.bit_count_count[width_index(operand_type)];
                } else if (opcode == spv::Op::OpCapability &&
                           instruction_word_count >= 2u) {
                    auto capability = words[offset + 1u];
                    facts.declares_fma |= capability ==
                                          static_cast<uint32_t>(spv::Capability::CapabilityFMAKHR);
                    facts.declares_abort |= capability ==
                                            static_cast<uint32_t>(spv::Capability::CapabilityAbortKHR);
                    facts.declares_mixed_float_dot_product |=
                        capability ==
                        static_cast<uint32_t>(spv::Capability::CapabilityDotProductFloat16AccFloat32VALVE);
                    facts.declares_split_barrier |=
                        capability ==
                        static_cast<uint32_t>(spv::Capability::CapabilitySplitBarrierEXT);
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

    "hlsl_bitwise_uses_natural_widths_only_with_the_maintenance9_capability"_test =
        [argv] {
        // VK_KHR_maintenance9 lifts the 32-bit operand restriction of
        // OpBitReverse/OpBitCount.  With maintenance9_bitwise the codegen hands
        // `reversebits` the natural 16/64-bit operand (and casts the result back
        // to the DSL uint); without it dxc widens/splits the operand into 32-bit
        // halves.  `countbits` always receives the natural operand, so the flag
        // only changes how dxc lowers it.  8-bit operands have no bitwise path
        // and keep the legacy widening either way.
        Kernel1D kernel = [](BufferUShort in16, BufferULong in64,
                             BufferVar<ubyte> in8, BufferUInt out) noexcept {
            auto i = dispatch_x();
            out.write(i, reverse(in16.read(i)));
            out.write(i + 1u, reverse(in64.read(i)));
            out.write(i + 2u, popcount(in64.read(i)));
            out.write(i + 3u, reverse(in8.read(i)));
        };
        auto function = kernel.function()->function();
        auto on_codegen = lc::hlsl::CodegenUtility{}
                              .Codegen(function, {}, 0u, /*isSpirV=*/true, false, false,
                                       false, false, false,
                                       lc::hlsl::SpirvShaderCapabilities{.maintenance9_bitwise = true});
        auto off_codegen = lc::hlsl::CodegenUtility{}
                               .Codegen(function, {}, 0u, /*isSpirV=*/true, false, false);
        auto on_program = on_codegen.result.view();
        auto off_program = off_codegen.result.view();
        // The natural-width path keeps the operand type and converts the result
        // to uint explicitly; the legacy path widens first (16/8-bit) or passes
        // the 64-bit operand through unchanged for dxc to split.
        expect(contains(on_program, "uint(reversebits("))
            << "the maintenance9 reverse must convert the natural-width result to uint";
        expect(contains(on_program, " << 16)"))
            << "16-bit reverse must preserve the legacy 32-bit result semantics";
        expect(contains(off_program, "reversebits(uint("))
            << "the legacy reverse must widen the narrow operand to uint first";
        // The 8-bit reverse expression is identical on both paths: the codegen
        // has no 8-bit bitwise path, so the source stays byte for byte.  Collect
        // every widened `reversebits(uint(...))` expression: on the maintenance9
        // path only the 8-bit reverse is widened, on the legacy path the 16-bit
        // reverse is widened too, and the 8-bit one must be unchanged.
        auto widened_reverse_expressions = [](luisa::string_view program) {
            luisa::vector<luisa::string> expressions;
            for (auto pos = program.find("reversebits(uint(");
                 pos != luisa::string_view::npos;
                 pos = program.find("reversebits(uint(", pos + 1u)) {
                auto end = pos;
                auto depth = 0u;
                do {
                    if (program[end] == '(') { ++depth; }
                    if (program[end] == ')') { --depth; }
                    ++end;
                } while (depth != 0u && end < program.size());
                expressions.emplace_back(program.substr(pos, end - pos));
            }
            return expressions;
        };
        auto on_widened = widened_reverse_expressions(on_program);
        auto off_widened = widened_reverse_expressions(off_program);
        expect(on_widened.size() == 1u)
            << "only the 8-bit reverse may stay widened on the maintenance9 path";
        expect(off_widened.size() == 2u)
            << "the legacy path widens both the 16-bit and the 8-bit reverse";
        expect(!on_widened.empty() &&
               std::find(off_widened.begin(), off_widened.end(), on_widened.front()) !=
                   off_widened.end())
            << "the 8-bit reverse expression changed between the two capability sets";

        auto const &runtime_directory = luisa::filesystem::path{argv[0]}.parent_path();
        auto on = compile_and_scan(on_program, runtime_directory,
                                   lc::hlsl::SpirvShaderCapabilities{.maintenance9_bitwise = true});
        expect(on.error.empty()) << on.error;
        expect(on.compiled);
        expect(on.validated);
        // reverse(ushort) and reverse(ulong) become one OpBitReverse each at
        // their own width; reverse(ubyte) stays a 32-bit OpBitReverse;
        // popcount(ulong) becomes one OpBitCount on the 64-bit operand.
        expect(eq(on.bit_reverse_count[0], 1u))
            << "expected one 16-bit OpBitReverse for reverse(ushort)";
        expect(eq(on.bit_reverse_count[1], 1u))
            << "expected one 32-bit OpBitReverse for reverse(ubyte)";
        expect(eq(on.bit_reverse_count[2], 1u))
            << "expected one 64-bit OpBitReverse for reverse(ulong)";
        expect(eq(on.bit_count_count[2], 1u))
            << "expected one OpBitCount on the 64-bit operand for popcount(ulong)";

        auto off = compile_and_scan(off_program, runtime_directory,
                                    lc::hlsl::SpirvShaderCapabilities{});
        expect(off.error.empty()) << off.error;
        expect(off.compiled);
        expect(off.validated);
        // Legacy emulation: the 16-bit reverse widens to 32 bits, the 64-bit
        // reverse splits into two 32-bit halves, and the 64-bit popcount splits
        // into two 32-bit counts; only the 8-bit reverse was 32-bit already.
        expect(eq(off.bit_reverse_count[0], 0u))
            << "the legacy path must not reverse at 16-bit width";
        expect(eq(off.bit_reverse_count[1], 4u))
            << "expected four 32-bit OpBitReverse (16-bit widened, 64-bit split, 8-bit widened)";
        expect(eq(off.bit_reverse_count[2], 0u))
            << "the legacy path must not reverse at 64-bit width";
        expect(eq(off.bit_count_count[1], 2u))
            << "the legacy popcount(ulong) must split into two 32-bit OpBitCount";
        expect(eq(off.bit_count_count[2], 0u))
            << "the legacy path must not count at 64-bit width";
    };

    "hlsl_dot2add_uses_the_valve_instruction_only_with_the_capability"_test =
        [argv] {
        // `dot2add(half2, half2, float)` is always the HLSL intrinsic: dxc
        // fuses it into OpFDot2MixAcc32VALVE (single rounding) only when
        // SPV_VALVE_mixed_float_dot_product is enabled, and otherwise supplies
        // the unfused sequence itself (products rounded to half, widened, then
        // accumulated in FP32). The same source must also compile to DXIL,
        // which covers a platform without the VALVE device feature.
        Kernel1D kernel = [](BufferHalf2 a, BufferHalf2 b, BufferFloat acc,
                             BufferFloat out) noexcept {
            auto i = dispatch_x();
            out.write(i, dot2add(a.read(i), b.read(i), acc.read(i)));
        };
        auto function = kernel.function()->function();
        auto on_codegen = lc::hlsl::CodegenUtility{}
                              .Codegen(function, {}, 0u, /*isSpirV=*/true, false, false,
                                       false, false, false,
                                       lc::hlsl::SpirvShaderCapabilities{.mixed_float_dot_product = true});
        auto off_codegen = lc::hlsl::CodegenUtility{}
                               .Codegen(function, {}, 0u, /*isSpirV=*/true, false, false);
        auto on_program = on_codegen.result.view();
        auto off_program = off_codegen.result.view();
        expect(contains(on_program, "dot2add("))
            << "the fused path must still emit the dot2add intrinsic";
        expect(contains(off_program, "dot2add("))
            << "the unfused path must emit the same dot2add intrinsic";

        auto const &runtime_directory = luisa::filesystem::path{argv[0]}.parent_path();
        auto on = compile_and_scan(on_program, runtime_directory,
                                   lc::hlsl::SpirvShaderCapabilities{.mixed_float_dot_product = true});
        expect(on.error.empty()) << on.error;
        expect(on.compiled);
        expect(on.validated);
        expect(eq(on.dot2add_count, 1u))
            << "expected exactly one OpFDot2MixAcc32VALVE for the dot2add of the kernel";
        expect(on.declares_mixed_float_dot_product)
            << "OpCapability DotProductFloat16AccFloat32VALVE is missing";

        auto off = compile_and_scan(off_program, runtime_directory,
                                    lc::hlsl::SpirvShaderCapabilities{});
        expect(off.error.empty()) << off.error;
        expect(off.compiled);
        expect(off.validated);
        expect(eq(off.dot2add_count, 0u))
            << "the unfused path must not declare OpFDot2MixAcc32VALVE";
        expect(!off.declares_mixed_float_dot_product)
            << "the unfused path must not declare the VALVE capability";

        // DXIL (SM 6.5): the intrinsic exists natively, so the off-capability
        // source compiles to a non-empty DXIL blob even on a platform without
        // VK_VALVE_shader_mixed_float_dot_product.
        lc::hlsl::ShaderCompiler dxil_compiler{runtime_directory, false};
        auto dxil_size = size_t{0u};
        vstd::string dxil_error;
        auto dxil = dxil_compiler.compile_compute(
            off_program, true, 65u, true, false, false);
        dxil.multi_visit(
            [&](const lc::hlsl::ComUniquePtr<IDxcBlob> &blob) {
                dxil_size = blob->GetBufferSize();
            },
            [&](const vstd::string &error) { dxil_error = error; });
        expect(dxil_error.empty()) << dxil_error;
        expect(dxil_size > 0u) << "the DXIL blob must not be empty";
    };

    "hlsl_split_barrier_lowers_to_the_ext_instructions_only_with_the_capability"_test =
        [argv] {
        // A split barrier publishes shared memory with a Release arrive, runs
        // independent work, and only then waits with an Acquire.  dxc emits
        // OpControlBarrierArriveEXT/OpControlBarrierWaitEXT with the same
        // Workgroup scopes as GroupMemoryBarrierWithGroupSync, and only when
        // SPV_EXT_split_barrier is enabled.
        Kernel1D kernel = [](BufferFloat in, BufferFloat out) noexcept {
            Shared<float> scratch{64u};
            auto i = dispatch_x();
            auto tid = thread_x();
            scratch.write(tid, in.read(i));
            split_barrier_arrive();
            // Independent ordinary arithmetic: reads only this invocation's
            // own input, never another invocation's shared-memory write.
            auto independent = in.read(i) * 2.0f + 1.0f;
            split_barrier_wait();
            out.write(i, scratch.read(tid ^ 1u) + independent);
        };
        auto codegen = lc::hlsl::CodegenUtility{}
                           .Codegen(kernel.function()->function(), {}, 0u,
                                    /*isSpirV=*/true, false, false,
                                    false, false, false,
                                    lc::hlsl::SpirvShaderCapabilities{.split_barrier = true});
        auto program = codegen.result.view();
        expect(contains(program, "#define LUISA_SPV_SPLIT_BARRIER 1"));
        expect(contains(program, "vk::SplitBarrierArrive()"))
            << "the arrive did not become a vk::SplitBarrierArrive call";
        expect(contains(program, "vk::SplitBarrierWait()"))
            << "the wait did not become a vk::SplitBarrierWait call";
        expect(!contains(program, "GroupMemoryBarrierWithGroupSync"))
            << "a split barrier must not degrade to a full workgroup barrier";

        auto const &runtime_directory = luisa::filesystem::path{argv[0]}.parent_path();
        auto facts = compile_and_scan(program, runtime_directory,
                                      lc::hlsl::SpirvShaderCapabilities{.split_barrier = true});
        expect(facts.error.empty()) << facts.error;
        expect(facts.compiled);
        expect(facts.validated);
        expect(eq(facts.barrier_arrive_count, 1u))
            << "expected exactly one OpControlBarrierArriveEXT for the arrive";
        expect(eq(facts.barrier_wait_count, 1u))
            << "expected exactly one OpControlBarrierWaitEXT for the wait";
        expect(facts.declares_split_barrier) << "OpCapability SplitBarrierEXT is missing";
        // Execution scope Workgroup (2), memory scope Workgroup (2),
        // semantics Release|WorkgroupMemory (4|256 = 260) on the arrive and
        // Acquire|WorkgroupMemory (2|256 = 258) on the wait.
        auto const workgroup = static_cast<uint32_t>(spv::Scope::ScopeWorkgroup);
        auto const release_workgroup =
            static_cast<uint32_t>(spv::MemorySemanticsReleaseMask) |
            static_cast<uint32_t>(spv::MemorySemanticsWorkgroupMemoryMask);
        auto const acquire_workgroup =
            static_cast<uint32_t>(spv::MemorySemanticsAcquireMask) |
            static_cast<uint32_t>(spv::MemorySemanticsWorkgroupMemoryMask);
        expect(eq(workgroup, 2u));
        expect(eq(release_workgroup, 260u));
        expect(eq(acquire_workgroup, 258u));
        expect(eq(facts.barrier_arrive_operands[0], workgroup))
            << "the arrive execution scope must be Workgroup";
        expect(eq(facts.barrier_arrive_operands[1], workgroup))
            << "the arrive memory scope must be Workgroup";
        expect(eq(facts.barrier_arrive_operands[2], release_workgroup))
            << "the arrive semantics must be Release|WorkgroupMemory";
        expect(eq(facts.barrier_wait_operands[0], workgroup))
            << "the wait execution scope must be Workgroup";
        expect(eq(facts.barrier_wait_operands[1], workgroup))
            << "the wait memory scope must be Workgroup";
        expect(eq(facts.barrier_wait_operands[2], acquire_workgroup))
            << "the wait semantics must be Acquire|WorkgroupMemory";

        // An ordinary sync_block() kernel is untouched: it keeps the classic
        // full workgroup barrier and declares no split-barrier capability.
        Kernel1D sync_kernel = [](BufferFloat in, BufferFloat out) noexcept {
            Shared<float> scratch{64u};
            auto i = dispatch_x();
            auto tid = thread_x();
            scratch.write(tid, in.read(i));
            sync_block();
            out.write(i, scratch.read(tid ^ 1u));
        };
        auto sync_codegen = lc::hlsl::CodegenUtility{}
                                .Codegen(sync_kernel.function()->function(), {}, 0u,
                                         /*isSpirV=*/true, false, false);
        auto sync_program = sync_codegen.result.view();
        expect(contains(sync_program, "GroupMemoryBarrierWithGroupSync()"))
            << "sync_block must keep the classic workgroup barrier";
        expect(!contains(sync_program, "vk::SplitBarrier"))
            << "sync_block must not emit a split barrier";
        expect(!contains(sync_program, "#define LUISA_SPV_SPLIT_BARRIER 1"))
            << "the macro must not be defined without the capability";
    };
}
