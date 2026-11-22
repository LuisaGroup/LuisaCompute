#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>
#include <spirv/unified1/spirv.hpp11>

#include "spirv_codegen/optimizer.h"
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>

using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr auto dead_arithmetic_module = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%function = OpTypeFunction %void
%uint = OpTypeInt 32 0
%one = OpConstant %uint 1
%main = OpFunction %void None %function
%entry = OpLabel
%dead = OpIAdd %uint %one %one
OpReturn
OpFunctionEnd
)";

constexpr auto dead_subgroup_arithmetic_module = R"(
OpCapability Shader
OpCapability GroupNonUniformArithmetic
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%function = OpTypeFunction %void
%uint = OpTypeInt 32 0
%subgroup = OpConstant %uint 3
%one = OpConstant %uint 1
%main = OpFunction %void None %function
%entry = OpLabel
%dead = OpGroupNonUniformIAdd %uint %subgroup Reduce %one
OpReturn
OpFunctionEnd
)";

constexpr auto dead_float64_arithmetic_module = R"(
OpCapability Shader
OpCapability Float64
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%function = OpTypeFunction %void
%double = OpTypeFloat 64
%one = OpConstant %double 1
%main = OpFunction %void None %function
%entry = OpLabel
%dead = OpFAdd %double %one %one
OpReturn
OpFunctionEnd
)";

constexpr auto storage_only_uniform_16_module = R"(
OpCapability Shader
OpCapability UniformAndStorageBuffer16BitAccess
OpExtension "SPV_KHR_16bit_storage"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %ubo
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpDecorate %ubo DescriptorSet 0
OpDecorate %ubo Binding 0
%void = OpTypeVoid
%function = OpTypeFunction %void
%short = OpTypeInt 16 1
%block = OpTypeStruct %short
%uniform_block_ptr = OpTypePointer Uniform %block
%ubo = OpVariable %uniform_block_ptr Uniform
%main = OpFunction %void None %function
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

constexpr auto unsupported_mixed_capability_module = R"(
OpCapability Shader
OpCapability GroupNonUniform
OpCapability GroupNonUniformArithmetic
OpCapability GroupNonUniformShuffle
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%function = OpTypeFunction %void
%uint = OpTypeInt 32 0
%subgroup = OpConstant %uint 3
%one = OpConstant %uint 1
%main = OpFunction %void None %function
%entry = OpLabel
%collective = OpGroupNonUniformIAdd %uint %subgroup Reduce %one
%dead = OpIAdd %uint %one %one
OpReturn
OpFunctionEnd
)";

// A loop-local floating-point definition may be referenced by a module-scope
// annotation while an independent loop-local definition escapes through the
// merge. The annotation is present in def-use, but is not an executable CFG
// use and therefore must neither invalidate LCSSA nor be rewritten by LCSSA
// construction.
constexpr auto annotated_loop_lcssa_module = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %condition_buffer
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %condition_block Block
OpMemberDecorate %condition_block 0 Offset 0
OpDecorate %condition_buffer DescriptorSet 0
OpDecorate %condition_buffer Binding 0
OpDecorate %sum NoContraction
%void = OpTypeVoid
%bool = OpTypeBool
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%function = OpTypeFunction %void
%condition_block = OpTypeStruct %uint
%condition_block_ptr = OpTypePointer Uniform %condition_block
%condition_uint_ptr = OpTypePointer Uniform %uint
%condition_buffer = OpVariable %condition_block_ptr Uniform
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%limit = OpConstant %uint 4
%float_zero = OpConstant %float 0
%float_one = OpConstant %float 1
%main = OpFunction %void None %function
%entry = OpLabel
%condition_address = OpAccessChain %condition_uint_ptr %condition_buffer %zero
%condition_value = OpLoad %uint %condition_address
%condition = OpINotEqual %bool %condition_value %zero
OpBranch %header
%header = OpLabel
%index = OpPhi %uint %zero %entry %next %continue
%accumulator = OpPhi %float %float_zero %entry %sum %continue
OpLoopMerge %merge %continue None
OpBranch %body
%body = OpLabel
%sum = OpFAdd %float %accumulator %float_one
OpSelectionMerge %selection_merge None
OpBranchConditional %condition %selection_true %selection_false
%selection_true = OpLabel
OpBranch %selection_merge
%selection_false = OpLabel
OpBranch %selection_merge
%selection_merge = OpLabel
OpBranch %continue
%continue = OpLabel
%next = OpIAdd %uint %index %one
%more = OpULessThan %bool %next %limit
OpBranchConditional %more %header %merge
%merge = OpLabel
%final = OpFAdd %float %accumulator %float_one
OpReturn
OpFunctionEnd
)";

[[nodiscard]] std::string replace_placeholder(
    std::string text, luisa::string_view placeholder,
    luisa::string_view replacement) {
    auto offset = text.find(placeholder.data(), 0u, placeholder.size());
    expect(offset != std::string::npos)
        << "the SPIR-V optimizer fixture template must contain its placeholder";
    if (offset != std::string::npos) {
        text.replace(offset, placeholder.size(),
                     replacement.data(), replacement.size());
    }
    return text;
}

// ---------------------------------------------------------------------------
// DXC-parity transformation fixtures
//
// Every fixture below is assembled and validated before optimization. The
// shared prologue declares one StorageBuffer block and loads its inputs; the
// body under test writes its result back into that block so the arithmetic
// cannot be removed by dead-code elimination and the assertion therefore
// observes a live transformation instead of registration bookkeeping.
// ---------------------------------------------------------------------------

// One StorageBuffer block of four float32 slots: three inputs and one output.
constexpr auto float_storage_module_prefix = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpMemberDecorate %block 1 Offset 4
OpMemberDecorate %block 2 Offset 8
OpMemberDecorate %block 3 Offset 12
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
$extra_decorations
%void = OpTypeVoid
%bool = OpTypeBool
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%function = OpTypeFunction %void
%block = OpTypeStruct %float %float %float %float
%block_ptr = OpTypePointer StorageBuffer %block
%float_ptr = OpTypePointer StorageBuffer %float
%out = OpVariable %block_ptr StorageBuffer
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%two = OpConstant %uint 2
%three = OpConstant %uint 3
%main = OpFunction %void None %function
%entry = OpLabel
%a_ptr = OpAccessChain %float_ptr %out %zero
%a = OpLoad %float %a_ptr
%b_ptr = OpAccessChain %float_ptr %out %one
%b = OpLoad %float %b_ptr
%c_ptr = OpAccessChain %float_ptr %out %two
%c = OpLoad %float %c_ptr
%dst_ptr = OpAccessChain %float_ptr %out %three
)";

// One StorageBuffer block of two uint32 slots: one input and one output.
constexpr auto uint_storage_module_prefix = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpMemberDecorate %block 1 Offset 4
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
$extra_decorations
%void = OpTypeVoid
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%function = OpTypeFunction %void
%block = OpTypeStruct %uint %uint
%block_ptr = OpTypePointer StorageBuffer %block
%uint_ptr = OpTypePointer StorageBuffer %uint
%out = OpVariable %block_ptr StorageBuffer
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%four = OpConstant %uint 4
%seven = OpConstant %uint 7
%eight = OpConstant %uint 8
%nine = OpConstant %uint 9
%false = OpConstantFalse %bool
%main = OpFunction %void None %function
%entry = OpLabel
%value_ptr = OpAccessChain %uint_ptr %out %zero
%value = OpLoad %uint %value_ptr
%dst_ptr = OpAccessChain %uint_ptr %out %one
)";

constexpr auto storage_module_suffix = R"(
OpReturn
OpFunctionEnd
)";

// %sum = %a * %b + %c, stored live.
constexpr auto fma_contraction_body = R"(
%product = OpFMul %float %a %b
%sum = OpFAdd %float %product %c
OpStore %dst_ptr %sum
)";

// Nested division (a / b) / c, stored live.
constexpr auto nested_division_body = R"(
%inner = OpFDiv %float %a %b
%outer = OpFDiv %float %inner %c
OpStore %dst_ptr %outer
)";

// Two quotients sharing one denominator, both stored live.
constexpr auto common_denominator_division_body = R"(
%first = OpFDiv %float %a %c
OpStore %dst_ptr %first
%second = OpFDiv %float %b %c
OpStore %b_ptr %second
)";

// Integer multiply by a power of two, stored live.
constexpr auto strength_reduction_body = R"(
%product = OpIMul %uint %value %eight
OpStore %dst_ptr %product
)";

// Boolean select consumed by a live uint select so neither the condition nor
// the result can be discarded. select(cond, x, false) is rewritten into
// OpLogicalAnd by BooleanSimplifyPass; the constant-arm forms are already
// folded by the shared instruction-folding rules, which is why this fixture
// uses a non-constant left arm to keep the BooleanSimplify-specific rewrite
// observable in the whole-pipeline result.
constexpr auto boolean_simplify_body = R"(
%zero_value = OpINotEqual %bool %value %zero
%one_value = OpINotEqual %bool %value %one
%selected = OpSelect %bool %zero_value %one_value %false
%picked = OpSelect %uint %selected %seven %nine
OpStore %dst_ptr %picked
)";

// Small eligible counted loop whose accumulator escapes through the storage
// block, so a folded trip count remains observable in the final binary.
constexpr auto accumulator_loop_body = R"(
OpBranch %header
%header = OpLabel
%index = OpPhi %uint %zero %entry %next %continue
%accumulator = OpPhi %uint %zero %entry %accumulated %continue
%condition = OpULessThan %bool %index %four
OpLoopMerge %merge %continue $loop_control
OpBranchConditional %condition %body %merge
%body = OpLabel
%accumulated = OpIAdd %uint %accumulator %index
OpBranch %continue
%continue = OpLabel
%next = OpIAdd %uint %index %one
OpBranch %header
%merge = OpLabel
OpStore %dst_ptr %accumulator
)";

[[nodiscard]] std::string float_storage_module(
    luisa::string_view body, luisa::string_view extra_decorations = {}) {
    auto module = replace_placeholder(
        std::string{float_storage_module_prefix},
        "$extra_decorations", extra_decorations);
    module += std::string{body};
    module += std::string{storage_module_suffix};
    return module;
}

[[nodiscard]] std::string uint_storage_module(
    luisa::string_view body, luisa::string_view extra_decorations = {}) {
    auto module = replace_placeholder(
        std::string{uint_storage_module_prefix},
        "$extra_decorations", extra_decorations);
    module += std::string{body};
    module += std::string{storage_module_suffix};
    return module;
}

[[nodiscard]] std::string make_accumulator_loop_module(
    luisa::string_view loop_control) {
    return uint_storage_module(replace_placeholder(
        std::string{accumulator_loop_body}, "$loop_control", loop_control));
}

// ---------------------------------------------------------------------------
// DXC c4adf384 pass-sequence description
//
// DXC's default route registers three structural prefix passes, then the stock
// SPIRV-Tools performance pipeline, then its own additional tail, then iterates
// the whole sequence to a fixed point of at most eight runs. compute and full
// map onto that pipeline unchanged.
// ---------------------------------------------------------------------------

// DXC c4adf384:tools/clang/lib/SPIRV/SpirvEmitter.cpp L20260-20262.
constexpr std::string_view dxc_prefix_passes[] = {
    "merge-return",
    "dead-branch-elim",
    "cfg-cleanup",
};

// The complete stock spvtools::Optimizer::RegisterPerformancePasses(false)
// pipeline at the vendored revision
// (src/ext/SPIRV-Tools/source/opt/optimizer.cpp L185-230), flattened in
// registration order. 44 passes.
constexpr std::string_view stock_performance_passes[] = {
    // structural normalization
    "wrap-op-kill",
    "dead-branch-elim",
    "merge-return",
    "inline-exhaustive",
    "eliminate-dead-functions",
    "adce",
    "private-to-local",
    // local memory, aggregates, and scalar SSA
    "local-single-block-load-store-elim",
    "local-single-store-elim",
    "adce",
    "scalar-replacement",
    "local-access-chain-convert",
    "local-single-block-load-store-elim",
    "local-single-store-elim",
    "adce",
    "local-multi-store-elim",
    "adce",
    "ccp",
    "adce",
    "loop-unroll",
    "dead-branch-elim",
    "redundancy-elimination",
    "combine-access-chains",
    "simplification",
    "scalar-replacement",
    "local-access-chain-convert",
    "local-single-block-load-store-elim",
    "local-single-store-elim",
    "adce",
    "ssa-rewrite",
    "adce",
    "vector-dce",
    "dead-insert-elim",
    "dead-branch-elim",
    "simplification",
    "if-conversion",
    "copy-propagate-arrays",
    "reduce-load-size",
    "adce",
    "block-merge",
    "redundancy-elimination",
    "dead-branch-elim",
    "block-merge",
    "simplification",
};

// DXC c4adf384's additional tail, strictly IEEE
// (SpirvEmitter.cpp L20264-20294 without the conditional FloatDivOptPass).
// 29 passes.
constexpr std::string_view dxc_tail_strict_passes[] = {
    "loop-unroll",
    "ccp",
    "simplification",
    "dead-branch-elim",
    "block-merge",
    "scalar-replacement",
    "local-access-chain-convert",
    "local-single-block-load-store-elim",
    "local-single-store-elim",
    "local-multi-store-elim",
    "ssa-rewrite",
    "strength-reduction",
    "boolean-simplify",
    "loop-unswitch",
    "loop-invariant-code-motion",
    "gvn-hoist",
    "new-gvn",
    "insert-extract-elim",
    "local-redundancy-elimination",
    "simplification",
    "adce",
    "dead-branch-elim",
    "block-merge",
    "cfg-cleanup",
    "eliminate-dead-members",
    "eliminate-dead-constant",
    "dead-variable-elimination",
    "spread-volatile-semantics",
    "compact-ids",
};

// The same tail with CreateFloatDivOptPass() registered between
// CreateBooleanSimplifyPass() and CreateLoopUnswitchPass()
// (SpirvEmitter.cpp L20277-20279, gated by !IEEEStrict). 30 passes.
constexpr std::string_view dxc_tail_fast_passes[] = {
    "loop-unroll",
    "ccp",
    "simplification",
    "dead-branch-elim",
    "block-merge",
    "scalar-replacement",
    "local-access-chain-convert",
    "local-single-block-load-store-elim",
    "local-single-store-elim",
    "local-multi-store-elim",
    "ssa-rewrite",
    "strength-reduction",
    "boolean-simplify",
    "float-div-opt",
    "loop-unswitch",
    "loop-invariant-code-motion",
    "gvn-hoist",
    "new-gvn",
    "insert-extract-elim",
    "local-redundancy-elimination",
    "simplification",
    "adce",
    "dead-branch-elim",
    "block-merge",
    "cfg-cleanup",
    "eliminate-dead-members",
    "eliminate-dead-constant",
    "dead-variable-elimination",
    "spread-volatile-semantics",
    "compact-ids",
};

// The lightweight preset is the intentionally minimal native chain.
constexpr std::string_view lightweight_passes[] = {
    "adce",
    "block-merge",
    "simplification",
    "dead-branch-elim",
};

// The only pass this backend appends after the DXC sequence for a module whose
// declarations the audited trim-capabilities contract accepts.
constexpr std::string_view native_capability_trim_pass = "trim-capabilities";

template<size_t N>
void append_pass_names(std::vector<std::string> &out,
                       const std::string_view (&names)[N]) {
    for (auto name : names) { out.emplace_back(name); }
}

// The literal DXC structure: 3 prefix passes, the complete 44-pass stock
// pipeline, and a 29- or 30-pass tail.
static_assert(std::size(dxc_prefix_passes) == 3u);
static_assert(std::size(stock_performance_passes) == 44u);
static_assert(std::size(dxc_tail_strict_passes) == 29u);
static_assert(std::size(dxc_tail_fast_passes) == 30u);
static_assert(std::size(lightweight_passes) == 4u);

[[nodiscard]] std::vector<std::string> expected_dxc_pass_sequence(
    bool enable_fast_math) {
    std::vector<std::string> expected;
    append_pass_names(expected, dxc_prefix_passes);
    append_pass_names(expected, stock_performance_passes);
    if (enable_fast_math) {
        append_pass_names(expected, dxc_tail_fast_passes);
    } else {
        append_pass_names(expected, dxc_tail_strict_passes);
    }
    expected.emplace_back(native_capability_trim_pass);
    return expected;
}

[[nodiscard]] std::vector<std::string> expected_lightweight_pass_sequence() {
    std::vector<std::string> expected;
    append_pass_names(expected, lightweight_passes);
    expected.emplace_back(native_capability_trim_pass);
    return expected;
}

void expect_exact_pass_sequence(
    const lc::spirv::SpirvOptimizerReport &report,
    const std::vector<std::string> &expected) {
    expect(report.registered_passes.size() == expected.size())
        << "the optimizer must register exactly the literal DXC sequence: "
           "expected only the DXC prefix, stock pipeline, DXC tail and native "
           "capability trim";
    auto shared = std::min<size_t>(report.registered_passes.size(),
                                   expected.size());
    for (auto index = 0u; index < shared; ++index) {
        expect(report.registered_passes[index] == expected[index].c_str())
            << "registered pass #" << index << " must be '"
            << expected[index].c_str() << "', found '"
            << report.registered_passes[index].c_str() << "'";
    }
}

void expect_optimizer_report_ok(const lc::spirv::SpirvOptimizerReport &report) {
    expect(report.attempted);
    expect(report.succeeded)
        << "the SPIR-V optimizer must succeed: "
        << report.diagnostics.c_str();
    expect(report.output_validated);
    expect(report.diagnostics.find("failed") == luisa::string::npos)
        << "a successful run must not report a failure diagnostic";
}

// ---------------------------------------------------------------------------
// SPIR-V inspection helpers (test-local)
// ---------------------------------------------------------------------------

constexpr size_t kSpirvHeaderWordCount = 5u;

struct SpirvInstructionRef {
    spv::Op opcode{spv::Op::OpNop};
    size_t offset{0u};
};

// Returns the first instruction with |opcode|, or nothing when the module has
// none.
[[nodiscard]] std::optional<SpirvInstructionRef> find_instruction(
    const std::vector<uint32_t> &words, spv::Op opcode) {
    for (size_t offset = kSpirvHeaderWordCount; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) { break; }
        if (static_cast<spv::Op>(words[offset] & 0xffffu) == opcode) {
            return SpirvInstructionRef{opcode, offset};
        }
        offset += word_count;
    }
    return std::nullopt;
}

[[nodiscard]] size_t count_instructions(const std::vector<uint32_t> &words,
                                        spv::Op opcode) {
    size_t count = 0u;
    for (size_t offset = kSpirvHeaderWordCount; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) { break; }
        if (static_cast<spv::Op>(words[offset] & 0xffffu) == opcode) { ++count; }
        offset += word_count;
    }
    return count;
}

[[nodiscard]] bool declares_capability(const std::vector<uint32_t> &words,
                                       spv::Capability capability) {
    for (size_t offset = kSpirvHeaderWordCount; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) { break; }
        if (static_cast<spv::Op>(words[offset] & 0xffffu) ==
                spv::Op::OpCapability &&
            word_count == 2u &&
            static_cast<spv::Capability>(words[offset + 1u]) == capability) {
            return true;
        }
        offset += word_count;
    }
    return false;
}

// |literal| is compared against the NUL-terminated operand of OpExtension.
[[nodiscard]] bool declares_extension(const std::vector<uint32_t> &words,
                                      std::string_view literal) {
    for (size_t offset = kSpirvHeaderWordCount; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) { break; }
        if (static_cast<spv::Op>(words[offset] & 0xffffu) ==
            spv::Op::OpExtension) {
            auto *bytes = reinterpret_cast<const char *>(words.data() + offset + 1u);
            auto byte_count = (word_count - 1u) * sizeof(uint32_t);
            auto length = size_t{0u};
            while (length < byte_count && bytes[length] != '\0') { ++length; }
            if (length < byte_count &&
                std::string_view{bytes, length} == literal) {
                return true;
            }
        }
        offset += word_count;
    }
    return false;
}

// Counts GLSL.std.450 Fma extended instructions (opcode 50) without depending
// on how a disassembler renders the extended-instruction literal.
[[nodiscard]] size_t count_glsl_fma(const std::vector<uint32_t> &words) {
    constexpr auto glsl_std_450_fma = 50u;
    size_t count = 0u;
    for (size_t offset = kSpirvHeaderWordCount; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) { break; }
        // OpExtInst: word 3 is the instruction set, word 4 the literal.
        if (static_cast<spv::Op>(words[offset] & 0xffffu) ==
                spv::Op::OpExtInst &&
            word_count >= 5u && words[offset + 4u] == glsl_std_450_fma) {
            ++count;
        }
        offset += word_count;
    }
    return count;
}

// Resolves the definition of |result_id| for the instruction kinds this test
// file inspects. Each listed instruction encodes a result type in word 2 and a
// result id in word 3.
[[nodiscard]] std::optional<SpirvInstructionRef> find_definition(
    const std::vector<uint32_t> &words, uint32_t result_id) {
    constexpr spv::Op result_defining_opcodes[] = {
        spv::Op::OpConstant,
        spv::Op::OpLoad,
        spv::Op::OpAccessChain,
        spv::Op::OpPhi,
        spv::Op::OpFAdd,
        spv::Op::OpFSub,
        spv::Op::OpFMul,
        spv::Op::OpFDiv,
        spv::Op::OpIAdd,
        spv::Op::OpIMul,
        spv::Op::OpShiftLeftLogical,
        spv::Op::OpSelect,
        spv::Op::OpLogicalNot,
        spv::Op::OpULessThan,
        spv::Op::OpINotEqual,
        spv::Op::OpExtInst,
        spv::Op::OpFmaKHR,
        spv::Op::OpLogicalAnd,
    };
    for (size_t offset = kSpirvHeaderWordCount; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) { break; }
        auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        for (auto candidate : result_defining_opcodes) {
            if (opcode == candidate && word_count >= 3u &&
                words[offset + 2u] == result_id) {
                return SpirvInstructionRef{opcode, offset};
            }
        }
        offset += word_count;
    }
    return std::nullopt;
}

// Returns the declaring opcode of a type id.
[[nodiscard]] spv::Op type_declaration_opcode(
    const std::vector<uint32_t> &words, uint32_t result_id) {
    for (size_t offset = kSpirvHeaderWordCount; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) { break; }
        auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        if ((opcode == spv::Op::OpTypeBool || opcode == spv::Op::OpTypeInt ||
             opcode == spv::Op::OpTypeFloat || opcode == spv::Op::OpTypeVector) &&
            word_count >= 2u && words[offset + 1u] == result_id) {
            return opcode;
        }
        offset += word_count;
    }
    return spv::Op::OpNop;
}

// Returns the stored object id of the module's single OpStore so a test can
// prove that the transformation it exercises reaches live memory.
[[nodiscard]] std::optional<uint32_t> single_store_object_id(
    const std::vector<uint32_t> &words) {
    auto instruction = find_instruction(words, spv::Op::OpStore);
    if (!instruction) { return std::nullopt; }
    return words[instruction->offset + 2u];
}

// Returns the OpFDiv denominator (in operand 1) of |ref|.
[[nodiscard]] uint32_t fdiv_denominator(const std::vector<uint32_t> &words,
                                        const SpirvInstructionRef &ref) {
    return words[ref.offset + 4u];
}

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
    luisa::optional<std::string> _previous;

public:
    ScopedEnvironmentVariable(const char *name, const char *value) noexcept
        : _name{name} {
        if (auto previous = std::getenv(name)) { _previous.emplace(previous); }
        set_environment_variable(name, value);
    }
    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(_name, _previous ? _previous->c_str() : nullptr);
    }
    ScopedEnvironmentVariable(const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(const ScopedEnvironmentVariable &) = delete;
};

// Removes the optimizer environment knobs that would otherwise change the
// registered sequence, the iteration cap, or the optimizer options behind a
// test's back.
class ScopedDefaultOptimizerEnvironment {
private:
    ScopedEnvironmentVariable _pass_flags{"LUISA_SPIRV_OPT_PASS_FLAGS", nullptr};
    ScopedEnvironmentVariable _max_iterations{
        "LUISA_SPIRV_OPT_MAX_ITERATIONS", nullptr};
    ScopedEnvironmentVariable _max_id_bound{
        "LUISA_SPIRV_OPT_MAX_ID_BOUND", nullptr};
    ScopedEnvironmentVariable _preserve_bindings{
        "LUISA_SPIRV_OPT_PRESERVE_BINDINGS", nullptr};

public:
    ScopedDefaultOptimizerEnvironment() noexcept = default;
};

[[nodiscard]] std::vector<uint32_t> assemble_test_module(
    luisa::string_view assembly = dead_arithmetic_module) {
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    std::vector<uint32_t> words;
    expect(tools.Assemble(std::string{assembly}, &words))
        << "failed to assemble the SPIR-V optimizer fixture";
    expect(tools.Validate(words))
        << "SPIR-V optimizer fixture must validate before optimization";
    return words;
}

[[nodiscard]] std::string disassemble(const std::vector<uint32_t> &words) {
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    std::string text;
    expect(tools.Disassemble(words.data(), words.size(), &text))
        << "failed to disassemble optimized SPIR-V";
    return text;
}

[[nodiscard]] bool validates(const std::vector<uint32_t> &words) {
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    return tools.Validate(words.data(), words.size());
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_optimizer_none_is_byte_exact"_test = [] {
        auto words = assemble_test_module();
        auto original = words;
        auto report = lc::spirv::optimize_spirv(
            words, {.level = 3, .preset = "none"});
        expect(!report.attempted);
        expect(report.succeeded);
        expect(!report.changed);
        expect(report.effective_preset == "none");
        expect(words == original)
            << "preset=none must not rewrite or renumber the module";
    };

    "spirv_validation_contract_rejects_malformed_modules"_test = [] {
        auto words = assemble_test_module();
        auto valid = lc::spirv::validate_spirv(words.data(), words.size());
        expect(valid.valid);

        auto malformed = words;
        malformed[0] = 0u;
        auto invalid = lc::spirv::validate_spirv(
            malformed.data(), malformed.size());
        expect(!invalid.valid);
        expect(!invalid.diagnostics.empty())
            << "validation failure must retain actionable diagnostics";

        auto empty = lc::spirv::validate_spirv(nullptr, 0u);
        expect(!empty.valid);
        expect(!empty.diagnostics.empty());
    };

    "spirv_optimizer_failure_retains_input_byte_exactly"_test = [] {
        auto words = assemble_test_module();
        words[0] = 0u;
        auto original = words;
        auto report = lc::spirv::optimize_spirv(
            words, {.level = 1, .preset = "lightweight"});
        expect(report.attempted);
        expect(!report.succeeded);
        expect(!report.output_validated);
        expect(!report.changed);
        expect(!report.diagnostics.empty())
            << "optimizer failure must retain actionable diagnostics";
        expect(eq(report.input_word_count, original.size()));
        expect(eq(report.output_word_count, original.size()));
        expect(words == original)
            << "optimizer failure must retain the caller's input byte-exactly";
    };

    "spirv_transform_rejects_successful_invalid_candidate_transactionally"_test = [] {
        auto words = assemble_test_module();
        auto original = words;
        auto invalid_candidate = words;
        invalid_candidate.pop_back();
        auto candidate_word_count = invalid_candidate.size();

        auto report = lc::spirv::validate_and_commit_spirv_transform(
            words, std::move(invalid_candidate));

        expect(!report.succeeded);
        expect(!report.output_validated);
        expect(!report.changed);
        expect(!report.diagnostics.empty())
            << "a rejected transform candidate must retain validator diagnostics";
        expect(eq(report.input_word_count, original.size()));
        expect(eq(report.candidate_word_count, candidate_word_count));
        expect(eq(report.output_word_count, original.size()));
        expect(words == original)
            << "a successful transform that produces invalid SPIR-V must not "
               "partially commit or mutate the caller's artifact";
    };

    "spirv_optimizer_lightweight_removes_dead_code"_test = [] {
        auto words = assemble_test_module();
        auto original = words;
        auto report = lc::spirv::optimize_spirv(
            words, {.level = 1, .preset = "lightweight"});
        expect(report.attempted);
        expect(report.succeeded);
        expect(report.output_validated);
        expect(report.changed);
        expect(words != original);
        expect(validates(words))
            << "SPIR-V must remain valid after the lightweight preset";
        expect(disassemble(words).find("OpIAdd") == std::string::npos)
            << "the deliberately dead arithmetic must be eliminated";
    };

    "spirv_lightweight_preset_stays_minimal"_test = [] {
        ScopedDefaultOptimizerEnvironment defaults;
        auto words = assemble_test_module();
        auto report = lc::spirv::optimize_spirv(
            words, {.level = 1, .preset = "lightweight"});
        expect_optimizer_report_ok(report);
        expect(report.effective_preset == "lightweight");
        expect(report.capability_trim_registered)
            << "a Shader-only module is inside the audited trim contract";
        expect(!report.loop_unroll_registered)
            << "lightweight must not run the DXC unroll analysis";
        expect(!report.fma_contraction_attempted)
            << "lightweight must not run the DXC post-convergence FMA stage";
        expect(report.max_iterations == 8u);
        expect_exact_pass_sequence(
            report, expected_lightweight_pass_sequence());
        expect(validates(words));
    };

    "spirv_dxc_parity_pass_sequence_is_exact"_test = [] {
        ScopedDefaultOptimizerEnvironment defaults;
        // compute and full share the complete DXC pipeline; the explicit
        // fast-math policy only inserts CreateFloatDivOptPass() at its literal
        // position and enables the post-convergence contraction stage.
        for (auto preset : {"compute", "full"}) {
            for (auto enable_fast_math : {false, true}) {
                auto words = assemble_test_module();
                auto report = lc::spirv::optimize_spirv(
                    words,
                    {.level = 2,
                     .preset = preset,
                     .enable_fast_math = enable_fast_math});
                expect_optimizer_report_ok(report);
                expect(report.effective_preset == preset);
                expect(report.loop_unroll_registered)
                    << "the DXC pipeline always registers loop-unroll analysis";
                expect(report.capability_trim_registered);
                expect(report.max_iterations == 8u)
                    << "DXC c4adf384 caps the fixed point at eight iterations";
                expect(report.iterations >= 1u);
                expect(report.iterations <= report.max_iterations);
                expect(report.fma_contraction_attempted == enable_fast_math)
                    << "the post-convergence contraction stage is fast-math "
                       "only";
                expect_exact_pass_sequence(
                    report, expected_dxc_pass_sequence(enable_fast_math));
                // float-div-opt must exist only in fast math and only at DXC's
                // exact registration point: after boolean-simplify and before
                // loop-unswitch, i.e. behind the 3 prefix and 44 stock passes plus
                // the 13 tail entries that precede it.
                std::vector<size_t> float_div_opt_positions;
                for (size_t index = 0u;
                     index < report.registered_passes.size(); ++index) {
                    if (report.registered_passes[index] == "float-div-opt") {
                        float_div_opt_positions.push_back(index);
                    }
                }
                expect(float_div_opt_positions.size() ==
                       (enable_fast_math ? 1u : 0u))
                    << "float-div-opt must be registered exactly in fast math";
                if (enable_fast_math && !float_div_opt_positions.empty()) {
                    auto position = float_div_opt_positions.front();
                    constexpr auto expected_position = 3u + 44u + 13u;
                    expect(position == expected_position)
                        << "float-div-opt must keep DXC's exact position";
                    expect(report.registered_passes[position - 1u] ==
                           "boolean-simplify");
                    expect(report.registered_passes[position + 1u] ==
                           "loop-unswitch");
                }
                expect(validates(words));
            }
        }
    };

    "spirv_dxc_parity_registers_both_unrolls_and_loop_unswitch"_test = [] {
        ScopedDefaultOptimizerEnvironment defaults;
        // DXC keeps the stock explicit-Unroll pass and adds its own bounded
        // automatic unroll, and it always registers loop-unswitch. Neither is
        // an opt-in policy switch any more.
        for (auto preset : {"compute", "full"}) {
            auto words = assemble_test_module(annotated_loop_lcssa_module);
            auto report = lc::spirv::optimize_spirv(
                words, {.level = 2, .preset = preset});
            expect_optimizer_report_ok(report);
            auto unroll_count = size_t{0u};
            auto unswitch_count = size_t{0u};
            for (const auto &pass : report.registered_passes) {
                unroll_count += pass == "loop-unroll" ? 1u : 0u;
                unswitch_count += pass == "loop-unswitch" ? 1u : 0u;
            }
            expect(unroll_count == 2u)
                << "the stock explicit-Unroll pass and DXC's tail automatic "
                   "unroll must both be registered";
            expect(unswitch_count == 1u)
                << "DXC parity requires the loop-unswitch pass unconditionally";
            expect(validates(words));
        }
    };

    "spirv_loop_unswitch_ignores_non_cfg_annotation_uses"_test = [] {
        auto words = assemble_test_module(annotated_loop_lcssa_module);
        spvtools::Optimizer optimizer{SPV_ENV_VULKAN_1_2};
        optimizer.RegisterPass(spvtools::CreateLoopUnswitchPass());
        std::vector<uint32_t> optimized;
        expect(optimizer.Run(words.data(), words.size(), &optimized))
            << "loop unswitch must accept annotations of loop-local values";
        expect(validates(optimized))
            << "loop unswitch and LCSSA construction must preserve valid SPIR-V";
        expect(disassemble(optimized).find("NoContraction") != std::string::npos)
            << "the non-semantic annotation must survive the CFG transform";
    };

    "spirv_optimizer_fixed_point_cap_defaults_to_dxc_eight"_test = [] {
        ScopedEnvironmentVariable clear_pass_flags{
            "LUISA_SPIRV_OPT_PASS_FLAGS", nullptr};
        {
            ScopedEnvironmentVariable clear_cap{
                "LUISA_SPIRV_OPT_MAX_ITERATIONS", nullptr};
            auto words = assemble_test_module();
            auto report = lc::spirv::optimize_spirv(
                words, {.level = 2, .preset = "compute"});
            expect_optimizer_report_ok(report);
            expect(report.max_iterations == 8u)
                << "DXC c4adf384 kSpirvOptMaxIterations is eight";
            expect(report.iterations >= 1u);
            expect(report.iterations <= report.max_iterations);
        }
        {
            ScopedEnvironmentVariable cap{
                "LUISA_SPIRV_OPT_MAX_ITERATIONS", "3"};
            auto words = assemble_test_module();
            auto report = lc::spirv::optimize_spirv(
                words, {.level = 2, .preset = "compute"});
            expect_optimizer_report_ok(report);
            expect(report.max_iterations == 3u)
                << "LUISA_SPIRV_OPT_MAX_ITERATIONS must bound the fixed point";
            expect(report.iterations >= 1u);
            expect(report.iterations <= 3u);
        }
        {
            ScopedEnvironmentVariable cap{
                "LUISA_SPIRV_OPT_MAX_ITERATIONS", "1"};
            auto words = assemble_test_module();
            auto report = lc::spirv::optimize_spirv(
                words, {.level = 2, .preset = "compute"});
            expect_optimizer_report_ok(report);
            expect(report.max_iterations == 1u);
            expect(report.iterations == 1u)
                << "a cap of one must run the pipeline exactly once";
        }
        // Invalid or out-of-range values (the accepted range is 1..10) keep the
        // DXC default instead of silently disabling the cap.
        for (auto text : {"0", "11", "-1", "abc", "3trailing", ""}) {
            ScopedEnvironmentVariable cap{
                "LUISA_SPIRV_OPT_MAX_ITERATIONS", text};
            auto words = assemble_test_module();
            auto report = lc::spirv::optimize_spirv(
                words, {.level = 2, .preset = "compute"});
            expect_optimizer_report_ok(report);
            expect(report.max_iterations == 8u)
                << "an invalid LUISA_SPIRV_OPT_MAX_ITERATIONS value must keep "
                   "the DXC default of eight";
        }
    };

    "spirv_fast_math_post_fma_stage_is_gated_by_preset_and_policy"_test = [] {
        ScopedDefaultOptimizerEnvironment defaults;
        auto fixture = float_storage_module(fma_contraction_body);
        // Strict math: the stage must not even be attempted, and the unfused
        // multiply/add must survive in the final binary.
        for (auto preset : {"compute", "full"}) {
            auto words = assemble_test_module(fixture);
            auto report = lc::spirv::optimize_spirv(
                words, {.level = 2, .preset = preset});
            expect_optimizer_report_ok(report);
            expect(!report.fma_contraction_attempted)
                << "strict math must never run DXC's post-convergence stage";
            expect(!report.fma_contraction_applied);
            expect(count_instructions(words, spv::Op::OpFMul) == 1u)
                << disassemble(words);
            expect(count_instructions(words, spv::Op::OpFAdd) == 1u);
            expect(count_glsl_fma(words) == 0u);
            expect(count_instructions(words, spv::Op::OpFmaKHR) == 0u);
            expect(validates(words));
        }
        // The lightweight preset has no DXC pipeline at all.
        {
            auto words = assemble_test_module(fixture);
            auto report = lc::spirv::optimize_spirv(
                words, {.level = 1, .preset = "lightweight"});
            expect_optimizer_report_ok(report);
            expect(!report.fma_contraction_attempted)
                << "lightweight must never run DXC's post-convergence stage";
            expect(!report.fma_contraction_applied);
            expect(count_instructions(words, spv::Op::OpFMul) == 1u);
            expect(count_instructions(words, spv::Op::OpFAdd) == 1u);
            expect(count_glsl_fma(words) == 0u);
            expect(validates(words));
        }
        // Fast math: the stage runs once after convergence and the final
        // binary really contains a contracted GLSL.std.450 Fma.
        for (auto preset : {"compute", "full"}) {
            auto words = assemble_test_module(fixture);
            auto report = lc::spirv::optimize_spirv(
                words,
                {.level = 2, .preset = preset, .enable_fast_math = true});
            expect_optimizer_report_ok(report);
            expect(report.fma_contraction_attempted)
                << "fast math enables DXC's post-convergence contraction stage";
            expect(report.fma_contraction_applied)
                << "a real contraction must report applied, not merely "
                   "attempted";
            expect(report.changed);
            expect(count_instructions(words, spv::Op::OpFMul) == 0u)
                << disassemble(words);
            expect(count_instructions(words, spv::Op::OpFAdd) == 0u);
            expect(count_glsl_fma(words) == 1u)
                << "use_khr_fma=false must contract to GLSL.std.450 Fma";
            expect(count_instructions(words, spv::Op::OpFmaKHR) == 0u);
            expect(!declares_capability(words, spv::Capability::FMAKHR));
            // The contracted value is the live store operand.
            auto stored = single_store_object_id(words);
            expect(stored.has_value());
            auto definition = stored ? find_definition(words, *stored) : std::nullopt;
            expect(definition.has_value() &&
                   definition->opcode == spv::Op::OpExtInst)
                << "the stored value must be the contracted instruction";
            expect(validates(words))
                << "the contracted module must still validate for Vulkan 1.2";
            // The final disassembly must show the contracted instruction, not
            // merely a successful report.
            auto text = disassemble(words);
            expect(text.find(" Fma ") != std::string::npos)
                << "the final disassembly must contain GLSL.std.450 Fma: "
                << text;
            expect(text.find("OpFMul") == std::string::npos)
                << "the contracted multiply must be gone from the disassembly";
        }
    };

    "spirv_fast_math_emits_khr_fma_when_selected"_test = [] {
        ScopedDefaultOptimizerEnvironment defaults;
        for (auto preset : {"compute", "full"}) {
            auto words = assemble_test_module(
                float_storage_module(fma_contraction_body));
            auto report = lc::spirv::optimize_spirv(
                words,
                {.level = 2,
                 .preset = preset,
                 .enable_fast_math = true,
                 .use_khr_fma = true});
            expect_optimizer_report_ok(report);
            expect(report.fma_contraction_attempted);
            expect(report.fma_contraction_applied)
                << "selecting OpFmaKHR is a real contraction, so the stage must "
                   "report applied";
            expect(count_instructions(words, spv::Op::OpFmaKHR) == 1u)
                << "use_khr_fma=true must select OpFmaKHR instead of "
                   "GLSL.std.450 Fma";
            expect(count_glsl_fma(words) == 0u);
            expect(count_instructions(words, spv::Op::OpFMul) == 0u);
            expect(declares_capability(words, spv::Capability::FMAKHR))
                << "OpFmaKHR requires the FMAKHR capability";
            expect(declares_extension(words, "SPV_KHR_fma"))
                << "OpFmaKHR requires the SPV_KHR_fma extension";
            expect(validates(words))
                << "the SPV_KHR_fma result must validate for Vulkan 1.2";
            auto validation =
                lc::spirv::validate_spirv(words.data(), words.size());
            expect(validation.valid) << validation.diagnostics.c_str();
            constexpr auto fma_requirement =
                lc::spirv::target_feature::shader_fma_float32;
            expect(lc::spirv::reconcile_spirv_target_features(
                       words.data(), words.size(), 0u) == fma_requirement)
                << "the emitted capability must be observable as the float32 "
                   "FMA target requirement";
            auto stored = single_store_object_id(words);
            expect(stored.has_value());
            auto definition = stored ? find_definition(words, *stored) :
                                       std::nullopt;
            expect(definition.has_value() &&
                   definition->opcode == spv::Op::OpFmaKHR);
            auto text = disassemble(words);
            expect(text.find("OpFmaKHR") != std::string::npos)
                << "the final disassembly must contain OpFmaKHR: " << text;
            expect(text.find("OpCapability FMAKHR") != std::string::npos);
            expect(text.find("OpExtension \"SPV_KHR_fma\"") !=
                   std::string::npos);
        }
    };

    "spirv_fast_math_respects_nocontraction"_test = [] {
        ScopedDefaultOptimizerEnvironment defaults;
        // A NoContraction decoration is a per-instruction veto that stays
        // effective even when the numeric policy enables fast math, on either
        // the multiply or the add. The stage is attempted (the policy asks for
        // it) but must report that it applied nothing: a successful no-op has
        // to be distinguishable from a real contraction.
        for (auto decoration : {"OpDecorate %sum NoContraction",
                                "OpDecorate %product NoContraction"}) {
            auto words = assemble_test_module(
                float_storage_module(fma_contraction_body, decoration));
            auto report = lc::spirv::optimize_spirv(
                words,
                {.level = 2, .preset = "compute", .enable_fast_math = true});
            expect_optimizer_report_ok(report);
            expect(report.fma_contraction_attempted)
                << "fast math must still attempt DXC's post-convergence stage";
            expect(!report.fma_contraction_applied)
                << "a vetoed contraction is a successful no-op, not an applied "
                   "transformation";
            expect(count_glsl_fma(words) == 0u)
                << "NoContraction must veto contraction in fast-math mode: "
                << disassemble(words);
            expect(count_instructions(words, spv::Op::OpFmaKHR) == 0u);
            expect(count_instructions(words, spv::Op::OpFMul) == 1u);
            expect(count_instructions(words, spv::Op::OpFAdd) == 1u);
            expect(validates(words));
        }
    };

    "spirv_float_div_opt_is_fast_math_only"_test = [] {
        ScopedDefaultOptimizerEnvironment defaults;
        // Strict math must leave the nested division intact: two live OpFDiv
        // instructions and no reassociation product.
        auto strict_words = assemble_test_module(
            float_storage_module(nested_division_body));
        auto strict_report = lc::spirv::optimize_spirv(
            strict_words, {.level = 2, .preset = "compute"});
        expect_optimizer_report_ok(strict_report);
        expect(count_instructions(strict_words, spv::Op::OpFDiv) == 2u)
            << disassemble(strict_words);
        expect(count_instructions(strict_words, spv::Op::OpFMul) == 0u)
            << "strict math must not reassociate floating-point division";
        expect(validates(strict_words));

        // Fast math rewrites (a / b) / c into a / (b * c) and stores the
        // rewritten quotient, so the transformation is live rather than dead.
        auto fast_words = assemble_test_module(
            float_storage_module(nested_division_body));
        auto fast_report = lc::spirv::optimize_spirv(
            fast_words,
            {.level = 2, .preset = "compute", .enable_fast_math = true});
        expect_optimizer_report_ok(fast_report);
        expect(count_instructions(fast_words, spv::Op::OpFDiv) == 1u)
            << disassemble(fast_words);
        expect(count_instructions(fast_words, spv::Op::OpFMul) == 1u);
        auto stored = single_store_object_id(fast_words);
        expect(stored.has_value());
        auto definition = stored ? find_definition(fast_words, *stored) : std::nullopt;
        expect(definition.has_value() &&
               definition->opcode == spv::Op::OpFDiv)
            << "the surviving quotient must be the live stored value";
        auto denominator = definition ? find_definition(
                                            fast_words,
                                            fdiv_denominator(fast_words, *definition)) :
                                        std::nullopt;
        expect(denominator.has_value() &&
               denominator->opcode == spv::Op::OpFMul)
            << "the rewrite must divide by the product of both denominators";
        expect(validates(fast_words));
    };

    "spirv_float_div_opt_factors_common_denominator"_test = [] {
        ScopedDefaultOptimizerEnvironment defaults;
        // Two quotients sharing one non-constant denominator stay untouched
        // under strict math and are factored into one reciprocal under fast
        // math (FloatDivOptPass common-denominator pattern).
        auto strict_words = assemble_test_module(
            float_storage_module(common_denominator_division_body));
        auto strict_report = lc::spirv::optimize_spirv(
            strict_words, {.level = 2, .preset = "compute"});
        expect_optimizer_report_ok(strict_report);
        expect(count_instructions(strict_words, spv::Op::OpFDiv) == 2u)
            << disassemble(strict_words);
        expect(count_instructions(strict_words, spv::Op::OpFMul) == 0u);
        expect(validates(strict_words));

        auto fast_words = assemble_test_module(
            float_storage_module(common_denominator_division_body));
        auto fast_report = lc::spirv::optimize_spirv(
            fast_words,
            {.level = 2, .preset = "compute", .enable_fast_math = true});
        expect_optimizer_report_ok(fast_report);
        expect(count_instructions(fast_words, spv::Op::OpFDiv) == 1u)
            << disassemble(fast_words);
        expect(count_instructions(fast_words, spv::Op::OpFMul) == 2u)
            << "both quotients must be rewritten as products with the "
               "reciprocal";
        expect(validates(fast_words));
    };

    "spirv_strength_reduction_rewrites_live_power_of_two_multiply"_test = [] {
        ScopedDefaultOptimizerEnvironment defaults;
        auto fixture = uint_storage_module(strength_reduction_body);
        // The lightweight preset has no strength reduction.
        auto minimal_words = assemble_test_module(fixture);
        auto minimal_report = lc::spirv::optimize_spirv(
            minimal_words, {.level = 1, .preset = "lightweight"});
        expect_optimizer_report_ok(minimal_report);
        expect(count_instructions(minimal_words, spv::Op::OpIMul) == 1u);
        expect(count_instructions(
                   minimal_words, spv::Op::OpShiftLeftLogical) == 0u);

        // The DXC tail rewrites the multiply by a power of two into a shift,
        // and the rewritten value is what the module stores.
        auto words = assemble_test_module(fixture);
        auto report = lc::spirv::optimize_spirv(
            words, {.level = 2, .preset = "compute"});
        expect_optimizer_report_ok(report);
        expect(count_instructions(words, spv::Op::OpIMul) == 0u)
            << disassemble(words);
        expect(count_instructions(words, spv::Op::OpShiftLeftLogical) == 1u);
        auto stored = single_store_object_id(words);
        expect(stored.has_value());
        auto definition = stored ? find_definition(words, *stored) : std::nullopt;
        expect(definition.has_value() &&
               definition->opcode == spv::Op::OpShiftLeftLogical)
            << "the strength-reduced value must be the live stored value";
        expect(validates(words));
    };

    "spirv_boolean_simplify_canonicalizes_bool_select"_test = [] {
        ScopedDefaultOptimizerEnvironment defaults;
        auto fixture = uint_storage_module(boolean_simplify_body);
        expect(count_instructions(assemble_test_module(fixture),
                                  spv::Op::OpSelect) == 2u)
            << "the fixture must contain both selects before optimization";
        // The lightweight preset keeps the redundant bool select.
        auto minimal_words = assemble_test_module(fixture);
        auto minimal_report = lc::spirv::optimize_spirv(
            minimal_words, {.level = 1, .preset = "lightweight"});
        expect_optimizer_report_ok(minimal_report);
        expect(count_instructions(minimal_words, spv::Op::OpSelect) == 2u)
            << disassemble(minimal_words);
        expect(count_instructions(minimal_words, spv::Op::OpLogicalAnd) == 0u);

        // BooleanSimplifyPass rewrites select(cond, x, false) into
        // OpLogicalAnd, leaving only the live uint select.
        auto words = assemble_test_module(fixture);
        auto report = lc::spirv::optimize_spirv(
            words, {.level = 2, .preset = "compute"});
        expect_optimizer_report_ok(report);
        expect(count_instructions(words, spv::Op::OpSelect) == 1u)
            << disassemble(words);
        expect(count_instructions(words, spv::Op::OpLogicalAnd) == 1u)
            << "the bool select must become a logical-and";
        auto select = find_instruction(words, spv::Op::OpSelect);
        expect(select.has_value());
        auto result_type = select ? words[select->offset + 1u] : 0u;
        expect(select && type_declaration_opcode(words, result_type) ==
                             spv::Op::OpTypeInt)
            << "the surviving select must be the uint select, not a bool select";
        auto condition = select ? words[select->offset + 3u] : 0u;
        auto condition_definition = select ? find_definition(words, condition) : std::nullopt;
        expect(condition_definition.has_value() &&
               condition_definition->opcode == spv::Op::OpLogicalAnd)
            << "the uint select must be driven by the canonicalized condition";
        auto stored = single_store_object_id(words);
        expect(stored.has_value());
        auto definition = stored ? find_definition(words, *stored) : std::nullopt;
        expect(definition.has_value() &&
               definition->opcode == spv::Op::OpSelect)
            << "the surviving select must be the live stored value";
        expect(validates(words));
    };

    "spirv_dxc_pipeline_auto_unrolls_counted_loop"_test = [] {
        ScopedDefaultOptimizerEnvironment defaults;
        // A small LoopControl None counted loop is now unrolled by DXC's bounded
        // automatic unroll pass instead of being left to the runtime.
        auto fixture = make_accumulator_loop_module("None");
        auto minimal_words = assemble_test_module(fixture);
        auto minimal_report = lc::spirv::optimize_spirv(
            minimal_words, {.level = 1, .preset = "lightweight"});
        expect_optimizer_report_ok(minimal_report);
        expect(count_instructions(minimal_words, spv::Op::OpLoopMerge) == 1u)
            << "the minimal preset must not unroll any loop";

        auto words = assemble_test_module(fixture);
        auto report = lc::spirv::optimize_spirv(
            words, {.level = 2, .preset = "compute"});
        expect_optimizer_report_ok(report);
        expect(count_instructions(words, spv::Op::OpLoopMerge) == 0u)
            << "the counted loop must be fully unrolled: "
            << disassemble(words);
        expect(count_instructions(words, spv::Op::OpBranchConditional) == 0u)
            << "a fully unrolled counted loop must leave no conditional "
               "back-edge behind";
        auto stored = single_store_object_id(words);
        expect(stored.has_value())
            << "the unrolled accumulator must still reach the output store";
        auto stored_definition = stored ? find_definition(words, *stored) : std::nullopt;
        expect(stored_definition.has_value() &&
               stored_definition->opcode == spv::Op::OpConstant &&
               words[stored_definition->offset + 3u] == 6u)
            << "the four-iteration accumulator 0+1+2+3 must be folded to the "
               "constant 6 and stored: "
            << disassemble(words);
        expect(validates(words));

        // The same loop marked Unroll keeps its established fully-unrolled
        // behavior through the stock explicit-Unroll pass.
        auto explicit_words = assemble_test_module(
            make_accumulator_loop_module("Unroll"));
        auto explicit_report = lc::spirv::optimize_spirv(
            explicit_words, {.level = 2, .preset = "compute"});
        expect_optimizer_report_ok(explicit_report);
        expect(count_instructions(explicit_words, spv::Op::OpLoopMerge) == 0u)
            << "an explicit Unroll control must stay fully unrolled";
        expect(single_store_object_id(explicit_words).has_value());
        expect(validates(explicit_words));
    };

    "spirv_capability_reconciliation_expands_implicit_subgroup_parent"_test = [] {
        // This fixture starts at the already-emitted SPIR-V boundary. It does
        // not claim that every feature-dependent lowering can emit when its
        // target feature is disabled; it verifies that a child capability also
        // recovers its implicit Vulkan subgroup-basic requirement.
        auto words = assemble_test_module(dead_subgroup_arithmetic_module);
        constexpr auto capability_requirements =
            lc::spirv::target_feature::subgroup_basic |
            lc::spirv::target_feature::subgroup_arithmetic;
        constexpr auto runtime_requirement =
            lc::spirv::target_feature::sampler_anisotropy;
        constexpr auto emitted_requirements =
            capability_requirements | runtime_requirement;
        expect(lc::spirv::spirv_target_feature_is_capability_owned(
            capability_requirements));
        expect(!lc::spirv::spirv_target_feature_is_capability_owned(
            runtime_requirement));
        expect(eq(
            lc::spirv::reconcile_spirv_target_features(
                words.data(), words.size(), emitted_requirements),
            emitted_requirements))
            << "an implicitly required subgroup parent capability must retain "
               "the complete Vulkan operation requirement";
    };

    "spirv_optimizer_reconciles_optimized_away_capabilities"_test = [] {
        // SPIRV-Tools deliberately does not classify group collectives as
        // safe-to-delete just because their SSA result is unused. Use ordinary
        // floating-point arithmetic here so the test really proves the DCE ->
        // capability-trim -> feature-reconciliation handoff.
        auto words = assemble_test_module(dead_float64_arithmetic_module);
        constexpr auto capability_requirement =
            lc::spirv::target_feature::shader_float64;
        constexpr auto runtime_requirement =
            lc::spirv::target_feature::sampler_anisotropy;
        constexpr auto emitted_requirements =
            capability_requirement | runtime_requirement;
        expect(lc::spirv::spirv_target_feature_is_capability_owned(
            capability_requirement));
        expect(!lc::spirv::spirv_target_feature_is_capability_owned(
            runtime_requirement));
        expect(eq(
            lc::spirv::reconcile_spirv_target_features(
                words.data(), words.size(), emitted_requirements),
            emitted_requirements));

        auto report = lc::spirv::optimize_spirv(
            words, {.level = 1, .preset = "lightweight"});
        expect(report.attempted);
        expect(report.succeeded);
        expect(report.capability_trim_registered);
        expect(report.output_validated);
        expect(validates(words))
            << "capability-trimmed SPIR-V must validate for Vulkan 1.2";
        auto text = disassemble(words);
        expect(text.find("OpCapability Shader") != std::string::npos)
            << "the mandatory compute capability must remain";
        expect(text.find("OpFAdd") == std::string::npos)
            << "the deliberately dead scalar arithmetic must be eliminated";
        expect(text.find("OpCapability Float64") == std::string::npos)
            << "the final artifact must not retain a dead float capability";
        expect(eq(
            lc::spirv::reconcile_spirv_target_features(
                words.data(), words.size(), emitted_requirements),
            runtime_requirement))
            << "dead capability requirements must be removed without "
               "guessing away emission-owned runtime requirements";
    };

    "spirv_optimizer_preserves_storage_only_uniform_16_capability"_test = [] {
        auto words = assemble_test_module(storage_only_uniform_16_module);
        auto report = lc::spirv::optimize_spirv(
            words, {.level = 1, .preset = "lightweight"});
        expect(report.attempted);
        expect(report.succeeded);
        expect(!report.capability_trim_registered)
            << "the vendored trim pass cannot prove storage-only Uniform "
               "16-bit capability liveness";
        expect(report.output_validated);
        expect(validates(words));
        auto text = disassemble(words);
        expect(text.find(
                   "OpCapability UniformAndStorageBuffer16BitAccess") !=
               std::string::npos)
            << "a live storage-only capability must not be trimmed";
        expect(text.find("OpCapability Int16") == std::string::npos)
            << "storage-only 16-bit access must not acquire arithmetic support";
    };

    "spirv_native_artifact_detects_capability_mask_tampering"_test = [] {
        auto words = assemble_test_module(storage_only_uniform_16_module);
        constexpr auto capability_requirement =
            lc::spirv::target_feature::uniform_storage_buffer_16bit_access;
        constexpr auto emission_owned_requirement =
            lc::spirv::target_feature::sampler_anisotropy;
        constexpr auto persisted =
            capability_requirement | emission_owned_requirement;

        expect(eq(
            lc::spirv::reconcile_spirv_target_features(
                words.data(), words.size(), persisted),
            persisted));

        constexpr auto omitted_capability = emission_owned_requirement;
        expect(neq(
            lc::spirv::reconcile_spirv_target_features(
                words.data(), words.size(), omitted_capability),
            omitted_capability))
            << "clearing a capability-owned bit in a persisted native mask "
               "must be detectable from the validated binary";

        constexpr auto fabricated_capability =
            persisted | lc::spirv::target_feature::shader_int64;
        expect(neq(
            lc::spirv::reconcile_spirv_target_features(
                words.data(), words.size(), fabricated_capability),
            fabricated_capability))
            << "injecting a capability-owned bit must also be rejected";

        constexpr auto changed_emission_owned = capability_requirement;
        expect(eq(
            lc::spirv::reconcile_spirv_target_features(
                words.data(), words.size(), changed_emission_owned),
            changed_emission_owned))
            << "requirements with no authoritative capability mapping remain "
               "owned by the serialized emission contract";
    };

    "spirv_native_multistage_artifact_unions_capability_requirements"_test = [] {
        auto subgroup = assemble_test_module(
            dead_subgroup_arithmetic_module);
        auto uniform_16 = assemble_test_module(
            storage_only_uniform_16_module);
        std::array<luisa::span<const uint32_t>, 2u> modules{
            luisa::span<const uint32_t>{subgroup},
            luisa::span<const uint32_t>{uniform_16}};
        constexpr auto emission_owned =
            lc::spirv::target_feature::sampler_anisotropy;
        constexpr auto provisional =
            emission_owned |
            lc::spirv::target_feature::shader_int64;
        constexpr auto expected =
            emission_owned |
            lc::spirv::target_feature::subgroup_basic |
            lc::spirv::target_feature::subgroup_arithmetic |
            lc::spirv::target_feature::uniform_storage_buffer_16bit_access;
        expect(eq(
            lc::spirv::reconcile_spirv_target_features(
                luisa::span<const luisa::span<const uint32_t>>{modules},
                provisional),
            expected))
            << "a multi-stage artifact must retain the union of every module's "
               "capability-owned requirements and the shared emission-owned mask";
    };

    "spirv_optimizer_skips_trim_for_unsupported_mixed_capabilities"_test = [] {
        auto words = assemble_test_module(
            unsupported_mixed_capability_module);
        auto report = lc::spirv::optimize_spirv(
            words, {.level = 1, .preset = "lightweight"});
        expect(report.attempted);
        expect(report.succeeded);
        expect(!report.capability_trim_registered)
            << "an unsupported capability must conservatively disable trimming";
        expect(report.output_validated);
        expect(validates(words));
        auto text = disassemble(words);
        expect(text.find("OpIAdd") == std::string::npos)
            << "ordinary DCE should remain enabled when capability trim is skipped";
        expect(text.find("OpGroupNonUniformIAdd") != std::string::npos)
            << "an unused subgroup collective is not classified as safe-to-delete";
        expect(text.find("OpCapability GroupNonUniformArithmetic") !=
               std::string::npos);
        expect(text.find("OpCapability GroupNonUniformShuffle") !=
               std::string::npos);
    };

    "spirv_optimizer_unknown_preset_is_explicit"_test = [] {
        auto words = assemble_test_module();
        auto report = lc::spirv::optimize_spirv(
            words, {.level = 2, .preset = "definitely-not-a-preset"});
        expect(report.attempted);
        expect(report.succeeded);
        expect(report.effective_preset == "compute");
        expect(!report.diagnostics.empty())
            << "falling back from an unknown preset must be observable";
        expect(validates(words));
    };

    "spirv_optimizer_environment_is_not_cached"_test = [] {
        ScopedEnvironmentVariable clear_preset{"LUISA_SPIRV_OPT_PASSES", nullptr};
        {
            ScopedEnvironmentVariable level{"LUISA_SPIRV_OPT_LEVEL", "0"};
            auto options = lc::spirv::spirv_optimizer_options_from_environment();
            expect(options.level == 0);
            expect(options.preset.empty());
        }
        {
            ScopedEnvironmentVariable level{"LUISA_SPIRV_OPT_LEVEL", "1"};
            ScopedEnvironmentVariable preset{"LUISA_SPIRV_OPT_PASSES", "lightweight"};
            auto options = lc::spirv::spirv_optimizer_options_from_environment();
            expect(options.level == 1);
            expect(options.preset == "lightweight");
        }
    };

    "spirv_optimizer_environment_level_parse_is_bounded"_test = [] {
        ScopedEnvironmentVariable clear_preset{
            "LUISA_SPIRV_OPT_PASSES", nullptr};
        auto above_int_max = std::to_string(
            std::numeric_limits<int>::max());
        above_int_max.push_back('0');
        auto below_int_min = std::to_string(
            std::numeric_limits<int>::lowest());
        below_int_min.push_back('0');
        std::vector<std::string> rejected{
            "999999999999999999999999999999999999999999999999",
            "-999999999999999999999999999999999999999999999999",
            std::move(above_int_max), std::move(below_int_min),
            "1trailing"};
        for (auto &&text : rejected) {
            ScopedEnvironmentVariable level{
                "LUISA_SPIRV_OPT_LEVEL", text.c_str()};
            auto options =
                lc::spirv::spirv_optimizer_options_from_environment();
            expect(options.level == 2)
                << "an invalid or out-of-int-range optimization level must preserve the default";
        }

        auto lowest = std::to_string(
            std::numeric_limits<int>::lowest());
        {
            ScopedEnvironmentVariable level{
                "LUISA_SPIRV_OPT_LEVEL", lowest.c_str()};
            auto options =
                lc::spirv::spirv_optimizer_options_from_environment();
            expect(options.level ==
                   std::numeric_limits<int>::lowest());
        }
        auto highest = std::to_string(
            std::numeric_limits<int>::max());
        {
            ScopedEnvironmentVariable level{
                "LUISA_SPIRV_OPT_LEVEL", highest.c_str()};
            auto options =
                lc::spirv::spirv_optimizer_options_from_environment();
            expect(options.level ==
                   std::numeric_limits<int>::max());
        }
    };
}
