#include "ut/ut.hpp"

#include <array>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <spirv-tools/libspirv.hpp>

#include "spirv_codegen/optimizer.h"

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

constexpr auto counted_loop_module = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%function = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%four = OpConstant %uint 4
%main = OpFunction %void None %function
%entry = OpLabel
OpBranch %header
%header = OpLabel
%index = OpPhi %uint %zero %entry %next %continue
%condition = OpULessThan %bool %index %four
OpLoopMerge %merge %continue $loop_control
OpBranchConditional %condition %body %merge
%body = OpLabel
OpBranch %continue
%continue = OpLabel
%next = OpIAdd %uint %index %one
OpBranch %header
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

[[nodiscard]] std::string make_counted_loop_module(
    std::string_view loop_control) {
    auto module = std::string{counted_loop_module};
    constexpr auto placeholder = std::string_view{"$loop_control"};
    auto offset = module.find(placeholder);
    expect(offset != std::string::npos);
    module.replace(offset, placeholder.size(), loop_control);
    return module;
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
    std::optional<std::string> _previous;

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

[[nodiscard]] std::vector<uint32_t> assemble_test_module(
    std::string_view assembly = dead_arithmetic_module) {
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

    "spirv_compute_optimizer_only_analyzes_requested_unrolls"_test = [] {
        auto ordinary = assemble_test_module(
            make_counted_loop_module("None"));
        auto ordinary_report = lc::spirv::optimize_spirv(
            ordinary, {.level = 2, .preset = "compute"});
        expect(ordinary_report.succeeded);
        expect(ordinary_report.output_validated);
        expect(!ordinary_report.loop_unroll_registered)
            << "a LoopControl None depth/runtime loop must not trigger the "
               "whole-module unroll analysis";
        expect(validates(ordinary));

        auto requested = assemble_test_module(
            make_counted_loop_module("Unroll"));
        auto requested_report = lc::spirv::optimize_spirv(
            requested, {.level = 2, .preset = "compute"});
        expect(requested_report.succeeded);
        expect(requested_report.output_validated);
        expect(requested_report.loop_unroll_registered)
            << "an explicit Unroll control must preserve the existing "
               "SPIRV-Tools behavior";
        expect(validates(requested));
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
