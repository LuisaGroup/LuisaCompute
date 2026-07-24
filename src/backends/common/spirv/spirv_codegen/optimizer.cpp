#include "optimizer.h"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <utility>

#include <spirv-tools/optimizer.hpp>
#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp11>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>

namespace lc::spirv {

namespace {

[[nodiscard]] luisa::string effective_preset(
    const SpirvOptimizerOptions &options) noexcept {
    if (!options.preset.empty()) { return options.preset; }
    if (options.level <= 0) { return "none"; }
    if (options.level == 1) { return "lightweight"; }
    if (options.level == 2) { return "compute"; }
    if (options.level == 3) { return "dxc"; }
    return "full";
}

void register_compute_passes(spvtools::Optimizer &optimizer) {
    optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
    optimizer.RegisterPass(spvtools::CreateBlockMergePass());
    optimizer.RegisterPass(spvtools::CreateSimplificationPass());
    optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
    optimizer.RegisterPass(spvtools::CreateLocalSingleStoreElimPass());
    optimizer.RegisterPass(spvtools::CreateLocalMultiStoreElimPass());
    optimizer.RegisterPass(spvtools::CreateRedundancyEliminationPass());
    optimizer.RegisterPass(spvtools::CreateLoopUnrollPass(true));
    optimizer.RegisterPass(spvtools::CreateCCPPass());
    optimizer.RegisterPass(spvtools::CreateScalarReplacementPass(100));
    optimizer.RegisterPass(spvtools::CreateIfConversionPass());
    optimizer.RegisterPass(spvtools::CreatePrivateToLocalPass());
    optimizer.RegisterPass(spvtools::CreateCopyPropagateArraysPass());
}

void register_dxc_passes(spvtools::Optimizer &optimizer) {
    // Performance passes (same as RegisterPerformancePasses)
    optimizer.RegisterPerformancePasses();
    // Extra passes from DXC
    optimizer.RegisterPass(spvtools::CreateLoopRerollPass());
    optimizer.RegisterPass(spvtools::CreateGVNHoistPass());
    optimizer.RegisterPass(spvtools::CreateConstraintEliminationPass());
    optimizer.RegisterPass(spvtools::CreateInferAlignmentPass());
    optimizer.RegisterPass(spvtools::CreateCalledValuePropagationPass());
    optimizer.RegisterPass(spvtools::CreateCallSiteSplittingPass());
    optimizer.RegisterPass(spvtools::CreateInferAddressSpacesPass());
    optimizer.RegisterPass(spvtools::CreateFunctionSpecializationPass());
    optimizer.RegisterPass(spvtools::CreateAttributorPass());
    optimizer.RegisterPass(spvtools::CreateLoadStoreMotionPass());
    optimizer.RegisterPass(spvtools::CreateLoadCombinePass());
    optimizer.RegisterPass(spvtools::CreateLoopUnswitchPass());
    optimizer.RegisterPass(spvtools::CreateSpreadVolatileSemanticsPass());
    optimizer.RegisterPass(spvtools::CreateCompactIdsPass());
}

// SPIRV-Tools' trim-capabilities pass is intentionally incomplete: its own
// contract warns that modules containing unsupported capabilities may produce
// incorrect results. Keep this allowlist narrower than the SPIR-V grammar and
// update it only after auditing the exact pass revision vendored here.
//
// UniformAndStorageBuffer16BitAccess needs one additional guard. The vendored
// pass only recognizes that capability on an OpTypePointer when Float16 or
// Int16 is also declared. SPIR-V permits storage-only 16-bit Uniform blocks
// without either arithmetic capability, so trimming such a module would
// remove a live capability and invalidate the result.
[[nodiscard]] bool can_safely_trim_capabilities(
    const uint32_t *words, size_t word_count) noexcept {
    constexpr auto header_word_count = 5u;
    if (words == nullptr || word_count < header_word_count) { return false; }
    auto has_uniform_storage_16 = false;
    auto has_16bit_arithmetic = false;
    for (size_t offset = header_word_count; offset < word_count;) {
        auto instruction = words[offset];
        auto instruction_word_count =
            static_cast<size_t>(instruction >> 16u);
        auto opcode = static_cast<spv::Op>(instruction & 0xffffu);
        if (instruction_word_count == 0u ||
            instruction_word_count > word_count - offset) {
            return false;
        }
        if (opcode == spv::Op::OpCapability) {
            if (instruction_word_count != 2u) { return false; }
            auto capability =
                static_cast<spv::Capability>(words[offset + 1u]);
            switch (capability) {
                case spv::Capability::Shader:
                case spv::Capability::Float64:
                case spv::Capability::Int64:
                case spv::Capability::MinLod:
                case spv::Capability::GroupNonUniform:
                case spv::Capability::GroupNonUniformVote:
                case spv::Capability::GroupNonUniformArithmetic:
                case spv::Capability::StorageBuffer16BitAccess:
                case spv::Capability::StorageImageReadWithoutFormat:
                case spv::Capability::StorageImageWriteWithoutFormat:
                case spv::Capability::RayQueryKHR:
                case spv::Capability::ShaderClockKHR:
                case spv::Capability::GroupNonUniformBallot:
                case spv::Capability::GroupNonUniformClustered:
                case spv::Capability::GroupNonUniformShuffle:
                case spv::Capability::StorageBufferArrayDynamicIndexing:
                case spv::Capability::StorageBufferArrayNonUniformIndexing:
                case spv::Capability::SampledImageArrayDynamicIndexing:
                case spv::Capability::SampledImageArrayNonUniformIndexing:
                case spv::Capability::RuntimeDescriptorArray:
                case spv::Capability::ImageQuery:
                case spv::Capability::ImageMSArray:
                    break;
                case spv::Capability::Float16:
                case spv::Capability::Int16:
                    has_16bit_arithmetic = true;
                    break;
                case spv::Capability::UniformAndStorageBuffer16BitAccess:
                    has_uniform_storage_16 = true;
                    break;
                default:
                    // Capabilities not explicitly enumerated above are not
                    // guaranteed safe for the vendored trim pass. Retain them
                    // for conservative presets (lightweight/compute).
                    return false;
            }
        }
        offset += instruction_word_count;
    }
    return !has_uniform_storage_16 || has_16bit_arithmetic;
}

// These requirements are owned entirely by OpCapability: if optimization
// removes the declaration, reconciliation may remove the provisional bit.
// A runtime-owned requirement may still be implied by a present capability
// in target_feature_from_capability() below. That lets artifact validation
// detect an omitted bit without treating capability absence as proof that a
// semantic or descriptor-layout requirement disappeared.
constexpr auto capability_owned_target_features =
    target_feature::sampled_image_array_dynamic_indexing |
    target_feature::sampled_image_array_non_uniform_indexing |
    target_feature::shader_resource_min_lod |
    target_feature::subgroup_basic |
    target_feature::subgroup_vote |
    target_feature::subgroup_arithmetic |
    target_feature::subgroup_ballot |
    target_feature::subgroup_shuffle |
    target_feature::storage_image_read_without_format |
    target_feature::storage_image_write_without_format |
    target_feature::shader_float8 |
    target_feature::shader_float16 |
    target_feature::shader_float64 |
    target_feature::shader_int8 |
    target_feature::shader_int16 |
    target_feature::shader_int64 |
    target_feature::storage_buffer_8bit_access |
    target_feature::uniform_storage_buffer_8bit_access |
    target_feature::storage_buffer_16bit_access |
    target_feature::uniform_storage_buffer_16bit_access |
    target_feature::storage_buffer_array_non_uniform_indexing |
    target_feature::storage_buffer_array_dynamic_indexing |
    target_feature::shader_device_clock;

[[nodiscard]] constexpr SpirvTargetFeatureMask
target_feature_from_capability(spv::Capability capability) noexcept {
    switch (capability) {
        case spv::Capability::SampledImageArrayDynamicIndexing:
            return target_feature::sampled_image_array_dynamic_indexing;
        case spv::Capability::SampledImageArrayNonUniformIndexing:
            return target_feature::sampled_image_array_non_uniform_indexing;
        case spv::Capability::MinLod:
            return target_feature::shader_resource_min_lod;
        case spv::Capability::GroupNonUniform:
            return target_feature::subgroup_basic;
        case spv::Capability::GroupNonUniformVote:
            // GroupNonUniform is an implicit capability dependency. The
            // trim-capabilities pass may therefore remove its redundant
            // explicit declaration while retaining this child capability,
            // but Vulkan still needs both subgroup operation classes.
            return target_feature::subgroup_basic |
                   target_feature::subgroup_vote;
        case spv::Capability::GroupNonUniformArithmetic:
            return target_feature::subgroup_basic |
                   target_feature::subgroup_arithmetic;
        case spv::Capability::GroupNonUniformBallot:
            return target_feature::subgroup_basic |
                   target_feature::subgroup_ballot;
        case spv::Capability::GroupNonUniformShuffle:
            return target_feature::subgroup_basic |
                   target_feature::subgroup_shuffle;
        case spv::Capability::StorageImageReadWithoutFormat:
            return target_feature::storage_image_read_without_format;
        case spv::Capability::StorageImageWriteWithoutFormat:
            return target_feature::storage_image_write_without_format;
        case spv::Capability::Float8EXT:
            return target_feature::shader_float8;
        case spv::Capability::Float16:
            return target_feature::shader_float16;
        case spv::Capability::Float64:
            return target_feature::shader_float64;
        case spv::Capability::Int8:
            return target_feature::shader_int8;
        case spv::Capability::Int16:
            return target_feature::shader_int16;
        case spv::Capability::Int64:
            return target_feature::shader_int64;
        case spv::Capability::StorageBuffer8BitAccess:
            return target_feature::storage_buffer_8bit_access;
        case spv::Capability::UniformAndStorageBuffer8BitAccess:
            return target_feature::uniform_storage_buffer_8bit_access;
        case spv::Capability::StorageBuffer16BitAccess:
            return target_feature::storage_buffer_16bit_access;
        case spv::Capability::UniformAndStorageBuffer16BitAccess:
            return target_feature::uniform_storage_buffer_16bit_access;
        case spv::Capability::StorageBufferArrayNonUniformIndexing:
            return target_feature::storage_buffer_array_non_uniform_indexing;
        case spv::Capability::StorageBufferArrayDynamicIndexing:
            return target_feature::storage_buffer_array_dynamic_indexing;
        case spv::Capability::RuntimeDescriptorArray:
            return target_feature::runtime_descriptor_array;
        case spv::Capability::RayQueryKHR:
            return target_feature::ray_query;
        case spv::Capability::ShaderClockKHR:
            return target_feature::shader_device_clock;
        default: return 0u;
    }
}

}// namespace

SpirvValidationReport validate_spirv(
    const uint32_t *words, size_t word_count) {
    SpirvValidationReport report;
    if (words == nullptr || word_count == 0u) {
        report.diagnostics = "SPIR-V module is empty.\n";
        return report;
    }
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    tools.SetMessageConsumer(
        [&report](spv_message_level_t level, const char *source,
                  const spv_position_t &position, const char *message) {
            report.has_warning |= level <= SPV_MSG_WARNING;
            report.diagnostics.append(luisa::format(
                "{} [{}:{}:{}]: {}\n", static_cast<int>(level),
                source == nullptr ? "" : source, position.line,
                position.column, message == nullptr ? "" : message));
        });
    spvtools::ValidatorOptions options;
    report.valid = tools.Validate(words, word_count, options);
    return report;
}

SpirvTransformCommitReport validate_and_commit_spirv_transform(
    std::vector<uint32_t> &artifact,
    std::vector<uint32_t> candidate) {
    SpirvTransformCommitReport report{
        .input_word_count = artifact.size(),
        .candidate_word_count = candidate.size(),
        .output_word_count = artifact.size()};
    auto validation = validate_spirv(
        candidate.data(), candidate.size());
    report.output_validated = validation.valid;
    report.diagnostics = std::move(validation.diagnostics);
    if (!validation.valid) { return report; }
    report.succeeded = true;
    report.changed = candidate != artifact;
    if (report.changed) { artifact = std::move(candidate); }
    report.output_word_count = artifact.size();
    return report;
}

SpirvOptimizerOptions spirv_optimizer_options_from_environment() noexcept {
    SpirvOptimizerOptions options;
    if (auto *env = std::getenv("LUISA_SPIRV_OPT_LEVEL")) {
        char *end = nullptr;
        errno = 0;
        auto value = std::strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' &&
            value >= std::numeric_limits<int>::lowest() &&
            value <= std::numeric_limits<int>::max()) {
            options.level = static_cast<int>(value);
        }
    }
    if (auto *env = std::getenv("LUISA_SPIRV_OPT_PASSES")) {
        options.preset = env;
    }
    if (auto *env = std::getenv("LUISA_SPIRV_OPT_MAX_ITERATIONS")) {
        char *end = nullptr;
        errno = 0;
        auto value = std::strtoul(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' &&
            value <= std::numeric_limits<unsigned>::max()) {
            options.max_iterations = static_cast<unsigned>(value);
        }
    }
    return options;
}

SpirvOptimizerReport optimize_spirv(
    std::vector<uint32_t> &words, SpirvOptimizerOptions options) {
    SpirvOptimizerReport report{
        .requested_level = options.level,
        .requested_preset = options.preset,
        .effective_preset = effective_preset(options),
        .input_word_count = words.size(),
        .output_word_count = words.size()};
    if (report.effective_preset == "none") { return report; }

    spvtools::Optimizer optimizer{SPV_ENV_VULKAN_1_2};
    spvtools::OptimizerOptions spv_options;
    spv_options.set_run_validator(false);
    spv_options.set_preserve_bindings(true);
    optimizer.SetMessageConsumer(
        [&report](spv_message_level_t level, const char *source,
                  const spv_position_t &position, const char *message) {
            report.diagnostics.append(luisa::format(
                "{} [{}:{}:{}]: {}\n", static_cast<int>(level),
                source == nullptr ? "" : source, position.line,
                position.column, message == nullptr ? "" : message));
        });

    if (report.effective_preset == "lightweight") {
        optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass());
        optimizer.RegisterPass(spvtools::CreateBlockMergePass());
        optimizer.RegisterPass(spvtools::CreateSimplificationPass());
        optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
    } else if (report.effective_preset == "compute") {
        register_compute_passes(optimizer);
    } else if (report.effective_preset == "dxc") {
        register_dxc_passes(optimizer);
    } else if (report.effective_preset == "full") {
        register_dxc_passes(optimizer);
        optimizer.RegisterPass(spvtools::CreatePrivateToLocalPass());
        optimizer.RegisterPass(spvtools::CreateCopyPropagateArraysPass());
    } else {
        report.diagnostics.append(luisa::format(
            "Unknown SPIR-V optimization preset '{}'; using compute.\n",
            report.effective_preset));
        report.effective_preset = "compute";
        register_compute_passes(optimizer);
    }

    // DCE can remove the last use of an optional capability, but ordinary
    // optimization passes deliberately do not edit module capabilities. Use
    // SPIRV-Tools' grammar-aware proof only inside its audited input domain;
    // outside that domain retaining a conservative requirement is preferable
    // to trusting an incomplete trim. Never strip capabilities with an ad-hoc
    // binary rewrite.
    //
    // For aggressive presets (dxc, full), skip the gate and run the trim pass
    // unconditionally — the post-optimization validation gate catches any
    // incorrect output. For conservative presets, gate on the allowlist.
    auto should_trim_capabilities = false;
    if (report.effective_preset == "dxc" ||
        report.effective_preset == "full") {
        should_trim_capabilities = true;
    } else {
        should_trim_capabilities = can_safely_trim_capabilities(
            words.data(), words.size());
    }
    if (should_trim_capabilities) {
        optimizer.RegisterPass(spvtools::CreateTrimCapabilitiesPass());
        report.capability_trim_registered = true;
    }

    report.attempted = true;
    auto max_iters = options.max_iterations;
    if (max_iters == 0u) { max_iters = 1u; }
    std::vector<uint32_t> optimized;
    report.succeeded = false;
    for (unsigned iter = 0u; iter < max_iters; ++iter) {
        if (!optimizer.Run(words.data(), words.size(),
                           &optimized, spv_options)) {
            report.diagnostics.append(luisa::format(
                "SPIR-V optimizer iteration {} failed.\n", iter + 1u));
            report.succeeded = false;
            break;
        }
        report.succeeded = true;
        if (optimized.size() == words.size()) {
            // Converged — no further shrinking.
            break;
        }
        words.swap(optimized);
        optimized.clear();
    }
    if (!report.succeeded) { return report; }
    // Validate the final optimizer output. If we exited via convergence,
    // |optimized| holds the converged result. If we completed the loop
    // without convergence, the last swap left |words| with the latest
    // result and |optimized| empty; in that case validate |words| directly.
    if (optimized.empty()) {
        // All iterations ran without convergence; |words| holds the latest
        // optimization result. Validate it directly.
        auto validation = validate_spirv(words.data(), words.size());
        report.output_validated = validation.valid;
        if (!validation.valid) {
            report.diagnostics.append(
                "SPIR-V optimizer output failed Vulkan 1.2 validation; "
                "retaining the input binary.\n");
            report.diagnostics.append(validation.diagnostics);
            report.succeeded = false;
            return report;
        }
        report.changed = true;
        report.output_word_count = words.size();
        if (!validation.diagnostics.empty()) {
            report.diagnostics.append(
                "SPIR-V optimizer output validation diagnostics:\n");
            report.diagnostics.append(validation.diagnostics);
        }
    } else {
        auto commit = validate_and_commit_spirv_transform(
            words, std::move(optimized));
        report.succeeded = commit.succeeded;
        report.output_validated = commit.output_validated;
        report.changed = commit.changed;
        report.output_word_count = commit.output_word_count;
        if (!commit.succeeded) {
            report.diagnostics.append(
                "SPIR-V optimizer output failed Vulkan 1.2 validation; "
                "retaining the input binary.\n");
            report.diagnostics.append(commit.diagnostics);
            return report;
        }
        if (!commit.diagnostics.empty()) {
            report.diagnostics.append(
                "SPIR-V optimizer output validation diagnostics:\n");
            report.diagnostics.append(commit.diagnostics);
        }
    }
    return report;
}

bool spirv_target_feature_is_capability_owned(
    SpirvTargetFeatureMask feature) noexcept {
    return feature != 0u &&
           (feature & ~capability_owned_target_features) == 0u;
}

SpirvTargetFeatureMask reconcile_spirv_target_features(
    const uint32_t *words, size_t word_count,
    SpirvTargetFeatureMask emitted_requirements) noexcept {
    constexpr auto header_word_count = 5u;
    LUISA_ASSERT(words != nullptr && word_count >= header_word_count,
                 "Validated SPIR-V module has no complete header.");
    SpirvTargetFeatureMask declared_capability_features{};
    for (size_t offset = header_word_count; offset < word_count;) {
        auto instruction = words[offset];
        auto instruction_word_count =
            static_cast<size_t>(instruction >> 16u);
        auto opcode = static_cast<spv::Op>(instruction & 0xffffu);
        LUISA_ASSERT(
            instruction_word_count != 0u &&
                instruction_word_count <= word_count - offset,
            "Validated SPIR-V module contains a malformed instruction at word {}.",
            offset);
        if (opcode == spv::Op::OpCapability) {
            LUISA_ASSERT(
                instruction_word_count == 2u,
                "Malformed OpCapability at SPIR-V word {}.", offset);
            declared_capability_features |= target_feature_from_capability(
                static_cast<spv::Capability>(words[offset + 1u]));
        }
        offset += instruction_word_count;
    }
    return (emitted_requirements &
            ~capability_owned_target_features) |
           declared_capability_features;
}

SpirvTargetFeatureMask reconcile_spirv_target_features(
    luisa::span<const luisa::span<const uint32_t>> modules,
    SpirvTargetFeatureMask emitted_requirements) noexcept {
    auto reconciled = emitted_requirements &
                      ~capability_owned_target_features;
    for (auto module : modules) {
        reconciled |= reconcile_spirv_target_features(
            module.data(), module.size(), 0u);
    }
    return reconciled;
}

}// namespace lc::spirv
