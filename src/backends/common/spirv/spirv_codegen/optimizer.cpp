#include "optimizer.h"

#include "../../env_flag.h"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <utility>

#include <spirv-tools/optimizer.hpp>
#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp11>

#include <luisa/core/clock.h>
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
    return "full";
}

[[nodiscard]] bool has_unroll_loop_control(
    const uint32_t *words, size_t word_count) noexcept {
    constexpr auto header_word_count = 5u;
    constexpr auto loop_merge_min_word_count = 4u;
    constexpr auto loop_control_word_index = 3u;
    constexpr auto unroll_mask =
        static_cast<uint32_t>(spv::LoopControlMask::Unroll);
    if (words == nullptr || word_count < header_word_count) { return false; }
    for (auto offset = header_word_count; offset < word_count;) {
        auto instruction = words[offset];
        auto instruction_word_count =
            static_cast<size_t>(instruction >> 16u);
        auto opcode = static_cast<spv::Op>(instruction & 0xffffu);
        if (instruction_word_count == 0u ||
            instruction_word_count > word_count - offset) {
            return false;
        }
        if (opcode == spv::Op::OpLoopMerge &&
            instruction_word_count >= loop_merge_min_word_count &&
            (words[offset + loop_control_word_index] & unroll_mask) != 0u) {
            return true;
        }
        offset += instruction_word_count;
    }
    return false;
}

// Records the pass name for report observability and consumes the token.
void register_pass(spvtools::Optimizer &optimizer,
                   SpirvOptimizerReport &report,
                   luisa::string name,
                   spvtools::Optimizer::PassToken pass) {
    report.registered_passes.emplace_back(std::move(name));
    optimizer.RegisterPass(std::move(pass));
}

// LUISA_SPIRV_OPT_MAX_ITERATIONS: fixed-point cap, bounded 1..10, default 5
// (DXC's kSpirvOptMaxIterations). Invalid or out-of-range values keep the
// default, mirroring the strict LUISA_SPIRV_OPT_LEVEL parsing.
[[nodiscard]] size_t
spirv_opt_max_iterations_from_environment() noexcept {
    if (auto *env = std::getenv("LUISA_SPIRV_OPT_MAX_ITERATIONS")) {
        char *end = nullptr;
        errno = 0;
        auto value = std::strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' &&
            value >= 1 && value <= 10) {
            return static_cast<size_t>(value);
        }
    }
    return 5u;
}

// LUISA_SPIRV_OPT_SROA_LIMIT: ScalarReplacement composite-size limit. Default
// 100 preserves the historical compute behavior; 0 means unlimited (DXC
// parity). Values must fit the uint32_t factory argument.
[[nodiscard]] uint32_t
spirv_opt_sroa_limit_from_environment() noexcept {
    if (auto *env = std::getenv("LUISA_SPIRV_OPT_SROA_LIMIT")) {
        char *end = nullptr;
        errno = 0;
        auto value = std::strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' &&
            value >= 0 && value <= std::numeric_limits<int>::max()) {
            return static_cast<uint32_t>(value);
        }
    }
    return 100u;
}

// LUISA_SPIRV_OPT_PASS_FLAGS: optional comma/space separated --pass flags
// appended after the preset (policy #5, -Oconfig-style). Returns false when
// the flag list is present but contains an invalid entry; the caller then
// fails closed and retains the input binary.
[[nodiscard]] bool register_custom_pass_flags(
    spvtools::Optimizer &optimizer, SpirvOptimizerReport &report,
    luisa::string_view flags) noexcept {
    std::vector<std::string> parsed;
    auto text = flags;
    while (true) {
        while (!text.empty() &&
               (text.front() == ' ' || text.front() == '\t' ||
                text.front() == ',')) {
            text.remove_prefix(1u);
        }
        if (text.empty()) { break; }
        auto split = text.find_first_of(" \t,");
        if (split == luisa::string_view::npos) {
            parsed.emplace_back(text);
            break;
        }
        parsed.emplace_back(text.substr(0u, split));
        text.remove_prefix(split);
    }
    if (parsed.empty()) { return true; }
    report.registered_passes.emplace_back(
        luisa::format("custom-pass-flags[{}]", flags));
    return optimizer.RegisterPassesFromFlags(parsed);
}

// Mirrors DXC's RegisterPerformancePasses ordering (vendored
// src/ext/SPIRV-Tools/source/opt/optimizer.cpp L185-230) minus the four passes
// architecturally covered at XIR level (WrapOpKill, MergeReturn,
// InlineExhaustive, EliminateDeadFunctions), keeping the Luisa-side
// conditional loop-unroll gate (policy #2). The target-independent
// SPIRV-Tools loop-unswitch pass is deliberately not a default native pass: k
// independent loop-invariant selectors can clone a loop 2^k times, and the
// pass has no target cost model. It remains available explicitly through
// LUISA_SPIRV_OPT_PASS_FLAGS=--loop-unswitch. sroa_limit threads the
// LUISA_SPIRV_OPT_SROA_LIMIT override into both ScalarReplacement passes.
void register_compute_passes(spvtools::Optimizer &optimizer,
                             SpirvOptimizerReport &report,
                             bool register_loop_unroll,
                             uint32_t sroa_limit) {
    register_pass(optimizer, report, "dead-branch-elim",
                  spvtools::CreateDeadBranchElimPass());
    register_pass(optimizer, report, "adce",
                  spvtools::CreateAggressiveDCEPass());
    register_pass(optimizer, report, "private-to-local",
                  spvtools::CreatePrivateToLocalPass());
    register_pass(optimizer, report, "local-single-block-load-store-elim",
                  spvtools::CreateLocalSingleBlockLoadStoreElimPass());
    register_pass(optimizer, report, "local-single-store-elim",
                  spvtools::CreateLocalSingleStoreElimPass());
    register_pass(optimizer, report, "adce",
                  spvtools::CreateAggressiveDCEPass());
    register_pass(optimizer, report, "scalar-replacement",
                  spvtools::CreateScalarReplacementPass(sroa_limit));
    register_pass(optimizer, report, "local-access-chain-convert",
                  spvtools::CreateLocalAccessChainConvertPass());
    register_pass(optimizer, report, "local-single-block-load-store-elim",
                  spvtools::CreateLocalSingleBlockLoadStoreElimPass());
    register_pass(optimizer, report, "local-single-store-elim",
                  spvtools::CreateLocalSingleStoreElimPass());
    register_pass(optimizer, report, "adce",
                  spvtools::CreateAggressiveDCEPass());
    register_pass(optimizer, report, "local-multi-store-elim",
                  spvtools::CreateLocalMultiStoreElimPass());
    register_pass(optimizer, report, "adce",
                  spvtools::CreateAggressiveDCEPass());
    register_pass(optimizer, report, "ccp",
                  spvtools::CreateCCPPass());
    register_pass(optimizer, report, "adce",
                  spvtools::CreateAggressiveDCEPass());
    // SPIRV-Tools only considers loops carrying the explicit Unroll control
    // bit. No preceding pass in this pipeline introduces that bit, so proving
    // its absence in the input makes registering the whole-module loop
    // analysis a pure cost. In particular, ordinary runtime/depth loops use
    // LoopControl None and must remain compact. (Deliberate divergence from
    // DXC, which unrolls unconditionally.)
    if (register_loop_unroll) {
        register_pass(optimizer, report, "loop-unroll",
                      spvtools::CreateLoopUnrollPass(true));
    }
    register_pass(optimizer, report, "dead-branch-elim",
                  spvtools::CreateDeadBranchElimPass());
    register_pass(optimizer, report, "redundancy-elimination",
                  spvtools::CreateRedundancyEliminationPass());
    register_pass(optimizer, report, "combine-access-chains",
                  spvtools::CreateCombineAccessChainsPass());
    register_pass(optimizer, report, "simplification",
                  spvtools::CreateSimplificationPass());
    register_pass(optimizer, report, "scalar-replacement",
                  spvtools::CreateScalarReplacementPass(sroa_limit));
    register_pass(optimizer, report, "local-access-chain-convert",
                  spvtools::CreateLocalAccessChainConvertPass());
    register_pass(optimizer, report, "local-single-block-load-store-elim",
                  spvtools::CreateLocalSingleBlockLoadStoreElimPass());
    register_pass(optimizer, report, "local-single-store-elim",
                  spvtools::CreateLocalSingleStoreElimPass());
    register_pass(optimizer, report, "adce",
                  spvtools::CreateAggressiveDCEPass());
    register_pass(optimizer, report, "ssa-rewrite",
                  spvtools::CreateSSARewritePass());
    register_pass(optimizer, report, "adce",
                  spvtools::CreateAggressiveDCEPass());
    register_pass(optimizer, report, "vector-dce",
                  spvtools::CreateVectorDCEPass());
    register_pass(optimizer, report, "dead-insert-elim",
                  spvtools::CreateDeadInsertElimPass());
    register_pass(optimizer, report, "dead-branch-elim",
                  spvtools::CreateDeadBranchElimPass());
    register_pass(optimizer, report, "simplification",
                  spvtools::CreateSimplificationPass());
    register_pass(optimizer, report, "if-conversion",
                  spvtools::CreateIfConversionPass());
    register_pass(optimizer, report, "copy-propagate-arrays",
                  spvtools::CreateCopyPropagateArraysPass());
    register_pass(optimizer, report, "reduce-load-size",
                  spvtools::CreateReduceLoadSizePass());
    register_pass(optimizer, report, "adce",
                  spvtools::CreateAggressiveDCEPass());
    register_pass(optimizer, report, "block-merge",
                  spvtools::CreateBlockMergePass());
    register_pass(optimizer, report, "redundancy-elimination",
                  spvtools::CreateRedundancyEliminationPass());
    register_pass(optimizer, report, "dead-branch-elim",
                  spvtools::CreateDeadBranchElimPass());
    register_pass(optimizer, report, "block-merge",
                  spvtools::CreateBlockMergePass());
    register_pass(optimizer, report, "simplification",
                  spvtools::CreateSimplificationPass());
    // Keep the two size-bounded DXC fork-tail transforms. Loop unswitch is an
    // explicit opt-in because its repeated whole-loop cloning is unbounded.
    register_pass(optimizer, report, "spread-volatile-semantics",
                  spvtools::CreateSpreadVolatileSemanticsPass());
    register_pass(optimizer, report, "compact-ids",
                  spvtools::CreateCompactIdsPass());
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
                    break;
                case spv::Capability::Float16:
                case spv::Capability::Int16:
                    has_16bit_arithmetic = true;
                    break;
                case spv::Capability::UniformAndStorageBuffer16BitAccess:
                    has_uniform_storage_16 = true;
                    break;
                default:
                    // This includes Int8/Float8, 8-bit storage, descriptor
                    // indexing, subgroup ballot/shuffle, image-query, and
                    // atomic capabilities. The vendored trim pass does not
                    // claim all of them, so leave every declaration intact.
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
    target_feature::shader_device_clock |
    target_feature::cooperative_vector;

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
        case spv::Capability::CooperativeVectorNV:
        case spv::Capability::CooperativeVectorTrainingNV:
            return target_feature::cooperative_vector;
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
    optimizer.SetMessageConsumer(
        [&report](spv_message_level_t level, const char *source,
                  const spv_position_t &position, const char *message) {
            report.diagnostics.append(luisa::format(
                "{} [{}:{}:{}]: {}\n", static_cast<int>(level),
                source == nullptr ? "" : source, position.line,
                position.column, message == nullptr ? "" : message));
        });

    if (report.effective_preset == "lightweight") {
        register_pass(optimizer, report, "adce",
                      spvtools::CreateAggressiveDCEPass());
        register_pass(optimizer, report, "block-merge",
                      spvtools::CreateBlockMergePass());
        register_pass(optimizer, report, "simplification",
                      spvtools::CreateSimplificationPass());
        register_pass(optimizer, report, "dead-branch-elim",
                      spvtools::CreateDeadBranchElimPass());
    } else if (report.effective_preset == "compute") {
        report.loop_unroll_registered =
            has_unroll_loop_control(words.data(), words.size());
        register_compute_passes(
            optimizer, report, report.loop_unroll_registered,
            spirv_opt_sroa_limit_from_environment());
    } else if (report.effective_preset == "full") {
        optimizer.RegisterPerformancePasses();
        report.registered_passes.emplace_back("RegisterPerformancePasses");
        // RegisterPerformancePasses owns a fixed upstream pipeline that
        // includes the loop-unroll pass.
        report.loop_unroll_registered = true;
        register_pass(optimizer, report, "private-to-local",
                      spvtools::CreatePrivateToLocalPass());
        register_pass(optimizer, report, "copy-propagate-arrays",
                      spvtools::CreateCopyPropagateArraysPass());
        // The bounded DXC fork-tail transforms remain enabled. See the
        // compute-preset rationale above for the explicit loop-unswitch gate.
        register_pass(optimizer, report, "spread-volatile-semantics",
                      spvtools::CreateSpreadVolatileSemanticsPass());
        register_pass(optimizer, report, "compact-ids",
                      spvtools::CreateCompactIdsPass());
    } else {
        report.diagnostics.append(luisa::format(
            "Unknown SPIR-V optimization preset '{}'; using compute.\n",
            report.effective_preset));
        report.effective_preset = "compute";
        report.loop_unroll_registered =
            has_unroll_loop_control(words.data(), words.size());
        register_compute_passes(
            optimizer, report, report.loop_unroll_registered,
            spirv_opt_sroa_limit_from_environment());
    }

    // DCE can remove the last use of an optional capability, but ordinary
    // optimization passes deliberately do not edit module capabilities. Use
    // SPIRV-Tools' grammar-aware proof only inside its audited input domain;
    // outside that domain retaining a conservative requirement is preferable
    // to trusting an incomplete trim. Never strip capabilities with an ad-hoc
    // binary rewrite.
    if (can_safely_trim_capabilities(words.data(), words.size())) {
        register_pass(optimizer, report, "trim-capabilities",
                      spvtools::CreateTrimCapabilitiesPass());
        report.capability_trim_registered = true;
    }

    // Optional -Oconfig-style custom pass list appended after the preset
    // (policy #5). Invalid flags fail closed and retain the input binary,
    // matching DXC's behavior of aborting on an invalid -Oconfig list.
    if (auto *env = std::getenv("LUISA_SPIRV_OPT_PASS_FLAGS")) {
        if (*env != '\0' &&
            !register_custom_pass_flags(optimizer, report, env)) {
            report.diagnostics.append(
                "LUISA_SPIRV_OPT_PASS_FLAGS contains an invalid pass flag; "
                "aborting optimization and retaining the input binary.\n");
            return report;
        }
    }

    report.attempted = true;
    report.max_iterations = spirv_opt_max_iterations_from_environment();
    spvtools::OptimizerOptions optimizer_options;
    // The module is already validated before optimization (entry.cpp
    // "pre-optimization" stage) and the final candidate is re-validated by
    // validate_and_commit_spirv_transform below; per-iteration validation is
    // redundant and expensive (DXC L17194 does the same).
    optimizer_options.set_run_validator(false);
    if (auto *env = std::getenv("LUISA_SPIRV_OPT_MAX_ID_BOUND")) {
        char *end = nullptr;
        errno = 0;
        auto value = std::strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' &&
            value > 0 && value <= std::numeric_limits<int>::max()) {
            optimizer_options.set_max_id_bound(static_cast<uint32_t>(value));
        }
    }
    if (auto *env = std::getenv("LUISA_SPIRV_OPT_PRESERVE_BINDINGS")) {
        luisa::string_view text{env};
        if (text == "1" || text == "true" || text == "TRUE") {
            optimizer_options.set_preserve_bindings(true);
        } else if (text == "0" || text == "false" || text == "FALSE") {
            optimizer_options.set_preserve_bindings(false);
        }
    }

    // ---- Fixed-point iteration loop (DXC mirror) ----
    // Run the optimizer repeatedly until the binary content stabilizes or the
    // maximum number of iterations is reached. The final iteration's candidate
    // is kept in `optimized` so the transactional commit below validates the
    // exact bytes that would replace the artifact. Intermediate iterations are
    // not validated, matching DXC.
    auto original = words;
    const auto profile =
        luisa::compute::detail::env_flag("LUISA_VULKAN_PROFILE_COMPILATION");
    luisa::Clock optimizer_clock;
    std::vector<uint32_t> optimized;
    for (report.iterations = 1u;
         report.iterations <= report.max_iterations; ++report.iterations) {
        optimizer_clock.tic();
        report.succeeded = optimizer.Run(
            words.data(), words.size(), &optimized, optimizer_options);
        if (profile) {
            LUISA_INFO(
                "Vulkan native SPIR-V optimizer iteration {}: {:.3f} ms",
                report.iterations, optimizer_clock.toc());
        }
        if (!report.succeeded) { return report; }
        report.converged = (optimized == words);
        if (profile) {
            LUISA_INFO(
                "Vulkan native SPIR-V optimizer iteration {} result: {} words{}",
                report.iterations, optimized.size(),
                report.converged ? " (converged)" : "");
        }
        if (report.converged) { break; }
        if (report.iterations == report.max_iterations) { break; }
        words.swap(optimized);
        optimized.clear();
    }
    if (profile) {
        LUISA_INFO(
            "Vulkan native SPIR-V optimizer fixed point: {} iteration(s), "
            "converged={}, {} words",
            report.iterations, report.converged, words.size());
    }

    optimizer_clock.tic();
    auto commit = validate_and_commit_spirv_transform(
        words, std::move(optimized));
    if (profile) {
        LUISA_INFO(
            "Vulkan native SPIR-V optimizer commit validation: {:.3f} ms",
            optimizer_clock.toc());
    }
    if (!commit.succeeded) {
        report.succeeded = false;
        report.output_validated = false;
        report.changed = false;
        words = std::move(original);
        report.output_word_count = words.size();
        report.diagnostics.append(
            "SPIR-V optimizer output failed Vulkan 1.2 validation; "
            "retaining the input binary.\n");
        report.diagnostics.append(commit.diagnostics);
        return report;
    }
    report.succeeded = commit.succeeded;
    report.output_validated = commit.output_validated;
    report.changed = words != original;
    report.output_word_count = commit.output_word_count;
    if (!commit.diagnostics.empty()) {
        report.diagnostics.append(
            "SPIR-V optimizer output validation diagnostics:\n");
        report.diagnostics.append(commit.diagnostics);
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
