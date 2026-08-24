#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "target_feature_mask.h"

namespace lc::spirv {

struct SpirvOptimizerOptions {
    int level{2};
    luisa::string preset;
};

struct SpirvOptimizerReport {
    int requested_level{2};
    luisa::string requested_preset;
    luisa::string effective_preset;
    luisa::string diagnostics;
    size_t input_word_count{0u};
    size_t output_word_count{0u};
    bool attempted{false};
    bool succeeded{true};
    bool changed{false};
    bool loop_unroll_registered{false};
    bool capability_trim_registered{false};
    bool output_validated{false};
    // Fixed-point iteration state (DXC mirrors this loop, <= 5 iterations).
    size_t iterations{0u};
    bool converged{false};
    size_t max_iterations{5u};
    // Every registered pass name in registration order (observability).
    luisa::vector<luisa::string> registered_passes;
};

struct SpirvValidationReport {
    luisa::string diagnostics;
    bool valid{false};
    bool has_warning{false};
};

struct SpirvTransformCommitReport {
    luisa::string diagnostics;
    size_t input_word_count{0u};
    size_t candidate_word_count{0u};
    size_t output_word_count{0u};
    bool succeeded{false};
    bool changed{false};
    bool output_validated{false};
};

// Validates a complete SPIR-V module for the Vulkan 1.2 environment used by
// this backend. Keeping this beside the optimizer gives code generation and
// cache/AOT loading one validation contract instead of subtly different
// checks at each boundary.
[[nodiscard]] SpirvValidationReport
validate_spirv(const uint32_t *words, size_t word_count);

// Transactional boundary for transformed SPIR-V artifacts. The candidate is
// committed only after Vulkan 1.2 validation succeeds; otherwise the caller's
// artifact remains byte-exact and the report retains validation diagnostics.
[[nodiscard]] SpirvTransformCommitReport
validate_and_commit_spirv_transform(
    std::vector<uint32_t> &artifact,
    std::vector<uint32_t> candidate);

[[nodiscard]] SpirvOptimizerOptions
spirv_optimizer_options_from_environment() noexcept;

[[nodiscard]] SpirvOptimizerReport
optimize_spirv(std::vector<uint32_t> &words,
               SpirvOptimizerOptions options);

// Returns true only when every requested bit has a one-to-one OpCapability
// representation and may therefore defer availability validation to the final
// artifact boundary.
[[nodiscard]] bool spirv_target_feature_is_capability_owned(
    SpirvTargetFeatureMask feature) noexcept;

// Reconciles requirements recorded while emitting instructions with the
// capabilities declared by the final, validated SPIR-V artifact. Capability-
// backed features are owned by the final binary: optimization may remove the
// last instruction that needs one, and the capability-trimming pass may then
// remove its declaration. Requirements that describe runtime layout or other
// semantics not recoverable from OpCapability remain emission-owned.
// This reconciliation happens after lowering; it does not promise that a
// feature-dependent planner can emit an unsupported source operation merely
// because later DCE might remove it.
[[nodiscard]] SpirvTargetFeatureMask
reconcile_spirv_target_features(
    const uint32_t *words, size_t word_count,
    SpirvTargetFeatureMask emitted_requirements) noexcept;

// Multi-stage artifacts use the union of capability-owned requirements from
// every validated module, while emission-owned requirements are retained once
// from the common artifact contract.
[[nodiscard]] SpirvTargetFeatureMask
reconcile_spirv_target_features(
    luisa::span<const luisa::span<const uint32_t>> modules,
    SpirvTargetFeatureMask emitted_requirements) noexcept;

}// namespace lc::spirv
