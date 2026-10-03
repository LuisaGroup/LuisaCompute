#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <luisa/tile/collective_partition.h>

namespace luisa::compute::cuda::native_tile {

inline constexpr luisa::string_view kPartitionCostProfile = "sm89-24-cuda134-partition-linear-v1";
inline constexpr luisa::string_view kPartitionCostFit = "63e0677c8554b46b707aa9f1fca3f29f223c653b1ec35fd57b5534b79595b6c2";
// Frozen before independent heldout validation. This is an uncalibrated
// ranking score, not a confidence bound or a physical occupancy estimate.
inline constexpr double kPartitionCostConstant = 0.9028899747245084;
inline constexpr double kPartitionCostPrograms = 0.0;
inline constexpr double kPartitionCostVolume = 0.00011327031090119965;

struct ProgramPartitionChoice {
    uint32_t target_rows{0u};
    uint32_t original_rows{0u};
    double original_score{0.0};
    double selected_score{0.0};
    bool has_score{false};
    luisa::string_view status{"ineligible"};
    luisa::string_view reason{"device-query"};
};

[[nodiscard]] inline bool partition_cost_score(const tile::IndependentCollectiveWorkFacts &facts,
                                               double &score) noexcept {
    if (!facts.ok() || facts.collective_input_elements_per_program == 0u) { return false; }
    auto programs = facts.geometry.programs;
    auto waves = programs / 24u + static_cast<uint64_t>(programs % 24u != 0u);
    auto volume = facts.collective_input_elements_per_program;
    if (waves > std::numeric_limits<uint64_t>::max() / volume) { return false; }
    auto demand = waves * volume;
    score = kPartitionCostConstant + kPartitionCostPrograms * static_cast<double>(programs) +
            kPartitionCostVolume * static_cast<double>(demand);
    return std::isfinite(score) && score > 0.0;
}

[[nodiscard]] inline bool partition_cost_supported_plan(const tile::IndependentCollectivePlan &plan,
                                                        uint32_t target_rows) noexcept {
    using tile::ScalarType;
    auto storage = plan.input_storage;
    return plan.ok() &&
           (storage == ScalarType::FLOAT32 || storage == ScalarType::FLOAT16 || storage == ScalarType::BFLOAT16) &&
           plan.output_storage == storage &&
           (plan.kind == tile::CollectiveKind::SUM || plan.kind == tile::CollectiveKind::MAXIMUM) &&
           plan.input_independent_axis == 0u && plan.input_contribution_axis == 1u &&
           plan.output_independent_axis == 0u && (plan.output_rank == 1u || plan.output_rank == 2u) &&
           (plan.original.independent_extent_per_program == 4u || plan.original.independent_extent_per_program == 8u) &&
           plan.candidate.independent_extent_per_program == target_rows &&
           plan.logical_contribution_extent > 0u && plan.logical_contribution_extent <= 65536u &&
           plan.tile_contribution_extent == std::bit_ceil(plan.logical_contribution_extent) &&
           plan.original.programs <= 0x7fffffffu && plan.candidate.programs <= 0x7fffffffu;
}

[[nodiscard]] inline ProgramPartitionChoice choose_program_partition(
    const tile::IndependentCollectivePlan &rows1, const tile::IndependentCollectivePlan &rows2,
    uint32_t compute_capability, uint32_t processors, uint32_t subgroup_width, uint32_t resident_threads,
    uint32_t driver_api_version, uint32_t toolkit_version, bool fast_math) noexcept {
    ProgramPartitionChoice choice;
    if (compute_capability != 89u || processors != 24u || subgroup_width != 32u || resident_threads != 1536u ||
        driver_api_version != 13040u || toolkit_version != 13040u) {
        choice.reason = "target-profile";
        return choice;
    }
    if (fast_math) {
        choice.reason = "fast-math";
        return choice;
    }
    std::array plans{&rows1, &rows2};
    const tile::IndependentCollectivePlan *original = nullptr;
    for (auto i = size_t{0u}; i < plans.size(); i++) {
        if (partition_cost_supported_plan(*plans[i], static_cast<uint32_t>(i + 1u))) {
            original = plans[i];
            break;
        }
    }
    if (original == nullptr) {
        choice.reason = "analysis-or-layout";
        return choice;
    }
    auto facts = tile::analyze_independent_collective_candidate(*original, tile::IndependentCollectiveGeometryKind::ORIGINAL);
    if (!partition_cost_score(facts, choice.original_score)) {
        choice.reason = "facts";
        return choice;
    }
    choice.original_rows = static_cast<uint32_t>(facts.geometry.independent_extent_per_program);
    choice.has_score = true;
    choice.selected_score = choice.original_score;
    auto best_score = choice.original_score;
    auto best_rows = 0u;
    // Strict comparisons retain the original on ties, then prefer rows1 over rows2.
    for (auto i = size_t{0u}; i < plans.size(); i++) {
        auto &plan = *plans[i];
        auto rows = static_cast<uint32_t>(i + 1u);
        if (!partition_cost_supported_plan(plan, rows) || plan.function != original->function ||
            plan.collective_operation_id != original->collective_operation_id ||
            plan.logical_independent_extent != original->logical_independent_extent ||
            plan.logical_contribution_extent != original->logical_contribution_extent ||
            plan.original.independent_extent_per_program != original->original.independent_extent_per_program) { continue; }
        auto candidate = tile::analyze_independent_collective_candidate(plan);
        double score{};
        if (partition_cost_score(candidate, score) && score < best_score) {
            best_score = score;
            best_rows = rows;
        }
    }
    if (best_rows != 0u && best_score / choice.original_score < 0.95) {
        choice.target_rows = best_rows;
        choice.selected_score = best_score;
        choice.status = "selected";
        choice.reason = "predicted-saving";
    } else {
        choice.status = "retained";
        choice.reason = "predicted-original";
    }
    return choice;
}

}// namespace luisa::compute::cuda::native_tile
