#pragma once

#include <array>
#include <luisa/tile/collective_plan.h>

namespace luisa::compute::tile {

inline constexpr size_t kCollectiveCostFeatureCount = 9u;
inline constexpr size_t kCollectiveCostMaxNodes = 256u;
inline constexpr size_t kCollectiveCostMaxFeatures = 32u;

struct CollectiveCostDeviceReference {
    uint64_t processor_count{0u};
    uint64_t subgroup_width{0u};
};

// Version 3 logical features, in fixed order. C is the checked sum of
// contribution_extent * independent_elements over all collectives.
// Entries 0..5 and 8 apply log2(1 + x); entries 6 and 7 are plain ratios:
// 0 programs/processors; 1 largest materialized Tile elements;
// 2 peak explicit Tile bytes/(subgroup_width*4); 3 elementwise elements/C;
// 4 maximum contribution extent/subgroup_width; 5 sum of independent elements;
// 6 SUM input elements/C; 7 MAXIMUM input elements/C; 8 (read+write) bytes/(4*C).
// The fixed 4-byte reference is FP32 arithmetic, not input storage width.
// These are model inputs, not measured occupancy, traffic or execution time.
struct CollectiveCostFeatures {
    std::array<double, kCollectiveCostFeatureCount> values{};
    luisa::string_view error;
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

// Root is node 0. feature==-1 denotes a leaf; its child indices are unused.
// Otherwise the feature value <= threshold selects left. All numeric fields must be
// finite, including unused thresholds/scores. Shared acyclic subtrees are valid.
struct CollectiveCostTreeNode {
    int32_t feature{-1};
    double threshold{0.0};
    uint32_t left{0u};
    uint32_t right{0u};
    double log_score{0.0};
};

struct CollectiveCostResult {
    double log_score{0.0};
    luisa::string_view error;
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

[[nodiscard]] LUISA_TILE_API CollectiveCostFeatures collective_cost_features(
    const CollectiveWorkAnalysis &analysis, CollectiveCostDeviceReference device) noexcept;

// Bounded, non-recursive validation includes unreachable components. A score
// grants no IR legality or numerical permission; candidate policy stays local
// to the backend. This API neither selects a device schedule nor exponentiates.
[[nodiscard]] LUISA_TILE_API CollectiveCostResult evaluate_collective_cost_tree(
    luisa::span<const CollectiveCostTreeNode> nodes, luisa::span<const double> features) noexcept;

}// namespace luisa::compute::tile
