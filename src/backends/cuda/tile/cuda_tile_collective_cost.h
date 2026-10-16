#pragma once

#include <array>
#include <luisa/tile/collective_cost.h>

namespace luisa::compute::cuda::native_tile {

inline constexpr luisa::string_view kCollectiveScheduleProfile = "sm89-24-cuda134-v3";
// Frozen before external heldout measurements. The fitted score is relative
// log time, not a calibrated confidence bound, register count or occupancy.
// Profile SHA256: 357c15e0c745797a75e8d11c9de9fafc0ab204f71d1619fedd9251323e7abb35.
inline constexpr std::array<tile::CollectiveCostTreeNode, 7u> kCollectiveScheduleTree{{{4, 4.565928480304397, 1u, 4u, 0.010218042504209173},
                                                                                       {8, 0.5863706188262989, 2u, 3u, 0.16022989736269946},
                                                                                       {-1, 0.0, 0u, 0u, 0.3131874161611729},
                                                                                       {-1, 0.0, 0u, 0u, 0.09904688984331009},
                                                                                       {0, 0.5981986064017516, 5u, 6u, -0.12104233049696983},
                                                                                       {-1, 0.0, 0u, 0u, -0.2564738724463534},
                                                                                       {-1, 0.0, 0u, 0u, 0.014389211452413706}}};

struct CollectiveScheduleChoice {
    uint32_t worker_warps{0u};
    double log_score{0.0};
    bool has_score{false};
    luisa::string_view status{"ineligible"};
    luisa::string_view reason{"device-query"};
};

[[nodiscard]] inline CollectiveScheduleChoice choose_collective_schedule(
    const tile::CollectiveWorkAnalysis &work, uint32_t compute_capability,
    uint32_t processors, uint32_t subgroup_width, uint32_t resident_threads,
    uint32_t driver_api_version, uint32_t toolkit_version, bool fast_math) noexcept {
    CollectiveScheduleChoice choice;
    if (compute_capability != 89u || processors != 24u || subgroup_width != 32u || resident_threads != 1536u ||
        driver_api_version != 13040u || toolkit_version != 13040u) {
        choice.reason = "target-profile";
        return choice;
    }
    if (fast_math) {
        choice.reason = "fast-math";
        return choice;
    }
    if (!work.ok()) {
        choice.reason = "analysis";
        return choice;
    }
    for (auto &&collective : work.collectives) {
        if (collective.kind == tile::CollectiveKind::INCLUSIVE_SUM) {
            choice.status = "default";
            choice.reason = "prefix";
            return choice;
        }
        if (collective.kind != tile::CollectiveKind::SUM && collective.kind != tile::CollectiveKind::MAXIMUM) {
            choice.reason = "unsupported-algebra";
            return choice;
        }
    }
    auto features = tile::collective_cost_features(work, {processors, subgroup_width});
    if (!features.ok()) {
        choice.reason = "features";
        return choice;
    }
    auto prediction = tile::evaluate_collective_cost_tree(kCollectiveScheduleTree, features.values);
    if (!prediction.ok()) {
        choice.reason = "model";
        return choice;
    }
    choice.has_score = true;
    choice.log_score = prediction.log_score;
    choice.worker_warps = prediction.log_score < -0.05129329438755058 ? 8u : 0u;
    choice.status = choice.worker_warps != 0u ? "selected" : "default";
    choice.reason = choice.worker_warps != 0u ? "predicted-saving" : "predicted-default";
    return choice;
}

}// namespace luisa::compute::cuda::native_tile
