#pragma once

#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
#include <embree3/rtcore.h>
#else
#include <embree4/rtcore.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <luisa/runtime/rhi/resource.h>

namespace luisa::compute::fallback {

struct FallbackEmbreeBvhStats {
    std::atomic<int64_t> live_bytes{0};
    std::atomic<int64_t> peak_bytes{0};
    std::atomic<int64_t> retained_after_accel_bytes{0};
    std::atomic<uint64_t> blas_commits{0};
    std::atomic<uint64_t> blas_commit_ns{0};
    std::atomic<uint64_t> tlas_commits{0};
    std::atomic<uint64_t> tlas_commit_ns{0};
    std::atomic<uint64_t> last_direct_meshes{0};
    std::atomic<uint64_t> last_instanced_slots{0};

    void record_alloc(ssize_t bytes) noexcept {
        const auto cur = live_bytes.fetch_add(static_cast<int64_t>(bytes), std::memory_order_relaxed) +
                         static_cast<int64_t>(bytes);
        auto prev_peak = peak_bytes.load(std::memory_order_relaxed);
        while (cur > prev_peak &&
               !peak_bytes.compare_exchange_weak(prev_peak, cur, std::memory_order_relaxed)) {}
    }
};

[[nodiscard]] inline FallbackEmbreeBvhStats &luisa_fallback_bvh_stats() noexcept {
    static FallbackEmbreeBvhStats stats;
    return stats;
}

[[nodiscard]] inline bool luisa_fallback_bvh_stats_enabled() noexcept {
    static const bool enabled = []() noexcept {
        const char *env = std::getenv("LUISA_FALLBACK_BVH_STATS");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

[[nodiscard]] inline bool luisa_fallback_mesh_sbvh_enabled() noexcept {
    if (const char *env = std::getenv("LUISA_FALLBACK_MESH_SBVH"); env != nullptr && env[0] != '\0') {
        return env[0] != '0';
    }
    return true;
}

[[nodiscard]] inline RTCBuildQuality luisa_fallback_embree_geometry_quality(
    RTCGeometryType geom_type, const AccelOption &option) noexcept {
    if (option.allow_update) { return RTC_BUILD_QUALITY_LOW; }
    if (geom_type == RTC_GEOMETRY_TYPE_TRIANGLE &&
        option.hint == AccelOption::UsageHint::FAST_TRACE &&
        luisa_fallback_mesh_sbvh_enabled()) {
        return RTC_BUILD_QUALITY_HIGH;
    }
    return RTC_BUILD_QUALITY_MEDIUM;
}

inline void luisa_fallback_accel_set_flags(RTCScene scene, const AccelOption &option) noexcept {
    // Preserve the hardware ray-tracing shared-edge ownership invariant: a ray
    // on a manifold shared edge may choose either adjacent primitive, but it
    // must not miss both. Embree's default fast mode can lose a sloped
    // shared-edge hit and expose a farther surface instead. Robust mode is the
    // documented Embree mechanism for enforcing this invariant, and must be
    // enabled on both BLAS and TLAS scenes.
    auto scene_flags = static_cast<unsigned>(RTC_SCENE_FLAG_ROBUST);
#if LUISA_COMPUTE_FALLBACK_EMBREE_VERSION == 3
    scene_flags |= RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION;
#else
    scene_flags |= RTC_SCENE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS;
#endif
    if (option.allow_compaction) { scene_flags |= RTC_SCENE_FLAG_COMPACT; }
    if (option.allow_update) { scene_flags |= RTC_SCENE_FLAG_DYNAMIC; }
    rtcSetSceneFlags(scene, static_cast<RTCSceneFlags>(scene_flags));
    if (option.allow_update) {
        rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_LOW);
        return;
    }
    switch (option.hint) {
        case AccelOption::UsageHint::FAST_TRACE:
            rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);
            break;
        case AccelOption::UsageHint::FAST_BUILD:
            rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_MEDIUM);
            break;
    }
}

}// namespace luisa::compute::fallback
