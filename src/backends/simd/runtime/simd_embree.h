#pragma once

#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
#include <embree3/rtcore.h>
#else
#include <embree4/rtcore.h>
#endif

#include <luisa/runtime/rhi/resource.h>

namespace luisa::compute::simd {

inline void simd_accel_set_flags(
    RTCScene scene, const AccelOption &option) noexcept {
    auto flags = static_cast<unsigned>(RTC_SCENE_FLAG_ROBUST);
#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
    flags |= RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION;
#endif
    if (option.allow_compaction) { flags |= RTC_SCENE_FLAG_COMPACT; }
    if (option.allow_update) { flags |= RTC_SCENE_FLAG_DYNAMIC; }
    rtcSetSceneFlags(scene, static_cast<RTCSceneFlags>(flags));
    rtcSetSceneBuildQuality(
        scene,
        option.hint == AccelOption::UsageHint::FAST_TRACE ?
            RTC_BUILD_QUALITY_HIGH :
            RTC_BUILD_QUALITY_MEDIUM);
}

}// namespace luisa::compute::simd
