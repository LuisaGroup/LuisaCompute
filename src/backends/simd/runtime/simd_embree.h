#pragma once

#include "simd_embree_packet_support.h"

#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
#include <embree3/rtcore.h>
#else
#include <embree4/rtcore.h>
#endif

#include <luisa/runtime/rhi/resource.h>

namespace luisa::compute::simd {

[[nodiscard]] inline auto simd_embree_native_ray_packet_support(
    RTCDevice device) noexcept {
    return SIMDEmbreeNativeRayPacketSupport{
        .w4 = rtcGetDeviceProperty(
                  device,
                  RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED) != 0,
        .w8 = rtcGetDeviceProperty(
                  device,
                  RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED) != 0,
        .w16 = rtcGetDeviceProperty(
                   device,
                   RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED) != 0,
    };
}

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
