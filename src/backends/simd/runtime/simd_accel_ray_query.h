#pragma once

#include <array>
#include <cstddef>

#include "../llvm/llvm_schedule_codegen.h"
#include "simd_embree.h"

namespace luisa::compute::simd::detail {

struct RayQueryBatchBuildState {
    bool heapified{false};
    bool ascending{true};
    bool descending{true};
};

#if LUISA_COMPUTE_SIMD_EMBREE_VERSION == 3
using RayQueryRTCContext = RTCIntersectContext;
#else
using RayQueryRTCContext = RTCRayQueryContext;
#endif

struct RayQueryScanContext {
    RayQueryRTCContext rtc{};
    uint32_t lane_count{0u};
    const SIMDHostAccelInstanceTable *instances{nullptr};
    std::array<SIMDHostRayQueryState *, 16u> states{};
    std::array<RayQueryBatchBuildState, 16u> batch_build{};
    std::array<RayQueryBatchBuildState, 16u> procedural_batch_build{};
};
static_assert(offsetof(RayQueryScanContext, rtc) == 0u);

void ray_query_filter_wide(
    const RTCFilterFunctionNArguments *arguments) noexcept;

}// namespace luisa::compute::simd::detail
