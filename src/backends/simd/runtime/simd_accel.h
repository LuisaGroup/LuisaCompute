#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/rtx/accel.h>

#include "../llvm/llvm_schedule_codegen.h"
#include "simd_embree.h"

namespace luisa::compute::simd {

namespace triangle_ray_query {

struct SIMDAccelAccess;
[[nodiscard]] bool triangle_only_ray_query_enabled() noexcept;
void ray_query_proceed_triangle_only(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept;
void ray_query_proceed_wide_triangle_only(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept;

}// namespace triangle_ray_query

class alignas(16) SIMDAccel {

private:
    using Instance = SIMDHostAccelInstance;

    struct MotionState {
        AccelMotionOption option{};
        luisa::vector<MotionInstanceTransform> keyframes;
    };

private:
    RTCScene _scene{nullptr};
    luisa::vector<Instance> _instances;
    luisa::vector<RTCGeometry> _geometries;
    luisa::vector<luisa::unique_ptr<MotionState>> _motion_states;
    SIMDHostAccelInstanceTable _instance_table{
        .ray_query_proceed_status = simd_host_ray_query_proceed_status,
        .ray_query_proceed_wide_status = simd_host_ray_query_proceed_status};
    bool _has_curve_instances{false};
    bool _has_procedural_instances{false};
    uint32_t _warp_width{8u};
    bool _enable_triangle_only_ray_query{true};
    bool _enable_procedural_dense_status{true};

    friend struct triangle_ray_query::SIMDAccelAccess;

private:
    static void _trace_closest(
        void *accel, uint32_t lane_count,
        void *ray_hit_packet) noexcept;
    static void _trace_any(
        void *accel, uint32_t lane_count,
        void *ray_packet) noexcept;
    static void _ray_query_proceed(
        uint32_t lane_count, uint64_t active_mask_bits,
        SIMDHostRayQueryState *const *states) noexcept;
    static void _ray_query_proceed_wide(
        uint32_t lane_count, uint64_t active_mask_bits,
        SIMDHostRayQueryState *const *states) noexcept;
    static LUISA_FORCE_INLINE void _ray_query_proceed_wide_lane(
        SIMDHostRayQueryState *state, uint32_t lane,
        uint64_t bit, uint64_t &pending) noexcept;

public:
    SIMDAccel(
        RTCDevice device, const AccelOption &option,
        uint32_t warp_width) noexcept;
    ~SIMDAccel() noexcept;

    void build(const AccelBuildCommand &command) noexcept;
    [[nodiscard]] auto native_handle() const noexcept { return _scene; }
    [[nodiscard]] SIMDHostAccelView host_view() noexcept {
        auto use_triangle_only_provider =
            !_has_procedural_instances && !_has_curve_instances &&
            _enable_triangle_only_ray_query;
        return {
            .accel = this,
            .trace_closest = _trace_closest,
            .trace_any = _trace_any,
            .instances = &_instance_table,
            .ray_query_proceed = use_triangle_only_provider ?
                                     triangle_ray_query::ray_query_proceed_triangle_only :
                                     _ray_query_proceed,
            .ray_query_proceed_wide = use_triangle_only_provider ?
                                          triangle_ray_query::ray_query_proceed_wide_triangle_only :
                                          _ray_query_proceed_wide,
        };
    }
};

}// namespace luisa::compute::simd
