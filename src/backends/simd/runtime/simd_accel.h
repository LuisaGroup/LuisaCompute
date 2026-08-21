#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/rtx/accel.h>

#include "../llvm/llvm_schedule_codegen.h"
#include "simd_embree.h"

namespace luisa::compute::simd {

class SIMDPrimitive;
class SIMDSRTMotionForwarder;

namespace triangle_ray_query {

struct SIMDAccelAccess;
struct SurfaceFilterPipelineAccelAccess;
[[nodiscard]] bool triangle_only_ray_query_enabled() noexcept;
void ray_query_proceed_triangle_only(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept;
void ray_query_proceed_wide_triangle_only(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states) noexcept;
void ray_query_surface_filter_pipeline_triangle_only(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    const SIMDPacketLaunchConfig *launch_config,
    SIMDHostRayQuerySurfaceFilterHandler *on_surface) noexcept;
void ray_query_surface_filter_packet_pipeline_triangle_only(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    void *ray_packet,
    const SIMDPacketLaunchConfig *launch_config,
    SIMDHostRayQuerySurfaceFilterHandler *on_surface,
    SIMDHostRayQueryDirectSurfaceFilterHandler *on_surface_direct) noexcept;
void ray_query_surface_filter_packet_pipeline_triangle_only_state_oracle(
    uint32_t lane_count, uint64_t active_mask_bits,
    SIMDHostRayQueryState *const *states,
    void *ray_packet,
    const SIMDPacketLaunchConfig *launch_config,
    SIMDHostRayQuerySurfaceFilterHandler *on_surface,
    SIMDHostRayQueryDirectSurfaceFilterHandler *on_surface_direct) noexcept;
void ray_query_empty_surface_filter_packet_pipeline_triangle_only(
    uint32_t lane_count, uint32_t ray_packet_width,
    uint64_t active_mask_bits,
    void *accel, SIMDHostRayQueryOutputPacket *outputs,
    void *ray_packet, uint32_t terminate_on_first) noexcept;
void ray_query_direct_output_surface_filter_packet_pipeline_triangle_only(
    uint32_t lane_count, uint64_t active_mask_bits,
    void *accel, SIMDHostRayQueryOutputPacket *outputs,
    void *ray_packet, uint32_t terminate_on_first,
    SIMDHostRayQueryDirectSurfaceFilterHandler *on_surface_direct) noexcept;

}// namespace triangle_ray_query

class alignas(16) SIMDAccel {

private:
    using Instance = SIMDHostAccelInstance;

    struct MotionState {
        AccelMotionOption option{};
        luisa::vector<MotionInstanceTransform> keyframes;
        uint64_t source_build_version{0u};
    };

    enum class GeometryRoute : uint8_t {
        instance,
        forwarded_srt,
    };

private:
    RTCScene _scene{nullptr};
    luisa::vector<Instance> _instances;
    // Desired primitive bindings visible through the public instance table.
    // They intentionally remain separate from _geometries: a buffer-only
    // update may consume a primitive modification without touching Embree,
    // and the next ordinary build must still be able to commit that binding.
    luisa::vector<SIMDPrimitive *> _primitives;
    luisa::vector<RTCGeometry> _geometries;
    luisa::vector<GeometryRoute> _geometry_routes;
    // Callback payloads belong to the committed Embree scene rather than the
    // desired public instance table. A buffer-only update must keep the old
    // payload alive until the following ordinary build detaches or replaces
    // its user geometry.
    luisa::vector<luisa::unique_ptr<SIMDSRTMotionForwarder>>
        _motion_forwarders;
    // Primitive kinds and last-built opacity for the committed Embree scene.
    // This remains at the committed geometry count across buffer-only
    // resize/rebind commands; current in-range opacity still comes directly
    // from the public table.
    luisa::vector<SIMDHostAccelCommittedInstance> _committed_instances;
    luisa::vector<luisa::unique_ptr<MotionState>> _motion_states;
    SIMDHostAccelInstanceTable _instance_table{
        .ray_query_proceed_status = simd_host_ray_query_proceed_status,
        .ray_query_proceed_wide_status = simd_host_ray_query_proceed_status};
    bool _has_curve_instances{false};
    bool _has_procedural_instances{false};
    // True when the desired instance topology no longer matches the query
    // provider summary for the last committed Embree scene.
    bool _instance_summary_dirty{true};
    uint32_t _warp_width{8u};
    bool _enable_coherent_w16_direct_trace{true};
    bool _enable_triangle_only_ray_query{true};
    bool _enable_surface_filter_pipeline{true};
    bool _enable_surface_filter_ray_packet{true};
    bool _enable_direct_surface_filter_candidate{true};
    bool _enable_output_only_empty_surface_filter{true};
    bool _enable_direct_output_surface_filter{true};
    bool _enable_narrow_shared_status{true};
    bool _enable_w8_wide_shared_status{true};
    bool _enable_procedural_dense_status{true};
    bool _enable_procedural_fused_status{true};

    friend struct triangle_ray_query::SIMDAccelAccess;
    friend struct triangle_ray_query::SurfaceFilterPipelineAccelAccess;
    friend uint64_t
    simd_host_ray_query_proceed_wide_procedural_fused_status(
        uint32_t lane_count, uint64_t active_mask_bits,
        SIMDHostRayQueryState *const *states) noexcept;

private:
    [[nodiscard]] static uint64_t _ray_query_proceed_narrow_shared(
        uint32_t lane_count, uint64_t active_mask_bits,
        SIMDHostRayQueryState *const *states,
        bool publish_status) noexcept;
    static void _trace_closest(
        void *accel, uint32_t lane_count,
        void *ray_hit_packet) noexcept;
    static void _trace_any(
        void *accel, uint32_t lane_count,
        void *ray_packet) noexcept;
    static void _ray_query_proceed(
        uint32_t lane_count, uint64_t active_mask_bits,
        SIMDHostRayQueryState *const *states) noexcept;
    [[nodiscard]] static uint64_t _ray_query_proceed_status(
        uint32_t lane_count, uint64_t active_mask_bits,
        SIMDHostRayQueryState *const *states) noexcept;
    [[nodiscard]] static uint64_t _ray_query_proceed_wide_shared(
        uint32_t lane_count, uint64_t active_mask_bits,
        SIMDHostRayQueryState *const *states,
        bool publish_status) noexcept;
    static void _ray_query_proceed_wide(
        uint32_t lane_count, uint64_t active_mask_bits,
        SIMDHostRayQueryState *const *states) noexcept;
    static void _ray_query_candidate_object_ray(
        uint32_t lane_count, uint64_t active_mask_bits,
        SIMDHostRayQueryState *const *states) noexcept;
    [[nodiscard]] static uint64_t _ray_query_proceed_wide_status(
        uint32_t lane_count, uint64_t active_mask_bits,
        SIMDHostRayQueryState *const *states) noexcept;
    static void _ray_query_pipeline_w1(
        SIMDHostRayQueryState *state, const void *capture,
        const SIMDPacketLaunchConfig *launch_config,
        SIMDHostRayQueryPipelineHandlerW1 *on_candidate) noexcept;
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
            .ray_query_candidate_object_ray =
                _ray_query_candidate_object_ray,
        };
    }
};

}// namespace luisa::compute::simd
