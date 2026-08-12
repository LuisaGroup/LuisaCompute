#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/memory.h>
#include <luisa/runtime/rtx/accel.h>

#include "../llvm/llvm_schedule_codegen.h"
#include "simd_embree.h"

namespace luisa::compute::simd {

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
    SIMDHostAccelInstanceTable _instance_table{};

private:
    static void _trace_closest(
        void *accel, uint32_t lane_count,
        uint64_t active_mask_bits, const float *ray_components,
        const uint32_t *visibility_masks,
        const float *times, uint32_t *hit_ids,
        float *hit_values) noexcept;
    static void _trace_any(
        void *accel, uint32_t lane_count,
        uint64_t active_mask_bits, const float *ray_components,
        const uint32_t *visibility_masks,
        const float *times, uint32_t *occluded) noexcept;

public:
    SIMDAccel(RTCDevice device, const AccelOption &option) noexcept;
    ~SIMDAccel() noexcept;

    void build(const AccelBuildCommand &command) noexcept;
    [[nodiscard]] auto native_handle() const noexcept { return _scene; }
    [[nodiscard]] SIMDHostAccelView host_view() noexcept {
        return {
            .accel = this,
            .trace_closest = _trace_closest,
            .trace_any = _trace_any,
            .instances = &_instance_table,
        };
    }
};

}// namespace luisa::compute::simd
