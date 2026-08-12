#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/runtime/rtx/accel.h>

#include "../llvm/llvm_schedule_codegen.h"
#include "simd_embree.h"

namespace luisa::compute::simd {

class alignas(16) SIMDAccel {

private:
    struct alignas(16) Instance {
        float affine[12]{};
        uint32_t user_id{0u};
        uint8_t mask{0xffu};
        bool opaque{true};
        bool dirty{false};
    };

private:
    RTCScene _scene{nullptr};
    luisa::vector<Instance> _instances;
    luisa::vector<RTCGeometry> _geometries;

private:
    static void _trace_closest(
        void *accel, uint32_t lane_count,
        uint64_t active_mask_bits, const float *ray_components,
        const uint32_t *visibility_masks,
        uint32_t *hit_ids, float *hit_values) noexcept;
    static void _trace_any(
        void *accel, uint32_t lane_count,
        uint64_t active_mask_bits, const float *ray_components,
        const uint32_t *visibility_masks,
        uint32_t *occluded) noexcept;

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
        };
    }
};

}// namespace luisa::compute::simd
