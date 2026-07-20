//
// Exact matrix and quaternion-SRT motion instances for HIPRT.
//

#pragma once

#include <cstddef>

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rtx/motion_instance.h>

#include "hip_geometry.h"

namespace luisa::compute::hip {

class HIPCommandEncoder;

// Device-visible metadata used by the instance-motion query/write operations.
// HIPRT copies the frames into its private scene storage during build/refit, so
// shaders mutate this authoritative input array and HIPAccel refits the nested
// motion scene before updating the outer TLAS.
struct alignas(16) HIPMotionInstanceDeviceData {
    uint64_t frames;
    uint32_t keyframe_count;
    uint32_t frame_stride;
    uint32_t mode;
    uint32_t reserved[3];
};

static_assert(sizeof(HIPMotionInstanceDeviceData) == 32u);
static_assert(alignof(HIPMotionInstanceDeviceData) == 16u);
static_assert(offsetof(HIPMotionInstanceDeviceData, frames) == 0u);
static_assert(offsetof(HIPMotionInstanceDeviceData, keyframe_count) == 8u);
static_assert(offsetof(HIPMotionInstanceDeviceData, frame_stride) == 12u);
static_assert(offsetof(HIPMotionInstanceDeviceData, mode) == 16u);

class HIPMotionInstance final : public HIPPrimitive {

private:
    AccelMotionOption _option;
    hiprtContext _hiprt_ctx{nullptr};
    hiprtScene _scene{nullptr};
    hipDeviceptr_t _input_buffer{};
    size_t _input_buffer_size{};
    size_t _transform_header_offset{};
    size_t _frame_offset{};
    size_t _frame_stride{};
    size_t _device_data_offset{};
    const HIPPrimitive *_child{nullptr};
    hiprtInstance _built_child_instance{};
    mutable spin_mutex _mutex;

private:
    [[nodiscard]] hiprtSceneBuildInput _make_build_input() const noexcept;

public:
    HIPMotionInstance(hiprtContext ctx, const AccelMotionOption &option) noexcept;
    ~HIPMotionInstance() noexcept override;
    void build(HIPCommandEncoder &encoder, MotionInstanceBuildCommand *command) noexcept;
    [[nodiscard]] Binding binding() const noexcept override;
    void prepare_for_tlas_build(HIPCommandEncoder &encoder) const noexcept override;
    [[nodiscard]] auto option() const noexcept { return _option; }
};

}// namespace luisa::compute::hip
