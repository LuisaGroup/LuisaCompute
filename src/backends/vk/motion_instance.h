#pragma once
#include <volk.h>
#include "primitive_base.h"
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/motion_transform.h>
namespace lc::vk {
class Blas;
using namespace luisa;
using namespace luisa::compute;

// MotionInstance wraps a child BLAS with per-keyframe transforms for motion blur.
// Uses VK_NV_ray_tracing_motion_blur extension for hardware-accelerated motion blur.
class MotionInstance : public PrimitiveBase {
    AccelMotionOption _option;
    Blas *_child{nullptr};
    luisa::vector<MotionInstanceTransform> _keyframes;

public:
    MotionInstance(Device *device, const AccelMotionOption &option);
    ~MotionInstance() override = default;

    void set_child(Blas *child) noexcept { _child = child; }
    void set_keyframes(luisa::vector<MotionInstanceTransform> keyframes) noexcept {
        _keyframes = std::move(keyframes);
    }

    [[nodiscard]] auto child() const noexcept { return _child; }
    [[nodiscard]] auto &option() const noexcept { return _option; }
    [[nodiscard]] auto &keyframes() const noexcept { return _keyframes; }
    [[nodiscard]] auto keyframe_count() const noexcept { return _option.keyframe_count; }
    [[nodiscard]] auto mode() const noexcept { return _option.mode; }

    // PrimitiveBase: a motion instance is not an acceleration structure of its
    // own — it only carries the per-keyframe transforms of the child BLAS it
    // wraps, so both answers are the child's. Defined in motion_instance.cpp,
    // where `Blas` is complete (blas.h is deliberately not pulled in here).
    [[nodiscard]] uint64_t get_accel_device_address() const override;
    [[nodiscard]] bool has_motion() const noexcept override;
};
}// namespace lc::vk
