#include "motion_instance.h"
#include "blas.h"
#include "device.h"
#include "log.h"
namespace lc::vk {

MotionInstance::MotionInstance(Device *device, const AccelMotionOption &option)
    : PrimitiveBase(device, PrimitiveBase::PrimTag::MOTION_INSTANCE), _option(option) {
    if (!device->enable_motion_blur()) [[unlikely]] {
        LUISA_ERROR("Motion blur not supported on this device. "
                    "VK_NV_ray_tracing_motion_blur extension is required.");
    }
    if (option.keyframe_count < 2) [[unlikely]] {
        LUISA_ERROR("Motion instance requires at least 2 keyframes, got {}.",
                    option.keyframe_count);
    }
    // Initialize keyframes with identity transforms
    _keyframes.resize(option.keyframe_count);
    if (option.mode == AccelMotionMode::SRT) {
        for (auto &kf : _keyframes) {
            kf.as_srt() = MotionInstanceTransformSRT{};
        }
    } else {
        for (auto &kf : _keyframes) {
            kf.as_matrix() = make_float4x4(1.f);
        }
    }
}

uint64_t MotionInstance::get_accel_device_address() const {
    LUISA_ASSERT(_child != nullptr,
                 "Motion instance has no child BLAS assigned.");
    return _child->get_accel_device_address();
}

bool MotionInstance::has_motion() const noexcept {
    // The wrapper's whole purpose is motion; report it even before a child is
    // bound so a TLAS that instanced this handle stays motion-capable.
    return true;
}

}// namespace lc::vk
