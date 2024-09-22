//
// Created by Mike on 2024/9/21.
//

#include "cuda_error.h"
#include "cuda_device.h"
#include "cuda_command_encoder.h"
#include "cuda_motion_instance.h"

namespace luisa::compute::cuda {

CUDAMotionInstance::CUDAMotionInstance(CUDADevice *device, const AccelMotionOption &option) noexcept
    : CUDAPrimitiveBase{CUDAPrimitiveBase::Tag::MOTION_INSTANCE}, _option{option} {
    auto [traversable_type, buffer_size] = [&] {
        switch (option.mode) {
            case AccelOption::MotionMode::STATIC: {
                LUISA_ASSERT(option.keyframe_count == 1u,
                             "Static motion instance should have only one keyframe.");
                auto size = sizeof(optix::StaticTransform);
                return std::make_pair(optix::TRAVERSABLE_TYPE_STATIC_TRANSFORM, size);
            }
            case AccelOption::MotionMode::MATRIX: {
                LUISA_ASSERT(option.keyframe_count >= 2u,
                             "Matrix motion instance should have at least two keyframes.");
                auto size = sizeof(optix::MatrixMotionTransform) -
                            sizeof(std::declval<optix::MatrixMotionTransform>().transform) +
                            sizeof(float[12]) * option.keyframe_count;
                return std::make_pair(optix::TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM, size);
            }
            case AccelOption::MotionMode::SRT: {
                LUISA_ASSERT(option.keyframe_count >= 2u,
                             "SRT motion instance should have at least two keyframes.");
                auto size = sizeof(optix::SRTMotionTransform) -
                            sizeof(std::declval<optix::SRTMotionTransform>().srtData) +
                            sizeof(optix::SRTData) * option.keyframe_count;
                return std::make_pair(optix::TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM, size);
            }
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Unsupported motion mode.");
    }();
    LUISA_CHECK_CUDA(cuMemAlloc(&_motion_buffer, buffer_size));
    _motion_buffer_size = buffer_size;
    auto optix_ctx = device->handle().optix_context();
    LUISA_CHECK_OPTIX(optix::api().convertPointerToTraversableHandle(
        optix_ctx, _motion_buffer, traversable_type, &_handle));
}

CUDAMotionInstance::~CUDAMotionInstance() noexcept {
    LUISA_CHECK_CUDA(cuMemFree(_motion_buffer));
}

void CUDAMotionInstance::build(CUDACommandEncoder &encoder,
                               MotionInstanceBuildCommand *command) noexcept {
    LUISA_ASSERT(command->keyframes().size() == _option.keyframe_count,
                 "Keyframe count mismatch.");
}

}// namespace luisa::compute::cuda
