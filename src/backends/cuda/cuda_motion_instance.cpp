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
    LUISA_VERBOSE("Created motion instance: buffer = {}, size = {}, handle = {}.",
                  reinterpret_cast<void *>(_motion_buffer), buffer_size,
                  reinterpret_cast<void *>(_handle));
}

CUDAMotionInstance::~CUDAMotionInstance() noexcept {
    LUISA_CHECK_CUDA(cuMemFree(_motion_buffer));
}

void CUDAMotionInstance::build(CUDACommandEncoder &encoder,
                               MotionInstanceBuildCommand *command) noexcept {
    // note that we do not need to lock the mutex here,
    // since none of the members change during build
    LUISA_ASSERT(command->keyframes().size() == _option.keyframe_count,
                 "Keyframe count mismatch.");
    // allocate host-pinned memory for keyframes
    auto child = reinterpret_cast<CUDAPrimitive *>(command->child());
    {
        std::scoped_lock lock{_mutex};
        _child = child;
    }
    encoder.with_upload_buffer(
        _motion_buffer_size,
        [&encoder, option = _option, child,
         buffer = _motion_buffer, size = _motion_buffer_size,
         keys = command->steal_keyframes()](auto view) noexcept {
            struct MotionTransform {
                optix::TraversableHandle child;
                optix::MotionOptions motionOptions;
                unsigned int pad[3];
                [[no_unique_address]] float data[];
            };
            static_assert(sizeof(MotionTransform) == 32u);
            auto p = reinterpret_cast<MotionTransform *>(view->address());
            memset(p, 0, sizeof(MotionTransform));
            {
                p->child = child->handle();
                p->motionOptions.numKeys = option.keyframe_count;
                p->motionOptions.flags = optix::MOTION_FLAG_NONE;
                if (option.should_vanish_start) { p->motionOptions.flags |= optix::MOTION_FLAG_START_VANISH; }
                if (option.should_vanish_end) { p->motionOptions.flags |= optix::MOTION_FLAG_END_VANISH; }
                p->motionOptions.timeBegin = option.time_start;
                p->motionOptions.timeEnd = option.time_end;
            }
            switch (option.mode) {
                case AccelOption::MotionMode::MATRIX: {
                    auto m = reinterpret_cast<std::array<float, 12> *>(p->data);
                    // m is a 3x4 row-major matrix, while the input is 4x4 column-major
                    for (auto i = 0u; i < option.keyframe_count; i++) {
                        m[i][0] = keys[i].as_matrix()[0][0];
                        m[i][1] = keys[i].as_matrix()[1][0];
                        m[i][2] = keys[i].as_matrix()[2][0];
                        m[i][3] = keys[i].as_matrix()[3][0];
                        m[i][4] = keys[i].as_matrix()[0][1];
                        m[i][5] = keys[i].as_matrix()[1][1];
                        m[i][6] = keys[i].as_matrix()[2][1];
                        m[i][7] = keys[i].as_matrix()[3][1];
                        m[i][8] = keys[i].as_matrix()[0][2];
                        m[i][9] = keys[i].as_matrix()[1][2];
                        m[i][10] = keys[i].as_matrix()[2][2];
                        m[i][11] = keys[i].as_matrix()[3][2];
                    }
                    break;
                }
                case AccelOption::MotionMode::SRT: {
                    auto m = reinterpret_cast<optix::SRTData *>(p->data);
                    for (auto i = 0u; i < option.keyframe_count; i++) {
                        m[i].pvx = keys[i].as_srt().pivot[0];
                        m[i].pvy = keys[i].as_srt().pivot[1];
                        m[i].pvz = keys[i].as_srt().pivot[2];
                        m[i].qx = keys[i].as_srt().quaternion[0];
                        m[i].qy = keys[i].as_srt().quaternion[1];
                        m[i].qz = keys[i].as_srt().quaternion[2];
                        m[i].qw = keys[i].as_srt().quaternion[3];
                        m[i].sx = keys[i].as_srt().scale[0];
                        m[i].sy = keys[i].as_srt().scale[1];
                        m[i].sz = keys[i].as_srt().scale[2];
                        m[i].a = keys[i].as_srt().shear[0];
                        m[i].b = keys[i].as_srt().shear[1];
                        m[i].c = keys[i].as_srt().shear[2];
                        m[i].tx = keys[i].as_srt().translation[0];
                        m[i].ty = keys[i].as_srt().translation[1];
                        m[i].tz = keys[i].as_srt().translation[2];
                    }
                    break;
                }
                default:
                    LUISA_ERROR_WITH_LOCATION("Unsupported motion mode.");
            }
            LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
                buffer, view->address(), size, encoder.stream()->handle()));
        });
}

}// namespace luisa::compute::cuda
