//
// Exact matrix and quaternion-SRT motion instances for HIPRT.
//

#include <cmath>
#include <cstring>
#include <limits>

#include <luisa/core/logging.h>

#include "hip_check.h"
#include "hip_command_encoder.h"
#include "hip_motion_instance.h"
#include "hip_stream.h"

namespace luisa::compute::hip {

namespace {

[[nodiscard]] constexpr size_t align_up(size_t size, size_t alignment) noexcept {
    return (size + alignment - 1u) & ~(alignment - 1u);
}

[[nodiscard]] uint64_t hiprt_instance_handle(const hiprtInstance &instance) noexcept {
    switch (instance.type) {
        case hiprtInstanceTypeGeometry:
            return reinterpret_cast<uint64_t>(instance.geometry);
        case hiprtInstanceTypeScene:
            return reinterpret_cast<uint64_t>(instance.scene);
    }
    LUISA_ERROR_WITH_LOCATION("Invalid HIPRT instance type.");
}

[[nodiscard]] bool same_hiprt_instance(
    const hiprtInstance &lhs, const hiprtInstance &rhs) noexcept {
    return lhs.type == rhs.type &&
           hiprt_instance_handle(lhs) == hiprt_instance_handle(rhs);
}

[[nodiscard]] hiprtBuildOptions motion_build_options() noexcept {
    hiprtBuildOptions options{};
    options.buildFlags = hiprtBuildFlagBitPreferFastBuild |
                         hiprtBuildFlagBitDisableSpatialSplits |
                         hiprtBuildFlagBitDisableOrientedBoundingBoxes;
    return options;
}

}// namespace

HIPMotionInstance::HIPMotionInstance(
    hiprtContext ctx, const AccelMotionOption &option) noexcept
    : _option{option}, _hiprt_ctx{ctx} {
    LUISA_ASSERT(option.mode == AccelMotionMode::MATRIX ||
                     option.mode == AccelMotionMode::SRT,
                 "HIP motion instances support MATRIX and SRT transforms only.");
    LUISA_ASSERT(option.keyframe_count >= 2u,
                 "HIP motion instances require at least two keyframes (got {}).",
                 option.keyframe_count);
    LUISA_ASSERT(!option.should_vanish_start && !option.should_vanish_end,
                 "HIP motion instances do not support start/end vanish flags.");
    LUISA_ASSERT(std::isfinite(option.time_start) &&
                     std::isfinite(option.time_end) &&
                     option.time_start < option.time_end,
                 "HIP motion instance time range must be finite and strictly increasing "
                 "(got [{}, {}]).",
                 option.time_start, option.time_end);

    static_assert(sizeof(hiprtFrameSRTQuaternion) == 80u);
    static_assert(alignof(hiprtFrameSRTQuaternion) == 16u);
    static_assert(offsetof(hiprtFrameSRTQuaternion, rotation) == 0u);
    static_assert(offsetof(hiprtFrameSRTQuaternion, pivot) == 16u);
    static_assert(offsetof(hiprtFrameSRTQuaternion, scale) == 28u);
    static_assert(offsetof(hiprtFrameSRTQuaternion, shear) == 40u);
    static_assert(offsetof(hiprtFrameSRTQuaternion, translation) == 52u);
    static_assert(offsetof(hiprtFrameSRTQuaternion, time) == 64u);
    auto frame_alignment = option.mode == AccelMotionMode::MATRIX ?
                               alignof(hiprtFrameMatrix) :
                               alignof(hiprtFrameSRTQuaternion);
    _frame_stride = option.mode == AccelMotionMode::MATRIX ?
                        sizeof(hiprtFrameMatrix) :
                        sizeof(hiprtFrameSRTQuaternion);
    LUISA_ASSERT(alignof(hiprtTransformHeader) <= frame_alignment);
    _transform_header_offset = align_up(sizeof(hiprtInstance), alignof(hiprtTransformHeader));
    _frame_offset = align_up(_transform_header_offset + sizeof(hiprtTransformHeader),
                             frame_alignment);
    LUISA_ASSERT(option.keyframe_count <=
                     (std::numeric_limits<size_t>::max() - _frame_offset) /
                         _frame_stride,
                 "HIP motion instance keyframe buffer size overflow.");
    auto frame_data_end = _frame_offset +
                          static_cast<size_t>(option.keyframe_count) * _frame_stride;
    _device_data_offset = align_up(
        frame_data_end, alignof(HIPMotionInstanceDeviceData));
    _input_buffer_size = _device_data_offset +
                         sizeof(HIPMotionInstanceDeviceData);
    LUISA_CHECK_HIP(hipMalloc(
        reinterpret_cast<void **>(&_input_buffer), _input_buffer_size));
}

HIPMotionInstance::~HIPMotionInstance() noexcept {
    if (_scene) {
        // Scene builds and traces are asynchronous, while HIPRT destruction and
        // hipFree below have no stream on which to order their deallocation.
        LUISA_CHECK_HIP(hipDeviceSynchronize());
    }
    if (_scene) {
        LUISA_CHECK_HIPRT(hiprtDestroyScene(_hiprt_ctx, _scene));
    }
    if (_input_buffer) {
        LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_input_buffer)));
    }
}

hiprtSceneBuildInput HIPMotionInstance::_make_build_input() const noexcept {
    auto base = reinterpret_cast<std::byte *>(_input_buffer);
    hiprtSceneBuildInput input{};
    input.instances = reinterpret_cast<hiprtDevicePtr>(base);
    input.instanceTransformHeaders = reinterpret_cast<hiprtDevicePtr>(
        base + _transform_header_offset);
    input.instanceFrames = reinterpret_cast<hiprtDevicePtr>(base + _frame_offset);
    input.instanceMasks = nullptr;
    input.instanceCount = 1u;
    input.frameCount = _option.keyframe_count;
    input.frameType = _option.mode == AccelMotionMode::MATRIX ?
                          hiprtFrameTypeMatrix :
                          hiprtFrameTypeSRTQuaternion;
    return input;
}

void HIPMotionInstance::build(
    HIPCommandEncoder &encoder, MotionInstanceBuildCommand *command) noexcept {
    std::scoped_lock lock{_mutex};

    auto keyframes = command->keyframes();
    LUISA_ASSERT(keyframes.size() == _option.keyframe_count,
                 "HIP motion instance keyframe count mismatch (expected {}, got {}).",
                 _option.keyframe_count, keyframes.size());

    auto child = reinterpret_cast<const HIPPrimitive *>(command->child());
    LUISA_ASSERT(child != nullptr, "HIP motion instance child is null.");
    auto child_binding = child->binding();
    LUISA_ASSERT(child_binding.instance.type == hiprtInstanceTypeGeometry,
                 "HIP motion instances must directly wrap a geometry.");
    LUISA_ASSERT(hiprt_instance_handle(child_binding.instance) != 0u,
                 "Cannot build a HIP motion instance from an unbuilt geometry.");
    // A refit may update transforms and bounds, but it cannot retarget the
    // existing instance leaf to a different HIPRT object.
    auto requires_full_build =
        _scene == nullptr ||
        !same_hiprt_instance(_built_child_instance, child_binding.instance);

    auto hip_stream = encoder.stream()->handle();
    encoder.with_upload_buffer(_input_buffer_size, [&](auto upload_buffer) noexcept {
        auto staging = static_cast<std::byte *>(upload_buffer->address());
        std::memset(staging, 0, _input_buffer_size);
        std::memcpy(staging, &child_binding.instance, sizeof(child_binding.instance));

        hiprtTransformHeader transform_header{};
        transform_header.frameIndex = 0u;
        transform_header.frameCount = _option.keyframe_count;
        std::memcpy(staging + _transform_header_offset,
                    &transform_header, sizeof(transform_header));

        auto denominator = static_cast<double>(_option.keyframe_count - 1u);
        auto previous_time = -std::numeric_limits<float>::infinity();
        hiprtFloat4 previous_quaternion{};
        auto has_previous_quaternion = false;
        for (auto i = 0u; i < _option.keyframe_count; i++) {
            auto alpha = static_cast<double>(i) / denominator;
            auto time = static_cast<float>(
                static_cast<double>(_option.time_start) +
                (static_cast<double>(_option.time_end) -
                 static_cast<double>(_option.time_start)) *
                    alpha);
            if (i == 0u) { time = _option.time_start; }
            if (i + 1u == _option.keyframe_count) { time = _option.time_end; }
            LUISA_ASSERT(std::isfinite(time) && time > previous_time,
                         "HIP motion instance keyframe times must remain finite and "
                         "strictly increasing after interpolation.");
            previous_time = time;

            auto destination = staging + _frame_offset +
                               static_cast<size_t>(i) * _frame_stride;
            if (_option.mode == AccelMotionMode::MATRIX) {
                const auto &source = keyframes[i].as_matrix();
                hiprtFrameMatrix frame{};
                for (auto row = 0u; row < 3u; row++) {
                    for (auto column = 0u; column < 4u; column++) {
                        auto value = source[column][row];
                        LUISA_ASSERT(std::isfinite(value),
                                     "HIP motion instance keyframe {} contains a non-finite "
                                     "matrix element at ({}, {}).",
                                     i, row, column);
                        frame.matrix[row][column] = value;
                    }
                }
                frame.time = time;
                std::memcpy(destination, &frame, sizeof(frame));
            } else {
                const auto &source = keyframes[i].as_srt();
                auto validate_finite = [i](const float *values, size_t count,
                                           const char *field) noexcept {
                    for (size_t j = 0u; j < count; j++) {
                        LUISA_ASSERT(std::isfinite(values[j]),
                                     "HIP SRT motion keyframe {} contains a non-finite "
                                     "{} component at index {}.",
                                     i, field, j);
                    }
                };
                validate_finite(source.pivot, 3u, "pivot");
                validate_finite(source.quaternion, 4u, "quaternion");
                validate_finite(source.scale, 3u, "scale");
                validate_finite(source.shear, 3u, "shear");
                validate_finite(source.translation, 3u, "translation");

                auto quaternion_norm_squared =
                    source.quaternion[0] * source.quaternion[0] +
                    source.quaternion[1] * source.quaternion[1] +
                    source.quaternion[2] * source.quaternion[2] +
                    source.quaternion[3] * source.quaternion[3];
                LUISA_ASSERT(quaternion_norm_squared > 0.0f &&
                                 std::abs(quaternion_norm_squared - 1.0f) <= 1e-3f,
                             "HIP SRT motion keyframe {} quaternion must be unit length "
                             "(squared length = {}).",
                             i, quaternion_norm_squared);

                hiprtFrameSRTQuaternion frame{};
                frame.rotation = {source.quaternion[0], source.quaternion[1],
                                  source.quaternion[2], source.quaternion[3]};
                frame.pivot = {source.pivot[0], source.pivot[1], source.pivot[2]};
                frame.scale = {source.scale[0], source.scale[1], source.scale[2]};
                frame.shear = {source.shear[0], source.shear[1], source.shear[2]};
                frame.translation = {source.translation[0], source.translation[1],
                                     source.translation[2]};
                frame.time = time;

                if (has_previous_quaternion) {
                    auto adjacent_dot =
                        previous_quaternion.x * frame.rotation.x +
                        previous_quaternion.y * frame.rotation.y +
                        previous_quaternion.z * frame.rotation.z +
                        previous_quaternion.w * frame.rotation.w;
                    LUISA_ASSERT(adjacent_dot > 0.0f,
                                 "HIP SRT motion keyframes {} and {} must use quaternion "
                                 "representatives less than 180 degrees apart "
                                 "(dot = {}).",
                                 i - 1u, i, adjacent_dot);
                }
                previous_quaternion = frame.rotation;
                has_previous_quaternion = true;
                std::memcpy(destination, &frame, sizeof(frame));
            }
        }

        auto frame_address = reinterpret_cast<uint64_t>(
            reinterpret_cast<std::byte *>(_input_buffer) + _frame_offset);
        HIPMotionInstanceDeviceData device_data{
            .frames = frame_address,
            .keyframe_count = _option.keyframe_count,
            .frame_stride = static_cast<uint32_t>(_frame_stride),
            .mode = static_cast<uint32_t>(_option.mode),
            .reserved = {0u, 0u, 0u}};
        std::memcpy(staging + _device_data_offset,
                    &device_data, sizeof(device_data));

        LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
            _input_buffer, staging, _input_buffer_size, hip_stream));
    });

    auto build_input = _make_build_input();
    auto build_options = motion_build_options();
    if (!requires_full_build) {
        LUISA_CHECK_HIPRT(hiprtBuildScene(
            _hiprt_ctx, hiprtBuildOperationUpdate,
            build_input, build_options, nullptr, hip_stream, _scene));
        _child = child;
        _built_child_instance = child_binding.instance;
        return;
    }

    if (!_scene) {
        LUISA_CHECK_HIPRT(hiprtCreateScene(
            _hiprt_ctx, build_input, build_options, _scene));
    }
    size_t temporary_buffer_size{};
    LUISA_CHECK_HIPRT(hiprtGetSceneBuildTemporaryBufferSize(
        _hiprt_ctx, build_input, build_options, temporary_buffer_size));
    hipDeviceptr_t temporary_buffer{};
    if (temporary_buffer_size > 0u) {
        LUISA_CHECK_HIP(hipMallocAsync(
            reinterpret_cast<void **>(&temporary_buffer),
            temporary_buffer_size, hip_stream));
    }
    LUISA_CHECK_HIPRT(hiprtBuildScene(
        _hiprt_ctx, hiprtBuildOperationBuild,
        build_input, build_options, temporary_buffer, hip_stream, _scene));
    if (temporary_buffer) {
        LUISA_CHECK_HIP(hipFreeAsync(
            reinterpret_cast<void *>(temporary_buffer), hip_stream));
    }
    _child = child;
    _built_child_instance = child_binding.instance;
}

HIPPrimitive::Binding HIPMotionInstance::binding() const noexcept {
    std::scoped_lock lock{_mutex};
    LUISA_ASSERT(_scene != nullptr && _child != nullptr,
                 "Cannot bind an unbuilt HIP motion instance.");
    auto binding = _child->binding();
    LUISA_ASSERT(binding.instance.type == hiprtInstanceTypeGeometry,
                 "HIP motion instance child is no longer a geometry.");
    LUISA_ASSERT(same_hiprt_instance(binding.instance, _built_child_instance),
                 "HIP motion instance child handle changed; rebuild the motion instance "
                 "before binding it to an acceleration structure.");
    binding.instance.type = hiprtInstanceTypeScene;
    binding.instance.scene = _scene;
    binding.motion_data = reinterpret_cast<uint64_t>(
        reinterpret_cast<std::byte *>(_input_buffer) + _device_data_offset);
    return binding;
}

void HIPMotionInstance::prepare_for_tlas_build(
    HIPCommandEncoder &encoder) const noexcept {
    std::scoped_lock lock{_mutex};
    LUISA_ASSERT(_scene != nullptr && _child != nullptr,
                 "Cannot refit an unbuilt HIP motion instance.");
    auto child_binding = _child->binding();
    LUISA_ASSERT(same_hiprt_instance(
                     child_binding.instance, _built_child_instance),
                 "HIP motion instance child changed before nested-scene refit.");
    auto build_input = _make_build_input();
    auto build_options = motion_build_options();
    LUISA_CHECK_HIPRT(hiprtBuildScene(
        _hiprt_ctx, hiprtBuildOperationUpdate,
        build_input, build_options, nullptr,
        encoder.stream()->handle(), _scene));
}

}// namespace luisa::compute::hip
