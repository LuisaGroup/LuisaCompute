//
// Created by mike on 4/8/26.
//

#include <limits>

#include "hip_check.h"
#include "hip_buffer.h"
#include "hip_command_encoder.h"
#include "hip_geometry_build_policy.h"
#include "hip_stream.h"
#include "hip_device.h"
#include "hip_procedural_primitive.h"

namespace luisa::compute::hip {

HIPProceduralPrimitive::HIPProceduralPrimitive(hiprtContext ctx, const AccelOption &option) noexcept
    : _option{option}, _hiprt_ctx{ctx} {
    LUISA_ASSERT(!_option.motion.is_enabled(),
                 "HIP procedural primitives do not yet support motion keyframes.");
}

HIPProceduralPrimitive::~HIPProceduralPrimitive() noexcept {
    if (_geometry) {
        // Geometry builds and traces are asynchronous, while HIPRT destruction
        // has no stream on which to order the deallocation.
        LUISA_CHECK_HIP(hipDeviceSynchronize());
        LUISA_CHECK_HIPRT(hiprtDestroyGeometry(_hiprt_ctx, _geometry));
    }
}

void HIPProceduralPrimitive::build(HIPCommandEncoder &encoder, ProceduralPrimitiveBuildCommand *command) noexcept {

    auto aabb_buffer = reinterpret_cast<const HIPBuffer *>(command->aabb_buffer());
    auto aabb_offset = command->aabb_buffer_offset();
    auto aabb_size = command->aabb_buffer_size();
    LUISA_ASSERT(aabb_offset <= aabb_buffer->size_bytes() &&
                     aabb_size <= aabb_buffer->size_bytes() - aabb_offset,
                 "AABB buffer offset + size exceeds buffer size {}.", aabb_buffer->size_bytes());
    LUISA_ASSERT(aabb_size % sizeof(AABB) == 0u,
                 "HIP procedural primitive buffer size {} is not divisible by {}.",
                 aabb_size, sizeof(AABB));
    auto aabb_count = aabb_size / sizeof(AABB);
    LUISA_ASSERT(aabb_count > 0u &&
                     aabb_count <= std::numeric_limits<uint32_t>::max(),
                 "HIP procedural primitives require a nonzero 32-bit AABB count.");

    std::scoped_lock lock{_mutex};

    auto new_aabb_buffer = static_cast<std::byte *>(aabb_buffer->handle()) + aabb_offset;

    auto requires_build =
        _geometry == nullptr ||
        !_option.allow_update ||
        command->request() == AccelBuildRequest::FORCE_BUILD ||
        reinterpret_cast<hipDeviceptr_t>(new_aabb_buffer) != _aabb_buffer ||
        aabb_size != _aabb_buffer_size;
    auto old_aabb_count = _aabb_buffer_size / sizeof(AABB);
    auto recreate_geometry =
        _geometry == nullptr || aabb_count != old_aabb_count ||
        (_option.allow_compaction && requires_build);

    _aabb_buffer = reinterpret_cast<hipDeviceptr_t>(new_aabb_buffer);
    _aabb_buffer_size = aabb_size;

    hiprtAABBListPrimitive aabb_prim{};
    aabb_prim.aabbs = reinterpret_cast<hiprtDevicePtr>(_aabb_buffer);
    aabb_prim.aabbCount = static_cast<uint32_t>(aabb_count);
    aabb_prim.aabbStride = sizeof(AABB);

    hiprtGeometryBuildInput build_input{};
    build_input.type = hiprtPrimitiveTypeAABBList;
    build_input.primitive.aabbList = aabb_prim;

    auto hip_stream = encoder.stream()->handle();

    if (requires_build) {
        auto policy = select_hiprt_geometry_build_policy(
            _hiprt_ctx, build_input, _option);
        recreate_geometry = recreate_geometry ||
                            (_geometry &&
                             policy.options.buildFlags != _build_flags);
        if (recreate_geometry) {
            if (_geometry) {
                LUISA_CHECK_HIP(hipStreamSynchronize(hip_stream));
                LUISA_CHECK_HIPRT(hiprtDestroyGeometry(_hiprt_ctx, _geometry));
                _geometry = nullptr;
            }
            LUISA_CHECK_HIPRT(hiprtCreateGeometry(_hiprt_ctx, build_input,
                                                  policy.options, _geometry));
            _build_flags = policy.options.buildFlags;
        }

        auto temp_buffer =
            encoder.stream()->rt_scratch_buffer(
                policy.temporary_buffer_size);

        LUISA_CHECK_HIPRT(hiprtBuildGeometry(_hiprt_ctx, hiprtBuildOperationBuild,
                                             build_input, policy.options,
                                             temp_buffer, hip_stream, _geometry));

        if (_option.allow_compaction) {
            LUISA_CHECK_HIPRT(hiprtCompactGeometry(
                _hiprt_ctx, hip_stream, _geometry, _geometry));
        }
    } else {
        hiprtBuildOptions build_options{};
        build_options.buildFlags = _build_flags;
        LUISA_CHECK_HIPRT(hiprtBuildGeometry(_hiprt_ctx, hiprtBuildOperationUpdate,
                                             build_input, build_options,
                                             nullptr, hip_stream, _geometry));
    }
}

}// namespace luisa::compute::hip
