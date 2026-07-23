//
// Created by mike on 1/30/26.
//

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <luisa/core/pool.h>
#include <luisa/runtime/rtx/aabb.h>

#include "hip_check.h"
#include "hip_buffer.h"
#include "hip_command_encoder.h"
#include "hip_motion_mesh_builtin.h"
#include "hip_stage_buffer_pool.h"
#include "hip_stream.h"
#include "hip_device.h"
#include "hip_mesh.h"

namespace luisa::compute::hip {

namespace {

class HIPMotionMeshValidationCallback final : public HIPCallbackContext {

private:
    HIPStageBufferPool::View *_host_status;
    uint32_t _vertices_per_keyframe;

    [[nodiscard]] static auto &_pool() noexcept {
        static Pool<HIPMotionMeshValidationCallback, true> pool;
        return pool;
    }

public:
    HIPMotionMeshValidationCallback(
        HIPStageBufferPool::View *host_status,
        uint32_t vertices_per_keyframe) noexcept
        : _host_status{host_status},
          _vertices_per_keyframe{vertices_per_keyframe} {}

    [[nodiscard]] static auto create(
        HIPStageBufferPool::View *host_status,
        uint32_t vertices_per_keyframe) noexcept {
        return _pool().create(host_status, vertices_per_keyframe);
    }

    void recycle() noexcept override {
        HIPMotionMeshValidationStatus status{};
        std::memcpy(&status, _host_status->address(), sizeof(status));
        auto vertices_per_keyframe = _vertices_per_keyframe;
        _host_status->recycle();
        _pool().destroy(this);
        if (status.valid()) { return; }
        switch (status.error()) {
            case HIPMotionMeshValidationError::NON_FINITE_POSITION: {
                auto flat_vertex = status.element_index();
                auto keyframe = flat_vertex / vertices_per_keyframe;
                auto local_vertex = flat_vertex % vertices_per_keyframe;
                LUISA_ERROR_WITH_LOCATION(
                    "HIP motion mesh keyframe {} vertex {} has a non-finite "
                    "position component at axis {}.",
                    keyframe, local_vertex, status.component());
            }
            case HIPMotionMeshValidationError::INVALID_TRIANGLE_INDEX:
                LUISA_ERROR_WITH_LOCATION(
                    "HIP motion mesh triangle {} has an invalid local vertex "
                    "index at corner {} (each keyframe has {} vertices).",
                    status.element_index(), status.component(),
                    vertices_per_keyframe);
            default:
                LUISA_ERROR_WITH_LOCATION(
                    "HIP motion mesh preprocessing reported unknown validation "
                    "error {} (element = {}, component = {}).",
                    luisa::to_underlying(status.error()),
                    status.element_index(), status.component());
        }
    }
};

}// namespace

HIPMesh::HIPMesh(hiprtContext ctx, const AccelOption &option) noexcept
    : _option{option}, _hiprt_ctx{ctx} {
    if (_option.motion.is_enabled()) {
        auto time_span = _option.motion.time_end - _option.motion.time_start;
        LUISA_ASSERT(std::isfinite(_option.motion.time_start) &&
                         std::isfinite(_option.motion.time_end) &&
                         _option.motion.time_start < _option.motion.time_end &&
                         std::isfinite(time_span),
                     "HIP motion mesh time range must be finite and strictly increasing "
                     "with a finite representable span (got [{}, {}]).",
                     _option.motion.time_start, _option.motion.time_end);
    }
}

HIPMesh::~HIPMesh() noexcept {
    if (_geometry || _motion_vertex_buffer || _motion_triangle_buffer ||
        _motion_aabb_buffer || _motion_device_data) {
        // Geometry builds and traces are asynchronous, while HIPRT destruction
        // and the backing-allocation frees have no stream on which to order.
        LUISA_CHECK_HIP(hipDeviceSynchronize());
    }
    if (_geometry) {
        LUISA_CHECK_HIPRT(hiprtDestroyGeometry(_hiprt_ctx, _geometry));
    }
    if (_motion_vertex_buffer) {
        LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_motion_vertex_buffer)));
    }
    if (_motion_triangle_buffer) {
        LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_motion_triangle_buffer)));
    }
    if (_motion_aabb_buffer) {
        LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_motion_aabb_buffer)));
    }
    if (_motion_device_data) {
        LUISA_CHECK_HIP(hipFree(reinterpret_cast<void *>(_motion_device_data)));
    }
}

void HIPMesh::build(HIPCommandEncoder &encoder, MeshBuildCommand *command) noexcept {

    auto vertex_buffer = reinterpret_cast<const HIPBuffer *>(command->vertex_buffer());
    auto triangle_buffer = reinterpret_cast<const HIPBuffer *>(command->triangle_buffer());
    auto vertex_offset = command->vertex_buffer_offset();
    auto vertex_size = command->vertex_buffer_size();
    auto vertex_stride = command->vertex_stride();
    auto triangle_offset = command->triangle_buffer_offset();
    auto triangle_size = command->triangle_buffer_size();
    LUISA_ASSERT(vertex_offset <= vertex_buffer->size_bytes() &&
                     vertex_size <= vertex_buffer->size_bytes() - vertex_offset,
                 "Vertex buffer offset + size exceeds buffer size {}.", vertex_buffer->size_bytes());
    LUISA_ASSERT(triangle_offset <= triangle_buffer->size_bytes() &&
                     triangle_size <= triangle_buffer->size_bytes() - triangle_offset,
                 "Triangle buffer offset + size exceeds buffer size {}.", triangle_buffer->size_bytes());
    constexpr auto position_size = sizeof(float) * 3u;
    LUISA_ASSERT(vertex_stride >= position_size &&
                     vertex_stride <= std::numeric_limits<uint32_t>::max(),
                 "Invalid HIP mesh vertex stride {}.", vertex_stride);
    LUISA_ASSERT(vertex_size % vertex_stride == 0u,
                 "HIP mesh vertex buffer size {} is not divisible by stride {}.",
                 vertex_size, vertex_stride);
    LUISA_ASSERT(triangle_size % sizeof(Triangle) == 0u,
                 "HIP mesh triangle buffer size {} is not divisible by {}.",
                 triangle_size, sizeof(Triangle));
    if (!_option.motion.is_enabled()) {
        LUISA_ASSERT(vertex_offset % alignof(float) == 0u &&
                         vertex_stride % alignof(float) == 0u,
                     "Static HIP mesh vertex offset {} and stride {} must be "
                     "{}-byte aligned for HIPRT.",
                     vertex_offset, vertex_stride, alignof(float));
        LUISA_ASSERT(triangle_offset % alignof(Triangle) == 0u,
                     "Static HIP mesh triangle offset {} must be {}-byte aligned "
                     "for HIPRT.",
                     triangle_offset, alignof(Triangle));
    }

    auto vertex_count = vertex_size / vertex_stride;
    auto triangle_count = triangle_size / sizeof(Triangle);
    auto motion_keyframe_count = _option.motion.is_enabled() ?
                                     static_cast<size_t>(_option.motion.keyframe_count) :
                                     1u;
    LUISA_ASSERT(vertex_count % motion_keyframe_count == 0u,
                 "HIP motion mesh vertex count {} is not divisible by its {} keyframes.",
                 vertex_count, motion_keyframe_count);
    auto vertices_per_keyframe = vertex_count / motion_keyframe_count;
    LUISA_ASSERT(vertices_per_keyframe >= 3u &&
                     vertices_per_keyframe <= std::numeric_limits<uint32_t>::max() &&
                     triangle_count > 0u &&
                     triangle_count <= std::numeric_limits<uint32_t>::max(),
                 "HIP meshes require at least three 32-bit-addressable vertices per "
                 "keyframe and a nonzero 32-bit triangle count.");

    std::scoped_lock lock{_mutex};

    auto new_vertex_buffer = static_cast<std::byte *>(vertex_buffer->handle()) + vertex_offset;
    auto new_triangle_buffer = static_cast<std::byte *>(triangle_buffer->handle()) + triangle_offset;

    auto requires_build =
        _geometry == nullptr ||
        !_option.allow_update ||
        command->request() == AccelBuildRequest::FORCE_BUILD ||
        new_vertex_buffer != _vertex_buffer ||
        vertex_size != _vertex_buffer_size ||
        vertex_stride != _vertex_stride ||
        new_triangle_buffer != _triangle_buffer ||
        triangle_size != _triangle_buffer_size;
    // HIPRT storage is sized by primitive count. Compacted allocations have no
    // spare capacity for another full build and must always be recreated.
    auto old_triangle_count = _triangle_buffer_size / sizeof(Triangle);
    auto recreate_geometry =
        _geometry == nullptr || triangle_count != old_triangle_count ||
        (_option.allow_compaction && requires_build);

    _vertex_buffer = new_vertex_buffer;
    _vertex_buffer_size = vertex_size;
    _vertex_stride = vertex_stride;
    _triangle_buffer = new_triangle_buffer;
    _triangle_buffer_size = triangle_size;

    if (_option.motion.is_enabled()) {
        auto hip_stream = encoder.stream()->handle();

        // HIPRT has no native deforming-triangle primitive. Build one custom
        // AABB leaf per triangle around the union of every keyframe instead.
        // Recreating a HIPRT allocation is rare (primitive-count changes or
        // explicit compaction). Destruction has no stream parameter, so retain
        // a barrier only on that path; ordinary builds and updates stay fully
        // asynchronous.
        if (requires_build && recreate_geometry && _geometry) {
            LUISA_CHECK_HIP(hipStreamSynchronize(hip_stream));
            LUISA_CHECK_HIPRT(hiprtDestroyGeometry(_hiprt_ctx, _geometry));
            _geometry = nullptr;
        }

        // GPU preprocessing snapshots the exact positions and indices used by
        // custom intersection into mesh-owned allocations, while computing the
        // swept bounds from that same snapshot. This preserves source-resource
        // independence without a bulk readback or host synchronization.
        constexpr auto packed_position_size = sizeof(float) * 3u;
        static_assert(sizeof(AABB) == sizeof(float) * 6u);
        auto required_vertex_size = vertex_count * packed_position_size;
        if (_motion_vertex_buffer_size < required_vertex_size) {
            if (_motion_vertex_buffer) {
                LUISA_CHECK_HIP(hipFreeAsync(
                    reinterpret_cast<void *>(_motion_vertex_buffer), hip_stream));
            }
            LUISA_CHECK_HIP(hipMallocAsync(
                reinterpret_cast<void **>(&_motion_vertex_buffer),
                required_vertex_size, hip_stream));
            _motion_vertex_buffer_size = required_vertex_size;
        }
        auto required_triangle_size = triangle_count * sizeof(Triangle);
        if (_motion_triangle_buffer_size < required_triangle_size) {
            if (_motion_triangle_buffer) {
                LUISA_CHECK_HIP(hipFreeAsync(
                    reinterpret_cast<void *>(_motion_triangle_buffer), hip_stream));
            }
            LUISA_CHECK_HIP(hipMallocAsync(
                reinterpret_cast<void **>(&_motion_triangle_buffer),
                required_triangle_size, hip_stream));
            _motion_triangle_buffer_size = required_triangle_size;
        }
        auto required_aabb_size = triangle_count * sizeof(AABB);
        if (_motion_aabb_buffer_size < required_aabb_size) {
            if (_motion_aabb_buffer) {
                LUISA_CHECK_HIP(hipFreeAsync(
                    reinterpret_cast<void *>(_motion_aabb_buffer), hip_stream));
            }
            LUISA_CHECK_HIP(hipMallocAsync(
                reinterpret_cast<void **>(&_motion_aabb_buffer),
                required_aabb_size, hip_stream));
            _motion_aabb_buffer_size = required_aabb_size;
        }
        if (!_motion_device_data) {
            LUISA_CHECK_HIP(hipMallocAsync(
                reinterpret_cast<void **>(&_motion_device_data),
                sizeof(HIPMotionTriangleDeviceData), hip_stream));
        }

        hipDeviceptr_t validation_status{};
        LUISA_CHECK_HIP(hipMallocAsync(
            reinterpret_cast<void **>(&validation_status),
            sizeof(HIPMotionMeshValidationStatus), hip_stream));
        auto &preprocess = encoder.stream()->device()->motion_mesh_builtin();
        preprocess.reset_validation_status(hip_stream, validation_status);
        preprocess.pack_motion_vertices(
            hip_stream, _vertex_buffer, _motion_vertex_buffer,
            vertex_count, static_cast<uint32_t>(vertex_stride),
            validation_status);
        preprocess.build_motion_triangle_aabbs(
            hip_stream, _triangle_buffer, _motion_vertex_buffer,
            _motion_triangle_buffer, _motion_aabb_buffer,
            static_cast<uint32_t>(triangle_count),
            static_cast<uint32_t>(vertices_per_keyframe),
            static_cast<uint32_t>(motion_keyframe_count),
            validation_status);

        uint32_t motion_flags = 0u;
        if (_option.motion.should_vanish_start) { motion_flags |= 1u << 0u; }
        if (_option.motion.should_vanish_end) { motion_flags |= 1u << 1u; }
        HIPMotionTriangleDeviceData device_data{
            .vertices = reinterpret_cast<uint64_t>(_motion_vertex_buffer),
            .triangles = reinterpret_cast<uint64_t>(_motion_triangle_buffer),
            .vertex_stride = packed_position_size,
            .vertices_per_keyframe = static_cast<uint32_t>(vertices_per_keyframe),
            .keyframe_count = static_cast<uint32_t>(motion_keyframe_count),
            .flags = motion_flags,
            .time_start = _option.motion.time_start,
            .time_end = _option.motion.time_end};

        encoder.with_upload_buffer(sizeof(device_data), [&](auto upload_buffer) noexcept {
            auto upload = upload_buffer->address();
            std::memcpy(upload, &device_data, sizeof(device_data));
            LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
                _motion_device_data, upload,
                sizeof(device_data), hip_stream));
        });

        hiprtAABBListPrimitive aabb_primitive{};
        aabb_primitive.aabbs = reinterpret_cast<hiprtDevicePtr>(
            _motion_aabb_buffer);
        aabb_primitive.aabbCount = static_cast<uint32_t>(triangle_count);
        aabb_primitive.aabbStride = sizeof(AABB);

        hiprtGeometryBuildInput build_input{};
        build_input.type = hiprtPrimitiveTypeAABBList;
        build_input.geomType = 0u;
        build_input.primitive.aabbList = aabb_primitive;

        hiprtBuildOptions build_options{};
        build_options.buildFlags = make_hiprt_build_flags(_option);

        if (requires_build) {
            if (!_geometry) {
                LUISA_CHECK_HIPRT(hiprtCreateGeometry(
                    _hiprt_ctx, build_input, build_options, _geometry));
            }
            size_t temporary_buffer_size{};
            LUISA_CHECK_HIPRT(hiprtGetGeometryBuildTemporaryBufferSize(
                _hiprt_ctx, build_input, build_options,
                temporary_buffer_size));
            hipDeviceptr_t temporary_buffer{};
            if (temporary_buffer_size > 0u) {
                LUISA_CHECK_HIP(hipMallocAsync(
                    reinterpret_cast<void **>(&temporary_buffer),
                    temporary_buffer_size, hip_stream));
            }
            LUISA_CHECK_HIPRT(hiprtBuildGeometry(
                _hiprt_ctx, hiprtBuildOperationBuild,
                build_input, build_options, temporary_buffer,
                hip_stream, _geometry));
            if (temporary_buffer) {
                LUISA_CHECK_HIP(hipFreeAsync(
                    reinterpret_cast<void *>(temporary_buffer), hip_stream));
            }
            if (_option.allow_compaction) {
                LUISA_CHECK_HIPRT(hiprtCompactGeometry(
                    _hiprt_ctx, hip_stream, _geometry, _geometry));
            }
        } else {
            LUISA_CHECK_HIPRT(hiprtBuildGeometry(
                _hiprt_ctx, hiprtBuildOperationUpdate,
                build_input, build_options, nullptr,
                hip_stream, _geometry));
        }

        // Report deterministic device-side validation after all preceding
        // stream work has completed. Invalid inputs were sanitized by the
        // kernels, so the asynchronous HIPRT build remains memory-safe.
        auto host_status = encoder.stream()->download_pool()->allocate(
            sizeof(HIPMotionMeshValidationStatus));
        LUISA_ASSERT(host_status != nullptr,
                     "Failed to allocate HIP motion-mesh validation staging memory.");
        LUISA_CHECK_HIP(hipMemcpyDtoHAsync(
            host_status->address(), validation_status,
            sizeof(HIPMotionMeshValidationStatus), hip_stream));
        LUISA_CHECK_HIP(hipFreeAsync(
            reinterpret_cast<void *>(validation_status), hip_stream));
        encoder.add_callback(HIPMotionMeshValidationCallback::create(
            host_status, static_cast<uint32_t>(vertices_per_keyframe)));
        return;
    }

    hiprtTriangleMeshPrimitive mesh_prim{};
    mesh_prim.vertices = reinterpret_cast<hiprtDevicePtr>(_vertex_buffer);
    mesh_prim.vertexCount = static_cast<uint32_t>(vertex_count);
    mesh_prim.vertexStride = static_cast<uint32_t>(_vertex_stride);
    mesh_prim.triangleIndices = reinterpret_cast<hiprtDevicePtr>(_triangle_buffer);
    mesh_prim.triangleCount = static_cast<uint32_t>(triangle_count);
    mesh_prim.triangleStride = sizeof(Triangle);

    hiprtGeometryBuildInput build_input{};
    build_input.type = hiprtPrimitiveTypeTriangleMesh;
    build_input.primitive.triangleMesh = mesh_prim;

    hiprtBuildOptions build_options{};
    build_options.buildFlags = make_hiprt_build_flags(_option);

    auto hip_stream = encoder.stream()->handle();

    if (requires_build) {
        if (recreate_geometry) {
            if (_geometry) {
                LUISA_CHECK_HIP(hipStreamSynchronize(hip_stream));
                LUISA_CHECK_HIPRT(hiprtDestroyGeometry(_hiprt_ctx, _geometry));
                _geometry = nullptr;
            }
            LUISA_CHECK_HIPRT(hiprtCreateGeometry(_hiprt_ctx, build_input,
                                                  build_options, _geometry));
        }

        size_t temp_size = 0;
        LUISA_CHECK_HIPRT(hiprtGetGeometryBuildTemporaryBufferSize(_hiprt_ctx, build_input, build_options, temp_size));

        hipDeviceptr_t temp_buffer{};
        if (temp_size > 0) {
            LUISA_CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&temp_buffer), temp_size, hip_stream));
        }

        LUISA_CHECK_HIPRT(hiprtBuildGeometry(_hiprt_ctx, hiprtBuildOperationBuild,
                                             build_input, build_options,
                                             temp_buffer, hip_stream, _geometry));

        if (temp_buffer) {
            LUISA_CHECK_HIP(hipFreeAsync(reinterpret_cast<void *>(temp_buffer), hip_stream));
        }
        if (_option.allow_compaction) {
            LUISA_CHECK_HIPRT(hiprtCompactGeometry(
                _hiprt_ctx, hip_stream, _geometry, _geometry));
        }
    } else {
        LUISA_CHECK_HIPRT(hiprtBuildGeometry(_hiprt_ctx, hiprtBuildOperationUpdate,
                                             build_input, build_options,
                                             nullptr, hip_stream, _geometry));
    }
}

}// namespace luisa::compute::hip
