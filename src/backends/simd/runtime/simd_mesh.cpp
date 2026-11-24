#include "simd_mesh.h"

#include <cstdlib>

#include <luisa/core/logging.h>

#include "simd_buffer.h"

namespace luisa::compute::simd {

[[nodiscard]] bool simd_direct_static_mesh_enabled() noexcept {
    if (const char *env = std::getenv("LUISA_SIMD_DIRECT_STATIC_MESH");
        env != nullptr && env[0] != '\0') {
        return env[0] != '0';
    }
    return true;
}

SIMDMesh::SIMDMesh(
    RTCDevice device, const AccelOption &option) noexcept
    : SIMDPrimitive{Kind::mesh},
      _scene{rtcNewScene(device)},
      _geometry{rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE)},
      _option{option} {
    simd_accel_set_flags(_scene, option);
    rtcSetGeometryBuildQuality(
        _geometry, simd_triangle_geometry_build_quality(option));
    simd_geometry_enable_argument_filter(_geometry);
    rtcSetGeometryMask(_geometry, ~0u);
    rtcAttachGeometry(_scene, _geometry);
    rtcReleaseGeometry(_geometry);
}

SIMDMesh::~SIMDMesh() noexcept {
    {
        std::scoped_lock lock{_blas_mutex};
        _imported_bvh.restore_and_release();
    }
    rtcReleaseScene(_scene);
}

void SIMDMesh::bind_shared_buffers(RTCGeometry dst_geometry) const noexcept {
    if (_option.motion) {
        auto vertices_per_keyframe =
            _v_count / _option.motion.keyframe_count;
        auto frame_stride = _v_stride * vertices_per_keyframe;
        rtcSetGeometryTimeRange(
            dst_geometry, _option.motion.time_start, _option.motion.time_end);
        rtcSetGeometryTimeStepCount(
            dst_geometry, _option.motion.keyframe_count);
        for (auto frame = 0u; frame < _option.motion.keyframe_count; frame++) {
            rtcSetSharedGeometryBuffer(
                dst_geometry, RTC_BUFFER_TYPE_VERTEX, frame,
                RTC_FORMAT_FLOAT3, _v_buffer,
                _v_offset + frame * frame_stride,
                _v_stride, vertices_per_keyframe);
        }
    } else {
        rtcSetGeometryTimeStepCount(dst_geometry, 1u);
        rtcSetSharedGeometryBuffer(
            dst_geometry, RTC_BUFFER_TYPE_VERTEX, 0u,
            RTC_FORMAT_FLOAT3, _v_buffer,
            _v_offset, _v_stride, _v_count);
    }
    rtcSetSharedGeometryBuffer(
        dst_geometry, RTC_BUFFER_TYPE_INDEX, 0u,
        RTC_FORMAT_UINT3, _t_buffer,
        _t_offset, sizeof(Triangle), _t_count);
}

void SIMDMesh::build(const MeshBuildCommand &command) noexcept {
    auto *vertices = static_cast<const std::byte *>(
        reinterpret_cast<SIMDBuffer *>(command.vertex_buffer())->data());
    auto *triangles = static_cast<const std::byte *>(
        reinterpret_cast<SIMDBuffer *>(command.triangle_buffer())->data());
    LUISA_ASSERT(
        command.vertex_stride() != 0u &&
            command.vertex_buffer_size() % command.vertex_stride() == 0u,
        "Invalid SIMD mesh vertex buffer size or stride.");
    LUISA_ASSERT(
        command.triangle_buffer_size() % sizeof(Triangle) == 0u,
        "Invalid SIMD mesh triangle buffer size.");
    auto vertex_count =
        command.vertex_buffer_size() / command.vertex_stride();
    auto triangle_count =
        command.triangle_buffer_size() / sizeof(Triangle);
    if (_option.motion) {
        LUISA_ASSERT(
            vertex_count % _option.motion.keyframe_count == 0u,
            "Invalid SIMD motion-mesh vertex count.");
    }
    {
        std::scoped_lock lock{_blas_mutex};
        _imported_bvh.restore_and_release();
        _v_buffer = vertices;
        _v_offset = command.vertex_buffer_offset();
        _v_stride = command.vertex_stride();
        _v_count = vertex_count;
        _t_buffer = triangles;
        _t_offset = command.triangle_buffer_offset();
        _t_count = triangle_count;
        _blas_dirty = true;
        _built = true;
        _is_imported_blas.store(false, std::memory_order_release);
        _build_revision.fetch_add(1u, std::memory_order_release);
    }
    if (!simd_direct_static_mesh_enabled() ||
        _option.motion || _option.allow_update) {
        static_cast<void>(handle());
    }
}

bool SIMDMesh::import_from_archive(
    const EmbreeResolvedTriangleGeometry &geom,
    luisa::span<const std::byte> bvh_payload,
    const EmbreeOutOfTreeABI &abi,
    size_t &out_unpacked_bytes,
    EmbreeArchiveStatus &out_status) noexcept {
    std::scoped_lock lock{_blas_mutex};
    EmbreeTriangleBufferBinding binding{_v_buffer, _v_offset, _v_stride, _v_count, _t_buffer, _t_offset, _t_count};
    auto sync_binding = [&](const EmbreeTriangleBufferBinding &b) noexcept {
        _v_buffer = b.v_buffer;
        _v_offset = b.v_offset;
        _v_stride = b.v_stride;
        _v_count = b.v_count;
        _t_buffer = b.t_buffer;
        _t_offset = b.t_offset;
        _t_count = b.t_count;
    };
    const bool ok = embree_mesh_import_from_archive_locked(
        abi,
        _scene,
        _geometry,
        geom,
        bvh_payload,
        _imported_bvh,
        binding,
        _blas_dirty,
        _built,
        _is_imported_blas,
        _build_revision,
        [&](const EmbreeTriangleBufferBinding &old_b) noexcept {
            sync_binding(old_b);
            bind_shared_buffers(_geometry);
            rtcCommitGeometry(_geometry);
        },
        out_unpacked_bytes,
        out_status);
    sync_binding(binding);
    return ok;
}

RTCScene SIMDMesh::handle() const noexcept {
    std::scoped_lock lock{_blas_mutex};
    if (_blas_dirty) {
        bind_shared_buffers(_geometry);
        rtcCommitGeometry(_geometry);
        rtcCommitScene(_scene);
        _blas_dirty = false;
    }
    return _scene;
}

}// namespace luisa::compute::simd
