#include <cstdlib>
#include <luisa/core/logging.h>

#include "fallback_buffer.h"
#include "fallback_mesh.h"

namespace luisa::compute::fallback {

[[nodiscard]] bool luisa_fallback_direct_static_mesh_enabled() noexcept {
    if (const char *env = std::getenv("LUISA_FALLBACK_DIRECT_STATIC_MESH");
        env != nullptr && env[0] != '\0') {
        return env[0] != '0';
    }
    return true;
}

FallbackMesh::FallbackMesh(RTCDevice device, const AccelOption &option) noexcept
    : FallbackPrim{device, RTC_GEOMETRY_TYPE_TRIANGLE, option} {}

void FallbackMesh::bind_shared_buffers(RTCGeometry dst_geom) const noexcept {
    if (auto m = motion()) {
        auto v_count_per_keyframe = _v_count / m.keyframe_count;
        auto v_buffer_stride = _v_stride * v_count_per_keyframe;
        rtcSetGeometryTimeRange(dst_geom, m.time_start, m.time_end);
        rtcSetGeometryTimeStepCount(dst_geom, m.keyframe_count);
        for (auto i = 0u; i < m.keyframe_count; i++) {
            rtcSetSharedGeometryBuffer(dst_geom, RTC_BUFFER_TYPE_VERTEX, i, RTC_FORMAT_FLOAT3,
                                       _v_buffer, _v_offset + i * v_buffer_stride,
                                       _v_stride, v_count_per_keyframe);
        }
    } else {
        rtcSetGeometryTimeStepCount(dst_geom, 1u);
        rtcSetSharedGeometryBuffer(dst_geom, RTC_BUFFER_TYPE_VERTEX, 0u, RTC_FORMAT_FLOAT3,
                                   _v_buffer, _v_offset, _v_stride, _v_count);
    }
    rtcSetSharedGeometryBuffer(dst_geom, RTC_BUFFER_TYPE_INDEX, 0u, RTC_FORMAT_UINT3,
                               _t_buffer, _t_offset, sizeof(Triangle), _t_count);
}

void FallbackMesh::build(luisa::unique_ptr<MeshBuildCommand> cmd) noexcept {
    auto v_buffer = static_cast<const std::byte *>(
        reinterpret_cast<FallbackBuffer *>(cmd->vertex_buffer())->data());
    auto t_buffer = static_cast<const std::byte *>(
        reinterpret_cast<FallbackBuffer *>(cmd->triangle_buffer())->data());
    LUISA_DEBUG_ASSERT(cmd->vertex_buffer_size() % cmd->vertex_stride() == 0u, "Invalid vertex buffer size.");
    LUISA_DEBUG_ASSERT(cmd->triangle_buffer_size() % sizeof(Triangle) == 0u, "Invalid triangle buffer size.");
    auto v_count = cmd->vertex_buffer_size() / cmd->vertex_stride();
    auto t_count = cmd->triangle_buffer_size() / sizeof(Triangle);
    if (auto m = motion()) {
        LUISA_DEBUG_ASSERT(v_count % m.keyframe_count == 0u, "Invalid vertex buffer size.");
    }
    {
        std::scoped_lock lock{_blas_mutex};
        _v_buffer = v_buffer;
        _v_offset = cmd->vertex_buffer_offset();
        _v_stride = cmd->vertex_stride();
        _v_count = v_count;
        _t_buffer = t_buffer;
        _t_offset = cmd->triangle_buffer_offset();
        _t_count = t_count;
        _blas_dirty = true;
        _build_revision.fetch_add(1u, std::memory_order_release);
    }
    // Mirror Cycles Geometry::need_build_bvh(BVH_LAYOUT_EMBREE): defer committing
    // the per-mesh BLAS until an RTC_GEOMETRY_TYPE_INSTANCE actually references
    // this mesh (shared/repeated instances, non-identity transforms, or motion).
    // Single-reference static identity meshes attached directly via
    // rtcSetSharedGeometryBuffer never pay BLAS build cost or BLAS memory.
    if (!luisa_fallback_direct_static_mesh_enabled() || motion() || option().allow_update) {
        (void)committed_handle();
    }
}

RTCScene FallbackMesh::committed_handle() const noexcept {
    std::scoped_lock lock{_blas_mutex};
    if (_blas_dirty) {
        const auto t0 = luisa_fallback_bvh_stats_enabled()
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        bind_shared_buffers(geometry());
        rtcCommitGeometry(geometry());
        rtcCommitScene(handle());
        _blas_dirty = false;
        if (luisa_fallback_bvh_stats_enabled()) {
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count();
            auto &s = luisa_fallback_bvh_stats();
            s.blas_commits.fetch_add(1u, std::memory_order_relaxed);
            s.blas_commit_ns.fetch_add(static_cast<uint64_t>(ns), std::memory_order_relaxed);
        }
    }
    return handle();
}

}// namespace luisa::compute::fallback
