#include "simd_mesh.h"

#include <luisa/core/logging.h>

#include "simd_buffer.h"

namespace luisa::compute::simd {

SIMDMesh::SIMDMesh(
    RTCDevice device, const AccelOption &option) noexcept
    : _scene{rtcNewScene(device)},
      _geometry{rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE)},
      _motion{option.motion} {
    simd_accel_set_flags(_scene, option);
    rtcSetGeometryMask(_geometry, ~0u);
    rtcAttachGeometry(_scene, _geometry);
    rtcReleaseGeometry(_geometry);
}

SIMDMesh::~SIMDMesh() noexcept { rtcReleaseScene(_scene); }

void SIMDMesh::build(const MeshBuildCommand &command) noexcept {
    auto *vertices = reinterpret_cast<SIMDBuffer *>(
                         command.vertex_buffer())
                         ->data();
    auto *triangles = reinterpret_cast<SIMDBuffer *>(
                          command.triangle_buffer())
                          ->data();
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
    if (_motion) {
        LUISA_ASSERT(
            vertex_count % _motion.keyframe_count == 0u,
            "Invalid SIMD motion-mesh vertex count.");
        auto vertices_per_keyframe =
            vertex_count / _motion.keyframe_count;
        auto frame_stride =
            command.vertex_stride() * vertices_per_keyframe;
        rtcSetGeometryTimeRange(
            _geometry, _motion.time_start, _motion.time_end);
        rtcSetGeometryTimeStepCount(
            _geometry, _motion.keyframe_count);
        for (auto frame = 0u; frame < _motion.keyframe_count; frame++) {
            rtcSetSharedGeometryBuffer(
                _geometry, RTC_BUFFER_TYPE_VERTEX, frame,
                RTC_FORMAT_FLOAT3, vertices,
                command.vertex_buffer_offset() + frame * frame_stride,
                command.vertex_stride(), vertices_per_keyframe);
        }
    } else {
        rtcSetSharedGeometryBuffer(
            _geometry, RTC_BUFFER_TYPE_VERTEX, 0u,
            RTC_FORMAT_FLOAT3, vertices,
            command.vertex_buffer_offset(), command.vertex_stride(),
            vertex_count);
    }
    rtcSetSharedGeometryBuffer(
        _geometry, RTC_BUFFER_TYPE_INDEX, 0u,
        RTC_FORMAT_UINT3, triangles,
        command.triangle_buffer_offset(), sizeof(Triangle),
        triangle_count);
    rtcCommitGeometry(_geometry);
    rtcCommitScene(_scene);
}

}// namespace luisa::compute::simd
