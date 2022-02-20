//
// Created by Mike Smith on 2022/2/11.
//

#include <rtx/mesh.h>
#include <backends/ispc/ispc_mesh.h>

namespace luisa::compute::ispc {

ISPCMesh::ISPCMesh(
    RTCDevice device, AccelBuildHint hint,
    uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
    uint64_t t_buffer, size_t t_offset, size_t t_count) noexcept
    : _handle{rtcNewScene(device)},
      _geometry{rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE)},
      _v_buffer{v_buffer}, _t_buffer{t_buffer} {

    rtcSetSharedGeometryBuffer(
        _geometry, RTC_BUFFER_TYPE_VERTEX, 0u, RTC_FORMAT_FLOAT3,
        reinterpret_cast<const void *>(v_buffer),
        v_offset, v_stride, v_count);
    rtcSetSharedGeometryBuffer(
        _geometry, RTC_BUFFER_TYPE_INDEX, 0u, RTC_FORMAT_UINT3,
        reinterpret_cast<const void *>(t_buffer),
        t_offset, sizeof(Triangle), t_count);
    switch (hint) {
        case AccelBuildHint::FAST_TRACE:
            rtcSetGeometryBuildQuality(_geometry, RTC_BUILD_QUALITY_HIGH);
            break;
        case AccelBuildHint::FAST_UPDATE:
        case AccelBuildHint::FAST_REBUILD:
            rtcSetGeometryBuildQuality(_geometry, RTC_BUILD_QUALITY_REFIT);
            break;
    }
    rtcAttachGeometry(_handle, _geometry);
    rtcSetSceneBuildQuality(_handle, RTC_BUILD_QUALITY_HIGH);
    rtcSetSceneFlags(_handle, RTC_SCENE_FLAG_COMPACT);
}

ISPCMesh::~ISPCMesh() noexcept {
    rtcReleaseScene(_handle);
    rtcReleaseGeometry(_geometry);
}

void ISPCMesh::commit() noexcept {
    rtcCommitGeometry(_geometry);
    rtcCommitScene(_handle);
}

}// namespace luisa::compute::ispc
