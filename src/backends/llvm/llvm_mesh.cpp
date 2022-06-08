//
// Created by Mike Smith on 2022/2/11.
//

#include <rtx/mesh.h>
#include <backends/llvm/llvm_mesh.h>

namespace luisa::compute::llvm {

LLVMMesh::LLVMMesh(
    RTCDevice device, AccelUsageHint hint,
    uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
    uint64_t t_buffer, size_t t_offset, size_t t_count) noexcept
    : _handle{rtcNewScene(device)},
      _geometry{rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE)},
      _v_buffer{v_buffer}, _v_offset{v_offset}, _v_stride{v_stride}, _v_count{v_count},
      _t_buffer{t_buffer}, _t_offset{t_offset}, _t_count{t_count}, _hint{hint} {}

LLVMMesh::~LLVMMesh() noexcept {
    rtcReleaseScene(_handle);
    rtcReleaseGeometry(_geometry);
}

void LLVMMesh::commit() noexcept {
    if (!_buffers_already_set.exchange(true)) [[unlikely]] {
        rtcSetSharedGeometryBuffer(
            _geometry, RTC_BUFFER_TYPE_VERTEX, 0u, RTC_FORMAT_FLOAT3,
            reinterpret_cast<const void *>(_v_buffer),
            _v_offset, _v_stride, _v_count);
        rtcSetSharedGeometryBuffer(
            _geometry, RTC_BUFFER_TYPE_INDEX, 0u, RTC_FORMAT_UINT3,
            reinterpret_cast<const void *>(_t_buffer),
            _t_offset, sizeof(Triangle), _t_count);
        switch (_hint) {
            case AccelUsageHint::FAST_TRACE:
                rtcSetGeometryBuildQuality(_geometry, RTC_BUILD_QUALITY_HIGH);
                break;
            case AccelUsageHint::FAST_UPDATE:
            case AccelUsageHint::FAST_BUILD:
                rtcSetGeometryBuildQuality(_geometry, RTC_BUILD_QUALITY_REFIT);
                break;
        }
        rtcAttachGeometry(_handle, _geometry);
        rtcSetSceneBuildQuality(_handle, RTC_BUILD_QUALITY_HIGH);
        rtcSetSceneFlags(_handle, RTC_SCENE_FLAG_COMPACT);
    }
    rtcCommitGeometry(_geometry);
    rtcCommitScene(_handle);
}

}// namespace luisa::compute::llvm
