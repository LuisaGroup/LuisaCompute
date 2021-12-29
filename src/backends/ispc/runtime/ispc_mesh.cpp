#include <backends/ispc/runtime/ispc_mesh.h>

namespace lc::ispc {

    ISPCMesh::ISPCMesh(uint64_t v_buffer, size_t v_offset, size_t v_stride, size_t v_count,
        uint64_t t_buffer, size_t t_offset, size_t t_count, AccelBuildHint hint, RTCDevice device) noexcept :
        _v_buffer(v_buffer), _v_offset(v_offset), _v_stride(v_stride), _v_count(v_count),
        _t_buffer(t_buffer), _t_offset(t_offset), _t_count(t_count), _hint(hint), 
        geometry(rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE)) {
            LUISA_INFO("{} {} {} {}", v_buffer, v_offset, v_stride, v_count);
        }

    ISPCMesh::~ISPCMesh() noexcept {
        rtcReleaseGeometry(geometry);
    }

    void ISPCMesh::build() noexcept {
        rtcSetSharedGeometryBuffer(
            geometry, RTCBufferType::RTC_BUFFER_TYPE_VERTEX, 
            0, RTCFormat::RTC_FORMAT_FLOAT3, // slot need to be clarified
            (void *)_v_buffer,
            _v_offset,
            _v_stride,
            _v_count
        );
        rtcSetSharedGeometryBuffer(
            geometry, RTCBufferType::RTC_BUFFER_TYPE_INDEX, 
            0, RTCFormat::RTC_FORMAT_UINT3, // slot need to be clarified
            (void *)_t_buffer,
            _t_offset,
            sizeof(Triangle),
            _t_count
        );
        rtcCommitGeometry(geometry);
    }

    void ISPCMesh::update() noexcept {
        rtcSetSharedGeometryBuffer(
            geometry, RTCBufferType::RTC_BUFFER_TYPE_VERTEX, 
            0, RTCFormat::RTC_FORMAT_FLOAT3, // slot need to be clarified
            (void *)_v_buffer,
            _v_offset,
            _v_stride,
            _v_count
        );
        rtcSetSharedGeometryBuffer(
            geometry, RTCBufferType::RTC_BUFFER_TYPE_INDEX, 
            0, RTCFormat::RTC_FORMAT_UINT3, // slot need to be clarified
            (void *)_t_buffer,
            _t_offset,
            sizeof(Triangle),
            _t_count
        );
        rtcCommitGeometry(geometry);
    }

}