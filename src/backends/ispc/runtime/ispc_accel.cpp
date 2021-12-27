#include <backends/ispc/runtime/ispc_accel.h>

namespace lc::ispc{

ISPCAccel::ISPCAccel(AccelBuildHint hint):_hint(hint) {
    _device = rtcNewDevice(nullptr);
    _scene = rtcNewScene(_device);
}

ISPCAccel::~ISPCAccel() {
    for(auto& instance: _mesh_instances) rtcReleaseGeometry(instance);
    rtcReleaseScene(_scene);
    rtcReleaseDevice(_device);
}

void ISPCAccel::addMesh(ISPCMesh* mesh, float4x4 transform, bool visible){
    _meshes.emplace_back(mesh);
    _mesh_transforms.emplace_back(transform);
    _mesh_visibilities.emplace_back(visible);
}

void ISPCAccel::setMesh(size_t index, ISPCMesh* mesh, float4x4 transform, bool visible) noexcept {
    _meshes[index] = mesh;
    _mesh_transforms[index] = transform;
    _mesh_visibilities[index] = visible;
}

void ISPCAccel::popMesh() noexcept {
    _meshes.pop_back();
    _mesh_transforms.pop_back();
    _mesh_visibilities.pop_back();
}

void ISPCAccel::setVisibility(size_t index, bool visible) noexcept {
    _mesh_visibilities[index] = visible;
    _dirty.emplace(index);
}

void ISPCAccel::setTransform(size_t index, float4x4 transform) noexcept {
    _mesh_transforms[index] = transform;
    _dirty.emplace(index);
}

[[nodiscard]] bool ISPCAccel::usesResource(uint64_t handle) const noexcept {
    for(auto& mesh: _meshes){
        if(reinterpret_cast<uint64_t>(mesh) == handle) return true;
        if(mesh->_t_buffer == handle) return true;
        if(mesh->_v_buffer == handle) return true;
    }
    return false;
}

inline void ISPCAccel::buildAllGeometry() noexcept {
    for(int k=0;k<_meshes.size();k++){
        auto geometry = rtcNewGeometry(_device, RTCGeometryType::RTC_GEOMETRY_TYPE_TRIANGLE);
        rtcSetSharedGeometryBuffer(
            geometry, RTCBufferType::RTC_BUFFER_TYPE_VERTEX, 
            0, RTCFormat::RTC_FORMAT_FLOAT3, // slot need to be clarified
            (void *)_meshes[k]->_v_buffer,
            _meshes[k]->_v_offset,
            _meshes[k]->_v_stride,
            _meshes[k]->_v_count
        );
        rtcSetSharedGeometryBuffer(
            geometry, RTCBufferType::RTC_BUFFER_TYPE_INDEX, 
            0, RTCFormat::RTC_FORMAT_UINT3, // slot need to be clarified
            (void *)_meshes[k]->_t_buffer,
            _meshes[k]->_t_offset,
            sizeof(Triangle),
            _meshes[k]->_t_count
        );
        float transMat[16];
        for(int i=0;i<4;i++)
            for(int j=0;j<4;j++)
                transMat[i*4 + j] = _mesh_transforms[k][i][j];
        rtcSetGeometryTransform(
            geometry, 0, // timeStep = 0
            RTCFormat::RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, (void *)transMat
        );
        rtcSetGeometryMask(geometry, _mesh_visibilities[k] ? 0xffffu : 0x0000u);
        rtcCommitGeometry(geometry);
        auto id = rtcAttachGeometry(_scene, geometry);
        _mesh_instances.push_back(geometry);
    }
}

inline void ISPCAccel::updateAllGeometry() noexcept {
    for(auto& k : _dirty){
        auto& geometry = _mesh_instances[k];
        rtcSetSharedGeometryBuffer(
            geometry, RTCBufferType::RTC_BUFFER_TYPE_VERTEX, 
            0, RTCFormat::RTC_FORMAT_FLOAT3, // slot need to be clarified
            (void *)_meshes[k]->_v_buffer,
            _meshes[k]->_v_offset,
            _meshes[k]->_v_stride,
            _meshes[k]->_v_count
        );
        rtcSetSharedGeometryBuffer(
            geometry, RTCBufferType::RTC_BUFFER_TYPE_INDEX, 
            0, RTCFormat::RTC_FORMAT_UINT3, // slot need to be clarified
            (void *)_meshes[k]->_t_buffer,
            _meshes[k]->_t_offset,
            sizeof(Triangle),
            _meshes[k]->_t_count
        );
        float transMat[16];
        for(int i=0;i<4;i++)
            for(int j=0;j<4;j++)
                transMat[i*4 + j] = _mesh_transforms[k][i][j];
        rtcSetGeometryTransform(
            geometry, 0, // timeStep = 0
            RTCFormat::RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, (void *)transMat
        );
        rtcSetGeometryMask(geometry, _mesh_visibilities[k] ? 0xffffu : 0x0000u);
        rtcCommitGeometry(geometry);
    }
    _dirty.clear();
}

void ISPCAccel::build() noexcept{
    buildAllGeometry();
    rtcCommitScene(_scene);
}

void ISPCAccel::update() noexcept {
    updateAllGeometry();
    rtcCommitScene(_scene);
}

}