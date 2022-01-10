#include <backends/ispc/runtime/ispc_accel.h>

namespace lc::ispc{

ISPCAccel::ISPCAccel(AccelBuildHint hint, RTCDevice device) noexcept : _hint(hint) {
    _scene = rtcNewScene(device);
    _device = device;
}

ISPCAccel::~ISPCAccel() noexcept {
    for(auto& instance: _mesh_instances) rtcReleaseGeometry(instance);
    rtcReleaseScene(_scene);
}

void ISPCAccel::addMesh(ISPCMesh* mesh, float4x4 transform, bool visible) noexcept{
    _meshes.emplace_back(mesh);
    _mesh_transforms.emplace_back(transform);
    _mesh_visibilities.emplace_back(visible);
    // _dirty.emplace(_meshes.size() - 1);
}

void ISPCAccel::setMesh(size_t index, ISPCMesh* mesh, float4x4 transform, bool visible) noexcept {
    if(_meshes[index] != mesh) {
        // TODO: detach instance
    }
    _meshes[index] = mesh;
    _mesh_transforms[index] = transform;
    _mesh_visibilities[index] = visible;
    // _dirty.emplace(index);
}

void ISPCAccel::popMesh() noexcept {
    _meshes.pop_back();
    _mesh_transforms.pop_back();
    _mesh_visibilities.pop_back();
    // TODO: detach instance
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
        auto& scene = _meshes[k]->scene;
        auto instance = rtcNewGeometry(_device, RTC_GEOMETRY_TYPE_INSTANCE);
        rtcSetGeometryInstancedScene(instance, scene);
        rtcSetGeometryTimeStepCount(instance, 1);
        rtcSetGeometryTransform(
            instance, 0, // timeStep = 0
            RTCFormat::RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, _mesh_transforms.data() + k
        );
        rtcSetGeometryMask(instance, _mesh_visibilities[k] ? 0xffffu : 0x0000u);
        rtcCommitGeometry(instance);
        rtcAttachGeometry(_scene, instance);
        _mesh_instances.push_back(instance);
        LUISA_INFO("Build {}", k);
    }
}

inline void ISPCAccel::updateAllGeometry() noexcept {
    for(auto& k : _dirty){
        rtcSetGeometryTransform(
            _mesh_instances[k], 0, // timeStep = 0
            RTCFormat::RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, _mesh_transforms.data() + k
        );
        rtcSetGeometryMask(_mesh_instances[k], _mesh_visibilities[k] ? 0xffffu : 0x0000u);
    }
    for(auto& instance : _mesh_instances) {
        rtcCommitGeometry(instance);
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