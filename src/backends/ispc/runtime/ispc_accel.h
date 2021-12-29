#pragma once

#include <embree3/rtcore.h>
#include <rtx/accel.h>
#include <backends/ispc/runtime/ispc_mesh.h>
#include <vector>
#include <core/dirty_range.h>

using namespace luisa::compute;

namespace lc::ispc{

class ISPCAccel{
private:
    RTCScene _scene;
    AccelBuildHint _hint;
    luisa::vector<ISPCMesh*> _meshes;
    luisa::vector<float4x4> _mesh_transforms;
    luisa::vector<bool> _mesh_visibilities;
    luisa::vector<RTCGeometry> _mesh_instances;
    luisa::set<size_t> _dirty;
    

    inline void buildAllGeometry() noexcept;
    inline void updateAllGeometry() noexcept;
public:
    explicit ISPCAccel(AccelBuildHint hint, RTCDevice device) noexcept;
    ~ISPCAccel() noexcept;

    void addMesh(ISPCMesh* mesh, float4x4 transform, bool visible) noexcept;
    void setMesh(size_t index, ISPCMesh* mesh, float4x4 transform, bool visible) noexcept;
    void popMesh() noexcept;
    void setVisibility(size_t index, bool visible) noexcept;
    void setTransform(size_t index, float4x4 transform) noexcept;
    [[nodiscard]] bool usesResource(uint64_t handle) const noexcept;
    void build() noexcept;
    void update() noexcept;
    [[nodiscard]] auto getScene() const noexcept { return _scene; }
};

}