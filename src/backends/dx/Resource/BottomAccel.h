#pragma once
#include <DXRuntime/Device.h>
#include <Resource/DefaultBuffer.h>
#include <Resource/Mesh.h>
#include <runtime/command.h>

namespace toolhub::directx {

class TopAccel;
class BottomAccel : public vstd::IOperatorNewBase {
    friend class TopAccel;
    vstd::optional<DefaultBuffer> accelBuffer;
    Device *device;
    Mesh mesh;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomStruct;
    D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc;

public:
    Mesh const *GetMesh() const { return &mesh; }
    Buffer const *GetAccelBuffer() const {
        return accelBuffer.GetPtr();
    }
    BottomAccel(
        Device *device,
        Buffer const *vHandle, size_t vOffset, size_t vStride, size_t vCount,
        Buffer const *iHandle, size_t iOffset, size_t iCount,
        luisa::compute::AccelBuildHint hint);
    size_t PreProcessStates(
        CommandBufferBuilder &builder,
        ResourceStateTracker &tracker,
        bool update);
    void UpdateStates(
        CommandBufferBuilder &builder,
        ResourceStateTracker &tracker,
        BufferView const& );
    ~BottomAccel();
};
}// namespace toolhub::directx