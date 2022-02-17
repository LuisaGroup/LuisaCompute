#pragma once
#include <DXRuntime/Device.h>
#include <Resource/DefaultBuffer.h>
#include <Resource/Mesh.h>
#include <DXRuntime/IUpdateState.h>
namespace toolhub::directx {
class Mesh;
class CommandBufferBuilder;
class ResourceStateTracker;
class TopAccel;
class BottomAccel : public vstd::IOperatorNewBase, public IUpdateState {
    friend class TopAccel;
    vstd::optional<DefaultBuffer> accelBuffer;
    Device *device;
    Mesh mesh;

public:
    Mesh const *GetMesh() const { return &mesh; }
    DefaultBuffer const *GetAccelBuffer() const {
        return accelBuffer ? accelBuffer.GetPtr() : nullptr;
    }
    BottomAccel(
        Device *device,
        Buffer const *vHandle, size_t vOffset, size_t vStride, size_t vCount,
        Buffer const *iHandle, size_t iOffset, size_t iCount);
    void PreProcessStates(
        CommandBufferBuilder &builder,
        ResourceStateTracker &tracker) const override;
    void UpdateStates(
        CommandBufferBuilder &builder,
        ResourceStateTracker &tracker) const override;
    ~BottomAccel();
};
}// namespace toolhub::directx