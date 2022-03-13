#pragma once
#include <DXRuntime/Device.h>
#include <EASTL/shared_ptr.h>
#include <runtime/command.h>
#include <runtime/device.h>

namespace toolhub::directx {
class DefaultBuffer;
class BottomAccel;
class BboxAccel;
class CommandBufferBuilder;
class ResourceStateTracker;
class Mesh;
class BottomAccel;

class TopAccel : public vstd::IOperatorNewBase {

    friend class BottomAccel;
    friend class BboxAccel;
    vstd::unique_ptr<DefaultBuffer> instBuffer;
    vstd::unique_ptr<DefaultBuffer> accelBuffer;
    vstd::HashMap<Buffer const *, size_t> resourceRefMap;
    Device *device;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc;
    vstd::vector<BottomAccel const *> accelMesh;
    vstd::vector<D3D12_RAYTRACING_INSTANCE_DESC> newInstanceDesc;
    using SetPlaceHolder = std::aligned_storage_t<12, 4>;
    vstd::vector<SetPlaceHolder> setDesc;
    void IncreRef(BottomAccel const *bf);
    void DecreRef(BottomAccel const *bf);

public:
    TopAccel(Device *device, luisa::compute::AccelBuildHint hint);
    uint Length() const { return topLevelBuildDesc.Inputs.NumDescs; }
    bool IsBufferInAccel(Buffer const *buffer) const;
    bool IsMeshInAccel(Mesh const *mesh) const;
    void Update(
        uint idx,
        BottomAccel const *accel);
    void Emplace(
        BottomAccel const *mesh,
        luisa::float4x4 transform,
        bool visible);
    void PopBack();
    DefaultBuffer const *GetAccelBuffer() const {
        return accelBuffer.get();
    }
    DefaultBuffer const *GetInstBuffer() const {
        return instBuffer.get();
    }
    size_t PreProcess(
        ResourceStateTracker &tracker,
        CommandBufferBuilder &builder,
        bool update);
    void Build(
        ResourceStateTracker &tracker,
        CommandBufferBuilder &builder,
        BufferView const& scratchBuffer);
    ~TopAccel();
};
}// namespace toolhub::directx