#pragma once
#include <DXRuntime/Device.h>
#include <EASTL/shared_ptr.h>
#include <runtime/command.h>
namespace toolhub::directx {
class DefaultBuffer;
class BottomAccel;
class CommandBufferBuilder;
class ResourceStateTracker;
class Mesh;
class BottomAccel;
class TopAccel : public vstd::IOperatorNewBase {
    struct Element {
        BottomAccel const *mesh = nullptr;
        float4x4 transform;
        uint mask;
    };
    friend class BottomAccel;
    eastl::shared_ptr<DefaultBuffer> instBuffer;
    eastl::shared_ptr<DefaultBuffer> accelBuffer;
    vstd::HashMap<Buffer const *, size_t> resourceRefMap;
    Device *device;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc;
    mutable std::mutex mtx;
    size_t capacity = 0;
    vstd::vector<Element> accelMap;
    struct CopyCommand {
        eastl::shared_ptr<DefaultBuffer> srcBuffer;
        eastl::shared_ptr<DefaultBuffer> dstBuffer;
    };
    struct UpdateCommand {
        D3D12_RAYTRACING_INSTANCE_DESC ist;
        BufferView buffer;
    };
    using Command = vstd::variant<
        CopyCommand,
        UpdateCommand>;
    vstd::vector<Command> delayCommands;
    void UpdateBottomAccel(uint idx, BottomAccel const *c);
    void IncreRef(Buffer const *bf);
    void DecreRef(Buffer const *bf);

public:
    TopAccel(Device *device, luisa::compute::AccelBuildHint hint);
    uint Length() const { return topLevelBuildDesc.Inputs.NumDescs; }
    bool IsBufferInAccel(Buffer const *buffer) const;
    bool IsMeshInAccel(Mesh const *mesh) const;
    bool Update(
        uint idx,
        BottomAccel const *accel,
        uint mask,
        float4x4 const &localToWorld);
    bool Update(
        uint idx,
        uint mask);
    bool Update(
        uint idx,
        float4x4 const &localToWorld);
    void Emplace(
        BottomAccel const *accel,
        uint mask,
        float4x4 const &localToWorld);
    void PopBack();
    DefaultBuffer const *GetAccelBuffer() const {
        return accelBuffer ? (DefaultBuffer const *)accelBuffer.get() : (DefaultBuffer const *)nullptr;
    }
    void Reserve(
        size_t newCapacity);
    void Build(
        ResourceStateTracker &tracker,
        CommandBufferBuilder &builder);
    ~TopAccel();
};
}// namespace toolhub::directx