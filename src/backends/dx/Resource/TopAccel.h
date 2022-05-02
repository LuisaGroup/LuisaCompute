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

using luisa::compute::AccelBuildCommand;

class TopAccel : public vstd::IOperatorNewBase {

    friend class BottomAccel;
    friend class BboxAccel;
    vstd::unique_ptr<DefaultBuffer> instBuffer;
    vstd::unique_ptr<DefaultBuffer> accelBuffer;
    Device *device;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc;
    vstd::vector<AccelBuildCommand::Modification> allInstance;
    vstd::HashMap<uint64, AccelBuildCommand::Modification> setMap;
    vstd::vector<AccelBuildCommand::Modification> setDesc;
    vstd::HashMap<BottomAccel const *, vstd::HashMap<uint64>> meshMap;
    void RemoveMesh(BottomAccel const *mesh, uint64 index);
    void AddMesh(BottomAccel const *mesh, uint64 index);
    uint compactSize = 0;
    bool requireBuild = false;
    bool update = false;

public:
    bool RequireCompact() const;
    static vstd::HashMap<TopAccel *> *TopAccels();
    TopAccel(Device *device, luisa::compute::AccelUsageHint hint,
             bool allow_compact, bool allow_update);
    uint Length() const { return topLevelBuildDesc.Inputs.NumDescs; }
    void UpdateMesh(
        BottomAccel const *mesh);
    DefaultBuffer const *GetAccelBuffer() const {
        return accelBuffer.get();
    }
    DefaultBuffer const *GetInstBuffer() const {
        return instBuffer.get();
    }
    size_t PreProcess(
        ResourceStateTracker &tracker,
        CommandBufferBuilder &builder,
        uint64 size,
        vstd::span<AccelBuildCommand::Modification const> const &modifications,
        bool update);
    void Build(
        ResourceStateTracker &tracker,
        CommandBufferBuilder &builder,
        BufferView const &scratchBuffer);
    void FinalCopy(
        CommandBufferBuilder &builder,
        BufferView const &scratchBuffer);
    bool CheckAccel(
        CommandBufferBuilder &builder);
    ~TopAccel();
};
}// namespace toolhub::directx