#pragma once
#include <DXRuntime/Device.h>
#include <EASTL/shared_ptr.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/device.h>
#include <Resource/Resource.h>
using namespace luisa::compute;
namespace lc::dx {
class DefaultBuffer;
class BottomAccel;
class BboxAccel;
class CommandBufferBuilder;
class ResourceStateTracker;
class Mesh;
class BottomAccel;
class MeshHandle;
class TopAccel : public Resource {

    friend class BottomAccel;
    friend class BboxAccel;
    vstd::unique_ptr<DefaultBuffer> instBuffer;
    vstd::unique_ptr<DefaultBuffer> accelBuffer;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc;
    struct Instance {
        MeshHandle *handle = nullptr;
    };
    vstd::vector<Instance> allInstance;
    vstd::unordered_map<uint64, MeshHandle *> setMap;
    vstd::vector<AccelBuildCommand::Modification> setDesc;
    void SetMesh(BottomAccel *mesh, uint64 index);
    uint compactSize = 0;
    bool requireBuild = false;
    bool update = false;
    void UpdateMesh(
        MeshHandle *handle);
    bool GenerateNewBuffer(
        ResourceStateTracker &tracker,
        CommandBufferBuilder &builder,
        vstd::unique_ptr<DefaultBuffer> &oldBuffer, size_t newSize, bool needCopy, D3D12_RESOURCE_STATES state);
    struct Buffers {
        vstd::unique_ptr<DefaultBuffer> v;
        Buffers(vstd::unique_ptr<DefaultBuffer> &&a)
            : v(std::move(a)) {}
        void operator()() const {}
    };

public:
    bool RequireCompact() const;
    TopAccel(Device *device, luisa::compute::AccelOption const &option);
    uint Length() const { return topLevelBuildDesc.Inputs.NumDescs; }
    Tag GetTag() const override { return Tag::Accel; }

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
    void PreProcessInst(
        ResourceStateTracker &tracker,
        CommandBufferBuilder &builder,
        uint64 size,
        vstd::span<AccelBuildCommand::Modification const> const &modifications);
    void Build(
        ResourceStateTracker &tracker,
        CommandBufferBuilder &builder,
        vstd::span<AccelBuildCommand::Modification const> const &modifications,
        BufferView const *scratchBuffer);
    void FinalCopy(
        CommandBufferBuilder &builder,
        BufferView const &scratchBuffer);
    bool CheckAccel(
        CommandBufferBuilder &builder);
    ~TopAccel();
};
}// namespace lc::dx
