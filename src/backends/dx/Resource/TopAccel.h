#pragma once
#include <DXRuntime/Device.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/device.h>
#include <Resource/Resource.h>
#include <DXRuntime/EnhancedBarrierTracker.h>
using namespace luisa::compute;
namespace lc::dx {
class DefaultBuffer;
class BottomAccel;
class BboxAccel;
class CommandBufferBuilder;
class EnhancedBarrierTracker;
class Mesh;
class BottomAccel;
class MeshHandle;
struct alignas(16) PackedModifier {
    float affine[12];
    uint64_t primitive;
    uint index : 24;
    uint vis_mask : 8;
    uint user_id : 24;
    uint flags : 8;
};
class TopAccel : public Resource {

    friend class BottomAccel;
    friend class BboxAccel;
    vstd::unique_ptr<DefaultBuffer> instBuffer;
    vstd::unique_ptr<DefaultBuffer> accelBuffer;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc;
    struct Instance {
        MeshHandle *handle = nullptr;
        // Pending BLAS-address refresh queued by a BLAS recreate
        // (SyncTopAccel). Stores the stable BottomAccel (never the pooled
        // MeshHandle) so a destroyed or recycled handle can never leave a
        // dangling entry behind. Lives in the slot itself because the key
        // space (instance index) is exactly the dense [0, allInstance.size())
        // range of the contiguous instance array: queue/consume/erase are
        // O(1) array operations instead of hash-map read/writes, and shrinking
        // the vector drops entries of removed slots automatically.
        BottomAccel *refresh = nullptr;
    };
    vstd::vector<Instance> allInstance;
    void ResizeAllInstance(size_t size);
    // Number of slots carrying a non-null `refresh`. Mirrors the size of the
    // former refresh hash map so ProcessSetMap can skip its scan in O(1) when
    // nothing is pending. Kept in sync by QueueRefresh/DropRefresh and the
    // process/resize paths.
    size_t pendingRefreshCount = 0;
    // Queues a refresh of the BLAS device address of one instance slot.
    void QueueRefresh(size_t index, BottomAccel *mesh) noexcept;
    // Drops the pending refresh of one slot if it references `mesh` (used by
    // ~BottomAccel so a destroyed BLAS never leaks a refresh entry behind).
    void DropRefresh(size_t index, BottomAccel *mesh) noexcept;
    vstd::vector<PackedModifier> setDesc;
    void SetMesh(BottomAccel *mesh, uint64 index);
    uint compactSize = 0;
    bool requireBuild = false;
    bool update = false;
    void UpdateMesh(
        MeshHandle *handle);
    bool GenerateNewBuffer(
        char const *name,
        EnhancedBarrierTracker &tracker,
        CommandBufferBuilder &builder,
        vstd::unique_ptr<DefaultBuffer> &oldBuffer, size_t newSize, bool needCopy, D3D12_RESOURCE_STATES state);
    void InitSetDesc(vstd::span<AccelBuildCommand::Modification const> const &modifications);
    void ProcessSetDesc(EnhancedBarrierTracker &tracker);
    void ProcessSetMap();

public:
    bool RequireCompact() const;
    TopAccel(Device *device, luisa::compute::AccelOption const &option);
    uint Length() const { return topLevelBuildDesc.Inputs.NumDescs; }
    Tag get_tag() const override { return Tag::Accel; }

    DefaultBuffer const *GetAccelBuffer() const {
        return accelBuffer.get();
    }
    DefaultBuffer const *GetInstBuffer() const {
        return instBuffer.get();
    }
    size_t PreProcess(
        EnhancedBarrierTracker &tracker,
        CommandBufferBuilder &builder,
        uint64 size,
        vstd::span<AccelBuildCommand::Modification const> const &modifications,
        bool update);
    void PreProcessInst(
        EnhancedBarrierTracker &tracker,
        CommandBufferBuilder &builder,
        uint64 size,
        vstd::span<AccelBuildCommand::Modification const> const &modifications);
    void Build(
        EnhancedBarrierTracker &tracker,
        CommandBufferBuilder &builder,
        BufferView const *scratchBuffer);
    void FinalCopy(
        CommandBufferBuilder &builder,
        BufferView const &scratchBuffer);
    bool CheckAccel(
        CommandBufferBuilder &builder);
    ~TopAccel();
};
}// namespace lc::dx
