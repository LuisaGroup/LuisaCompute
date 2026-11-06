#pragma once
#include <DXRuntime/Device.h>
#include <Resource/DefaultBuffer.h>
#include <luisa/vstl/spin_mutex.h>
#include <luisa/runtime/rhi/command.h>
#include <DXRuntime/EnhancedBarrierTracker.h>
namespace lc::dx {
class CommandBufferBuilder;
class EnhancedBarrierTracker;
class TopAccel;
struct BottomAccelData {
    D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomStruct;
};
struct BottomBuffer {
    DefaultBuffer defaultBuffer;
    uint64 compactSize;
    template<typename... Args>
        requires(luisa::is_constructible_v<DefaultBuffer, Args && ...>)
    BottomBuffer(Args &&...args)
        : defaultBuffer(std::forward<Args>(args)...) {}
};

class BottomAccel;
class MeshHandle {
public:
    BottomAccel *mesh;
    TopAccel *accel;
    size_t accelIndex;
    size_t meshIndex;
    static MeshHandle *AllocateHandle();
    static void DestroyHandle(MeshHandle *handle);
};
class BottomAccel : public Resource {
    friend class TopAccel;

protected:
    // ClusterAccel reuses the TLAS back-reference machinery and publishes its
    // Cluster BLAS through the same buffer/address interface.
    vstd::unique_ptr<DefaultBuffer> accelBuffer;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS hint;
    void SyncTopAccel();

private:
    uint64 compactSize;
    bool update = false;
    vstd::fixed_vector<MeshHandle *, 2> handles;
    vstd::spin_mutex handleMtx;
    MeshHandle *AddAccelRef(TopAccel *accel, uint index);
    void RemoveAccelRef(MeshHandle *handle);

public:
    Tag get_tag() const override { return Tag::Mesh; }
    struct MeshOptions {
        Buffer const *vHandle;
        size_t vOffset;
        size_t vStride;
        size_t vSize;
        Buffer const *iHandle;
        size_t iOffset;
        size_t iSize;
    };
    struct AABBOptions {
        Buffer const *aabbBuffer;
        size_t offset;
        size_t size;
    };
    bool RequireCompact() const;
    Buffer const *GetAccelBuffer() const;
    BottomAccel(
        Device *device,
        luisa::compute::AccelOption const &option);
    size_t PreProcessStates(
        CommandBufferBuilder &builder,
        EnhancedBarrierTracker &tracker,
        bool update,
        vstd::variant<MeshOptions, AABBOptions> const &options,
        BottomAccelData &bottomData);
    void UpdateStates(
        EnhancedBarrierTracker &tracker,
        CommandBufferBuilder &builder,
        BufferView const &scratchBuffer,
        BottomAccelData &accelData) const;
    bool CheckAccel(
        CommandBufferBuilder &builder);
    void FinalCopy(
        CommandBufferBuilder &builder,
        BufferView const &scratchBuffer);
    ~BottomAccel();
};
}// namespace lc::dx
