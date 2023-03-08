#pragma once
#include <DXRuntime/Device.h>
#include <Resource/DefaultBuffer.h>
#include <runtime/rhi/command.h>
namespace toolhub::directx {
class CommandBufferBuilder;
class ResourceStateTracker;
class TopAccel;
struct BottomAccelData {
    D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomStruct;
};
struct BottomBuffer {
    DefaultBuffer defaultBuffer;
    uint64 compactSize;
    template<typename... Args>
        requires(std::is_constructible_v<DefaultBuffer, Args && ...>)
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
class BottomAccel : public vstd::IOperatorNewBase {
    friend class TopAccel;
    vstd::unique_ptr<DefaultBuffer> accelBuffer;
    uint64 compactSize;
    Device *device;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS hint;
    bool update = false;
    vstd::fixed_vector<MeshHandle *, 2> handles;
    vstd::spin_mutex handleMtx;
    MeshHandle *AddAccelRef(TopAccel *accel, uint index);
    void RemoveAccelRef(MeshHandle *handle);
    void SyncTopAccel();

public:
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
        size_t count;
    };
    bool RequireCompact() const;
    Buffer const *GetAccelBuffer() const {
        return accelBuffer.get();
    }
    BottomAccel(
        Device *device,
        luisa::compute::AccelOption const &option);
    size_t PreProcessStates(
        CommandBufferBuilder &builder,
        ResourceStateTracker &tracker,
        bool update,
        vstd::variant<MeshOptions, AABBOptions> const &options,
        BottomAccelData &bottomData);
    void UpdateStates(
        ResourceStateTracker &tracker,
        CommandBufferBuilder &builder,
        BufferView const &scratchBuffer,
        BottomAccelData &accelData);
    bool CheckAccel(
        CommandBufferBuilder &builder);
    void FinalCopy(
        CommandBufferBuilder &builder,
        BufferView const &scratchBuffer);
    ~BottomAccel();
};
}// namespace toolhub::directx