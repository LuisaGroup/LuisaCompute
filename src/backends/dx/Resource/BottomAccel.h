#pragma once
#include <DXRuntime/Device.h>
#include <Resource/DefaultBuffer.h>
#include <Resource/Mesh.h>
#include <runtime/command.h>

namespace toolhub::directx {

class TopAccel;
struct BottomAccelData {
    D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomStruct;
};
struct BottomBuffer {
    DefaultBuffer defaultBuffer;
    uint64 compactSize;
    template<typename... Args>
        requires(std::is_constructible_v<DefaultBuffer, Args &&...>)
    BottomBuffer(Args &&...args)
        : defaultBuffer(std::forward<Args>(args)...) {}
};
class BottomAccel : public vstd::IOperatorNewBase {
    friend class TopAccel;
    vstd::unique_ptr<DefaultBuffer> accelBuffer;
    uint64 compactSize;
    Device *device;
    Mesh mesh;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS hint;
    bool update = false;
    void SyncTopAccel() const;

public:
    bool RequireCompact() const;
    Mesh const *GetMesh() const { return &mesh; }
    Buffer const *GetAccelBuffer() const {
        return accelBuffer.get();
    }
    BottomAccel(
        Device *device,
        Buffer const *vHandle, size_t vOffset, size_t vStride, size_t vCount,
        Buffer const *iHandle, size_t iOffset, size_t iCount,
        luisa::compute::AccelUsageHint hint,
        bool allow_compact, bool allow_update);
    size_t PreProcessStates(
        CommandBufferBuilder &builder,
        ResourceStateTracker &tracker,
        bool update,
        Buffer const *vHandle,
        Buffer const *iHandle, 
        BottomAccelData &bottomData);
    void UpdateStates(
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