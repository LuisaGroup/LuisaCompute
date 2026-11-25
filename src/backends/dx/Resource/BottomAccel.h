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
    uint64 _logicalSize{0};
    uint64 _lastTriangleCount{0};
    luisa::compute::AccelOption _option;
    bool _built{false};
    bool _isTriangleMesh{false};
    bool _isCompacted{false};
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
    [[nodiscard]] luisa::compute::AccelOption const &Option() const noexcept { return _option; }
    [[nodiscard]] bool IsBuilt() const noexcept { return _built; }
    [[nodiscard]] bool IsTriangleMesh() const noexcept { return _isTriangleMesh; }
    [[nodiscard]] bool IsCompacted() const noexcept { return _isCompacted; }
    [[nodiscard]] uint64 LogicalSize() const noexcept { return _logicalSize; }
    [[nodiscard]] uint64 LastTriangleCount() const noexcept { return _lastTriangleCount; }
    [[nodiscard]] bool HasAccelBuffer() const noexcept { return static_cast<bool>(accelBuffer); }
    [[nodiscard]] DefaultBuffer const *TryGetAccelBuffer() const noexcept { return accelBuffer.get(); }
    static void EmitPostCopyBarrier(CommandBufferBuilder &builder, Buffer const *buffer);
 // Build-to-read ordering for acceleration structures.
 //
 // The enhanced-barrier tracker folds every use of a resource inside one
 // layer (the commands a command list is split into) into a single barrier
 // emitted *before* the layer, so a build followed by a ray query in the same
 // command list never gets its WRITE -> READ transition where it is needed and
 // DirectX Raytracing then answers every query of that list with a miss.
 // Builds therefore emit this barrier themselves, immediately after
 // BuildRaytracingAccelerationStructure.
    void ReplaceFromArchive(
        class CommandAllocator *alloc,
        vstd::unique_ptr<DefaultBuffer> newBuffer,
        uint64 logicalSize,
        bool isCompacted,
        uint64 triangleCount);
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
 void EmitBuildToReadBarrier(CommandBufferBuilder &builder, Buffer const *accel);
}// namespace lc::dx
