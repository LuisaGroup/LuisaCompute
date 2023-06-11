#pragma once
#include <DXRuntime/Device.h>
namespace D3D12MA {
class Allocator;
}
namespace lc::dx {
class GpuAllocator : public vstd::IOperatorNewBase {
    D3D12MA::Allocator *allocator = nullptr;

public:
    enum class Tag : uint8_t {
        None,
        GpuAllocator
    };
    uint64 AllocateBufferHeap(
        Device *device,
        uint64_t targetSizeInBytes,
        D3D12_HEAP_TYPE heapType,
        ID3D12Heap **heap, uint64_t *offset,
        uint64 custom_pool = 0);
    uint64 AllocateTextureHeap(
        Device *device,
        size_t sizeBytes,
        ID3D12Heap **heap, uint64_t *offset,
        bool isRenderTexture,
        uint64 custom_pool = 0);
    uint64 CreatePool(D3D12_HEAP_TYPE heap_type);
    void DestroyPool(uint64 pool);
    void Release(uint64 handle);
	GpuAllocator(Device* device);
    ~GpuAllocator();
};
}// namespace lc::dx
