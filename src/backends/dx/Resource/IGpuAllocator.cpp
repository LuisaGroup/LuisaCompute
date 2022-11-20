
//#endif
#include <Resource/IGpuAllocator.h>
#include <Resource/Resource.h>
#include <Resource/D3D12MemoryAllocator/D3D12MemAlloc.h>
namespace toolhub::directx {
namespace ma_detail {
class AllocateCallback {
public:
    D3D12MA::ALLOCATION_CALLBACKS callbacks;
    AllocateCallback() {
        callbacks.pAllocate = [](size_t Size, size_t Alignment,
                                 void *pPrivateData) -> void * {
            return luisa::detail::allocator_allocate(Size, Alignment);
        };
        callbacks.pFree = [](void *pMemory, void *) -> void {
            luisa::detail::allocator_deallocate(pMemory, 0);
        };
    }
};
static AllocateCallback gAllocateCallback;
}// namespace ma_detail
class DefaultAllocator final : public IGpuAllocator {
private:
    D3D12MA::Allocator *allocator = nullptr;

public:
    VSTD_SELF_PTR
    ~DefaultAllocator() {
        if (allocator)
            allocator->Release();
    }
    uint64 AllocateTextureHeap(
        Device *device,
        GFXFormat format,
        uint32_t width,
        uint32_t height,
        uint32_t depthSlice,
        TextureDimension dimension,
        uint32_t mipCount,
        ID3D12Heap **heap, uint64_t *offset,
        bool isRenderTexture) override {
        D3D12_HEAP_FLAGS heapFlag = isRenderTexture ? D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES : D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
        D3D12MA::ALLOCATION_DESC desc;
        desc.HeapType = D3D12_HEAP_TYPE_DEFAULT;
        desc.Flags = D3D12MA::ALLOCATION_FLAGS::ALLOCATION_FLAG_STRATEGY_BEST_FIT;
        desc.ExtraHeapFlags = heapFlag;
        desc.CustomPool = nullptr;
        D3D12_RESOURCE_ALLOCATION_INFO info;
        info.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        info.SizeInBytes = Resource::GetTextureSize(
            device,
            width,
            height,
            format,
            dimension,
            depthSlice,
            mipCount);
        D3D12MA::Allocation *alloc;
        allocator->AllocateMemory(&desc, &info, &alloc);
        *heap = alloc->GetHeap();
        *offset = alloc->GetOffset();
        return reinterpret_cast<uint64>(alloc);
    }
    uint64 AllocateBufferHeap(
        Device *device,
        uint64_t targetSizeInBytes,
        D3D12_HEAP_TYPE heapType,
        ID3D12Heap **heap, uint64_t *offset) override {
        D3D12MA::ALLOCATION_DESC desc;
        desc.HeapType = heapType;
        desc.Flags = D3D12MA::ALLOCATION_FLAGS::ALLOCATION_FLAG_STRATEGY_BEST_FIT;
        desc.ExtraHeapFlags = D3D12_HEAP_FLAGS::D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
        desc.CustomPool = nullptr;
        D3D12_RESOURCE_ALLOCATION_INFO info;
        info.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        info.SizeInBytes = CalcPlacedOffsetAlignment(targetSizeInBytes);
        D3D12MA::Allocation *alloc;
        allocator->AllocateMemory(&desc, &info, &alloc);
        *heap = alloc->GetHeap();
        *offset = alloc->GetOffset();
        return reinterpret_cast<uint64>(alloc);
    }
    void Release(uint64 alloc) override {
        reinterpret_cast<D3D12MA::Allocation *>(alloc)->Release();
    }
    DefaultAllocator(
        Device *device) {

        D3D12MA::ALLOCATOR_DESC desc;
        desc.Flags = D3D12MA::ALLOCATOR_FLAGS::ALLOCATOR_FLAG_DEFAULT_POOLS_NOT_ZEROED;
        desc.pAdapter = device->adapter.Get();
        desc.pAllocationCallbacks = &ma_detail::gAllocateCallback.callbacks;
        desc.pDevice = device->device.Get();
        desc.PreferredBlockSize = 1;
        desc.PreferredBlockSize <<= 30;//1G
        D3D12MA::CreateAllocator(&desc, &allocator);
    }
};
IGpuAllocator *IGpuAllocator::CreateAllocator(
    Device *device,
    Tag tag) {
    switch (tag) {
        case Tag::DefaultAllocator:
            return new DefaultAllocator(device);
        default: return nullptr;
    }
}
}// namespace toolhub::directx