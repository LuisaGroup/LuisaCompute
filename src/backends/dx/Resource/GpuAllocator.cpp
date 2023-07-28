#include <Resource/D3D12MemoryAllocator/D3D12MemAlloc.h>
#include <Resource/GpuAllocator.h>
#include <Resource/Resource.h>
#include <luisa/core/logging.h>

namespace lc::dx {
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
GpuAllocator::~GpuAllocator() {
    if (allocator)
        allocator->Release();
}
uint64 GpuAllocator::AllocateTextureHeap(
    Device *device,
    size_t sizeBytes,
    ID3D12Heap **heap, uint64_t *offset,
    bool isRenderTexture,
    uint64 custom_pool) {
    using namespace D3D12MA;
    D3D12_HEAP_FLAGS heapFlag =
        isRenderTexture ? D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES : D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
    ALLOCATION_DESC desc;
    desc.HeapType = D3D12_HEAP_TYPE_DEFAULT;
    desc.Flags = ALLOCATION_FLAGS::ALLOCATION_FLAG_STRATEGY_BEST_FIT;
    desc.ExtraHeapFlags = heapFlag;
    desc.CustomPool = reinterpret_cast<Pool *>(custom_pool);
    D3D12_RESOURCE_ALLOCATION_INFO info;
    info.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    info.SizeInBytes = sizeBytes;
    Allocation *alloc;
    allocator->AllocateMemory(&desc, &info, &alloc);
    *heap = alloc->GetHeap();
    *offset = alloc->GetOffset();
    return reinterpret_cast<uint64>(alloc);
}
uint64 GpuAllocator::AllocateBufferHeap(
    Device *device, uint64_t targetSizeInBytes,
    D3D12_HEAP_TYPE heapType, ID3D12Heap **heap,
    uint64_t *offset,
    uint64 custom_pool) {
    using namespace D3D12MA;
    ALLOCATION_DESC desc;
    desc.HeapType = heapType;
    desc.Flags = ALLOCATION_FLAGS::ALLOCATION_FLAG_STRATEGY_BEST_FIT;
    desc.ExtraHeapFlags = D3D12_HEAP_FLAGS::D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    desc.CustomPool = reinterpret_cast<Pool *>(custom_pool);
    D3D12_RESOURCE_ALLOCATION_INFO info;
    info.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    info.SizeInBytes = CalcPlacedOffsetAlignment(targetSizeInBytes);
    Allocation *alloc;
    allocator->AllocateMemory(&desc, &info, &alloc);
    *heap = alloc->GetHeap();
    *offset = alloc->GetOffset();
    return reinterpret_cast<uint64>(alloc);
}
void GpuAllocator::Release(uint64 alloc) {
    using namespace D3D12MA;
    reinterpret_cast<Allocation *>(alloc)->Release();
}
GpuAllocator::GpuAllocator(Device *device) {
    using namespace D3D12MA;
    ALLOCATOR_DESC desc;
    desc.Flags = ALLOCATOR_FLAGS::ALLOCATOR_FLAG_DEFAULT_POOLS_NOT_ZEROED;
    desc.pAdapter = device->adapter.Get();
    desc.pAllocationCallbacks = &ma_detail::gAllocateCallback.callbacks;
    desc.pDevice = device->device.Get();
    desc.PreferredBlockSize = 0;
    D3D12MA::CreateAllocator(&desc, &allocator);
}
uint64 GpuAllocator::CreatePool(D3D12_HEAP_TYPE heap_type) {
    D3D12MA::POOL_DESC pool_desc{
        .HeapProperties = {
            .Type = heap_type}};
    D3D12MA::Pool *pool;
    allocator->CreatePool(&pool_desc, &pool);
    return reinterpret_cast<uint64>(pool);
}
void GpuAllocator::DestroyPool(uint64 pool) {
    reinterpret_cast<D3D12MA::Pool *>(pool)->Release();
}
}// namespace lc::dx
