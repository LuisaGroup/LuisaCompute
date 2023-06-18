#include "SparseAllocator.h"
namespace lc::dx {
SparseAllocator::SparseAllocator(Device *device, bool isTex, uint tileCapacity) : device(device), tileCapacity(tileCapacity), isTex(isTex) {
}
SparseAllocator::~SparseAllocator() {
}
double time = 0;
auto SparseAllocator::AllocateHeap() -> PtrMap::iterator {
    ComPtr<ID3D12Heap> heap;
    D3D12_HEAP_DESC heapDesc{
        .SizeInBytes = tileCapacity * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES,
        .Properties = {
            .Type = D3D12_HEAP_TYPE_DEFAULT},
        .Flags = isTex ? (D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES) : (D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS)};
    heapDesc.Flags |= D3D12_HEAP_FLAG_CREATE_NOT_ZEROED;
    ThrowIfFailed(device->device->CreateHeap(&heapDesc, IID_PPV_ARGS(heap.GetAddressOf())));
    auto heapPtr = heap.Get();
    auto fullIte = fullMaps.emplace(std::move(heap));
    Heap *ptr = &fullIte.value();
    auto ite = waitingMaps.try_emplace(heapPtr, ptr);
    ptr->waitingPools.push_back_uninitialized(tileCapacity);
    for (uint i = 0; i < tileCapacity; ++i) {
        ptr->waitingPools[i] = i;
    }
    ptr->capacity = tileCapacity;
    tileCapacity *= 2;
    return std::move(ite.first);
}
void SparseAllocator::AllocateTiles(
    uint tileCount,
    vstd::FuncRef<void(ID3D12Heap *heap, vstd::span<uint const> offsets)> const &allocate) {
    uint tileIndex = 0;
    while (true) {
        decltype(waitingMaps)::iterator heap;
        if (waitingMaps.empty()) {
            heap = AllocateHeap();
        } else {
            heap = waitingMaps.begin();
        }
        auto heapPtr = heap->first;
        auto &heapVec = heap->second->waitingPools;
        if (tileIndex + heapVec.size() <= tileCount) {
            allocate(heapPtr, heapVec);
            tileIndex += heapVec.size();
            heapVec.clear();
            waitingMaps.erase(heap);
            if (tileIndex >= tileCount) return;
        }
        // pool still left
        else {
            auto size = tileCount - tileIndex;
            auto leftedSize = heapVec.size() - size;
            allocate(heapPtr, {heapVec.data() + leftedSize, size});
            heapVec.resize(leftedSize);
            return;
        }
    }
}
void SparseAllocator::Deallocate(ID3D12Heap *heap, vstd::span<uint const> offsets) {
    auto ite = fullMaps.find(heap);
    Heap *ptr = &ite.value();
    waitingMaps.try_emplace(heap, ptr);
    vstd::push_back_all(
        ptr->waitingPools,
        offsets);
}
void SparseAllocator::ClearAllocate() {
    waitingMaps.clear();
    for (auto &&i : fullMaps) {
        auto &heap = i.second;
        heap.waitingPools.clear();
        heap.waitingPools.push_back_uninitialized(tileCapacity);
        for (uint i = 0; i < tileCapacity; ++i) {
            heap.waitingPools[i] = i;
        }
        waitingMaps.try_emplace(i.first.Get(), &heap);
    }
}
}// namespace lc::dx
