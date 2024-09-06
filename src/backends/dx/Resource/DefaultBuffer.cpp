#include <Resource/DefaultBuffer.h>
namespace lc::dx {
DefaultBuffer::DefaultBuffer(
    Device *device,
    uint64 byteSize,
    GpuAllocator *allocator,
    D3D12_RESOURCE_STATES initState,
    bool shared_adaptor,
    char const* name)
    : Buffer(device),
      allocHandle(allocator),
      byteSize(byteSize),
      initState(initState) {
    if (allocator) {
        ID3D12Heap *heap;
        uint64 offset;
        allocHandle.allocateHandle = allocHandle.allocator->AllocateBufferHeap(
            device, name ? name : "default buffer", byteSize, D3D12_HEAP_TYPE_DEFAULT, &heap, &offset,
            shared_adaptor ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE);
        auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ThrowIfFailed(device->device->CreatePlacedResource(
            heap, offset,
            &buffer,
            initState,
            nullptr,
            IID_PPV_ARGS(&allocHandle.resource)));
        _is_heap_resource = true;
    } else {
        auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ThrowIfFailed(device->device->CreateCommittedResource(
            &prop,
            shared_adaptor ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE,
            &buffer,
            initState,
            nullptr,
            IID_PPV_ARGS(&allocHandle.resource)));
        _is_heap_resource = false;
    }
}
DefaultBuffer::DefaultBuffer(
    Device *device,
    uint64 byteSize,
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES initState)
    : Buffer(device),
      allocHandle(nullptr),
      byteSize(byteSize),
      initState(initState) {
    allocHandle.resource = resource;
}
vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> DefaultBuffer::GetColorSrvDesc(bool isRaw) const {
    return GetColorSrvDesc(0, byteSize, isRaw);
}
vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> DefaultBuffer::GetColorUavDesc(bool isRaw) const {
    return GetColorUavDesc(0, byteSize, isRaw);
}
vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> DefaultBuffer::GetColorSrvDesc(uint64 offset, uint64 byteSize, bool isRaw) const {
    return GetColorSrvDescBase(offset, byteSize, isRaw);
}
vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> DefaultBuffer::GetColorUavDesc(uint64 offset, uint64 byteSize, bool isRaw) const {
    return GetColorUavDescBase(offset, byteSize, isRaw);
}
DefaultBuffer::~DefaultBuffer() {
}
}// namespace lc::dx
