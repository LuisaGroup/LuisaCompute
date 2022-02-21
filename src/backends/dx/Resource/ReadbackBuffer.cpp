#pragma vengine_package vengine_directx
#include <Resource/ReadbackBuffer.h>
namespace toolhub::directx {
ReadbackBuffer::ReadbackBuffer(
	Device* device,
	uint64 byteSize,
	IGpuAllocator* allocator)
	: Buffer(device),
	  byteSize(byteSize),
	  allocHandle(allocator) {
	if (allocator) {
		ID3D12Heap* heap;
		uint64 offset;
		allocHandle.allocateHandle = allocator->AllocateBufferHeap(
			device, byteSize, D3D12_HEAP_TYPE_READBACK, &heap, &offset);
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
		ThrowIfFailed(device->device->CreatePlacedResource(
			heap, offset,
			&buffer,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(&allocHandle.resource)));
	} else {
		auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
		ThrowIfFailed(device->device->CreateCommittedResource(
			&prop,
			D3D12_HEAP_FLAG_NONE,
			&buffer,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(&allocHandle.resource)));
	}
}
ReadbackBuffer::~ReadbackBuffer() {
}
void ReadbackBuffer::CopyData(
	uint64 offset, 
	vstd::span<vbyte> data) const {
	void* mapPtr;
	D3D12_RANGE range;
	range.Begin = offset;
    range.End = offset + data.size();
	ThrowIfFailed(allocHandle.resource->Map(0, &range, (void**)(&mapPtr)));
	auto d = vstd::create_disposer([&] { allocHandle.resource->Unmap(0, nullptr); });
    memcpy(data.data(), reinterpret_cast<vbyte const *>(mapPtr) + offset, data.size());
}

}// namespace toolhub::directx