//#endif
#include <RenderComponent/ReadbackBuffer.h>
ReadbackBuffer::ReadbackBuffer(GFXDevice* device, uint64 elementCount, size_t stride, IBufferAllocator* allocator)
	: allocator(allocator), IBuffer(device, allocator) {
	mElementCount = elementCount;
	mStride = stride;
	auto byteSize = stride * elementCount;
	auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);

	if (allocator) {
		ID3D12Heap* heap;
		uint64 offset;
		allocator->AllocateTextureHeap(
			device, byteSize, D3D12_HEAP_TYPE_DEFAULT, &heap, &offset, GetInstanceID());
		auto buf = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
		ThrowIfFailed(device->device()->CreatePlacedResource(
			heap, offset,
			&buf,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	} else {

		auto buf = CD3DX12_RESOURCE_DESC::Buffer(byteSize);
		ThrowIfFailed(device->device()->CreateCommittedResource(
			&prop,
			D3D12_HEAP_FLAG_NONE,
			&buf,
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	}
	//	ThrowIfFailed(Resource->Map(0, nullptr, reinterpret_cast<void**>(&mMappedData)));
	// We do not need to unmap until we are done with the resource.  However, we must not write to
	// the resource while it is in use by the GPU (so we must use synchronization techniques).
}
ReadbackBuffer ::~ReadbackBuffer() {
	if (allocator) {
		allocator->Release(GetInstanceID());
	}
	if (Resource != nullptr && mMappedData != nullptr)
		Resource->Unmap(0, nullptr);
}
void ReadbackBuffer::Map() {
	D3D12_RANGE range;
	range.Begin = 0;
	range.End = mElementCount * mStride;
	ThrowIfFailed(Resource->Map(0, &range, (void**)(&mMappedData)));
}
void ReadbackBuffer::UnMap() {
	D3D12_RANGE range;
	range.Begin = 0;
	range.End = mElementCount * mStride;
	Resource->Unmap(0, nullptr);
	mMappedData = nullptr;
}
