
#include <Resource/DefaultBuffer.h>
namespace toolhub::directx {
DefaultBuffer::DefaultBuffer(
	Device* device,
	uint64 byteSize,
	IGpuAllocator* allocator,
	D3D12_RESOURCE_STATES initState)
	: Buffer(device),
	  initState(initState),
	  byteSize(byteSize),
	  allocHandle(allocator) {
	if (allocator) {
		ID3D12Heap* heap;
		uint64 offset;
		allocHandle.allocateHandle = allocHandle.allocator->AllocateBufferHeap(
			device, byteSize, D3D12_HEAP_TYPE_DEFAULT, &heap, &offset);
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		ThrowIfFailed(device->device->CreatePlacedResource(
			heap, offset,
			&buffer,
			initState,
			nullptr,
			IID_PPV_ARGS(&allocHandle.resource)));
	} else {
		auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		ThrowIfFailed(device->device->CreateCommittedResource(
			&prop,
			D3D12_HEAP_FLAG_NONE,
			&buffer,
			initState,
			nullptr,
			IID_PPV_ARGS(&allocHandle.resource)));
	}
}
vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> DefaultBuffer::GetColorSrvDesc(bool isRaw) const {
	return GetColorSrvDesc(0, byteSize, isRaw);
}
vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> DefaultBuffer::GetColorUavDesc(bool isRaw) const {
	return GetColorUavDesc(0, byteSize, isRaw);
}
vstd::optional<D3D12_SHADER_RESOURCE_VIEW_DESC> DefaultBuffer::GetColorSrvDesc(uint64 offset, uint64 byteSize, bool isRaw) const {
	D3D12_SHADER_RESOURCE_VIEW_DESC res;
	res.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	res.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	if (isRaw) {
		res.Format = DXGI_FORMAT_R32_TYPELESS;
		assert((offset & 15) == 0);
		res.Buffer.FirstElement = offset / 4;
		res.Buffer.NumElements = byteSize / 4;
		res.Buffer.StructureByteStride = 0;
		res.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
	} else {
		res.Format = DXGI_FORMAT_UNKNOWN;
		assert((offset & 3) == 0);
		res.Buffer.FirstElement = offset / 4;
		res.Buffer.NumElements = byteSize / 4;
		res.Buffer.StructureByteStride = 4;
		res.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
	}
	return res;
}
vstd::optional<D3D12_UNORDERED_ACCESS_VIEW_DESC> DefaultBuffer::GetColorUavDesc(uint64 offset, uint64 byteSize, bool isRaw) const {
	D3D12_UNORDERED_ACCESS_VIEW_DESC res;
	res.Format = DXGI_FORMAT_R32_TYPELESS;
	res.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	res.Buffer.CounterOffsetInBytes = 0;
	if (isRaw) {
		assert((offset & 15) == 0);
		res.Buffer.FirstElement = offset / 4;
		res.Buffer.NumElements = byteSize / 4;
		res.Buffer.StructureByteStride = 0;
		res.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
	} else {
		assert((offset & 3) == 0);
		res.Buffer.FirstElement = offset / 4;
		res.Buffer.NumElements = byteSize / 4;
		res.Buffer.StructureByteStride = 4;
		res.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
	}
	return res;
}
DefaultBuffer::~DefaultBuffer() {
}
}// namespace toolhub::directx