//#endif
#include "StructuredBuffer.h"
#include "DescriptorHeap.h"
#include "UploadBuffer.h"
#include "../PipelineComponent/ThreadCommand.h"
#include "../Singleton/Graphics.h"
D3D12_RESOURCE_STATES StructuredBuffer::GetGFXResourceState(GPUResourceState gfxState) const {
	if (usedAsMesh && gfxState == GPUResourceState_GenericRead) {
		uint v = ((uint)D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
				  | (uint)D3D12_RESOURCE_STATE_INDEX_BUFFER
				  | (uint)D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
				  | (uint)D3D12_RESOURCE_STATE_COPY_SOURCE);
		return (D3D12_RESOURCE_STATES)v;

	} else {
		return (D3D12_RESOURCE_STATES)gfxState;
	}
}
StructuredBuffer::StructuredBuffer(
	GFXDevice* device,
	StructuredBufferElement* elementsArray,
	uint elementsCount,
	bool isIndirect,
	bool isReadable,
	IBufferAllocator* allocator,
	bool usedAsMesh) : elements(elementsCount), offsets(elementsCount), allocator(allocator),
					   usedAsMesh(usedAsMesh) {
	memcpy(elements.data(), elementsArray, sizeof(StructuredBufferElement) * elementsCount);
	for (uint i = 0; i < elementsCount; ++i) {
		offsets[i] = byteSize;
		auto& ele = elements[i];
		byteSize += ele.stride * ele.elementCount;
	}
	initState = isReadable ? GPUResourceState_GenericRead : (isIndirect ? GPUResourceState_IndirectArg : GPUResourceState_UnorderedAccess);
	if (allocator) {
		ID3D12Heap* heap;
		uint64 offset;
		allocator->AllocateTextureHeap(
			device, byteSize, D3D12_HEAP_TYPE_DEFAULT, &heap, &offset, GetInstanceID());
		auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		ThrowIfFailed(device->CreatePlacedResource(
			heap, offset,
			&bufferDesc,
			GetGFXResourceState(initState),
			nullptr,
			IID_PPV_ARGS(&Resource)));
	} else {
		auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		ThrowIfFailed(device->CreateCommittedResource(
			&heap,
			D3D12_HEAP_FLAG_NONE,
			&buffer,
			GetGFXResourceState(initState),
			nullptr,
			IID_PPV_ARGS(&Resource)));
	}
}
void StructuredBuffer::BindSRVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const {
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Format = (DXGI_FORMAT)GFXFormat_Unknown;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.FirstElement = 0;
	srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
	srvDesc.Buffer.NumElements = elements[0].elementCount;
	srvDesc.Buffer.StructureByteStride = elements[0].stride;
	targetHeap->CreateSRV(device, this, &srvDesc, index);
}
void StructuredBuffer::BindUAVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const {
	D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
	uavDesc.Format = (DXGI_FORMAT)GFXFormat_Unknown;
	uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uavDesc.Buffer.FirstElement = 0;
	uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
	uavDesc.Buffer.NumElements = elements[0].elementCount;
	uavDesc.Buffer.CounterOffsetInBytes = 0;
	uavDesc.Buffer.StructureByteStride = elements[0].stride;
	targetHeap->CreateUAV(device, this, &uavDesc, index);
}
uint StructuredBuffer::GetSRVDescIndex(GFXDevice* device) const {
	InitGlobalSRVHeap(device);
	return srvDescIndex;
}
uint StructuredBuffer::GetUAVDescIndex(GFXDevice* device) const {
	InitGlobalUAVHeap(device);
	return uavDescIndex;
}
void StructuredBuffer::InitGlobalSRVHeap(GFXDevice* device) const {
	{
		std::lock_guard lck(descLock);
		if (srvDescIndex != -1) return;
		srvDescIndex = Graphics::GetDescHeapIndexFromPool();
	}
	BindSRVToHeap(Graphics::GetGlobalDescHeapNonConst(), srvDescIndex, device);
}
void StructuredBuffer::InitGlobalUAVHeap(GFXDevice* device) const {
	{
		std::lock_guard lck(descLock);
		if (uavDescIndex != -1) return;
		uavDescIndex = Graphics::GetDescHeapIndexFromPool();
	}
	BindUAVToHeap(Graphics::GetGlobalDescHeapNonConst(), uavDescIndex, device);
}
void StructuredBuffer::DisposeGlobalHeap() const {
	if (srvDescIndex != -1)
		Graphics::ReturnDescHeapIndexToPool(srvDescIndex);
	if (uavDescIndex != -1)
		Graphics::ReturnDescHeapIndexToPool(uavDescIndex);
}
StructuredBuffer::StructuredBuffer(
	GFXDevice* device,
	const std::initializer_list<StructuredBufferElement>& elementsArray,
	bool isIndirect,
	bool isReadable,
	IBufferAllocator* allocator,
	bool usedAsMesh) : elements(elementsArray.size()), offsets(elementsArray.size()), allocator(allocator), usedAsMesh(usedAsMesh) {
	memcpy(elements.data(), elementsArray.begin(), sizeof(StructuredBufferElement) * elementsArray.size());
	for (uint i = 0; i < elementsArray.size(); ++i) {
		offsets[i] = byteSize;
		auto& ele = elements[i];
		byteSize += ele.stride * ele.elementCount;
	}
	initState = isReadable ? GPUResourceState_GenericRead : (isIndirect ? GPUResourceState_IndirectArg : GPUResourceState_UnorderedAccess);
	if (allocator) {
		ID3D12Heap* heap;
		uint64 offset;
		allocator->AllocateTextureHeap(
			device, byteSize, D3D12_HEAP_TYPE_DEFAULT, &heap, &offset, GetInstanceID());
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		ThrowIfFailed(device->CreatePlacedResource(
			heap, offset,
			&buffer,
			GetGFXResourceState(initState),
			nullptr,
			IID_PPV_ARGS(&Resource)));
	} else {
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		ThrowIfFailed(device->CreateCommittedResource(
			&heap,
			D3D12_HEAP_FLAG_NONE,
			&buffer,
			GetGFXResourceState(initState),
			nullptr,
			IID_PPV_ARGS(&Resource)));
	}
}
StructuredBuffer::~StructuredBuffer() {
	if (allocator) {
		allocator->ReturnBuffer(GetInstanceID());
	}
	DisposeGlobalHeap();
}
StructuredBuffer::StructuredBuffer(
	GFXDevice* device,
	StructuredBufferElement* elementsArray,
	uint elementsCount,
	GPUResourceState targetState,
	IBufferAllocator* allocator,
	bool usedAsMesh) : elements(elementsCount), offsets(elementsCount), allocator(allocator), usedAsMesh(usedAsMesh) {
	memcpy(elements.data(), elementsArray, sizeof(StructuredBufferElement) * elementsCount);
	for (uint i = 0; i < elementsCount; ++i) {
		offsets[i] = byteSize;
		auto& ele = elements[i];
		byteSize += ele.stride * ele.elementCount;
	}
	initState = targetState;
	if (allocator) {
		ID3D12Heap* heap;
		uint64 offset;
		allocator->AllocateTextureHeap(
			device, byteSize, D3D12_HEAP_TYPE_DEFAULT, &heap, &offset, GetInstanceID());
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		ThrowIfFailed(device->CreatePlacedResource(
			heap, offset,
			&buffer,
			GetGFXResourceState(initState),
			nullptr,
			IID_PPV_ARGS(&Resource)));
	} else {
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		ThrowIfFailed(device->CreateCommittedResource(
			&heap,
			D3D12_HEAP_FLAG_NONE,
			&buffer,
			GetGFXResourceState(initState),
			nullptr,
			IID_PPV_ARGS(&Resource)));
	}
}
StructuredBuffer::StructuredBuffer(
	GFXDevice* device,
	const std::initializer_list<StructuredBufferElement>& elementsArray,
	GPUResourceState targetState,
	IBufferAllocator* allocator,
	bool usedAsMesh) : elements(elementsArray.size()), offsets(elementsArray.size()), allocator(allocator), usedAsMesh(usedAsMesh) {
	memcpy(elements.data(), elementsArray.begin(), sizeof(StructuredBufferElement) * elementsArray.size());
	for (uint i = 0; i < elementsArray.size(); ++i) {
		offsets[i] = byteSize;
		auto& ele = elements[i];
		byteSize += ele.stride * ele.elementCount;
	}
	initState = targetState;
	if (allocator) {
		ID3D12Heap* heap;
		uint64 offset;
		allocator->AllocateTextureHeap(
			device, byteSize, D3D12_HEAP_TYPE_DEFAULT, &heap, &offset, GetInstanceID());
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		ThrowIfFailed(device->CreatePlacedResource(
			heap, offset,
			&buffer,
			GetGFXResourceState(initState),
			nullptr,
			IID_PPV_ARGS(&Resource)));
	} else {
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(byteSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
		ThrowIfFailed(device->CreateCommittedResource(
			&heap,
			D3D12_HEAP_FLAG_NONE,
			&buffer,
			GetGFXResourceState(initState),
			nullptr,
			IID_PPV_ARGS(&Resource)));
	}
}
uint64_t StructuredBuffer::GetStride(uint64 index) const {
	return elements[index].stride;
}
uint64_t StructuredBuffer::GetElementCount(uint64 index) const {
	return elements[index].elementCount;
}
GpuAddress StructuredBuffer::GetAddress(uint64 element, uint64 index) const {
#ifdef NDEBUG
	auto& ele = elements[element];
	return {Resource->GetGPUVirtualAddress() + offsets[element] + ele.stride * index};
#else
	if (element >= elements.size()) {
		throw "Element Out of Range Exception";
	}
	auto& ele = elements[element];
	if (index >= ele.elementCount) {
		throw "Index Out of Range Exception";
	}
	return {Resource->GetGPUVirtualAddress() + offsets[element] + ele.stride * index};
#endif
}
uint64_t StructuredBuffer::GetAddressOffset(uint64 element, uint64 index) const {
#ifdef NDEBUG
	auto& ele = elements[element];
	return offsets[element] + ele.stride * index;
#else
	if (element >= elements.size()) {
		throw "Element Out of Range Exception";
	}
	auto& ele = elements[element];
	if (index >= ele.elementCount) {
		throw "Index Out of Range Exception";
	}
	return offsets[element] + ele.stride * index;
#endif
}
