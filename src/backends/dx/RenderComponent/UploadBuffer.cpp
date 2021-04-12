//#endif
#include <RenderComponent/UploadBuffer.h>
#include <RenderComponent/Utility/IBufferAllocator.h>
#include <RenderComponent/DescriptorHeap.h>
#include <Singleton/Graphics.h>
UploadBuffer::UploadBuffer(GFXDevice* device, uint64 elementCount, bool isConstantBuffer, uint64_t stride, IBufferAllocator* allocator) : allocator(allocator) {
	mIsConstantBuffer = isConstantBuffer;
	// Constant buffer elements need to be multiples of 256 bytes.
	// This is because the hardware can only view constant data
	// at m*256 byte offsets and of n*256 byte lengths.
	// typedef struct D3D12_CONSTANT_BUFFER_VIEW_DESC {
	// UINT64 OffsetInBytes; // multiple of 256
	// uint   SizeInBytes;   // multiple of 256
	// } D3D12_CONSTANT_BUFFER_VIEW_DESC;
	mElementCount = elementCount;
	if (isConstantBuffer)
		mElementByteSize = GFXUtil::CalcConstantBufferByteSize(stride);
	else
		mElementByteSize = stride;
	mStride = stride;
	if (Resource) {
		Resource->Unmap(0, nullptr);
		Resource = nullptr;
	}
	uint64 size = (uint64)mElementByteSize * (uint64)elementCount;
	if (allocator) {
		ID3D12Heap* heap;
		uint64 offset;
		allocator->AllocateTextureHeap(
			device, size, D3D12_HEAP_TYPE_UPLOAD, &heap, &offset, GetInstanceID());
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size);
		ThrowIfFailed(device->CreatePlacedResource(
			heap, offset,
			&buffer,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	} else {
		auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size);
		ThrowIfFailed(device->CreateCommittedResource(
			&prop,
			D3D12_HEAP_FLAG_NONE,
			&buffer,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	}
	ThrowIfFailed(Resource->Map(0, nullptr, reinterpret_cast<void**>(&mMappedData)));
	// We do not need to unmap until we are done with the resource.  However, we must not write to
	// the resource while it is in use by the GPU (so we must use synchronization techniques).
}
uint UploadBuffer::GetSRVDescIndex(GFXDevice* device) const {
	InitGlobalDesc(device);
	return srvDescIndex;
}
void UploadBuffer::BindSRVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const {
	D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
	srvDesc.Format = (DXGI_FORMAT)GFXFormat_Unknown;
	srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.FirstElement = 0;
	srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
	srvDesc.Buffer.NumElements = mElementCount;
	srvDesc.Buffer.StructureByteStride = mStride;
	targetHeap->CreateSRV(device, this, &srvDesc, index);
}
void UploadBuffer::Create(GFXDevice* device, uint64 elementCount, bool isConstantBuffer, uint64_t stride, IBufferAllocator* allocator) {
	ReturnGlobalDesc();
	if (this->allocator) {
		this->allocator->ReturnBuffer(GetInstanceID());
	}
	this->allocator = allocator;
	mIsConstantBuffer = isConstantBuffer;
	// Constant buffer elements need to be multiples of 256 bytes.
	// This is because the hardware can only view constant data
	// at m*256 byte offsets and of n*256 byte lengths.
	// typedef struct D3D12_CONSTANT_BUFFER_VIEW_DESC {
	// UINT64 OffsetInBytes; // multiple of 256
	// uint   SizeInBytes;   // multiple of 256
	// } D3D12_CONSTANT_BUFFER_VIEW_DESC;
	mElementCount = elementCount;
	if (isConstantBuffer)
		mElementByteSize = GFXUtil::CalcConstantBufferByteSize(stride);
	else
		mElementByteSize = stride;
	mStride = stride;
	if (Resource) {
		Resource->Unmap(0, nullptr);
		Resource = nullptr;
	}
	uint64 size = (uint64)mElementByteSize * (uint64)elementCount;
	if (allocator) {
		ID3D12Heap* heap;
		uint64 offset;
		allocator->AllocateTextureHeap(
			device, size, D3D12_HEAP_TYPE_UPLOAD, &heap, &offset, GetInstanceID());
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size);
		ThrowIfFailed(device->CreatePlacedResource(
			heap, offset,
			&buffer,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	} else {
		auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size);
		auto prop = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
		ThrowIfFailed(device->CreateCommittedResource(
			&prop,
			D3D12_HEAP_FLAG_NONE,
			&buffer,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&Resource)));
	}
	ThrowIfFailed(Resource->Map(0, nullptr, reinterpret_cast<void**>(&mMappedData)));
}
UploadBuffer::~UploadBuffer() {
	ReturnGlobalDesc();
	if (allocator) {
		allocator->ReturnBuffer(GetInstanceID());
	}
	if (Resource != nullptr)
		Resource->Unmap(0, nullptr);
}
void UploadBuffer::InitGlobalDesc(GFXDevice* device) const {
	{
		std::lock_guard lck(srvDescLock);
		if (srvDescIndex != -1) return;
		srvDescIndex = Graphics::GetDescHeapIndexFromPool();
	}
	BindSRVToHeap(Graphics::GetGlobalDescHeapNonConst(), srvDescIndex, device);
}
void UploadBuffer::ReturnGlobalDesc() {
	if (srvDescIndex == -1) return;
	Graphics::ReturnDescHeapIndexToPool(srvDescIndex);
	srvDescIndex = -1;
}

void UploadBuffer::CopyData(uint64 elementIndex, const void* data) const {
	char* dataPos = (char*)mMappedData;
	uint64_t offset = (uint64_t)elementIndex * mElementByteSize;
	dataPos += offset;
	memcpy(dataPos, data, mStride);
}
void UploadBuffer::CopyData(uint64 elementIndex, const void* data, uint64 byteSize) const {
	char* dataPos = (char*)mMappedData;
	uint64_t offset = (uint64_t)elementIndex * mElementByteSize;
	dataPos += offset;
	memcpy(dataPos, data, byteSize);
}
void UploadBuffer::CopyData(uint64 elementIndex, uint64 bufferByteOffset, const void* data, uint64 byteSize) const {
	char* dataPos = (char*)mMappedData;
	uint64_t offset = (uint64_t)elementIndex * mElementByteSize;
	dataPos += offset + bufferByteOffset;
	memcpy(dataPos, data, byteSize);
}
void UploadBuffer::CopyDatas(uint64 startElementIndex, uint64 elementCount, const void* data) const {
	char* dataPos = (char*)mMappedData;
	uint64_t offset = startElementIndex * (uint64_t)mElementByteSize;
	dataPos += offset;
	memcpy(dataPos, data, (elementCount - 1) * (uint64_t)mElementByteSize + mStride);
}
GpuAddress UploadBuffer::GetAddress(uint64 elementCount) const {
	return {Resource->GetGPUVirtualAddress() + GetAddressOffset(elementCount)};
}
uint64_t UploadBuffer::GetAddressOffset(uint64 elementCount) const {
	return elementCount * (uint64_t)mElementByteSize;
}
void* UploadBuffer::GetMappedDataPtr(uint64 element) const {
	char* dataPos = (char*)mMappedData;
	uint64_t offset = element * (uint64_t)mElementByteSize;
	dataPos += offset;
	return dataPos;
}