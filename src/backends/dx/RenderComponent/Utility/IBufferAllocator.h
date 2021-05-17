#pragma once
#include <Common/GFXUtil.h>
#include <Common/MetaLib.h>
#include <RenderComponent/Utility/IGPUAllocator.h>
class GPUResourceBase;
class IBufferAllocator : public IGPUAllocator {
public:
	virtual void AllocateTextureHeap(
		GFXDevice* device,
		uint64_t targetSizeInBytes,
		D3D12_HEAP_TYPE heapType,
		ID3D12Heap** heap, uint64_t* offset,
		uint64 instanceID) = 0;
	virtual ~IBufferAllocator() {}
};