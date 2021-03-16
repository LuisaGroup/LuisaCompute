#pragma once
#include "IBuffer.h"
#include "Utility/IBufferAllocator.h"
class ThreadCommand;
class DescriptorHeap;
struct StructuredBufferElement {
	uint64_t stride;
	uint64_t elementCount;
	static StructuredBufferElement Get(uint64_t stride, uint64_t elementCount) {
		return {stride, elementCount};
	}
};

enum StructuredBufferState {
	StructuredBufferState_UAV = GFXResourceState_UnorderedAccess,
	StructuredBufferState_Indirect = GFXResourceState_IndirectArg,
	StructuredBufferState_Read = GFXResourceState_GenericRead
};

class StructuredBuffer final : public IBuffer {
private:
	ArrayList<StructuredBufferElement> elements;
	ArrayList<uint64_t> offsets;
	GFXResourceState initState;
	mutable spin_mutex descLock;
	mutable uint srvDescIndex = -1;
	mutable uint uavDescIndex = -1;
	IBufferAllocator* allocator;
	uint64 byteSize = 0;
	void InitGlobalSRVHeap(GFXDevice* device) const;
	void InitGlobalUAVHeap(GFXDevice* device) const;
	void DisposeGlobalHeap() const;

public:
	StructuredBuffer(
		GFXDevice* device,
		StructuredBufferElement* elementsArray,
		uint elementsCount,
		bool isIndirect = false,
		bool isReadable = false,
		IBufferAllocator* allocator = nullptr);
	StructuredBuffer(
		GFXDevice* device,
		const std::initializer_list<StructuredBufferElement>& elementsArray,
		bool isIndirect = false,
		bool isReadable = false,
		IBufferAllocator* allocator = nullptr);
	StructuredBuffer(
		GFXDevice* device,
		StructuredBufferElement* elementsArray,
		uint elementsCount,
		GFXResourceState targetState,
		IBufferAllocator* allocator = nullptr);
	StructuredBuffer(
		GFXDevice* device,
		const std::initializer_list<StructuredBufferElement>& elementsArray,
		GFXResourceState targetState,
		IBufferAllocator* allocator = nullptr);
	KILL_COPY_CONSTRUCT(StructuredBuffer)
	uint GetSRVDescIndex(GFXDevice* device) const;
	uint GetUAVDescIndex(GFXDevice* device) const;
	~StructuredBuffer();
	uint64 GetByteSize() const override { return byteSize; }
	void BindSRVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const;
	void BindUAVToHeap(DescriptorHeap* targetHeap, uint64 index, GFXDevice* device) const;
	virtual GFXResourceState GetInitState() const { return initState; }
	uint64_t GetStride(uint64 index) const;
	uint64_t GetElementCount(uint64 index) const;
	GpuAddress GetAddress(uint64 element, uint64 index) const;
	uint64_t GetAddressOffset(uint64 element, uint64 index) const;
};
/*
class StructuredBufferStateBarrier final
{
	StructuredBuffer* sbuffer;
	GFXResourceState beforeState;
	GFXResourceState afterState;
	ThreadCommand* commandList;
public:
	StructuredBufferStateBarrier(ThreadCommand* commandList, StructuredBuffer* sbuffer, StructuredBufferState beforeState, StructuredBufferState afterState) :
		beforeState((GFXResourceState)beforeState), afterState((GFXResourceState)afterState), sbuffer(sbuffer),
		commandList(commandList)
	{
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
			sbuffer->GetResource(),
			this->beforeState,
			this->afterState
		));
	}
	~StructuredBufferStateBarrier()
	{
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
			sbuffer->GetResource(),
			afterState,
			beforeState
		));
	}
};*/