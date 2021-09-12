#pragma once
#include <Common/GFXUtil.h>
#include <vstl/VObject.h>
#include <vstl/MetaLib.h>
#include <RenderComponent/DescriptorHeap.h>
struct DescHeapElement
{
	DescriptorHeap* heap;
	uint offset;
	DescHeapElement() : heap(nullptr), offset(0) {}
	DescHeapElement(DescriptorHeap* const heap,
	uint const offset) : heap(heap), offset(offset) {}
	void operator=(const DescHeapElement& other)
	{
		heap = other.heap;
		offset = other.offset;
	}
};
class VENGINE_DLL_RENDERER DescHeapPool
{
private:
	vstd::vector<std::unique_ptr<DescriptorHeap>> arr;
	ArrayList<DescHeapElement> poolValue;
	uint capacity;
	uint size;
	D3D12_DESCRIPTOR_HEAP_TYPE type;
	void Add(GFXDevice* device);
public:
	DescHeapPool(uint size, uint initCapacity, D3D12_DESCRIPTOR_HEAP_TYPE type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	DescHeapElement Get(GFXDevice* device);
	void Return(const DescHeapElement& target);
	VSTL_DELETE_COPY_CONSTRUCT(DescHeapPool)
};
