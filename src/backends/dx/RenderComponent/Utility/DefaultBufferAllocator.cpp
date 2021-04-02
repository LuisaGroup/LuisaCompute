#include "DefaultBufferAllocator.h"
#include "D3D12MemoryAllocator/D3D12MemAlloc.h"
#include "../GPUResourceBase.h"
DefaultBufferAllocator::DefaultBufferAllocator(GFXDevice* device, IDXGIAdapter* adapter) : allocatedTexs(32) {
	D3D12MA::ALLOCATOR_DESC desc;
	desc.Flags = D3D12MA::ALLOCATOR_FLAGS::ALLOCATOR_FLAG_SINGLETHREADED;
	desc.pAdapter = adapter;
	desc.pAllocationCallbacks = nullptr;
	desc.pDevice = device;
	desc.PreferredBlockSize = 1;
	desc.PreferredBlockSize <<= 30;//1G
	D3D12MA::CreateAllocator(&desc, &allocator);
}
void DefaultBufferAllocator::AllocateTextureHeap(
	GFXDevice* device,
	uint64_t targetSizeInBytes,
	D3D12_HEAP_TYPE heapType,
	ID3D12Heap** heap, uint64_t* offset,
	uint64 instanceID) {
	D3D12MA::ALLOCATION_DESC desc;
	desc.HeapType = heapType;
	desc.Flags = D3D12MA::ALLOCATION_FLAGS::ALLOCATION_FLAG_NONE;
	desc.ExtraHeapFlags = D3D12_HEAP_FLAGS::D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
	desc.CustomPool = nullptr;
	D3D12_RESOURCE_ALLOCATION_INFO info;
	info.Alignment = 65536;
	info.SizeInBytes = GFXUtil::CalcPlacedOffsetAlignment(targetSizeInBytes);
	D3D12MA::Allocation* alloc;
	lockGuard lck(mtx);
	allocator->AllocateMemory(&desc, &info, &alloc);
	allocatedTexs.Insert(instanceID, alloc);
	*heap = alloc->GetHeap();
	*offset = alloc->GetOffset();
}
DefaultBufferAllocator::~DefaultBufferAllocator() {
	if (allocator) allocator->Release();
}
void DefaultBufferAllocator::ReturnBuffer(uint64 instanceID) {
	lockGuard lck(mtx);
	auto ite = allocatedTexs.Find(instanceID);
	if (!ite) {
		VEngine_Log("Empty Key!\n");
		VENGINE_EXIT;
	}
	ite.Value()->Release();
	allocatedTexs.Remove(ite);
}
