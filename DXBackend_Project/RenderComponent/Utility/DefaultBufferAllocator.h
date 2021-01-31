#pragma once
#include "IBufferAllocator.h"
#include "../../Common/DLL.h"
#include <mutex>
namespace D3D12MA
{
	class Allocator;
	class Allocation;
}
class DefaultBufferAllocator final: public IBufferAllocator
{
private:
	HashMap<uint64, D3D12MA::Allocation*> allocatedTexs;
	D3D12MA::Allocator* allocator;
	std::mutex mtx;
public:
	DefaultBufferAllocator(GFXDevice* device, IDXGIAdapter* adapter);
	virtual void AllocateTextureHeap(
		GFXDevice* device,
		uint64_t targetSizeInBytes,
		D3D12_HEAP_TYPE heapType,
		ID3D12Heap** heap, uint64_t* offset,
		uint64 instanceID);
	virtual void ReturnBuffer(uint64 instanceID);
	~DefaultBufferAllocator();
};