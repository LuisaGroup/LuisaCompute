#pragma once
#include <vstl/vstlconfig.h>
#include <RenderComponent/Utility/IBufferAllocator.h>
#include <mutex>
namespace D3D12MA {
class Allocator;
class Allocation;
}// namespace D3D12MA
class VENGINE_DLL_RENDERER DefaultBufferAllocator final : public IBufferAllocator {
private:
	HashMap<uint64, D3D12MA::Allocation*> allocatedBuffers;
	D3D12MA::Allocator* allocator;
	std::mutex mtx;

public:
	DefaultBufferAllocator(GFXDevice* device);
	void AllocateTextureHeap(
		GFXDevice* device,
		uint64_t targetSizeInBytes,
		D3D12_HEAP_TYPE heapType,
		ID3D12Heap** heap, uint64_t* offset,
		uint64 instanceID) override;
	void Release(uint64 instanceID) override;
	~DefaultBufferAllocator();
};
