#pragma once
#include <Resource/IGpuAllocator.h>
namespace toolhub::directx {
class AllocHandle {
public:
	uint64 allocateHandle = 0;
	IGpuAllocator* allocator;
	Microsoft::WRL::ComPtr<ID3D12Resource> resource;
	AllocHandle(
		IGpuAllocator* allocator)
		: allocator(allocator) {}
	~AllocHandle() {
		if (allocator) {
			resource = nullptr;
			allocator->Release(allocateHandle);
		}
	}
};
}// namespace toolhub::directx