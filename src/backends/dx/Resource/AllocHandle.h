#pragma once
#include <Resource/GpuAllocator.h>
namespace toolhub::directx {
class AllocHandle {
public:
    uint64 allocateHandle = 0;
    GpuAllocator *allocator;
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    AllocHandle(
        GpuAllocator *allocator)
        : allocator(allocator) {}
    AllocHandle(AllocHandle const &) = delete;
    AllocHandle(AllocHandle &&rhs) : allocateHandle(rhs.allocateHandle), allocator(rhs.allocator), resource(std::move(rhs.resource)) {
        rhs.allocator = nullptr;
    }
    ~AllocHandle() {
        if (allocator) {
            resource = nullptr;
            allocator->Release(allocateHandle);
        }
    }
};
}// namespace toolhub::directx