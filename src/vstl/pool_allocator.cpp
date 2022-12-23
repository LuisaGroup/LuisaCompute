#include <vstl/pool_allocator.h>
namespace vstd {
PoolAllocator::AllocateHandle PoolAllocator::Allocate() {
    if (allocatedData.empty()) {
        auto ptr = data.emplace_back(visitor->Allocate(stride * elementCount));
        push_back_func(
            allocatedData,
            elementCount, [&](size_t i) -> std::pair<void *, size_t> {
                return {ptr, i};
            });
    }
    auto result = allocatedData.back();
    allocatedData.pop_back();
    return {
        result.first,
        result.second};
}
void PoolAllocator::DeAllocate(AllocateHandle const &handle) {
    allocatedData.emplace_back(handle.resource, handle.offset);
}
PoolAllocator::~PoolAllocator() {
    for (auto &&i : data) {
        visitor->DeAllocate(i);
    }
}

}// namespace vstd
