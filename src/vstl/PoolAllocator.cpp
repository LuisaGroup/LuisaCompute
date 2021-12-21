#pragma vengine_package vengine_dll
#include <vstl/PoolAllocator.h>
namespace vstd {
PoolAllocator::AllocateHandle PoolAllocator::Allocate() {
	if (allocatedData.empty()) {
		auto ptr = data.emplace_back(visitor->Allocate(stride * elementCount));
		allocatedData.push_back_func(
			[&](size_t i) -> std::pair<void*, size_t> {
				return {ptr, i};
			},
			elementCount);
	}
	auto result = allocatedData.erase_last();
	return {
		result.first,
		result.second};
}
void PoolAllocator::DeAllocate(AllocateHandle const& handle) {
	allocatedData.emplace_back(handle.resource, handle.offset);
}
PoolAllocator::~PoolAllocator() {
	for (auto&& i : data) {
		visitor->DeAllocate(i);
	}
}

}// namespace vstd