#pragma once
#include <vstl/Common.h>
namespace vstd {
class VENGINE_DLL_COMMON PoolAllocator {
public:
	class Visitor {
	public:
		virtual void* Allocate(size_t size) = 0;
		virtual void DeAllocate(void* ptr) = 0;
		virtual ~Visitor() = default;
	};
	struct AllocateHandle {
		void* const resource;
		size_t const offset;
	};

private:
	vstd::vector<std::pair<void*, size_t>> allocatedData;
	vstd::vector<void*> data;
	size_t stride;
	size_t elementCount;
	vstd::SBO<Visitor> visitor;

public:
	template<typename Func>
    requires(decltype(visitor)::ConstructibleFunc<Func>)
		PoolAllocator(
			size_t stride,
			size_t elementCount,
			Func&& visitor) : visitor(std::forward<Func>(visitor)),
							  stride(stride),
							  elementCount(elementCount) {
		allocatedData.reserve(elementCount);
	}
	~PoolAllocator();
	AllocateHandle Allocate();
	void DeAllocate(AllocateHandle const& handle);
};
}// namespace vstd