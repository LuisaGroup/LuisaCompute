#pragma once
#include <util/AllocateType.h>
#include <util/Memory.h>
#include <util/StackAllocator.h>

template<VEngine_AllocType tt>
struct VAllocHandle {
	VAllocHandle() {}
	VAllocHandle(VAllocHandle const& v) : VAllocHandle() {}
	VAllocHandle(VAllocHandle&& v) : VAllocHandle() {}
	void* Malloc(size_t sz) const {
		if constexpr (tt == VEngine_AllocType::Default) {
			return vengine_default_malloc(sz);
		} else if constexpr (tt == VEngine_AllocType::VEngine) {
			return vengine_malloc(sz);
		}
	}
	void Free(void* ptr) const {
		if constexpr (tt == VEngine_AllocType::Default) {
			vengine_default_free(ptr);
		} else if constexpr (tt == VEngine_AllocType::VEngine) {
			return vengine_free(ptr);
		}
	}
};
template<>
struct VAllocHandle<VEngine_AllocType::Stack> {
	size_t handle;
	VAllocHandle() {
		handle = reinterpret_cast<size_t>(StackBuffer::GetCurrentPtr());
	}
	VAllocHandle(VAllocHandle const& v) : VAllocHandle() {}
	VAllocHandle(VAllocHandle&& v) : VAllocHandle() {}
	~VAllocHandle() {
		StackBuffer::stack_free(reinterpret_cast<uint8_t*>(handle));
	}
	void* Malloc(size_t sz) const { return StackBuffer::stack_malloc(sz); }
	void Free(void* ptr) const {}

private:
};
