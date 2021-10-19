#pragma once
#include <vstl/AllocateType.h>
#include <vstl/Memory.h>

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