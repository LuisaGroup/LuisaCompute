#pragma once

#include <vstl/Memory.h>

enum class VEngine_AllocType : uint8_t {
    Default,
    VEngine
};

namespace vstd {

template<VEngine_AllocType tt>
struct VAllocHandle {
    VAllocHandle() {}
    VAllocHandle(VAllocHandle const &v) : VAllocHandle() {}
    VAllocHandle(VAllocHandle &&v) : VAllocHandle() {}
    void *Malloc(size_t sz) const {
        if constexpr (tt == VEngine_AllocType::Default) {
            return vstl_default_malloc(sz);
        } else if constexpr (tt == VEngine_AllocType::VEngine) {
            return vstl_malloc(sz);
        }
    }
    void Free(void *ptr) const {
        if constexpr (tt == VEngine_AllocType::Default) {
            vstl_default_free(ptr);
        } else if constexpr (tt == VEngine_AllocType::VEngine) {
            return vstl_free(ptr);
        }
    }
};

}// namespace vstd