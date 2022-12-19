//
// Created by Mike on 2022/9/30.
//

#include <core/stl/memory.h>

namespace luisa {

LUISA_EXPORT_API void *allocator_allocate(size_t size, size_t alignment) noexcept {
    return eastl::GetDefaultAllocator()->allocate(size, alignment, 0u);
}

LUISA_EXPORT_API void allocator_deallocate(void *p, size_t) noexcept {
    eastl::GetDefaultAllocator()->deallocate(p, 0u);
}

LUISA_EXPORT_API void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept {
    return eastl::GetDefaultAllocator()->reallocate(p, size);
}

}// namespace luisa
