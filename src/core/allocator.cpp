//
// Created by Mike Smith on 2021/9/13.
//

#include <EASTL/allocator.h>
#include <core/platform.h>
#include <core/allocator.h>

namespace luisa::detail {

void *allocator_allocate(size_t size, size_t alignment) noexcept {
    return eastl::GetDefaultAllocator()->allocate(size, alignment, 0u);
}

void allocator_deallocate(void *p, size_t alignment) noexcept {
    eastl::GetDefaultAllocator()->deallocate(p, )
}

void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept {
    auto aligned_realloc = [p, size, alignment] {
        luisa::aligned_free(p);
        return luisa::aligned_alloc(alignment, size);
    };
    return alignment <= 16u ? realloc(p, size) : aligned_realloc();
}

}// namespace luisa::detail
