//
// Created by Mike Smith on 2021/9/13.
//

#include <EASTL/allocator.h>
#include <core/stl.h>

namespace luisa::detail {

void *allocator_allocate(size_t size, size_t alignment) noexcept {
    return eastl::GetDefaultAllocator()->allocate(size, alignment, 0u);
}

void allocator_deallocate(void *p, size_t) noexcept {
    eastl::GetDefaultAllocator()->deallocate(p, 0u);
}

void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept {
    return eastl::GetDefaultAllocator()->reallocate(p, size);
}

}// namespace luisa::detail
