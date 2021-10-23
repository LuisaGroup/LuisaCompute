//
// Created by Mike Smith on 2021/9/13.
//

#ifdef LUISA_ENABLE_MIMALLOC
#include <mimalloc.h>
#endif

#include <core/platform.h>
#include <core/allocator.h>

namespace luisa::detail {

#ifdef LUISA_ENABLE_MIMALLOC
void *allocator_allocate(size_t size, size_t alignment) noexcept {
    return alignment <= 16u ? mi_malloc(size) : mi_malloc_aligned(size, alignment);
}
void allocator_deallocate(void *p, size_t alignment) noexcept {
    alignment <= 16u ? mi_free(p) : mi_free_aligned(p, alignment);
}
void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept {
    alignment <= 16u ? mi_realloc(p, size) : mi_realloc_aligned(p, size, alignment);
}

#else
void *allocator_allocate(size_t size, size_t alignment) noexcept {
    return alignment <= 16u ? malloc(size) :luisa::aligned_alloc(alignment, size);
}
void allocator_deallocate(void *p, size_t alignment) noexcept {
    alignment <= 16u ? free(p) : luisa::aligned_free(p);
}
void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept {
    auto realloc_align = [&]() {
        luisa::aligned_free(p);
        return luisa::aligned_alloc(alignment, size);
    };
    alignment <= 16u ? realloc(p, size) : realloc_align();
}
#endif

}
