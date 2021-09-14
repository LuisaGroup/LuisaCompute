#include <core/allocator.h>
#include <vstl/config.h>
#include <vstl/Memory.h>

// TODO: add mimalloc back after fixing it on macOS
//#define VSTL_ENABLE_MIMALLOC

void *vstl_default_malloc(size_t sz) { return malloc(sz); }
void vstl_default_free(void *ptr) { free(ptr); }

void *vstl_malloc(size_t size) { return luisa::allocator<std::byte>{}.allocate(size); }
void vstl_free(void *ptr) { luisa::allocator<std::byte>{}.deallocate(static_cast<std::byte *>(ptr), 0u); }
void *operator new(size_t size) { return vstl_malloc(size); }
void operator delete(void *ptr) noexcept { vstl_free(ptr); }
