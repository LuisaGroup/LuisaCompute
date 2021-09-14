#include <core/allocator.h>
#include <vstl/config.h>
#include <vstl/Memory.h>

// TODO: add mimalloc back after fixing it on macOS
//#define VSTL_ENABLE_MIMALLOC

void *vstl_default_malloc(size_t sz) { return malloc(sz); }
void vstl_default_free(void *ptr) { free(ptr); }

void *vstl_malloc(size_t size) { return luisa::allocate<std::byte>(size); }
void vstl_free(void *ptr) { luisa::deallocate(static_cast<std::byte *>(ptr)); }
