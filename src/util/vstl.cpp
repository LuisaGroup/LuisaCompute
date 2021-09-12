#include <util/vstl_config.h>
#include <util/Memory.h>

// TODO: add mimalloc back after fixing it on macOS
//#define MIMALLOC_ENABLED

void *vstl_default_malloc(size_t sz) {
    return malloc(sz);
}
void vstl_default_free(void *ptr) {
    free(ptr);
}

void *vstl_default_realloc(void *ptr, size_t size) {
    return realloc(ptr, size);
}

#ifdef MIMALLOC_ENABLED
#include <ext/mimalloc/include/mimalloc.h>
void *vstl_malloc(size_t size) { return mi_malloc(size); }
void vstl_free(void *ptr) { mi_free(ptr); }
void *operator new(size_t size) { return mi_malloc(size); }
void operator delete(void *ptr) noexcept { mi_free(ptr); }
void *vstl_realloc(void *ptr, size_t size) { return mi_realloc(ptr, size); }
#else
void *vstl_malloc(size_t size) { return malloc(size); }
void *vstl_realloc(void *ptr, size_t size) { return realloc(ptr, size); }
void vstl_free(void *ptr) { free(ptr); }
void *operator new(size_t size) { return malloc(size); }
void operator delete(void *ptr) noexcept { free(ptr); }
#endif