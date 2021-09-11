#include <util/vstl_config.h>
#include <ext/mimalloc/include/mimalloc.h>
#include <util/Memory.h>
void *vengine_default_malloc(size_t sz) {
    return malloc(sz);
}
void vengine_default_free(void *ptr) {
    free(ptr);
}

void *vengine_default_realloc(void *ptr, size_t size) {
    return realloc(ptr, size);
}

void *vengine_malloc(size_t size) {
    return mi_malloc(size);
}
void vengine_free(void *ptr) {
    mi_free(ptr);
}
void *operator new(size_t size) noexcept {
    return mi_malloc(size);
}
void operator delete(void *ptr) noexcept {
    mi_free(ptr);
}
void *vengine_realloc(void *ptr, size_t size) {
    return mi_realloc(ptr, size);
}