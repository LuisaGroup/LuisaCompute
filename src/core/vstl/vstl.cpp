#include <core/vstl/Memory.h>
void *vengine_default_malloc(size_t sz) {
    return malloc(sz);
}
void vengine_default_free(void *ptr) {
    free(ptr);
}

void *vengine_malloc(size_t size) {
    return malloc(size);
}
void vengine_free(void *ptr) {
    free(ptr);
}

void *vengine_realloc(void *ptr, size_t size) {
    return realloc(ptr, size);
}