
#include <vstl/vstring.h>
#include <vstl/pool.h>
#include <mutex>
#include <vstl/functional.h>
#include <vstl/memory.h>
#include <vstl/vector.h>
#include <vstl/meta_lib.h>
//#include "BinaryLinkedAllocator.h"
void *vengine_default_malloc(size_t sz) {
    return malloc(sz);
}
void vengine_default_free(void *ptr) {
    free(ptr);
}

void *vengine_default_realloc(void *ptr, size_t size) {
    return realloc(ptr, size);
}
#ifdef EXPORT_UNITY_FUNCTION
VENGINE_UNITY_EXTERN void vengine_memcpy(void *dest, void *src, uint64 sz) {
    memcpy(dest, src, sz);
}
VENGINE_UNITY_EXTERN void vengine_memset(void *dest, byte b, uint64 sz) {
    memset(dest, b, sz);
}
VENGINE_UNITY_EXTERN void vengine_memmove(void *dest, void *src, uint64 sz) {
    memmove(dest, src, sz);
}
#endif
