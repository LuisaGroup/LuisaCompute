#pragma once

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <vstl/config.h>
#include <vstl/MetaLib.h>

void *vstl_default_malloc(size_t sz);
void vstl_default_free(void *ptr);
void *vstl_default_realloc(void *ptr, size_t size);

void *vstl_malloc(size_t size);
void vstl_free(void *ptr);
void *vstl_realloc(void *ptr, size_t size);

namespace vstd {

template<typename T, typename... Args>
inline T *vstl_new(Args &&...args) noexcept {
    T *tPtr = (T *)vstl_malloc(sizeof(T));// TODO: consider alignment?
    return new (tPtr) T(std::forward<Args>(args)...);
}

template<typename T>
inline void vstl_delete(T *ptr) noexcept {
    if (ptr != nullptr) {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            ((T *)ptr)->~T();
        }
        vstl_free(ptr);
    }
}

#define VSTL_OVERRIDE_OPERATOR_NEW                                \
    static void *operator new(size_t size) noexcept {             \
        return vstl_malloc(size);                                 \
    }                                                             \
    static void operator delete(void *p) noexcept {               \
        vstl_free(p);                                             \
    }                                                             \
    static void *operator new(size_t, void *place) noexcept {     \
        return place;                                             \
    }                                                             \
    static void operator delete(void *pdead, size_t) noexcept {   \
        vstl_free(pdead);                                         \
    }                                                             \
    static void *operator new[](size_t size) noexcept {           \
        return vstl_malloc(size);                                 \
    }                                                             \
    static void operator delete[](void *p) noexcept {             \
        vstl_free(p);                                             \
    }                                                             \
    static void operator delete[](void *pdead, size_t) noexcept { \
        vstl_free(pdead);                                         \
    }

class IOperatorNewBase {
public:
    virtual ~IOperatorNewBase() noexcept = default;
    VSTL_OVERRIDE_OPERATOR_NEW
};

#define VSTL_DELETE_COPY_CONSTRUCT(clsName)     \
    clsName(clsName const &) noexcept = delete; \
    clsName &operator=(clsName const &) noexcept = delete;

#define VSTL_DELETE_MOVE_CONSTRUCT(clsName) \
    clsName(clsName &&) noexcept = delete;  \
    clsName &operator=(clsName &&) noexcept = delete;

}// namespace vstd