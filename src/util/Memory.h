#pragma once

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <util/vstl_config.h>
#include <util/MetaLib.h>

VENGINE_C_FUNC_COMMON void *vengine_default_malloc(size_t sz);
VENGINE_C_FUNC_COMMON void vengine_default_free(void *ptr);
VENGINE_C_FUNC_COMMON void *vengine_default_realloc(void *ptr, size_t size);

VENGINE_C_FUNC_COMMON void *vengine_malloc(size_t size);
VENGINE_C_FUNC_COMMON void vengine_free(void *ptr);
VENGINE_C_FUNC_COMMON void *vengine_realloc(void *ptr, size_t size);

namespace vstd {

template<typename T, typename... Args>
inline T *vengine_new(Args &&...args) noexcept {
    T *tPtr = (T *)vengine_malloc(sizeof(T));
    new (tPtr) T(std::forward<Args>(args)...);
    return tPtr;
}

template<typename T>
inline void vengine_delete(T *ptr) noexcept {
    if constexpr (!std::is_trivially_destructible_v<T>)
        ((T *)ptr)->~T();
    vengine_free(ptr);
}

#define VSTL_OVERRIDE_OPERATOR_NEW                                \
    static void *operator new(size_t size) noexcept {             \
        return vengine_malloc(size);                              \
    }                                                             \
    static void operator delete(void *p) noexcept {               \
        vengine_free(p);                                          \
    }                                                             \
    static void *operator new(size_t, void *place) noexcept {     \
        return place;                                             \
    }                                                             \
    static void operator delete(void *pdead, size_t) noexcept {   \
        vengine_free(pdead);                                      \
    }                                                             \
    static void *operator new[](size_t size) noexcept {           \
        return vengine_malloc(size);                              \
    }                                                             \
    static void operator delete[](void *p) noexcept {             \
        vengine_free(p);                                          \
    }                                                             \
    static void operator delete[](void *pdead, size_t) noexcept { \
        vengine_free(pdead);                                      \
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