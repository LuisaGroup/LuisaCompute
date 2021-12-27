#pragma once
#include <vstl/config.h>
#include <cstdlib>
#include <stdint.h>
#include <type_traits>
#include <vstl/MetaLib.h>
VENGINE_C_FUNC_COMMON void *vengine_default_malloc(size_t sz);
VENGINE_C_FUNC_COMMON void vengine_default_free(void *ptr);
VENGINE_C_FUNC_COMMON void *vengine_default_realloc(void *ptr, size_t size);
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
#define DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW           \
    static void *operator new(                          \
        size_t size) noexcept {                         \
        return vengine_malloc(size);                    \
    }                                                   \
    static void *operator new(                          \
        size_t,                                         \
        void *place) noexcept {                         \
        return place;                                   \
    }                                                   \
    static void *operator new[](                        \
        size_t size) noexcept {                         \
        return vengine_malloc(size);                    \
    }                                                   \
    static void *operator new(                          \
        size_t size, const std::nothrow_t &) noexcept { \
        return vengine_malloc(size);                    \
    }                                                   \
    static void *operator new(                          \
        size_t,                                         \
        void *place, const std::nothrow_t &) noexcept { \
        return place;                                   \
    }                                                   \
    static void *operator new[](                        \
        size_t size, const std::nothrow_t &) noexcept { \
        return vengine_malloc(size);                    \
    }                                                   \
    static void operator delete(                        \
        void *pdead) noexcept {                         \
        vengine_free(pdead);                            \
    }                                                   \
    static void operator delete[](                      \
        void *pdead) noexcept {                         \
        vengine_free(pdead);                            \
    }
namespace vstd {
// Not correctly implemented and could lead to memory leaks
class IOperatorNewBase {
public:
    DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
};
class ISelfPtr {
public:
    virtual ~ISelfPtr() = default;
    virtual void *SelfPtr() = 0;
};
#define VSTD_SELF_PTR \
    void *SelfPtr() override { return this; }
template<typename T>
struct DynamicObject {
    template<typename... Args>
    static constexpr T *CreateObject(
        funcPtr_t<T *(
            funcPtr_t<void *(size_t)> operatorNew,
            Args...)>
            createFunc,
        Args... args) {
        return createFunc(
            T::operator new,
            std::forward<Args>(args)...);
    }
};
}// namespace vstd
#define KILL_COPY_CONSTRUCT(clsName)   \
    clsName(clsName const &) = delete; \
    clsName &operator=(clsName const &) = delete;

#define KILL_MOVE_CONSTRUCT(clsName) \
    clsName(clsName &&) = delete;    \
    clsName &operator=(clsName &&) = delete;
