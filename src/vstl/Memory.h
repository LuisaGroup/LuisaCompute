#pragma once
#include <vstl/config.h>
#include <cstdlib>
#include <stdint.h>
#include <type_traits>
#include <vstl/ranges.h>
VENGINE_C_FUNC_COMMON void *vengine_default_malloc(size_t sz);
VENGINE_C_FUNC_COMMON void vengine_default_free(void *ptr);
VENGINE_C_FUNC_COMMON void *vengine_default_realloc(void *ptr, size_t size);
template<typename T, typename... Args>
    requires(std::is_constructible_v<T, Args &&...>)
inline T *vengine_new(Args &&...args) noexcept {
    T *tPtr = (T *)vengine_malloc(sizeof(T));
    new (tPtr) T(std::forward<Args>(args)...);
    return tPtr;
}
template<typename T, typename... Args>
    requires(std::is_constructible_v<T, Args &&...>)
inline T *vengine_new_array(size_t arrayCount, Args &&...args) noexcept {
    T *tPtr = (T *)vengine_malloc(sizeof(T) * arrayCount);
    for (auto &&i : vstd::ptr_range(tPtr, tPtr + arrayCount)) {
        new (&i) T(std::forward<Args>(args)...);
    }
    return tPtr;
}
template<typename T>
inline void vengine_delete(T *ptr) noexcept {
    vstd::destruct(ptr);
    vengine_free(ptr);
}

namespace vstd {
// Not correctly implemented and could lead to memory leaks

class ISelfPtr : public IOperatorNewBase {
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
