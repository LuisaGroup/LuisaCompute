#pragma once

#include <new>
#include <utility>
#include <type_traits>
#include <EASTL/internal/type_pod.h>

namespace luisa {

template<class T, class... Args>
constexpr bool is_constructible_v = eastl::is_constructible_v<T, Args...>;

template<typename T>
constexpr void destruct(T *ptr) noexcept {
    if constexpr (!std::is_void_v<T> && !std::is_trivially_destructible_v<T>) {
        ptr->~T();
    }
}

template<typename T, typename... Args>
    requires(luisa::is_constructible_v<T, Args && ...>)
constexpr T *construct(T *ptr, Args &&...args) noexcept {
    return ::new (std::launder(ptr)) T(std::forward<Args>(args)...);
}

}// namespace luisa
