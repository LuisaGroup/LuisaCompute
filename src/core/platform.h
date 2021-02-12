//
// Created by Mike Smith on 2021/2/11.
//

#pragma once

#include <cstdlib>
#include <memory>
#include <type_traits>
#include <limits>

#ifdef _MSC_VER

#include <intrin.h>

#define LUISA_FORCE_INLINE __forceinline inline

namespace luisa {

[[nodiscard]] inline auto aligned_alloc(size_t alignment, size_t size) noexcept {
    return _aligned_malloc(size, alignment);
}

inline void aligned_free(void *p) noexcept {
    _aligned_free(p);
}

}// namespace luisa

#else

#define LUISA_FORCE_INLINE [[gnu::always_inline, gnu::hot]] inline

namespace luisa {

inline void aligned_free(void *p) noexcept {
    free(p);
}

}// namespace luisa

#endif

namespace luisa {

template<class T, class... Args>
constexpr T *construct_at(T *p, Args &&...args) {
    return ::new (const_cast<void *>(static_cast<const volatile void *>(p)))
        T(std::forward<Args>(args)...);
}

template<typename T, std::enable_if_t<std::is_unsigned_v<T> && (sizeof(T) == 4u || sizeof(T) == 8u), int> = 0>
[[nodiscard]] constexpr auto next_pow2(T v) noexcept {
    v--;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    if constexpr (sizeof(T) == 64u) { v |= v >> 32u; }
    return v + 1u;
}

}// namespace luisa
