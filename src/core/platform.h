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

template<typename T, typename... Args>
constexpr T *construct_at(T *p, Args &&...args) {
    return ::new (const_cast<void *>(static_cast<const volatile void *>(p)))
        T(std::forward<Args>(args)...);
}

}// namespace luisa
