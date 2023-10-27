#pragma once
#include <type_traits>
namespace luisa {
/// @brief  forget a value. similar to std::mem::forget in rust.
template<class T>
void forget(T &&value) noexcept {
    struct AlignedStorage {
        alignas(T) std::byte _[sizeof(T)];
    };
    static_assert(sizeof(AlignedStorage) == sizeof(T));
    static_assert(alignof(AlignedStorage) == alignof(T));
    AlignedStorage s{};
    new (s._) T{std::forward<T>(value)};
}
}// namespace luisa