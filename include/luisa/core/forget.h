#pragma once

#include <type_traits>

namespace luisa {

/// @brief  forget a value. similar to std::mem::forget in rust.
template<typename T>
    requires std::is_rvalue_reference_v<T &&>
void forget(T &&value) noexcept {
    struct AlignedStorage {
        alignas(T) std::byte _[sizeof(T)];
    };
    static_assert(sizeof(AlignedStorage) == sizeof(T));
    static_assert(alignof(AlignedStorage) == alignof(T));
    AlignedStorage s{};
    new (s._) T{std::move(value)};
}

}// namespace luisa
