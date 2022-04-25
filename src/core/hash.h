//
// Created by Mike Smith on 2021/2/6.
//

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <span>

#include <xxhash.h>
#include <core/logging.h>
#include <core/concepts.h>

namespace luisa {

namespace detail {

/// xxh3_hash64
[[nodiscard]] inline uint64_t xxh3_hash64(const void *data, size_t size, uint64_t seed) noexcept {
    return XXH3_64bits_withSeed(data, size, seed);
}

/// Type T has method hash()
template<typename T>
concept hashable_with_hash_method = requires(T x) {
    x.hash();
};

/// Type T has method hash_code()
template<typename T>
concept hashable_with_hash_code_method = requires(T x) {
    x.hash_code();
};

}// namespace detail

[[nodiscard]] LC_CORE_API std::string_view hash_to_string(uint64_t hash) noexcept;

/// Hash 64 calculator
class LC_CORE_API Hash64 {

public:
    static constexpr auto default_seed = 19980810ull;
    using is_transparent = void;// to enable heterogeneous lookup

private:
    uint64_t _seed;

public:
    /// Constructor, set seed
    explicit constexpr Hash64(uint64_t seed = default_seed) noexcept
        : _seed{seed} {}

    /// Calculate hash, return uint64
    template<typename T>
    [[nodiscard]] uint64_t operator()(T &&s) const noexcept -> uint64_t {
        if constexpr (detail::hashable_with_hash_method<T>) {
            return (*this)(std::forward<T>(s).hash());
        } else if constexpr (detail::hashable_with_hash_code_method<T>) {
            return (*this)(std::forward<T>(s).hash_code());
        } else if constexpr (concepts::string_viewable<T>) {
            std::string_view sv{std::forward<T>(s)};
            return detail::xxh3_hash64(sv.data(), sv.size(), _seed);
        } else if constexpr (is_vector3_v<T>) {
            auto x = s;
            return detail::xxh3_hash64(&x, sizeof(vector_element_t<T>) * 3u, _seed);
        } else if constexpr (is_matrix3_v<T>) {
            auto x = luisa::make_float4x4(s);
            return (*this)(x);
        } else if constexpr (
            std::is_arithmetic_v<std::remove_cvref_t<T>> ||
            std::is_enum_v<std::remove_cvref_t<T>> ||
            is_basic_v<T>) {
            auto x = s;
            return detail::xxh3_hash64(&x, sizeof(x), _seed);
        } else {
            static_assert(always_false_v<T>);
        }
    }
};

/// Calculate hash
template<typename T>
[[nodiscard]] inline uint64_t hash64(T &&v, uint64_t seed = Hash64::default_seed) noexcept {
    return Hash64{seed}(std::forward<T>(v));
}

}// namespace luisa
