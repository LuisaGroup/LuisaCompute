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

[[nodiscard]] inline auto xxh3_hash64(const void *data, size_t size, uint64_t seed) noexcept {
    return XXH3_64bits_withSeed(data, size, seed);
}

}// namespace detail

[[nodiscard]] std::string_view hash_to_string(uint64_t hash) noexcept;

class Hash64 {

public:
    static constexpr auto default_seed = 19980810ull;

private:
    uint64_t _seed;

public:
    explicit constexpr Hash64(uint64_t seed = default_seed) noexcept
        : _seed{seed} {}

    template<typename T>
    [[nodiscard]] auto operator()(T &&s) const noexcept {
        if constexpr (requires { std::forward<T>(s).hash(); }) {
            return std::forward<T>(s).hash();
        } else if constexpr (requires { std::forward<T>(s).hash_code(); }) {
            return std::forward<T>(s).hash_code();
        } else if constexpr (concepts::string_viewable<T>) {
            std::string_view sv{std::forward<T>(s)};
            return detail::xxh3_hash64(sv.data(), sv.size(), _seed);
        } else if constexpr (concepts::span_convertible<T>) {
            std::span v{std::forward<T>(s)};
            return detail::xxh3_hash64(v.data(), v.size_bytes(), _seed);
        } else if constexpr (concepts::iterable<T>) {
            auto seed = _seed;
            for (auto &&a : std::forward<T>(s)) { seed = Hash64{seed}(a); }
            return seed;
        } else {
            using V = std::remove_cvref_t<T>;
            static_assert(std::is_standard_layout_v<V>);
            auto v = s;
            return detail::xxh3_hash64(std::addressof(v), sizeof(V), _seed);
        }
    }
};

template<typename T>
[[nodiscard]] inline auto hash64(T &&v, uint64_t seed = Hash64::default_seed) noexcept {
    return Hash64{seed}(std::forward<T>(v));
}

}// namespace luisa
