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

[[nodiscard]] inline uint64_t xxh3_hash64(const void *data, size_t size, uint64_t seed = 19980810u) noexcept {
    return XXH3_64bits_withSeed(data, size, seed);
}

[[nodiscard]] std::string_view hash_to_string(uint64_t hash) noexcept;

struct Hash {

    template<typename T>
    requires requires { std::declval<T>().hash(); }
    [[nodiscard]] auto operator()(T &&s) const noexcept {
        return std::forward<T>(s).hash();
    }

    template<concepts::string_viewable T>
    [[nodiscard]] auto operator()(T &&s) const noexcept {
        std::string_view sv{std::forward<T>(s)};
        if (sv.empty()) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Computing hash for empty std::string."); }
        return xxh3_hash64(sv.data(), sv.size());
    }

    template<concepts::span_convertible T>
    requires(!concepts::string_viewable<T>) [[nodiscard]] auto
    operator()(T &&s) const noexcept {
        std::span v{std::forward<T>(s)};
        if (v.empty()) [[unlikely]] { LUISA_ERROR_WITH_LOCATION("Computing hash for empty std::span."); }
        return xxh3_hash64(v.data(), v.size_bytes());
    }

    template<typename T, std::enable_if_t<std::is_standard_layout_v<std::remove_cvref_t<T>>, int> = 0>
    [[nodiscard]] uint64_t operator()(T &&v) const noexcept {
        return xxh3_hash64(std::addressof(v), sizeof(std::remove_cvref_t<T>));
    }
};

}// namespace luisa
