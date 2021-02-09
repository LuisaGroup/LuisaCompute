//
// Created by Mike Smith on 2021/2/6.
//

#pragma once

#include "core/logging.h"
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <span>

#include <xxh3.h>

namespace luisa {

[[nodiscard]] uint32_t xxh32_hash32(const void *data, size_t size, uint32_t seed = 19980810u) noexcept;
[[nodiscard]] uint64_t xxh3_hash64(const void *data, size_t size, uint64_t seed = 19980810u) noexcept;
[[nodiscard]] XXH128_hash_t xxh3_hash128(const void *data, size_t size, uint64_t seed = 19980810u) noexcept;

struct Hash {

    [[nodiscard]] uint64_t operator()(std::string_view s) const noexcept {
        if (s.empty()) { LUISA_ERROR_WITH_LOCATION("Computing hash for empty std::string_view."); }
        return xxh3_hash64(s.data(), s.size());
    }

    [[nodiscard]] uint64_t operator()(const std::string &s) const noexcept {
        if (s.empty()) { LUISA_ERROR_WITH_LOCATION("Computing hash for empty std::string."); }
        return xxh3_hash64(s.data(), s.size());
    }

    template<typename T, std::enable_if_t<std::is_standard_layout_v<std::remove_cvref_t<T>>, int> = 0>
    [[nodiscard]] uint64_t operator()(T &&v) const noexcept {
        return xxh3_hash64(std::addressof(v), sizeof(std::decay_t<T>));
    }

    template<typename T, std::enable_if_t<std::is_standard_layout_v<std::decay_t<T>>, int> = 0>
    [[nodiscard]] uint64_t operator()(const std::vector<T> &v) const noexcept {
        if (v.empty()) { LUISA_ERROR_WITH_LOCATION("Computing hash for empty std::vector."); }
        return xxh3_hash64(v.data(), v.size() * sizeof(T));
    }

    template<typename T, size_t extent, std::enable_if_t<std::is_standard_layout_v<std::decay_t<T>>, int> = 0>
    [[nodiscard]] uint64_t operator()(const std::span<T, extent> &v) const noexcept {
        if (v.empty()) { LUISA_ERROR_WITH_LOCATION("Computing hash for empty std::span."); }
        return xxh3_hash64(v.data(), v.size_bytes());
    }
};

}// namespace luisa
