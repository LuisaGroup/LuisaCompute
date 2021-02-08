//
// Created by Mike Smith on 2021/2/6.
//

#pragma once

#include <xxh3.h>
#include <string>
#include <string_view>

namespace luisa {

[[nodiscard]] uint32_t xxh32_hash32(const void *data, size_t size, uint32_t seed = 19980810u) noexcept;
[[nodiscard]] uint64_t xxh3_hash64(const void *data, size_t size, uint64_t seed = 19980810u) noexcept;
[[nodiscard]] XXH128_hash_t xxh3_hash128(const void *data, size_t size, uint64_t seed = 19980810u) noexcept;

struct Hash {
    
    [[nodiscard]] uint64_t operator()(std::string_view s) const noexcept {
        return xxh3_hash64(s.data(), s.size());
    }
    
    [[nodiscard]] uint64_t operator()(const std::string &s) const noexcept {
        return xxh3_hash64(s.data(), s.size());
    }
    
    template<typename T, std::enable_if_t<std::is_standard_layout_v<std::decay_t<T>>, int> = 0>
    [[nodiscard]] uint64_t operator()(T &&v) const noexcept {
        return xxh3_hash64(std::addressof(v), sizeof(std::decay_t<T>));
    }
};



}

