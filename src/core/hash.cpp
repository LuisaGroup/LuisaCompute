//
// Created by Mike Smith on 2021/2/6.
//

#include <fmt/format.h>
#include <core/hash.h>

#include <span>

namespace luisa {

uint32_t xxh32_hash32(const void *data, size_t size, uint32_t seed) noexcept {
    return XXH_INLINE_XXH32(data, size, seed);
}

uint64_t xxh3_hash64(const void *data, size_t size, uint64_t seed) noexcept {
    return XXH_INLINE_XXH3_64bits_withSeed(data, size, seed);
}

XXH128_hash_t xxh3_hash128(const void *data, size_t size, uint64_t seed) noexcept {
    return XXH_INLINE_XXH3_128bits_withSeed(data, size, seed);
}

std::string_view hash_to_string(uint64_t hash) noexcept {
    static thread_local std::array<char, 16u> temp;
    fmt::format_to_n(temp.data(), temp.size(), "{:016X}", hash);
    return std::string_view{temp.data(), temp.size()};
}

std::string_view hash_to_string(uint32_t hash) noexcept {
    static thread_local std::array<char, 8u> temp;
    fmt::format_to_n(temp.data(), temp.size(), "{:08X}", hash);
    return std::string_view{temp.data(), temp.size()};
}

std::string_view hash_to_string(XXH_INLINE_XXH128_hash_t hash) noexcept {
    static thread_local std::array<char, 32u> temp;
    fmt::format_to_n(temp.data(), temp.size(), "{:016X}{:016X}", hash.high64, hash.low64);
    return std::string_view{temp.data(), temp.size()};
}

}// namespace luisa
