#ifndef XXH_INLINE_ALL
#define XXH_INLINE_ALL
#endif
#include <xxhash.h>

#include <core/stl/hash.h>

namespace luisa {

LC_CORE_API uint64_t hash64(const void *ptr, size_t size, uint64_t seed) noexcept {
    return XXH_INLINE_XXH3_64bits_withSeed(ptr, size, seed);
}

luisa::string Hash128::to_string() const noexcept {
    constexpr const char *hex = "0123456789abcdef";
    std::array<char, 32u> s{};
    for (auto i = 0u; i < 16u; ++i) {
        s[i * 2u + 0u] = hex[(_data[i] >> 4u) & 0x0fu];
        s[i * 2u + 1u] = hex[_data[i] & 0xfu];
    }
    return {s.data(), s.size()};
}

Hash128::Hash128(luisa::span<std::uint8_t> data) noexcept {
    assert(data.size() == 16u);
    memcpy(_data.data(), data.data(), 16u);
}

Hash128::Hash128(luisa::string_view s) noexcept {
    assert(s.size() == 32u);
    constexpr auto hex_digit = [](char c) noexcept {
        assert(c >= '0' && c <= '9' ||
               c >= 'a' && c <= 'f' ||
               c >= 'A' && c <= 'F');
        return c >= '0' && c <= '9' ? c - '0' :
               c >= 'a' && c <= 'f' ? c - 'a' + 10u :
                                      c - 'A' + 10u;
    };
    for (auto i = 0u; i < 16u; i++) {
        auto hi = hex_digit(s[i * 2u + 0u]);
        auto lo = hex_digit(s[i * 2u + 1u]);
        _data[i] = static_cast<uint8_t>(hi << 4u | lo);
    }
}

}// namespace luisa
