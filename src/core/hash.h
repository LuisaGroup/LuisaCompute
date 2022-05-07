//
// Created by Mike Smith on 2021/2/6.
//

#pragma once

#include <memory>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <span>
#include <cstdint>

#include <core/dll_export.h>
#include <core/concepts.h>

namespace luisa {

namespace detail {

/// Murmur2 hash. Implementation from PBRT-v4.
[[nodiscard]] inline uint64_t murmur2_hash64(const void *data, size_t len, uint64_t seed) noexcept {
    auto key = static_cast<const std::byte *>(data);
    constexpr auto m = 0xc6a4a7935bd1e995ull;
    constexpr auto r = 47u;
    auto h = seed ^ (len * m);
    auto end = key + 8u * (len / 8u);
    while (key != end) {
        uint64_t k;
        memcpy(&k, key, sizeof(uint64_t));
        key += 8u;
        k *= m;
        k ^= k >> r;
        k *= m;
        h ^= k;
        h *= m;
    }
    switch (len & 7u) {
        case 7: h ^= uint64_t(key[6]) << 48u; [[fallthrough]];
        case 6: h ^= uint64_t(key[5]) << 40u; [[fallthrough]];
        case 5: h ^= uint64_t(key[4]) << 32u; [[fallthrough]];
        case 4: h ^= uint64_t(key[3]) << 24u; [[fallthrough]];
        case 3: h ^= uint64_t(key[2]) << 16u; [[fallthrough]];
        case 2: h ^= uint64_t(key[1]) << 8u; [[fallthrough]];
        case 1: h = (h ^ uint64_t(key[0])) * m;
    };
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h;
}

}// namespace detail

/// Hash 64 calculator
class LC_CORE_API Hash64 {

public:
    static constexpr auto default_seed = 19980810ull;
    using is_transparent = void;// to enable heterogeneous lookup

public:
    [[nodiscard]] uint64_t operator()(const char *s, uint64_t seed) const noexcept {
        return detail::murmur2_hash64(s, strlen(s), seed);
    }
    [[nodiscard]] uint64_t operator()(std::string_view s, uint64_t seed) const noexcept {
        return detail::murmur2_hash64(s.data(), s.size(), seed);
    }
    template<typename C, typename CT, typename Alloc>
    [[nodiscard]] uint64_t operator()(const std::basic_string<C, CT, Alloc> &s, uint64_t seed) const noexcept {
        return detail::murmur2_hash64(s.data(), s.size() * sizeof(C), seed);
    }
    [[nodiscard]] uint64_t operator()(int3 v, uint64_t seed) const noexcept {
        auto vv = make_int4(v, 0);
        return detail::murmur2_hash64(&vv, sizeof(vv), seed);
    }
    [[nodiscard]] uint64_t operator()(uint3 v, uint64_t seed) const noexcept {
        auto vv = make_uint4(v, 0u);
        return detail::murmur2_hash64(&vv, sizeof(vv), seed);
    }
    [[nodiscard]] uint64_t operator()(float3 v, uint64_t seed) const noexcept {
        auto vv = make_float4(v, 0.f);
        return detail::murmur2_hash64(&vv, sizeof(vv), seed);
    }
    [[nodiscard]] uint64_t operator()(bool3 v, uint64_t seed) const noexcept {
        auto vv = make_bool4(v, false);
        return detail::murmur2_hash64(&vv, sizeof(vv), seed);
    }
    [[nodiscard]] uint64_t operator()(float3x3 m, uint64_t seed) const noexcept {
        auto mm = make_float4x4(m);
        return detail::murmur2_hash64(&mm, sizeof(mm), seed);
    }
    /// Calculate hash, return uint64
    template<typename T>
    [[nodiscard]] uint64_t operator()(T s, uint64_t seed) const noexcept {
        static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T> || is_basic_v<T>);
        return detail::murmur2_hash64(&s, sizeof(s), seed);
    }
    /// Calculate hash, return uint64
    template<typename T>
    [[nodiscard]] uint64_t operator()(T &&s) const noexcept {
        return (*this)(std::forward<T>(s), default_seed);
    }
};

struct PointerHash {
    [[nodiscard]] uint64_t operator()(const void *p) const noexcept {
        auto x = reinterpret_cast<uint64_t>(p);
        return detail::murmur2_hash64(&x, sizeof(x), Hash64::default_seed);
    }
};

/// Calculate hash
template<typename T>
[[nodiscard]] inline uint64_t hash64(T &&v, uint64_t seed = Hash64::default_seed) noexcept {
    return Hash64{}(std::forward<T>(v), seed);
}

}// namespace luisa
