//
// Created by Mike Smith on 2021/2/6.
//

#pragma once

#include <array>
#include <type_traits>

#include <core/stl/hash_fwd.h>
#include <core/stl/string.h>

namespace luisa {

template<typename T>
    requires std::disjunction_v<std::is_arithmetic<T>,
                                std::is_enum<T>>
struct hash<T> {
    using is_avalaunching = void;
    [[nodiscard]] constexpr uint64_t operator()(T value, uint64_t seed = hash64_default_seed) const noexcept {
        return hash64(&value, sizeof(T), seed);
    }
};

template<typename T>
    requires requires(const T v) { v.hash(); }
struct hash<T> {
    using is_avalaunching = void;
    [[nodiscard]] constexpr uint64_t operator()(const T &value) const noexcept {
        return value.hash();
    }
};

template<typename T>
[[nodiscard]] inline uint64_t hash_value(T &&value) noexcept {
    return hash<std::remove_cvref_t<T>>{}(std::forward<T>(value));
}

template<typename T>
[[nodiscard]] inline uint64_t hash_value(T &&value, uint64_t seed) noexcept {
    return hash<std::remove_cvref_t<T>>{}(std::forward<T>(value), seed);
}

[[nodiscard]] inline uint64_t hash_combine(std::initializer_list<uint64_t> values,
                                           uint64_t seed = hash64_default_seed) noexcept {
    return luisa::hash64(std::data(values), std::size(values) * sizeof(uint64_t), seed);
}

[[nodiscard]] inline uint64_t hash_combine(luisa::span<const uint64_t> values,
                                           uint64_t seed = hash64_default_seed) noexcept {
    return luisa::hash64(values.data(), values.size_bytes(), seed);
}

class LC_CORE_API Hash128 {

private:
    std::array<uint8_t, 16u> _data{};

public:
    Hash128() noexcept = default;
    explicit Hash128(span<std::uint8_t> data) noexcept;
    explicit Hash128(string_view s) noexcept;
    [[nodiscard]] auto data() noexcept { return luisa::span{_data}; }
    [[nodiscard]] auto data() const noexcept { return luisa::span{_data}; }
    [[nodiscard]] luisa::string to_string() const noexcept;
    [[nodiscard]] auto operator==(const Hash128 &rhs) const noexcept { return _data == rhs._data; }
};

}// namespace luisa
